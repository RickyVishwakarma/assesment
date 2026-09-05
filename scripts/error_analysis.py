"""Collects and auto-classifies extraction errors across approaches.

Run:
    python scripts/error_analysis.py --split gold_synth --approach all

The assignment asks for at least 20 analysed errors with causes. Hand-labelling causes
invites hindsight bias, so most of the taxonomy is assigned from evidence the pipeline
already has rather than from opinion:

``OCR_CORRUPTION``
    The gold value does not occur in the extracted text at all. No extractive system
    could have got this right -- the information was destroyed before extraction. This
    is a *ceiling* violation, not a model failure, and separating it out is essential:
    without it, OCR damage is silently charged to the model.
``HALLUCINATION``
    The predicted value occurs nowhere in the source text. The system invented it.
``TRUNCATION`` / ``OVER_CAPTURE``
    The prediction is a strict sub/superstring of the gold value -- the right region was
    found but the boundary was wrong.
``FORMAT_NORMALISATION``
    Prediction and gold agree once normalised but differ literally. Only ever counted
    against the strict metric, never the headline one.
``LABEL_CONFUSION``
    The predicted value is the gold value of a *different* field on the same document --
    the classic semi-structured failure, where a value was read from the wrong column.
``MISSED_ENTIRELY``
    Gold present, nothing predicted.
``WRONG_VALUE``
    Everything else: a real, substantive extraction error.

Each error also carries its layout, condition and language, so the report can say which
causes concentrate where instead of listing anecdotes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.cli import APPROACH_NAMES, build_extractor  # noqa: E402
from docintel.eval.runner import load_gold  # noqa: E402
from docintel.normalize import appears_in_text, normalize_field  # noqa: E402
from docintel.schema import DocType, OcrDocument  # noqa: E402
from docintel.gen.render import TEMPLATES  # noqa: E402


def classify(
    field: str,
    gold: str | None,
    pred: str | None,
    text: str,
    other_gold: dict[str, str | None],
) -> str:
    """Assign a cause from evidence, not from intuition."""
    norm_gold = normalize_field(field, gold) if gold else None
    norm_pred = normalize_field(field, pred) if pred else None

    if gold and gold not in text:
        # Check the normalised form too before blaming OCR.
        flat = "".join(c for c in text.lower() if c.isalnum())
        flat_gold = "".join(c for c in (norm_gold or "") if c.isalnum())
        if not flat_gold or flat_gold not in flat:
            return "OCR_CORRUPTION"

    if pred and not gold:
        return "SPURIOUS_EXTRACTION"
    if gold and not pred:
        return "MISSED_ENTIRELY"
    if not gold and not pred:
        return "OK"
    if norm_gold == norm_pred:
        return "FORMAT_NORMALISATION" if gold.strip() != pred.strip() else "OK"

    # Use the evaluator's own detector rather than a second, subtly different rule.
    #
    # An earlier version compared the *normalised* prediction against the text, which
    # flagged every correctly-extracted date as a hallucination ("5/9/25" normalises to
    # "2025-05-09", which of course does not appear on the page). That put 37 phantom
    # hallucinations against the rules approach in this report while the evaluator
    # reported zero for the same run. One definition, one number.
    if pred and not appears_in_text(pred, text, field):
        return "HALLUCINATION"

    if norm_gold and norm_pred:
        if norm_pred in norm_gold:
            return "TRUNCATION"
        if norm_gold in norm_pred:
            return "OVER_CAPTURE"

    for other_field, other_value in other_gold.items():
        if other_field == field or not other_value:
            continue
        if normalize_field(other_field, other_value) == norm_pred:
            return "LABEL_CONFUSION"

    return "WRONG_VALUE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--gold-dir", dest="gold_dir", default="data/gold", type=Path)
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--approach", nargs="+", default=["all"])
    parser.add_argument("--out", default="reports", type=Path)
    args = parser.parse_args()

    names = list(APPROACH_NAMES) if "all" in args.approach else args.approach
    records = load_gold(args.corpus, args.split, args.gold_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in names:
        try:
            extractor = build_extractor(name)
        except Exception as exc:
            print(f"  [{name}] unavailable: {exc}", file=sys.stderr)
            continue
        print(f"  analysing {name} ...", file=sys.stderr, flush=True)

        for record in records:
            doc = OcrDocument.model_validate_json(
                (args.corpus / record["text"]).read_text(encoding="utf-8")
            )
            try:
                prediction = extractor.extract(doc)
            except Exception as exc:
                print(f"    {record['doc_id']}: {exc}", file=sys.stderr)
                continue

            for field, gold in record["truth"].items():
                pred = prediction.value(field)
                cause = classify(field, gold, pred, doc.text, record["truth"])
                if cause == "OK":
                    continue
                rows.append({
                    "approach": name,
                    "doc_id": record["doc_id"],
                    "field": field,
                    "gold": gold,
                    "pred": pred,
                    "cause": cause,
                    "layout": TEMPLATES[record["template_id"]]["layout"],
                    "template_id": record["template_id"],
                    "condition": record["condition"],
                    "lang": record["lang"],
                    "doc_type": record["doc_type"],
                })

    csv_path = args.out / f"errors_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["approach"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- summaries --------------------------------------------------------------
    by_approach_cause: dict[str, Counter] = defaultdict(Counter)
    by_cause_layout: dict[str, Counter] = defaultdict(Counter)
    by_approach_field: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_approach_cause[row["approach"]][row["cause"]] += 1
        by_cause_layout[row["cause"]][row["layout"]] += 1
        by_approach_field[row["approach"]][row["field"]] += 1

    summary = {
        "split": args.split,
        "total_errors": len(rows),
        "by_approach_cause": {k: dict(v) for k, v in by_approach_cause.items()},
        "by_cause_layout": {k: dict(v) for k, v in by_cause_layout.items()},
        "worst_fields": {
            k: dict(v.most_common(6)) for k, v in by_approach_field.items()
        },
    }
    (args.out / f"error_summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {len(rows)} errors -> {csv_path}")
    for approach, counter in sorted(by_approach_cause.items()):
        total = sum(counter.values())
        print(f"\n{approach}  ({total} errors)")
        for cause, count in counter.most_common():
            print(f"    {cause:22s} {count:4d}  ({count / total:5.1%})")


if __name__ == "__main__":
    main()
