"""Demo: run every available approach over documents nothing has ever touched.

    python scripts/demo_unseen.py
    python scripts/demo_unseen.py --limit 3 --show-text

The ``demo`` split is held out twice over. Its documents use templates 11-14, which the
small model never trained on and the rules dictionary never saw wording from; and unlike
``gold_synth`` they are never scored during development, so no decision anywhere in this
project was tuned against them. That makes this the closest thing to "a document arrives
tomorrow" that the corpus can offer.

Scores printed here are measured against generator truth rather than human-verified gold,
which is stated on the output so the numbers are not mistaken for the headline result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.normalize import values_match  # noqa: E402
from docintel.schema import OcrDocument  # noqa: E402


def load_approaches(names: list[str], quiet: bool = False) -> dict:
    """Load whichever approaches are actually available on this machine."""
    available = {}
    for name in names:
        try:
            if name == "nlp":
                from docintel.approaches.nlp import NlpExtractor

                available[name] = NlpExtractor()
            elif name == "small_model":
                from docintel.approaches.small_model import SmallModelExtractor

                extractor = SmallModelExtractor()
                extractor.token_model  # force load; raises if untrained
                available[name] = extractor
            elif name == "llm_local":
                from docintel.approaches.llm_local import LocalLlmExtractor, ollama_available

                if not ollama_available():
                    raise RuntimeError("ollama is not running")
                available[name] = LocalLlmExtractor()
            elif name == "llm_frontier":
                from docintel.approaches.llm_frontier import FrontierLlmExtractor

                available[name] = FrontierLlmExtractor()
        except Exception as exc:
            if not quiet:
                print(f"  (skipping {name}: {type(exc).__name__}: {exc})", file=sys.stderr)
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--split", default="demo")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument(
        "--approach", nargs="+",
        default=["nlp", "small_model", "llm_local", "llm_frontier"],
    )
    args = parser.parse_args()

    corpus = args.corpus
    records = [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == args.split
    ]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"No documents in split {args.split}", file=sys.stderr)
        return 1

    print(f"Loading approaches ...")
    approaches = load_approaches(args.approach)

    # The frontier tier is a *cached* reference annotated only on the gold split. Run on
    # any other split it would silently return nothing for every field, which on screen
    # is indistinguishable from a model that failed at the task. Drop it explicitly and
    # say why, rather than printing a column of blanks that reads as a bad score.
    frontier = approaches.get("llm_frontier")
    if frontier is not None:
        covered = sum(1 for r in records if r["doc_id"] in getattr(frontier, "_cache", {}))
        if covered == 0:
            approaches.pop("llm_frontier")
            print(f"  (skipping llm_frontier: no cached annotations for split "
                  f"'{args.split}' — this tier is annotated on gold_synth only)",
                  file=sys.stderr)
        elif covered < len(records):
            print(f"  (llm_frontier covers {covered}/{len(records)} of these documents)",
                  file=sys.stderr)

    if not approaches:
        print("No approaches available.", file=sys.stderr)
        return 1
    print(f"Running {len(approaches)} approach(es) over {len(records)} unseen "
          f"documents from split '{args.split}'.\n")

    totals = {name: [0, 0, 0.0] for name in approaches}  # hits, total, seconds

    for record in records:
        doc = OcrDocument.model_validate_json(
            (corpus / record["text"]).read_text(encoding="utf-8")
        )
        truth = {k: v for k, v in record["truth"].items() if v}

        print("=" * 96)
        print(f"{record['doc_id']}")
        print(f"  type={record['doc_type']}  template={record['template_id']} "
              f"(held out)  condition={record['condition']}  lang={record['lang']}")
        if args.show_text:
            print("  " + "-" * 92)
            for line in doc.text.splitlines()[:14]:
                print(f"  | {line}")
            print("  " + "-" * 92)

        results = {}
        for name, extractor in approaches.items():
            started = time.perf_counter()
            try:
                results[name] = extractor.extract(doc)
            except Exception as exc:
                print(f"  {name} FAILED: {exc}")
                continue
            totals[name][2] += time.perf_counter() - started

        header = f"  {'field':<26}{'gold':<30}" + "".join(
            f"{n[:14]:<20}" for n in results
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        for field in sorted(truth):
            gold = truth[field]
            row = f"  {field:<26}{gold[:28]:<30}"
            for name, result in results.items():
                got = result.fields.get(field)
                value = got.value if got else None
                ok = values_match(field, gold, value)
                totals[name][0] += ok
                totals[name][1] += 1
                mark = "OK " if ok else "-- "
                row += f"{mark}{(value or '')[:16]:<17}"
            print(row)

        row = f"  {'DOCUMENT TYPE':<26}{record['doc_type'][:28]:<30}"
        for name, result in results.items():
            predicted = result.doc_type.value if result.doc_type else "?"
            mark = "OK " if predicted == record["doc_type"] else "-- "
            row += f"{mark}{predicted[:16]:<17}"
        print(row)
        print()

    print("=" * 96)
    print(f"SUMMARY over {len(records)} previously unseen documents")
    print("  scored against generator truth (not the human-verified gold set)\n")
    print(f"  {'approach':<18}{'field accuracy':<20}{'ms/doc':>10}")
    print("  " + "-" * 46)
    for name, (hits, total, seconds) in totals.items():
        if not total:
            continue
        print(f"  {name:<18}{hits}/{total} = {hits / total:6.1%}      "
              f"{seconds / len(records) * 1000:8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
