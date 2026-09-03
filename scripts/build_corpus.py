"""Builds the synthetic corpus: PDFs, extracted text, and the split manifest.

Run:
    python scripts/build_corpus.py --out data/corpus --seed 20250902

Split design (the anti-inflation control):

    train / val   -> templates 1-10   (the model may see these layouts)
    gold_synth    -> templates 11-14  (held out; never trained on)
    demo          -> templates 11-14  (held out; also never scored, for the live demo)

Because train and gold use structurally different layouts, a model that merely memorised
template geometry will visibly fall over on gold. That gap is a headline result, not an
embarrassment — it is the number that tells you whether the system generalises.

The ``scanned`` condition needs an OCR engine. If none is installed the builder still
runs, produces every document in the ``clean`` condition, and records the shortfall in
the manifest rather than pretending the corpus is what it is not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.gen.document import sample_document  # noqa: E402
from docintel.gen.render import (  # noqa: E402
    HELDOUT_TEMPLATES,
    TRAIN_TEMPLATES,
    render_pdf,
)
from docintel.ocr import ocr_available, read_clean, read_scanned  # noqa: E402
from docintel.schema import DOC_TYPES  # noqa: E402

#: split -> (number of documents, templates it may draw from)
SPLITS: dict[str, tuple[int, list[int]]] = {
    "train": (1000, TRAIN_TEMPLATES),
    "val": (200, TRAIN_TEMPLATES),
    "gold_synth": (90, HELDOUT_TEMPLATES),
    "demo": (10, HELDOUT_TEMPLATES),
}

SPANISH_RATE = 0.08
SCANNED_RATE = 0.50
SEVERITIES = ["light", "medium", "medium", "heavy"]


def build(out_dir: Path, seed: int, scanned_rate: float, limit: int | None) -> None:
    rng = random.Random(seed)
    pdf_dir = out_dir / "pdfs"
    text_dir = out_dir / "text"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    have_ocr = ocr_available()
    if not have_ocr and scanned_rate > 0:
        print(
            "WARNING: no OCR engine installed -> building every document in the "
            "'clean' condition. Install rapidocr-onnxruntime and re-run to add the "
            "scanned condition.",
            file=sys.stderr,
        )

    manifest_path = out_dir / "manifest.jsonl"
    records: list[dict] = []
    stats: Counter = Counter()

    for split, (count, templates) in SPLITS.items():
        if limit:
            count = min(count, limit)
        for i in range(count):
            doc_type = DOC_TYPES[i % len(DOC_TYPES)]
            template_id = templates[i % len(templates)]
            lang = "es" if rng.random() < SPANISH_RATE else "en"
            doc_id = f"{split}-{i:04d}-{doc_type.value}-t{template_id}"

            doc = sample_document(doc_id, doc_type, template_id, rng, lang=lang)
            pdf_path = pdf_dir / f"{doc_id}.pdf"
            render_pdf(doc, str(pdf_path), rng)

            want_scanned = have_ocr and rng.random() < scanned_rate
            severity = rng.choice(SEVERITIES) if want_scanned else None
            if want_scanned:
                ocr_doc = read_scanned(
                    str(pdf_path), doc_id, lang=lang, severity=severity,
                    seed=hash(doc_id) & 0xFFFF,
                )
            else:
                ocr_doc = read_clean(str(pdf_path), doc_id, lang=lang)

            (text_dir / f"{doc_id}.json").write_text(
                ocr_doc.model_dump_json(), encoding="utf-8"
            )

            records.append({
                "doc_id": doc_id,
                "split": split,
                "doc_type": doc_type.value,
                "template_id": template_id,
                "lang": lang,
                "condition": ocr_doc.condition,
                "severity": severity,
                "date_style": doc.date_style,
                "money_style": doc.money_style,
                "truth": doc.truth,
                "service_lines": [sl.__dict__ for sl in doc.service_lines],
                "org_roles": doc.org_roles,
                "pdf": str(pdf_path.relative_to(out_dir)),
                "text": f"text/{doc_id}.json",
            })
            stats[f"{split}/{ocr_doc.condition}"] += 1
            stats[f"lang/{lang}"] += 1

            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{count}", file=sys.stderr)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} documents -> {manifest_path}")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")

    # Fail loudly if the split control was violated.
    train_templates = {
        r["template_id"] for r in records if r["split"] in ("train", "val")
    }
    gold_templates = {
        r["template_id"] for r in records if r["split"] in ("gold_synth", "demo")
    }
    assert train_templates.isdisjoint(gold_templates), (
        f"TEMPLATE LEAK: {train_templates & gold_templates}"
    )
    print(
        f"\nTemplate split verified: train/val={sorted(train_templates)} "
        f"disjoint from gold/demo={sorted(gold_templates)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/corpus", type=Path)
    parser.add_argument("--seed", default=20250902, type=int)
    parser.add_argument("--scanned-rate", default=SCANNED_RATE, type=float)
    parser.add_argument("--limit", type=int, help="cap docs per split (for smoke runs)")
    args = parser.parse_args()
    build(args.out, args.seed, args.scanned_rate, args.limit)


if __name__ == "__main__":
    main()
