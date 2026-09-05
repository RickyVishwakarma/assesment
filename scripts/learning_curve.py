"""How much silver data does the small model actually need?

    python scripts/learning_curve.py --sizes 100 300 600 1000

Trains the token classifier from scratch at several silver-set sizes and scores each on
the same gold set. This is the measurement that turns "the small model scored X" into an
engineering answer: if the curve is still climbing at 1,000 documents, more teacher
inference buys accuracy; if it has flattened, the ceiling is the teacher's label quality
or the architecture, and spending more on labelling is wasted.

Every point uses identical hyperparameters and the same held-out gold set, and each is
trained from the pretrained checkpoint rather than continuing a previous run, so the
points are comparable to each other.

Models are written under ``models/curve/<n>/`` so the shipped ``models/small`` is left
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.approaches.small_model import (  # noqa: E402
    SmallModelExtractor,
    train_doctype_classifier,
    train_token_classifier,
)
from docintel.eval.runner import load_gold, run_approach, summarise  # noqa: E402


def load_silver(path: Path, limit: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", default="data/silver/train.jsonl", type=Path)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 300, 600, 1000])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--doctype-epochs", type=int, default=16)
    parser.add_argument("--out", default="reports/learning_curve.json", type=Path)
    args = parser.parse_args()

    all_rows = load_silver(args.silver, None)
    gold = load_gold(args.corpus, args.split, Path("data/gold"))
    print(f"{len(all_rows)} silver examples available; scoring on {len(gold)} gold docs\n")

    results = []
    for n in args.sizes:
        if n > len(all_rows):
            print(f"skipping n={n}: only {len(all_rows)} examples available")
            continue
        rows = all_rows[:n]
        out_dir = Path("models/curve") / str(n)
        print(f"=== n={n} ===")

        started = time.perf_counter()
        examples = [
            {"text": r["text"], "labels": {k: tuple(v) for k, v in r["labels"].items()}}
            for r in rows
        ]
        stats = train_token_classifier(
            examples, str(out_dir / "tokens"), epochs=args.epochs
        )
        train_doctype_classifier(
            [r["text"] for r in rows], [r["doc_type"] for r in rows],
            str(out_dir / "doctype"), epochs=args.doctype_epochs,
        )
        train_seconds = time.perf_counter() - started

        extractor = SmallModelExtractor(model_dir=out_dir)
        scored = summarise(run_approach(extractor, gold, args.corpus), bootstrap=200)
        micro = scored["overall"]
        point = {
            "n": n,
            "f1": round(micro["f1"], 4),
            "precision": round(micro["precision"], 4),
            "recall": round(micro["recall"], 4),
            "doctype_accuracy": round(scored["doc_type"]["accuracy"], 4),
            "macro_f1": round(micro["macro_f1"], 4),
            "train_seconds": round(train_seconds, 1),
            "final_loss": stats["history"][-1]["loss"],
        }
        results.append(point)
        print(f"  n={n:5}  F1={point['f1']:.3f}  doctype={point['doctype_accuracy']:.3f}"
              f"  ({train_seconds / 60:.1f} min to train)\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"{'n':>6}  {'F1':>7}  {'precision':>10}  {'recall':>8}  {'doctype':>8}")
    print("-" * 66)
    for p in results:
        print(f"{p['n']:>6}  {p['f1']:>7.3f}  {p['precision']:>10.3f}  "
              f"{p['recall']:>8.3f}  {p['doctype_accuracy']:>8.3f}")
    print("=" * 66)

    if len(results) >= 2:
        first, last = results[0], results[-1]
        gain = last["f1"] - first["f1"]
        tail = results[-1]["f1"] - results[-2]["f1"]
        print(f"\n{first['n']} -> {last['n']} examples: F1 {gain:+.3f}")
        print(f"last step ({results[-2]['n']} -> {last['n']}): F1 {tail:+.3f}", end="  ")
        print("-> still climbing; more silver data would help"
              if tail > 0.01 else
              "-> flattening; the limit is teacher quality or architecture, not volume")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
