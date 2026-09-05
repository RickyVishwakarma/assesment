"""Re-score the existing learning-curve checkpoints against the current gold set.

    python scripts/rescore_curve.py

The curve was first measured before the last round of gold adjudication, so its points
were scored against a slightly different test set than every other number in the report.
Nothing about the models changed -- only the yardstick did. This reloads each checkpoint
under ``models/curve/<n>/`` and re-scores it, preserving the original training statistics
so the curve stays a record of one training run measured consistently.

Re-training instead would have been wrong: it would introduce a fresh random
initialisation and confound "the gold set changed" with "the model came out differently
this time".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.approaches.small_model import SmallModelExtractor  # noqa: E402
from docintel.eval.runner import load_gold, run_approach, summarise  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--gold-dir", dest="gold_dir", default="data/gold", type=Path)
    parser.add_argument("--curve", default="reports/learning_curve.json", type=Path)
    parser.add_argument("--models", default="models/curve", type=Path)
    args = parser.parse_args()

    points = json.loads(args.curve.read_text(encoding="utf-8"))
    gold = load_gold(args.corpus, args.split, args.gold_dir)
    print(f"re-scoring {len(points)} checkpoints on {len(gold)} gold documents\n")

    for point in points:
        n = point["n"]
        model_dir = args.models / str(n)
        if not model_dir.exists():
            print(f"  n={n:5}  checkpoint missing at {model_dir}; leaving as-is")
            continue

        extractor = SmallModelExtractor(model_dir=model_dir)
        scored = summarise(run_approach(extractor, gold, args.corpus), bootstrap=200)
        micro = scored["overall"]

        before = point["f1"]
        point["f1"] = round(micro["f1"], 4)
        point["precision"] = round(micro["precision"], 4)
        point["recall"] = round(micro["recall"], 4)
        point["macro_f1"] = round(micro["macro_f1"], 4)
        point["doctype_accuracy"] = round(scored["doc_type"]["accuracy"], 4)
        point["scored_against"] = "gold_synth, final human-verified"
        print(f"  n={n:5}  F1 {before:.4f} -> {point['f1']:.4f}"
              f"  doctype={point['doctype_accuracy']:.3f}")

    args.curve.write_text(json.dumps(points, indent=2), encoding="utf-8")
    print(f"\nwrote {args.curve}")

    print("\n" + "=" * 58)
    print(f"{'n':>6}  {'F1':>7}  {'precision':>10}  {'recall':>8}  {'doctype':>8}")
    print("-" * 58)
    for p in points:
        print(f"{p['n']:>6}  {p['f1']:>7.3f}  {p['precision']:>10.3f}  "
              f"{p['recall']:>8.3f}  {p['doctype_accuracy']:>8.3f}")
    print("=" * 58)
    if len(points) >= 2:
        tail = points[-1]["f1"] - points[-2]["f1"]
        print(f"\nlast step ({points[-2]['n']} -> {points[-1]['n']}): F1 {tail:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
