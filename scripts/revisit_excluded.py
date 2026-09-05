"""Second look at gold fields excluded as unreadable that a matcher can still locate.

    python scripts/revisit_excluded.py --split gold_synth

Marking a destroyed value ``unreadable`` is correct and necessary: grading any approach
on a value OCR annihilated measures the scanner, not the extractor. But excluding a value
that is merely *messy* -- ``Medicaid   Managed.   Care``, ``CMY.929930559`` -- does the
opposite kind of damage. Those are the hard-but-winnable cases that separate a good
extractor from a lucky one, and dropping them quietly makes every score drift upward and
the approaches look more alike than they are.

So this pass shows only the fields where both are true:

* the reviewer marked the field unreadable, and
* the span aligner can still find the generator's value on the page

For each one it prints exactly what survives, so the decision is made against evidence.
Anything no matcher can locate is left alone -- those exclusions are not in question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.align import find_span  # noqa: E402
from docintel.schema import OcrDocument  # noqa: E402


def context(text: str, start: int, end: int, width: int = 100) -> str:
    lo = max(0, start - width // 2)
    hi = min(len(text), end + width // 2)
    return (
        text[lo:start] + ">>>" + text[start:end] + "<<<" + text[end:hi]
    ).replace("\n", " | ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    parser.add_argument("--gold", default="data/gold", type=Path)
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--list-only", action="store_true",
                        help="print the candidates and exit without prompting")
    args = parser.parse_args()

    gold_path = args.gold / f"{args.split}.jsonl"
    records = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = {
        json.loads(line)["doc_id"]: json.loads(line)
        for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    # Build the candidate list: excluded, but the value is still on the page.
    candidates = []
    for record in records:
        source = manifest.get(record["doc_id"])
        if not source:
            continue
        doc = None
        for field, original in source["truth"].items():
            if not original or record["fields"].get(field) is not None:
                continue  # not excluded
            if doc is None:
                doc = OcrDocument.model_validate_json(
                    (args.corpus / source["text"]).read_text(encoding="utf-8")
                )
            alignment = find_span(doc.text, original, field)
            if alignment.ok:
                candidates.append((record, field, original, doc, alignment))

    if not candidates:
        print("No excluded field is still locatable. Nothing to revisit.")
        return 0

    print(f"{len(candidates)} excluded field(s) can still be located on the page.\n")
    if args.list_only:
        for _, field, original, doc, a in candidates:
            print(f"  {field:24} {original!r:34} -> on page {doc.text[a.start:a.end]!r}")
        return 0

    restored = confirmed = 0
    for index, (record, field, original, doc, a) in enumerate(candidates, 1):
        found = doc.text[a.start:a.end]
        print("=" * 100)
        print(f"[{index}/{len(candidates)}] {record['doc_id']}  ({record['condition']})")
        print(f"\n  field      : {field}")
        print(f"  generator  : {original!r}")
        print(f"  on page    : {found!r}   (match={a.method}, score={a.score:.0f})")
        print(f"  context    : ...{context(doc.text, a.start, a.end)}...")
        print("\n  Is that value still readable to you?")
        try:
            answer = input("  [Enter]=yes, put it back  u=no, keep it excluded: ").strip()
        except EOFError:
            print("\nNo interactive input. Run this in a real terminal.", file=sys.stderr)
            return 2

        if answer.lower() == "u":
            confirmed += 1
            record.setdefault("corrections", []).append(
                f"{field}: UNREADABLE (confirmed on second look)"
            )
        else:
            record["fields"][field] = original
            restored += 1
            record.setdefault("corrections", []).append(
                f"{field}: restored to {original!r} (readable on second look)"
            )

        with gold_path.open("w", encoding="utf-8") as fh:
            for item in records:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = sum(1 for r in records for v in r["fields"].values() if v)
    print("\n" + "=" * 100)
    print("SECOND-LOOK SUMMARY")
    print(f"  restored to scoring   : {restored}")
    print(f"  kept excluded         : {confirmed}")
    print(f"  scoreable values now  : {total}")
    print(f"\nWrote {gold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
