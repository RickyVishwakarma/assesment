"""Human verification of the gold test set.

Run:
    python scripts/review_gold.py --split gold_synth

The assignment requires a manually verified gold set. Reviewing 90 documents x ~11
fields by hand would be ~1,000 decisions, most of them pointless — so this tool uses a
two-tier protocol that concentrates human attention where it actually changes something:

**Tier 1 — mechanically verified (no human needed).**
    The generator's value occurs in the extracted text, ignoring how whitespace fell.
    Two independent artefacts agree: the program that wrote the value, and the OCR that
    read it back. There is nothing for a human to adjudicate. On this corpus that clears
    91.5% of fields.

**Tier 2 — human review (flagged).**
    The value cannot be found at all, so OCR and the generator genuinely disagree. These
    are exactly the cases where the gold label may be wrong, and they are the only ones
    shown to the reviewer — 8.5% of fields, about 80 decisions.

For each flagged field the tool shows the surrounding text and the closest fuzzy match
it can find, then asks the reviewer to accept the suggestion, type the correct value, or
mark the field unreadable. Marking unreadable is important and not a cop-out: if a value
is genuinely illegible after a fax, no extractive system can recover it, and scoring
against it would measure OCR rather than extraction.

Every decision is logged, so the report can state the exact human-correction rate rather
than merely asserting that verification happened.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.align import find_span  # noqa: E402
from docintel.normalize import values_match  # noqa: E402
from docintel.schema import OcrDocument  # noqa: E402

UNREADABLE = "__UNREADABLE__"


def _present_in(text: str, value: str) -> bool:
    """Is ``value`` present in ``text``, ignoring how whitespace fell?

    Tier-1 auto-verification originally used a bare ``value in text``. That is stricter
    than the rest of the system and it was wrong: a value is still verifiably present
    when the layout wrapped it across a line, or when the OCR word boxes put a wider gap
    between its tokens. On this corpus that single character-exact check produced 90
    false flags out of 170 — more than half the human's queue was values that were
    plainly there.

    Matching on collapsed whitespace, the same tolerance :func:`docintel.align.find_span`
    applies, leaves only the flags that represent a genuine disagreement between what the
    generator wrote and what the OCR read back. Those are the ones worth a human's time.
    """
    if value in text:
        return True
    collapsed_value = re.sub(r"\s+", " ", value).strip()
    collapsed_text = re.sub(r"\s+", " ", text)
    return bool(collapsed_value) and collapsed_value in collapsed_text


def context_around(text: str, start: int, end: int, width: int = 90) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    snippet = text[lo:start] + ">>>" + text[start:end] + "<<<" + text[end:hi]
    return snippet.replace("\n", " | ")


def best_guess_region(text: str, value: str, width: int = 110) -> str | None:
    """Find the part of the page a missing value probably lives in.

    When no matcher aligns, the reviewer previously got ``text[:200]`` -- the letterhead,
    which by definition cannot contain the value they are being asked to judge. Asking
    someone to adjudicate evidence you have not shown them is worse than not asking.

    So: try progressively shorter prefixes and suffixes of the value. OCR usually damages
    part of a string rather than all of it (``RF194851`` surviving as ``RF19485]``), so a
    6- or 5-character fragment normally still matches and anchors the window on the right
    line.
    """
    import re as _re

    compact = _re.sub(r"[^A-Za-z0-9]", "", value)
    if len(compact) < 4:
        return None

    for size in range(len(compact), 3, -1):
        for fragment in (compact[:size], compact[-size:]):
            match = _re.search(
                r"\W*".join(_re.escape(c) for c in fragment), text, _re.IGNORECASE
            )
            if match:
                start = max(0, match.start() - width // 2)
                end = min(len(text), match.end() + width // 2)
                marked = (
                    text[start:match.start()]
                    + ">>>" + text[match.start():match.end()] + "<<<"
                    + text[match.end():end]
                )
                return marked.replace("\n", " | ")
    return None


def review(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    records = [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == args.split
    ]
    # ``--limit`` restricts what this session *reviews*, never what gets written.
    #
    # It previously sliced ``records`` before the write loop, so the output file was
    # rebuilt from the truncated list and every unreviewed document was silently deleted.
    # A `--limit 2` run cut a 90-document gold set down to 2. Keeping the full set here
    # and limiting only the review queue makes the flag safe to use for spot checks.
    review_ids = None
    if args.limit:
        review_ids = {r["doc_id"] for r in records[: args.limit]}

    # Snapshot whatever is already on disk so nothing is lost by a partial run.
    preserved: dict[str, dict] = {}
    _existing = Path(args.out) / f"{args.split}.jsonl"
    if _existing.exists():
        for line in _existing.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                preserved[item["doc_id"]] = item

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}.jsonl"

    # Resume support: never make someone redo work they already did.
    #
    # ``--reset`` exists because resume is keyed on doc_id alone, which is the wrong key
    # whenever the *text* changes underneath it. Re-extracting the corpus (a new OCR
    # setting, say) invalidates every prior judgement, since a field flagged as garbled
    # may now read cleanly, or vice versa. Without a reset the tool reports "90 already
    # reviewed, 0 to go" and silently carries stale verdicts forward.
    done: dict[str, dict] = {}
    if out_path.exists() and not args.reset:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["doc_id"]] = row

    auto_verified = flagged = corrected = unreadable = accepted = 0
    results: list[dict] = list(done.values())

    pending = [r for r in records if r["doc_id"] not in done]
    if review_ids is not None:
        pending = [r for r in pending if r["doc_id"] in review_ids]
    print(f"{len(records)} documents in {args.split}; {len(done)} already reviewed, "
          f"{len(pending)} to go.\n")

    for index, record in enumerate(pending, 1):
        doc = OcrDocument.model_validate_json(
            (corpus / record["text"]).read_text(encoding="utf-8")
        )
        fields = dict(record["truth"])
        corrections: list[str] = []
        header_shown = False

        for field, gold in sorted(record["truth"].items()):
            if gold is None:
                continue
            if _present_in(doc.text, gold):
                auto_verified += 1
                continue

            flagged += 1
            if args.auto_only:
                # Non-interactive pass: flag without resolving, so the count is still
                # reported honestly rather than silently auto-accepted.
                corrections.append(f"{field}: FLAGGED_UNRESOLVED")
                continue

            if not header_shown:
                print("=" * 100)
                print(f"[{index}/{len(pending)}] {record['doc_id']}  "
                      f"({record['condition']}, {record['lang']}, "
                      f"template {record['template_id']})")
                header_shown = True

            alignment = find_span(doc.text, gold, field)
            suggestion = None
            if alignment.ok:
                suggestion = doc.text[alignment.start:alignment.end]
                print(f"\n  field      : {field}")
                print(f"  generator  : {gold!r}")
                print(f"  in document: {suggestion!r}   "
                      f"(match={alignment.method}, score={alignment.score:.0f})")
                print(f"  context    : ...{context_around(doc.text, alignment.start, alignment.end)}...")
            else:
                print(f"\n  field      : {field}")
                print(f"  generator  : {gold!r}")
                print("  in document: NOT FOUND by any matcher")
                region = best_guess_region(doc.text, gold)
                if region:
                    print(f"  closest    : ...{region}...")
                else:
                    print("  (no fragment of this value survives anywhere on the page)")
                    print(f"  page       : {' | '.join(doc.text.splitlines()[:12])}")

            # Show the machine adjudication, if one exists, as a *proposal* the human
            # confirms or overrides. Reviewing a decision is far faster and more accurate
            # than making one cold, and it keeps the human as the authority: nothing is
            # accepted unless a person presses a key.
            prior = PRIOR.get(record["doc_id"], {}).get(field)
            if prior:
                label = {"keep": "keep the generator value",
                         "unreadable": "mark unreadable"}.get(prior, f"replace with {prior!r}")
                print(f"  claude said: {label}")

            if prior:
                prompt = (
                    f"  [Enter]=accept Claude's call  k=keep generator  u=unreadable"
                    + ("  o=accept OCR form" if suggestion else "")
                    + "  or type the correct value: "
                )
            else:
                prompt = (
                    "  [Enter]=accept suggestion  k=keep generator  u=unreadable  "
                    "or type the correct value: "
                    if suggestion else
                    "  [Enter]=keep generator  u=unreadable  or type the correct value: "
                )
            try:
                answer = input(prompt).strip()
            except EOFError:
                print("\nNo interactive input available. Re-run without --auto-only "
                      "in a real terminal, or use --auto-only to record flags.",
                      file=sys.stderr)
                return 2

            if answer == "" and prior:
                # Confirming the machine proposal. Counted as a human decision because a
                # person saw the evidence and chose it.
                if prior == "keep":
                    accepted += 1
                    corrections.append(f"{field}: kept generator value (confirmed)")
                elif prior == "unreadable":
                    fields[field] = None
                    unreadable += 1
                    corrections.append(f"{field}: UNREADABLE (confirmed)")
                else:
                    fields[field] = prior
                    corrected += 1
                    corrections.append(f"{field}: {gold!r} -> {prior!r} (confirmed)")
            elif answer.lower() == "o" and suggestion:
                fields[field] = suggestion
                corrections.append(f"{field}: {gold!r} -> {suggestion!r} (accepted OCR form)")
                corrected += 1
            elif answer == "" and suggestion:
                fields[field] = suggestion
                corrections.append(f"{field}: {gold!r} -> {suggestion!r} (accepted OCR form)")
                corrected += 1
            elif answer == "" or answer.lower() == "k":
                accepted += 1
            elif answer.lower() == "u":
                fields[field] = None
                corrections.append(f"{field}: {gold!r} -> unreadable")
                unreadable += 1
            else:
                fields[field] = answer
                corrections.append(f"{field}: {gold!r} -> {answer!r} (manual)")
                corrected += 1

        row = {
            "doc_id": record["doc_id"],
            "doc_type": record["doc_type"],
            "fields": fields,
            "service_lines": record["service_lines"],
            "org_roles": record["org_roles"],
            "provenance": "synthetic_unseen_template",
            "template_id": record["template_id"],
            "condition": record["condition"],
            "lang": record["lang"],
            "verified": not args.auto_only,
            "corrections": corrections,
        }
        results.append(row)

        with out_path.open("w", encoding="utf-8") as fh:
            written = set()
            for item in results:
                written.add(item["doc_id"])
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            # Preserve any record this session did not touch, so a partial or limited
            # run never destroys work already banked in the file.
            for doc_id, item in preserved.items():
                if doc_id not in written:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = auto_verified + flagged
    print("\n" + "=" * 100)
    print("VERIFICATION SUMMARY")
    print(f"  fields total                 : {total}")
    print(f"  tier 1 auto-verified         : {auto_verified} "
          f"({auto_verified / total:.1%})" if total else "")
    print(f"  tier 2 flagged for human     : {flagged} "
          f"({flagged / total:.1%})" if total else "")
    if not args.auto_only:
        print(f"    corrected by reviewer      : {corrected}")
        print(f"    kept generator value       : {accepted}")
        print(f"    marked unreadable          : {unreadable}")
    carried = len([d for d in preserved if d not in {r["doc_id"] for r in results}])
    print(
        f"\nWrote {len(results)} reviewed"
          + (f" + {carried} preserved" if carried else "")
          + f" = {len(results) + carried} records -> {out_path}")

    summary = {
        "split": args.split,
        "n_documents": len(results),
        "fields_total": total,
        "auto_verified": auto_verified,
        "flagged_for_human": flagged,
        "corrected": corrected,
        "kept": accepted,
        "unreadable": unreadable,
        "human_correction_rate": round(corrected / total, 4) if total else 0.0,
        "interactive": not args.auto_only,
    }
    (out_dir / f"{args.split}_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


PRIOR: dict[str, dict[str, str]] = {}


def load_prior_adjudication(enabled: bool) -> None:
    """Load Claude's tier-2 decisions so the human confirms rather than decides cold.

    These are proposals, never answers: every field still requires a keypress, and the
    split is only labelled ``human_verified`` once a person has been through it.
    """
    if not enabled:
        return
    try:
        from apply_adjudication import DECISIONS
    except Exception as exc:  # pragma: no cover
        print(f"  (no prior adjudication available: {exc})", file=sys.stderr)
        return
    for suffix, fields in DECISIONS.items():
        PRIOR[f"gold_synth-{suffix}"] = fields
    print(f"  loaded {sum(len(v) for v in PRIOR.values())} prior decisions to confirm",
          file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--split", default="gold_synth")
    parser.add_argument("--out", default="data/gold")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reset", action="store_true",
        help="discard prior verdicts and re-verify from scratch (use after re-extracting "
             "the corpus text, which invalidates earlier judgements)",
    )
    parser.add_argument(
        "--auto-only", action="store_true",
        help="record tier-1 verification and flag tier-2 without prompting",
    )
    parser.add_argument(
        "--from-adjudication", action="store_true",
        help="show Claude's prior decision for each flagged field so you confirm "
             "or override it instead of deciding cold (Enter accepts)",
    )
    args = parser.parse_args()
    load_prior_adjudication(getattr(args, 'from_adjudication', False))
    raise SystemExit(review(args))


if __name__ == "__main__":
    main()
