"""Export / merge workflow for the frontier-LLM reference tier.

The frontier tier is annotated by Claude working through the Claude Code session that
builds this project, because no paid API key is available on this machine. That makes it
a *reference point*, not a runnable baseline — see :mod:`docintel.approaches.llm_frontier`
for the honesty constraints that carries.

This script makes that process reproducible and auditable instead of ad-hoc:

    python scripts/frontier_tier.py export --split gold_synth --batch-size 15
    # ... annotate each .tmp/frontier_todo/batch_N.json into .tmp/frontier_done/ ...
    python scripts/frontier_tier.py merge

``export`` writes exactly the text the model is shown — the same ``OcrDocument.text``
every other approach receives, so no tier gets a cleaner view of the document than
another. ``merge`` assembles the annotations into the cache the extractor replays, and
records how token counts were arrived at.

**Token accounting.** Claude's tokeniser is not available offline, so prompt/completion
token counts are *estimated* from character length at a documented ratio rather than
metered. Every record carries ``token_basis`` saying so, and the cost column in the
report inherits that caveat. Overstating the rigour of a cost number would undermine
precisely the comparison this tier exists to support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.schema import DOC_TYPES, FIELD_NAMES, OcrDocument  # noqa: E402

TODO_DIR = Path(".tmp/frontier_todo")
DONE_DIR = Path(".tmp/frontier_done")
CACHE = Path("data/frontier/annotations.json")

#: Characters per token. A conventional English-prose approximation; JSON output runs a
#: little denser, so this slightly *over*-estimates completion tokens, which errs toward
#: over-stating the frontier tier's cost rather than flattering it.
CHARS_PER_TOKEN = 3.6
TOKEN_BASIS = f"estimated at {CHARS_PER_TOKEN} chars/token; not metered by a tokeniser"

#: The instruction the annotator works from. Recorded here so the prompt is part of the
#: artefact rather than living only in a conversation.
INSTRUCTION = f"""Extract these fields from the document text, returning strict JSON.

Fields: {", ".join(FIELD_NAMES)}
document_type: one of {", ".join(t.value for t in DOC_TYPES)}

Rules:
- Copy values EXACTLY as they appear in the text, including OCR errors. Do not correct,
  normalise or reformat anything.
- Use null for any field the document does not contain.
- service_lines: list of {{procedure_code, date_of_service, units, charge, paid}}.
- Wide runs of spaces mark column boundaries; do not merge across them.
"""


def load_split(corpus: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["split"] == split
    ]


def export(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    records = load_split(corpus, args.split)
    TODO_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    for old in TODO_DIR.glob("batch_*.json"):
        old.unlink()

    batches = [
        records[i:i + args.batch_size]
        for i in range(0, len(records), args.batch_size)
    ]
    for index, batch in enumerate(batches):
        payload = {}
        for record in batch:
            doc = OcrDocument.model_validate_json(
                (corpus / record["text"]).read_text(encoding="utf-8")
            )
            payload[record["doc_id"]] = {"text": doc.text}
        path = TODO_DIR / f"batch_{index}.json"
        path.write_text(
            json.dumps({"instruction": INSTRUCTION, "documents": payload},
                       indent=1, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(batch)} documents)")
    print(f"\n{len(records)} documents in {len(batches)} batches -> {TODO_DIR}")
    print(f"Annotate each into {DONE_DIR}/batch_N.json, then run: merge")
    return 0


def merge(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    records = {r["doc_id"]: r for r in load_split(corpus, args.split)}
    texts: dict[str, str] = {}
    for doc_id, record in records.items():
        doc = OcrDocument.model_validate_json(
            (corpus / record["text"]).read_text(encoding="utf-8")
        )
        texts[doc_id] = doc.text

    merged: dict[str, dict] = {}
    for path in sorted(DONE_DIR.glob("batch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        annotations = data.get("documents", data)
        for doc_id, annotation in annotations.items():
            if doc_id not in texts:
                print(f"  WARNING: {doc_id} is not in split {args.split}; skipping")
                continue
            prompt_chars = len(INSTRUCTION) + len(texts[doc_id])
            completion_chars = len(json.dumps(annotation, ensure_ascii=False))
            annotation["prompt_tokens"] = int(prompt_chars / CHARS_PER_TOKEN)
            annotation["completion_tokens"] = int(completion_chars / CHARS_PER_TOKEN)
            annotation["token_basis"] = TOKEN_BASIS
            merged[doc_id] = annotation

    missing = sorted(set(texts) - set(merged))
    if missing:
        print(f"\n{len(missing)} documents NOT annotated, e.g. {missing[:3]}")
        if not args.allow_partial:
            print("Refusing to write a partial cache. Pass --allow-partial to override.")
            return 1

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(merged)} annotations -> {CACHE}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)

    exporter = sub.add_parser("export", help="write batches of document text to annotate")
    exporter.add_argument("--split", default="gold_synth")
    exporter.add_argument("--batch-size", default=15, type=int)
    exporter.set_defaults(func=export)

    merger = sub.add_parser("merge", help="assemble annotated batches into the cache")
    merger.add_argument("--split", default="gold_synth")
    merger.add_argument("--allow-partial", action="store_true")
    merger.set_defaults(func=merge)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
