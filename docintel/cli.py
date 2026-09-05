"""Command-line interface.

Implements the flow the assignment asks for:

    Document -> OCR/Text -> NLP / Small Model / General LLM -> Structured Output

    python -m docintel extract  --file doc.pdf --approach all
    python -m docintel evaluate --split gold_synth --approach all
    python -m docintel train    --what all
    python -m docintel doctor
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .ocr import ocr_available, read_document
from .schema import DocType, ExtractionResult, OcrDocument

APPROACH_NAMES = ["nlp", "small_model", "llm_local", "llm_frontier"]


def build_extractor(name: str, **kwargs):
    """Instantiate one approach by name, importing lazily so a missing optional
    dependency only breaks the approach that needs it."""
    if name == "nlp":
        from .approaches.nlp import NlpExtractor

        return NlpExtractor(model_dir=kwargs.get("nlp_dir", "models/nlp"))
    if name == "small_model":
        from .approaches.small_model import SmallModelExtractor

        return SmallModelExtractor(model_dir=kwargs.get("small_dir", "models/small"))
    if name == "llm_local":
        from .approaches.llm_local import LocalLlmExtractor

        return LocalLlmExtractor(model=kwargs.get("llm_model", None) or "qwen2.5:3b-instruct")
    if name == "llm_frontier":
        from .approaches.llm_frontier import FrontierLlmExtractor

        return FrontierLlmExtractor(
            cache_path=kwargs.get("frontier_cache", "data/frontier/annotations.json")
        )
    raise ValueError(f"unknown approach: {name}")


def _resolve(names: list[str]) -> list[str]:
    if "all" in names:
        return list(APPROACH_NAMES)
    return names


def _print_result(result: ExtractionResult, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    print(f"\n=== {result.approach} ===")
    if result.meta.get("note"):
        # An approach that legitimately produced nothing must say so. A column of
        # blank fields is otherwise indistinguishable from a model that failed the
        # document, which is the worst way to present a non-result.
        print(f"  NOTE          : {result.meta['note']}")
    print(f"  document_type : {result.doc_type.value}"
          + (f"  (conf {result.doc_type_confidence:.2f})"
             if result.doc_type_confidence is not None else ""))
    if result.latency_ms is not None:
        print(f"  latency       : {result.latency_ms:.0f} ms")
    for name in sorted(result.fields):
        value = result.fields[name]
        print(f"  {name:26s}: {value.value if value else '-'}")
    if result.service_lines:
        print("  service_lines :")
        for line in result.service_lines:
            print(f"      {line.procedure_code or '-':8s} {line.date_of_service or '-':12s} "
                  f"{line.charge or '-':10s} {line.paid or '-'}")


def cmd_extract(args: argparse.Namespace) -> int:
    """Run one document through one or more approaches."""
    path = Path(args.file)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    doc = read_document(
        str(path), condition=args.condition, lang=args.lang, severity=args.severity
    )
    print(
        f"Read {doc.doc_id}: {len(doc.words)} words, {len(doc.text)} chars, "
        f"condition={doc.condition}",
        file=sys.stderr,
    )

    results = []
    for name in _resolve(args.approach):
        try:
            extractor = build_extractor(name, llm_model=args.llm_model)
            results.append(extractor.extract(doc))
        except Exception as exc:
            print(f"  [{name}] unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
            if _lfs_pointers(Path("models/small/tokens")):
                print(
                    "  [{}] the weights are unfetched Git LFS pointers. "
                    "Run: git lfs pull".format(name),
                    file=sys.stderr,
                )

    if args.json:
        print(json.dumps([json.loads(r.model_dump_json()) for r in results], indent=2))
    else:
        for result in results:
            _print_result(result, as_json=False)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score every approach on the same split and write the comparison report."""
    from .eval.runner import (
        ApproachProfile,
        compare,
        load_gold,
        measure_determinism,
        run_approach,
        summarise,
    )

    corpus = Path(args.corpus)
    records = load_gold(corpus, args.split, Path(args.gold_dir) if args.gold_dir else None)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"No records for split {args.split}", file=sys.stderr)
        return 2

    sources = {r.get("gold_source") for r in records}
    print(f"Split {args.split}: {len(records)} documents (gold source: {sources})",
          file=sys.stderr)

    results, summaries = {}, {}
    for name in _resolve(args.approach):
        try:
            extractor = build_extractor(name, llm_model=args.llm_model)
        except Exception as exc:
            print(f"  [{name}] unavailable: {exc}", file=sys.stderr)
            continue

        profile = _profile_for(name, extractor)
        if args.determinism:
            deterministic, rate = measure_determinism(extractor, records, corpus)
            profile.deterministic, profile.determinism_rate = deterministic, rate

        print(f"  running {name} ...", file=sys.stderr, flush=True)
        run = run_approach(extractor, records, corpus, profile)
        results[name] = run
        summaries[name] = summarise(run, bootstrap=args.bootstrap)
        overall = summaries[name]["overall"]
        print(
            f"    F1={overall['f1']:.3f} "
            f"CI{overall['f1_ci95']} "
            f"doctype_acc={summaries[name]['doc_type']['accuracy']:.3f} "
            f"halluc={overall['hallucination_rate']:.3f}",
            file=sys.stderr,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"eval_{args.split}.json"

    merged = dict(summaries)
    if args.merge and target.exists():
        # Carry forward approaches this pass did not run.
        #
        # This exists because of a hard constraint on the build machine rather than as a
        # convenience: with 7.4GB of RAM, torch cannot map its CUDA DLLs while Ollama is
        # resident, so evaluating `small_model` and `llm_local` in one process makes the
        # former fail with WinError 1455 on every document -- and a crashed approach
        # scores 0.000, which looks like a catastrophic model rather than an environment
        # limit. Splitting the run across passes and merging is the honest way to get one
        # four-way table on hardware that cannot hold both at once.
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("split") != args.split:
            print(f"  refusing to merge: existing report is for split "
                  f"{previous.get('split')!r}", file=sys.stderr)
            return 2
        carried = [k for k in previous.get("approaches", {}) if k not in merged]
        for name in carried:
            merged[name] = previous["approaches"][name]
        if carried:
            print(f"  merged in previous results for: {', '.join(sorted(carried))}",
                  file=sys.stderr)

    report = {
        "split": args.split,
        "n_documents": len(records),
        "gold_sources": sorted(s for s in sources if s),
        "approaches": merged,
        # Paired comparisons need both runs in memory, so they are only computed across
        # approaches this pass actually ran.
        "comparisons": compare(results, bootstrap=args.bootstrap) if len(results) > 1 else {},
        "merged": bool(args.merge),
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {target}")

    _print_comparison_table(merged)
    return 0


def _profile_for(name: str, extractor) -> "ApproachProfile":
    from .eval.runner import ApproachProfile

    if name == "nlp":
        size = _dir_size(Path("models/nlp"))
        return ApproachProfile(
            name=name, model_size_mb=size, offline=True, cost_per_doc_usd=0.0,
            notes="rules + dictionaries + TF-IDF doc-type classifier",
        )
    if name == "small_model":
        return ApproachProfile(
            name=name, model_size_mb=_dir_size(Path("models/small")), offline=True,
            cost_per_doc_usd=0.0,
            notes="distilbert-base-cased token + sequence classifier",
        )
    if name == "llm_local":
        from .approaches.llm_local import COST_PER_DOC_USD

        return ApproachProfile(
            name=name, model_size_mb=1900.0, offline=True,
            cost_per_doc_usd=COST_PER_DOC_USD,
            notes="Qwen2.5-3B-Instruct via Ollama, schema-constrained",
        )
    from .approaches.llm_frontier import COST_BASIS, MODEL_NAME, mean_cost_per_doc

    return ApproachProfile(
        name=name, model_size_mb=None, offline=False, cost_basis=COST_BASIS,
        cost_per_doc_usd=round(mean_cost_per_doc(), 6),
        notes=f"{MODEL_NAME}; reference tier replayed from cache",
    )


def _dir_size(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def _print_comparison_table(summaries: dict) -> None:
    if not summaries:
        return
    print("\n" + "=" * 96)
    print(f"{'approach':<14}{'F1':>7}{'prec':>7}{'rec':>7}{'macroF1':>9}"
          f"{'exact':>7}{'doctype':>9}{'halluc':>8}{'p50 ms':>9}{'size MB':>9}")
    print("-" * 96)
    for name, summary in summaries.items():
        overall = summary["overall"]
        profile = summary.get("profile") or {}
        size = profile.get("model_size_mb")
        p50 = profile.get("latency_p50_ms")
        print(
            f"{name:<14}{overall['f1']:>7.3f}{overall['precision']:>7.3f}"
            f"{overall['recall']:>7.3f}{overall['macro_f1']:>9.3f}"
            f"{overall['exact_match_rate']:>7.3f}"
            f"{summary['doc_type']['accuracy']:>9.3f}"
            f"{overall['hallucination_rate']:>8.3f}"
            f"{(f'{p50:.0f}' if p50 is not None else '-'):>9}"
            f"{(f'{size:.0f}' if size is not None else '-'):>9}"
        )
    print("=" * 96)


def cmd_train(args: argparse.Namespace) -> int:
    """Train the NLP doc-type classifier and/or the small model."""
    corpus = Path(args.corpus)

    if args.what in ("nlp", "all"):
        from .approaches.nlp import NlpExtractor
        from .eval.runner import load_split

        rows = load_split(corpus, "train")
        texts, labels = [], []
        for record in rows:
            doc = OcrDocument.model_validate_json(
                (corpus / record["text"]).read_text(encoding="utf-8")
            )
            texts.append(doc.text)
            labels.append(record["doc_type"])
        # The calibrated classifier cross-validates 3 ways, so every document type needs
        # at least 3 training examples. Below that sklearn raises a ValueError about
        # n_splits that says nothing about which class is short -- which is confusing on
        # a reduced corpus (`build_corpus.py --limit`), the only place it happens.
        counts = Counter(labels)
        thin = {k: n for k, n in counts.items() if n < 3}
        if thin:
            print(
                f"Not enough training data for the doc-type classifier: "
                f"{', '.join(f'{k} has {n}' for k, n in sorted(thin.items()))}. "
                f"It cross-validates 3 ways, so each of the {len(counts)} types needs 3+ "
                f"documents. Rebuild the corpus with a larger --limit.",
                file=sys.stderr,
            )
            return 2

        print(f"Training NLP doc-type classifier on {len(texts)} documents ...")
        NlpExtractor(model_dir="models/nlp").fit_doctype(texts, labels)
        print("  saved -> models/nlp/doctype.pkl")

    if args.what in ("small", "all"):
        from .approaches.small_model import (
            train_doctype_classifier,
            train_token_classifier,
        )

        silver = Path(args.silver) / "train.jsonl"
        if not silver.exists():
            print(f"No silver data at {silver}. Run scripts/make_silver.py first.",
                  file=sys.stderr)
            return 2
        rows = [
            json.loads(line)
            for line in silver.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.limit:
            rows = rows[: args.limit]
        examples = [
            {"text": r["text"], "labels": {k: tuple(v) for k, v in r["labels"].items()}}
            for r in rows
        ]
        print(f"Training token classifier on {len(examples)} silver examples ...")
        stats = train_token_classifier(
            examples, "models/small/tokens", epochs=args.epochs
        )
        print(f"  {stats}")
        # The doc-type head needs MORE passes than the token head, not fewer. The token
        # head gets a few hundred labelled tokens per document; the doc-type head gets
        # exactly one label per document, so the same corpus carries two orders of
        # magnitude less supervision for it, and its classifier is randomly initialised
        # rather than pretrained. Starving it (the previous `epochs - 2`) left it at
        # 0.367 accuracy on six classes — barely above the 0.167 chance baseline.
        doctype_epochs = args.doctype_epochs or max(8, args.epochs * 2)
        print(f"Training doc-type head ({doctype_epochs} epochs) ...")
        train_doctype_classifier(
            [r["text"] for r in rows], [r["doc_type"] for r in rows],
            "models/small/doctype", epochs=doctype_epochs,
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report which parts of the system are actually available on this machine."""
    from .approaches.llm_local import list_models, ollama_available

    print("environment")
    try:
        import torch

        print(f"  torch            : {torch.__version__}")
        print(f"  cuda available   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  gpu              : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"  torch            : MISSING ({exc})")
    print(f"  tesseract OCR    : {'yes' if ocr_available() else 'NO'}")
    try:
        import spacy  # noqa: F401

        print("  spacy            : yes")
    except Exception:
        print("  spacy            : NO")
    print(f"  ollama           : {'yes' if ollama_available() else 'NO'}")
    if ollama_available():
        print(f"  ollama models    : {list_models()}")

    print("\nartefacts")
    lfs_stubs: list[str] = []
    for label, path in [
        ("corpus", Path("data/corpus/manifest.jsonl")),
        ("silver train", Path("data/silver/train.jsonl")),
        ("nlp doctype", Path("models/nlp/doctype.pkl")),
        ("small tokens", Path("models/small/tokens")),
        ("small doctype", Path("models/small/doctype")),
        ("frontier cache", Path("data/frontier/annotations.json")),
        ("gold verified", Path("data/gold/gold_synth.jsonl")),
    ]:
        if not path.exists():
            status = "absent"
        elif _lfs_pointers(path):
            status = "LFS POINTER -- run: git lfs pull"
            lfs_stubs.append(label)
        else:
            status = "present"
        print(f"  {label:16s} : {status}")

    if lfs_stubs:
        # Without this the failure surfaces much later as
        # "SafetensorError: header too large", which explains nothing.
        print(
            "\nThe model weights are Git LFS objects and this clone has only the\n"
            "pointer files. Install Git LFS (https://git-lfs.com), then run:\n"
            "    git lfs pull\n"
            "Until then the small model is unavailable; every other approach works."
        )
    return 0


def _lfs_pointers(path: Path) -> bool:
    """True if ``path`` is, or contains, an unfetched Git LFS pointer file.

    A pointer is a ~130-byte text file opening with the LFS version line, so a
    weights file that small is a clone without ``git lfs pull`` rather than a
    corrupt download. Worth distinguishing: the errors look nothing alike.
    """
    candidates = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    for candidate in candidates:
        try:
            if candidate.stat().st_size > 1024:
                continue
            with candidate.open("rb") as handle:
                if handle.read(42).startswith(b"version https://git-lfs.github.com"):
                    return True
        except OSError:
            continue
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docintel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="run one document through the approaches")
    p.add_argument("--file", required=True)
    p.add_argument("--approach", nargs="+", default=["all"])
    p.add_argument("--condition", default="auto", choices=["auto", "clean", "scanned"])
    p.add_argument("--severity", default="medium", choices=["light", "medium", "heavy"])
    p.add_argument("--lang", default="en")
    p.add_argument("--llm-model", dest="llm_model", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("evaluate", help="score approaches on a split")
    p.add_argument("--split", default="gold_synth")
    p.add_argument("--approach", nargs="+", default=["all"])
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--gold-dir", dest="gold_dir", default="data/gold")
    p.add_argument("--out", default="reports")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--determinism", action="store_true")
    p.add_argument(
        "--merge", action="store_true",
        help="keep approaches from a previous report instead of overwriting them; "
             "needed when memory limits force approaches to be run in separate passes",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--llm-model", dest="llm_model", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="train the NLP classifier and/or small model")
    p.add_argument("--what", default="all", choices=["nlp", "small", "all"])
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--silver", default="data/silver")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--doctype-epochs", type=int, default=None,
        help="epochs for the document-type head (default: max(8, 2x --epochs); it sees "
             "one label per document rather than one per token, so it needs more)",
    )
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("doctor", help="report what is available on this machine")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
