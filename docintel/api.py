"""FastAPI service exposing the same pipeline as the CLI.

    uvicorn docintel.api:app --reload
    curl -F "file=@doc.pdf" "http://127.0.0.1:8000/extract?approach=all"

On Windows use ``curl.exe``: PowerShell aliases ``curl`` to ``Invoke-WebRequest``,
which does not accept ``-F`` and reports a misleading parameter-binding error.

Realises the flow the assignment specifies:

    Document -> OCR/Text -> NLP / Small Model / General LLM -> Structured Output

Approaches are constructed once at startup and reused, because loading DistilBERT per
request would dominate the latency numbers and make the service useless for the
latency comparison it exists to demonstrate.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from .cli import APPROACH_NAMES, build_extractor
from .ocr import ocr_available, read_document
from .schema import ExtractionResult

_EXTRACTORS: dict[str, object] = {}
_LOAD_ERRORS: dict[str, str] = {}


def _load_approaches() -> None:
    """Build each approach once. A failure disables that approach, not the service."""
    for name in APPROACH_NAMES:
        try:
            # Build and warm up *before* publishing. The constructors are lazy, so
            # building an approach whose weights are missing still succeeds and only
            # fails when the model is first touched. Registering it first meant a broken
            # approach appeared in both `approaches_ready` and `approaches_unavailable`,
            # which is worse than either — /health is what a caller checks to decide
            # whether an approach is usable.
            extractor = build_extractor(name)
            if name == "small_model":
                extractor.token_model  # noqa: B018  (forces the weights to load)
            _EXTRACTORS[name] = extractor
        except Exception as exc:
            _EXTRACTORS.pop(name, None)
            _LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load every approach once at startup rather than per request.

    Model loading dominates latency for the small model, so paying it here keeps the
    reported per-document timings about extraction rather than about disk I/O.
    """
    _load_approaches()
    yield


app = FastAPI(
    title="Document Intelligence API",
    description="Compare NLP, a fine-tuned small model, and LLM baselines on the "
                "same healthcare document.",
    version="1.0.0",
    lifespan=lifespan,
)


class ExtractResponse(BaseModel):
    doc_id: str
    condition: str
    n_words: int
    n_chars: int
    ocr_ms: float
    results: list[ExtractionResult]
    unavailable: dict[str, str] = {}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "approaches_ready": sorted(_EXTRACTORS),
        "approaches_unavailable": _LOAD_ERRORS,
        "ocr": ocr_available(),
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(...),
    approach: list[str] = Query(default=["all"]),
    condition: str = Query(default="auto", pattern="^(auto|clean|scanned)$"),
    lang: str = Query(default="en"),
    severity: str = Query(default="medium", pattern="^(light|medium|heavy)$"),
) -> ExtractResponse:
    """Run one uploaded document through the requested approaches."""
    names = list(APPROACH_NAMES) if "all" in approach else approach
    unknown = [n for n in names if n not in APPROACH_NAMES]
    if unknown:
        raise HTTPException(400, f"unknown approach(es): {unknown}")

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "empty upload")

    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = handle.name

    try:
        started = time.perf_counter()
        doc = read_document(
            temp_path,
            doc_id=Path(file.filename or "upload").stem,
            condition=condition,
            lang=lang,
            severity=severity,
        )
        ocr_ms = (time.perf_counter() - started) * 1000

        results, unavailable = [], {}
        for name in names:
            extractor = _EXTRACTORS.get(name)
            if extractor is None:
                unavailable[name] = _LOAD_ERRORS.get(name, "not loaded")
                continue
            try:
                results.append(extractor.extract(doc))
            except Exception as exc:
                unavailable[name] = f"{type(exc).__name__}: {exc}"

        return ExtractResponse(
            doc_id=doc.doc_id,
            condition=doc.condition,
            n_words=len(doc.words),
            n_chars=len(doc.text),
            ocr_ms=round(ocr_ms, 2),
            results=results,
            unavailable=unavailable,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)
