"""Turns a PDF into an :class:`OcrDocument`: plain text, plus word boxes.

Two conditions are supported, and the distinction is central to the evaluation:

``clean``
    The PDF's digital text layer, read with PyMuPDF. This is the best case: no
    character errors at all. Represents documents that arrive as real PDFs.

``scanned``
    The page is rasterised, degraded (skew, blur, noise, JPEG artefacts, fax-style
    thresholding), then passed through an OCR engine. Represents the fax-in reality
    that motivates this problem, and is where the OCR error category comes from.

Both paths produce the same structure, so no downstream code needs to branch on it.
Text is rebuilt from the word list rather than taken from a separate call, which
guarantees that every word's ``start``/``end`` offsets index correctly into ``text`` —
the span aligner and the token classifier both depend on that being exact.
"""

from __future__ import annotations

import io
import os
import math
import random
from dataclasses import dataclass

import fitz
from pathlib import Path

from .schema import OcrDocument, Word


@dataclass
class WordBox:
    """An engine-agnostic word with a box, before offsets are assigned."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0
    line_key: tuple = ()


#: A horizontal gap wider than this many "space widths" is treated as a column break
#: rather than a word space.
COLUMN_GAP_SPACES = 2.2
#: Cap on the run of spaces emitted for a column break, so one stray wide gap cannot
#: blow up the text length.
MAX_GAP_SPACES = 10


def _assemble(words: list[WordBox]) -> tuple[str, list[Word]]:
    """Join words into text while recording each word's exact char offsets.

    Lines are separated by a newline. Within a line, words are separated by a single
    space *unless* the horizontal gap between them is wide enough to be a column
    boundary, in which case a proportional run of spaces is emitted instead.

    That gap matters. In the two-column templates a row reads
    ``Patient Name:        Reference #:`` — visually two cells, but collapsed to a
    single space it becomes indistinguishable from an ordinary phrase, and both the LLM
    and the rules extractor then merge the two columns' values into one field. Keeping
    the gap (the same trick ``pdftotext -layout`` uses) preserves the only evidence
    that a column boundary was ever there.

    Building the string here, rather than trusting a separate text call to match, is
    what makes ``text[w.start:w.end] == w.text`` an invariant instead of a hope.
    """
    parts: list[str] = []
    out: list[Word] = []
    pos = 0
    last_key: tuple | None = None
    last_box: WordBox | None = None

    heights = sorted(w.y1 - w.y0 for w in words) if words else []
    median_height = heights[len(heights) // 2] if heights else 10.0
    # Rough advance width of a space for this page's dominant font size.
    space_width = max(2.0, median_height * 0.30)

    for wb in words:
        if last_key is not None:
            if wb.line_key != last_key:
                sep = "\n"
            else:
                gap = wb.x0 - (last_box.x1 if last_box else wb.x0)
                if gap > space_width * COLUMN_GAP_SPACES:
                    sep = " " * min(MAX_GAP_SPACES, max(2, int(round(gap / space_width))))
                else:
                    sep = " "
            parts.append(sep)
            pos += len(sep)
        start = pos
        parts.append(wb.text)
        pos += len(wb.text)
        out.append(
            Word(
                text=wb.text, start=start, end=pos, page=wb.page,
                x0=wb.x0, y0=wb.y0, x1=wb.x1, y1=wb.y1,
            )
        )
        last_key = wb.line_key
        last_box = wb

    return "".join(parts), out


def group_into_lines(boxes: list[WordBox], tolerance: float | None = None) -> list[WordBox]:
    """Cluster words into visual lines by vertical position, then order left-to-right.

    PyMuPDF's own block/line numbering follows the order content was *drawn*, not how
    the page *reads*. A template that draws a label and its value as two separate
    ``drawString`` calls therefore comes back as two lines, which made the ``inline``
    layouts indistinguishable from the ``below`` layouts in the extracted text — and
    that distinction is exactly what the held-out template split is meant to test.

    Clustering on the y-centre instead reproduces what a human (or an OCR engine)
    sees, so a value printed beside its label stays beside it.
    """
    if not boxes:
        return []
    if tolerance is None:
        heights = sorted(b.y1 - b.y0 for b in boxes)
        median_h = heights[len(heights) // 2] or 8.0
        tolerance = max(3.0, median_h * 0.6)

    ordered = sorted(boxes, key=lambda b: (b.page, (b.y0 + b.y1) / 2, b.x0))
    lines: list[list[WordBox]] = []
    current: list[WordBox] = []
    line_page = None
    line_y = None

    for box in ordered:
        y_centre = (box.y0 + box.y1) / 2
        if (
            current
            and box.page == line_page
            and abs(y_centre - line_y) <= tolerance
        ):
            current.append(box)
            # Track the running centre so a gently sloping line stays one line.
            line_y = sum((b.y0 + b.y1) / 2 for b in current) / len(current)
        else:
            if current:
                lines.append(current)
            current = [box]
            line_page, line_y = box.page, y_centre
    if current:
        lines.append(current)

    out: list[WordBox] = []
    for index, line in enumerate(lines):
        for box in sorted(line, key=lambda b: b.x0):
            box.line_key = (box.page, index)
            out.append(box)
    return out


def read_clean(path: str, doc_id: str, lang: str = "en") -> OcrDocument:
    """Read a digital PDF's text layer with word boxes — the no-OCR-error condition."""
    pdf = fitz.open(path)
    boxes: list[WordBox] = []
    for pno, page in enumerate(pdf):
        # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            if not text.strip():
                continue
            boxes.append(WordBox(text=text, x0=x0, y0=y0, x1=x1, y1=y1, page=pno))
    boxes = group_into_lines(boxes)
    text, words = _assemble(boxes)
    return OcrDocument(
        doc_id=doc_id, text=text, words=words, page_count=pdf.page_count,
        condition="clean", lang=lang, source_path=path,
    )


# --------------------------------------------------------------------------------------
# Scanned path: rasterise -> degrade -> OCR
# --------------------------------------------------------------------------------------

def render_page_image(path: str, dpi: int = 200, page: int = 0):
    """Rasterise one PDF page to a PIL image."""
    from PIL import Image

    pdf = fitz.open(path)
    pix = pdf[page].get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")


def degrade(image, rng: random.Random, severity: str = "medium"):
    """Apply fax/scanner-style degradation to a page image.

    Tuned to land around 92-96% character accuracy: hard enough that OCR errors are a
    real error category worth analysing, not so hard the task collapses and every
    approach fails for the same uninteresting reason.
    """
    from PIL import Image, ImageFilter

    params = {
        "light":  dict(angle=0.4, blur=0.3, noise=4,  quality=70, threshold=False),
        "medium": dict(angle=1.0, blur=0.6, noise=9,  quality=45, threshold=False),
        "heavy":  dict(angle=1.6, blur=0.9, noise=15, quality=30, threshold=True),
    }[severity]

    img = image.rotate(
        rng.uniform(-params["angle"], params["angle"]),
        resample=Image.BICUBIC, expand=False, fillcolor=255,
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=params["blur"]))

    # Salt-and-pepper speckle, the signature artefact of a fax transmission.
    px = img.load()
    w, h = img.size
    n_speckle = int(w * h * params["noise"] / 10000)
    for _ in range(n_speckle):
        x, y = rng.randrange(w), rng.randrange(h)
        px[x, y] = 0 if rng.random() < 0.5 else 255

    if params["threshold"]:
        img = img.point(lambda p: 255 if p > 128 else 0)

    buf = io.BytesIO()
    img.convert("L").save(buf, format="JPEG", quality=params["quality"])
    buf.seek(0)
    return Image.open(buf).convert("L")


#: Local tessdata directory, so the Spanish model ships with the repo rather than
#: depending on what happens to be in Program Files.
TESSDATA_DIR = Path(__file__).resolve().parent.parent / "models" / "tessdata"

#: Tesseract is installed outside PATH by the standard Windows installer.
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def _configure_tesseract() -> bool:
    """Point pytesseract at the installed binary. Returns whether one was found."""
    try:
        import pytesseract
    except Exception:
        return False

    from shutil import which

    found = which("tesseract")
    if not found:
        for candidate in _TESSERACT_CANDIDATES:
            if Path(candidate).exists():
                found = candidate
                break
    if not found:
        return False
    pytesseract.pytesseract.tesseract_cmd = found
    return True


def ocr_available() -> bool:
    """Whether an OCR engine is installed. The scanned condition needs one."""
    return _configure_tesseract()


#: Map our language codes onto Tesseract's.
_TESS_LANG = {"en": "eng", "es": "spa"}


def read_scanned(
    path: str,
    doc_id: str,
    lang: str = "en",
    severity: str = "medium",
    seed: int | None = None,
) -> OcrDocument:
    """Rasterise, degrade and OCR a PDF — the noisy condition.

    Tesseract is used rather than a PP-OCR-family model because it returns genuine
    *word*-level boxes. An earlier attempt with RapidOCR produced line-level output
    whose recognition model dropped inter-word spaces ("Patient Name" -> "PatientName"),
    which destroyed both the word geometry the rules approach depends on and the
    exact-match scoring. OCR noise should corrupt characters, not tokenisation.
    """
    if not _configure_tesseract():
        raise RuntimeError(
            "Tesseract not found. Install with: "
            "winget install --id UB-Mannheim.TesseractOCR"
        )
    import pytesseract
    from pytesseract import Output

    rng = random.Random(seed if seed is not None else doc_id)
    pdf = fitz.open(path)
    boxes: list[WordBox] = []

    tess_lang = _TESS_LANG.get(lang, "eng")
    # TESSDATA_PREFIX rather than --tessdata-dir: pytesseract splits the config string
    # on spaces, so a quoted path arrives at Tesseract with its quotes intact and the
    # lookup fails. The env var also survives paths that contain spaces.
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    config = "--oem 3 --psm 6"

    for pno in range(pdf.page_count):
        img = degrade(render_page_image(path, page=pno), rng, severity)
        data = pytesseract.image_to_data(
            img, lang=tess_lang, config=config, output_type=Output.DICT
        )
        scale = 72.0 / 200.0  # rasterised at 200 dpi; report boxes in PDF points
        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            left, top = data["left"][i] * scale, data["top"][i] * scale
            width, height = data["width"][i] * scale, data["height"][i] * scale
            boxes.append(
                WordBox(
                    text=word, x0=left, y0=top, x1=left + width, y1=top + height,
                    page=pno,
                )
            )

    boxes = group_into_lines(boxes)
    text, words = _assemble(boxes)
    return OcrDocument(
        doc_id=doc_id, text=text, words=words, page_count=pdf.page_count,
        condition="scanned", lang=lang, source_path=path,
    )


def read_document(
    path: str,
    doc_id: str | None = None,
    condition: str = "auto",
    lang: str = "en",
    severity: str = "medium",
) -> OcrDocument:
    """Front door used by the CLI and API.

    ``auto`` prefers the digital text layer and falls back to OCR when the PDF has
    none — which is the correct behaviour for a real intake pipeline receiving a mix
    of digital PDFs and scanned faxes.
    """
    doc_id = doc_id or path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if condition == "scanned":
        return read_scanned(path, doc_id, lang=lang, severity=severity)
    if condition == "clean":
        return read_clean(path, doc_id, lang=lang)

    doc = read_clean(path, doc_id, lang=lang)
    if len(doc.text.strip()) >= 40:
        return doc
    if ocr_available():
        return read_scanned(path, doc_id, lang=lang, severity="light")
    return doc


def character_accuracy(reference: str, hypothesis: str) -> float:
    """Character-level accuracy of an OCR result against the true text.

    Used to report how noisy the ``scanned`` condition actually is, so the error
    analysis can attribute failures to OCR with a number rather than an assertion.
    Levenshtein is computed with a rolling row to stay within this machine's memory.
    """
    ref, hyp = " ".join(reference.split()), " ".join(hypothesis.split())
    if not ref:
        return 1.0 if not hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(ref))
