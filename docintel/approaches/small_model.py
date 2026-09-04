"""The small-model approach: a fine-tuned DistilBERT token classifier.

Two heads, both built on ``distilbert-base-cased`` (~66M parameters, ~260MB on disk):

* **token classification** over 31 BIO labels, for the 15 extraction fields
* **sequence classification** over the 6 document types

Design decisions worth defending:

* **Cased, not uncased.** Half these fields are proper nouns — names, organisations,
  ICD-10 codes. Lowercasing throws away the single strongest surface cue the model has.
* **Extractive, not generative.** The model can only label spans that exist in the
  input, so it is *structurally incapable* of hallucinating a value. That is not a
  tuning result, it is an architectural guarantee, and it is a real argument for this
  approach in a clinical setting.
* **Sliding window with stride.** Documents exceed 512 tokens. Windows overlap by
  ``stride`` tokens and predictions are merged by confidence, so an entity that
  straddles a window boundary is still recoverable.
* **Trained on silver labels only.** Training on the generator's ground truth would
  measure something this pipeline could never have in production. The gold-trained
  variant is reported separately, as an upper bound.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..align import bio_to_values, pick_best_value
from ..schema import (
    BIO_LABELS,
    ID2LABEL,
    LABEL2ID,
    DocType,
    ExtractedField,
    ExtractionResult,
    OcrDocument,
    ServiceLine,
    expected_fields,
)

BASE_MODEL = "distilbert-base-cased"
MAX_LENGTH = 512
STRIDE = 128


def _device(override: str | None = None) -> str:
    """Pick a device, honouring an explicit override.

    An override is genuinely needed rather than a nicety: the 4GB GPU on this machine
    also hosts the local LLM during silver-label generation, so training has to be able
    to run on CPU on demand. It is also how the report measures CPU-only inference
    latency, which is the number that matters for "can this ship without a GPU?".
    """
    import torch

    if override:
        return override
    if os.environ.get("DOCINTEL_DEVICE"):
        return os.environ["DOCINTEL_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


class SmallModelExtractor:
    """Runs the fine-tuned token classifier + document-type classifier."""

    name = "small_model"

    def __init__(
        self,
        model_dir: str | Path = "models/small",
        device: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device or _device()
        self._tokenizer = None
        self._token_model = None
        self._doctype_model = None
        self._doctype_labels: list[str] = []

    # -- loading -----------------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            source = self.model_dir / "tokens"
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(source) if source.exists() else BASE_MODEL
            )
        return self._tokenizer

    @property
    def token_model(self):
        if self._token_model is None:
            from transformers import AutoModelForTokenClassification

            path = self.model_dir / "tokens"
            if not path.exists():
                raise FileNotFoundError(
                    f"No trained token model at {path}. Run: python -m docintel train"
                )
            self._token_model = AutoModelForTokenClassification.from_pretrained(
                str(path)
            ).to(self.device).eval()
        return self._token_model

    @property
    def doctype_model(self):
        if self._doctype_model is None:
            from transformers import AutoModelForSequenceClassification

            path = self.model_dir / "doctype"
            if not path.exists():
                return None
            self._doctype_model = AutoModelForSequenceClassification.from_pretrained(
                str(path)
            ).to(self.device).eval()
            labels = json.loads((path / "labels.json").read_text(encoding="utf-8"))
            self._doctype_labels = labels
        return self._doctype_model

    # -- inference ---------------------------------------------------------------
    def _encode_windows(self, text: str):
        return self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding="max_length",
            return_tensors="pt",
        )

    def predict_tokens(self, text: str) -> dict[str, list[tuple[str, int, int]]]:
        """Run the token classifier over all windows and merge the decoded spans."""
        import torch

        encoded = self._encode_windows(text)
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.token_model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)

        # Merge windows at the character level: for each character position keep the
        # label from whichever window was most confident about it. This is what makes
        # an entity spanning a window boundary recoverable instead of half-labelled.
        best_conf: dict[int, float] = {}
        best_label: dict[int, str] = {}
        merged_offsets: dict[int, tuple[int, int]] = {}

        for w in range(predicted.shape[0]):
            for t in range(predicted.shape[1]):
                start, end = int(offsets[w][t][0]), int(offsets[w][t][1])
                if start == end == 0:
                    continue
                score = float(confidence[w][t])
                if score > best_conf.get(start, -1.0):
                    best_conf[start] = score
                    best_label[start] = ID2LABEL[int(predicted[w][t])]
                    merged_offsets[start] = (start, end)

        ordered = sorted(merged_offsets)
        labels = [best_label[s] for s in ordered]
        spans = [merged_offsets[s] for s in ordered]
        return bio_to_values(text, labels, spans)

    def predict_doctype(self, text: str) -> tuple[DocType, float]:
        import torch

        model = self.doctype_model
        if model is None:
            return DocType.unknown, 0.0
        encoded = self.tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        index = int(probabilities.argmax())
        return DocType(self._doctype_labels[index]), float(probabilities[index])

    def extract(self, doc: OcrDocument) -> ExtractionResult:
        started = time.perf_counter()
        candidates = self.predict_tokens(doc.text)
        doc_type, confidence = self.predict_doctype(doc.text)
        wanted = expected_fields(doc_type) or set(LABEL2ID)

        fields: dict[str, ExtractedField | None] = {}
        for name in wanted:
            found = candidates.get(name) or []
            value = pick_best_value(found, name)
            if value:
                span = next((c for c in found if c[0] == value), None)
                fields[name] = ExtractedField(
                    value=value,
                    raw=value,
                    start=span[1] if span else None,
                    end=span[2] if span else None,
                    confidence=None,
                )
            else:
                fields[name] = None

        return ExtractionResult(
            doc_id=doc.doc_id,
            approach=self.name,
            doc_type=doc_type,
            doc_type_confidence=confidence,
            fields=fields,
            service_lines=self._service_lines(candidates),
            org_roles={
                "referring_org": None,
                "servicing_org": (fields.get("servicing_facility") or ExtractedField(value="")).value or None,
                "payer_org": (fields.get("payer_name") or ExtractedField(value="")).value or None,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
            meta={"device": self.device, "condition": doc.condition},
        )

    @staticmethod
    def _service_lines(candidates: dict) -> list[ServiceLine]:
        """Pair up procedure codes with charges by document order.

        A token classifier has no notion of table rows, so this zips the tagged
        procedure codes against the tagged charges positionally. It is a deliberate
        weakness of the extractive approach on tabular relationships, and the
        evaluation is expected to show it.
        """
        codes = candidates.get("procedure_code") or []
        charges = candidates.get("total_charge") or []
        dates = candidates.get("date_of_service") or []
        out = []
        for i, (code, _, _) in enumerate(codes):
            out.append(
                ServiceLine(
                    procedure_code=code,
                    date_of_service=dates[i][0] if i < len(dates) else None,
                    charge=charges[i][0] if i < len(charges) else None,
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_token_classifier(
    examples: list[dict],
    output_dir: str | Path,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    base_model: str = BASE_MODEL,
    seed: int = 42,
    eval_examples: list[dict] | None = None,
    device: str | None = None,
) -> dict:
    """Fine-tune the token classifier.

    ``examples`` are ``{"text": str, "labels": {field: (start, end)}}`` records, which
    is what the silver-labelling pipeline emits after span alignment.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    device = _device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    class SpanDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            encoded = tokenizer(
                row["text"], truncation=True, max_length=MAX_LENGTH,
                padding="max_length", return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            labels = [-100] * len(offsets)
            for i, (start, end) in enumerate(offsets):
                if start == end == 0:
                    continue
                labels[i] = LABEL2ID["O"]
            for field, (span_start, span_end) in row["labels"].items():
                first = True
                for i, (start, end) in enumerate(offsets):
                    if start == end == 0 or start >= span_end or end <= span_start:
                        continue
                    key = f"{'B' if first else 'I'}-{field}"
                    if key in LABEL2ID:
                        labels[i] = LABEL2ID[key]
                        first = False
            item = {k: torch.tensor(v) for k, v in encoded.items()}
            item["labels"] = torch.tensor(labels)
            return item

    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(device)

    loader = DataLoader(SpanDataset(examples), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
        mean_loss = running / max(1, len(loader))
        history.append({"epoch": epoch + 1, "loss": round(mean_loss, 4)})
        print(f"  epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}", flush=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    size_mb = sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) / 1e6
    return {
        "history": history,
        "n_examples": len(examples),
        "model_size_mb": round(size_mb, 1),
        "params": sum(p.numel() for p in model.parameters()),
        "device": device,
    }


def train_doctype_classifier(
    texts: list[str],
    labels: list[str],
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    base_model: str = BASE_MODEL,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Fine-tune the document-type classification head."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = _device(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    classes = sorted(set(labels))
    class_to_id = {c: i for i, c in enumerate(classes)}

    class DocDataset(Dataset):
        def __init__(self, xs, ys):
            self.xs, self.ys = xs, ys

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            encoded = tokenizer(
                self.xs[i], truncation=True, max_length=MAX_LENGTH,
                padding="max_length",
            )
            item = {k: torch.tensor(v) for k, v in encoded.items()}
            item["labels"] = torch.tensor(class_to_id[self.ys[i]])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(classes)
    ).to(device)
    loader = DataLoader(DocDataset(texts, labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    model.train()
    history = []
    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss)
        history.append({"epoch": epoch + 1, "loss": round(running / max(1, len(loader)), 4)})
        print(f"  doctype epoch {epoch + 1}/{epochs} loss={history[-1]['loss']:.4f}", flush=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "labels.json").write_text(json.dumps(classes), encoding="utf-8")
    return {"history": history, "classes": classes, "n_examples": len(texts)}
