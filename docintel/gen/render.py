"""Renders a :class:`GeneratedDoc` to PDF via reportlab, under one of 14 layouts.

The template split is the single most important control in this project. Templates
1-10 are used for training; 11-14 are reserved for the gold test set and are never
trained on. If a template only varied by a colour or a font, that split would be
worthless, so the layouts differ structurally:

* where the value sits relative to its label (right of it, below it, in a table cell)
* what the label is called ("Patient Name" / "PT NAME" / "Name of Patient" / "Paciente")
* whether the page is one column, two columns, prose, or a bordered grid
* whether the header is a letterhead block, a rule-separated banner, or absent

The unseen templates 11-14 lean hardest on the layouts that break naive positional
rules: values below labels, values in grid cells, and prose with values inline.
"""

from __future__ import annotations

import random

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as _canvas

from ..schema import DocType
from .document import GeneratedDoc

PAGE_W, PAGE_H = LETTER
MARGIN = 0.6 * inch

#: Label wording variants per field. Index into these by template so a given template
#: consistently uses one dialect of label naming.
LABELS: dict[str, list[str]] = {
    "patient_name": ["Patient Name", "PATIENT", "Name of Patient", "Patient", "PT NAME"],
    "patient_dob": ["DOB", "Date of Birth", "Birth Date", "D.O.B.", "BIRTHDATE"],
    "referring_provider_name": [
        "Referring Provider", "Referred By", "Ordering Physician",
        "Provider", "REFERRING PHYSICIAN",
    ],
    "referring_provider_npi": ["NPI", "Provider NPI", "NPI #", "National Provider ID"],
    "servicing_facility": [
        "Facility", "Servicing Facility", "Performing Facility",
        "Service Location", "FACILITY NAME",
    ],
    "payer_name": ["Insurance", "Payer", "Insurance Carrier", "Health Plan", "PAYER"],
    "member_id": ["Member ID", "Policy #", "Subscriber ID", "Insurance ID", "MEMBER NO"],
    "document_reference": ["Reference #", "Ref No.", "Document ID", "Control #", "REF"],
    "document_date": ["Date", "Document Date", "Date Issued", "Issued", "DATE"],
    "date_of_service": ["Date of Service", "DOS", "Service Date", "Svc Date", "DATE OF SVC"],
    "diagnosis_code": ["Diagnosis", "ICD-10", "Dx Code", "Diagnosis Code", "DX"],
    "procedure_code": ["Procedure", "CPT", "CPT Code", "Procedure Code", "PROC"],
    "total_charge": ["Total Charge", "Total Charges", "Amount Billed", "Total", "CHARGES"],
    "amount_paid": ["Amount Paid", "Paid", "Plan Paid", "Payment", "PAID AMT"],
    "patient_responsibility": [
        "Patient Responsibility", "Patient Resp.", "Balance Due",
        "You Owe", "PT RESP",
    ],
}

LABELS_ES: dict[str, list[str]] = {
    "patient_name": ["Nombre del Paciente", "Paciente"],
    "patient_dob": ["Fecha de Nacimiento", "F. Nac."],
    "referring_provider_name": ["Médico Remitente", "Proveedor"],
    "referring_provider_npi": ["NPI", "NPI del Proveedor"],
    "servicing_facility": ["Centro", "Centro de Servicio"],
    "payer_name": ["Seguro", "Aseguradora"],
    "member_id": ["N.º de Miembro", "Póliza"],
    "document_reference": ["N.º de Referencia", "Referencia"],
    "document_date": ["Fecha", "Fecha del Documento"],
    "date_of_service": ["Fecha de Servicio", "F. Servicio"],
    "diagnosis_code": ["Diagnóstico", "Código CIE-10"],
    "procedure_code": ["Procedimiento", "Código CPT"],
    "total_charge": ["Cargo Total", "Total"],
    "amount_paid": ["Monto Pagado", "Pagado"],
    "patient_responsibility": ["Responsabilidad del Paciente", "Saldo"],
}

DOC_TITLES: dict[DocType, list[str]] = {
    DocType.referral_letter: ["PATIENT REFERRAL", "REFERRAL LETTER", "SPECIALIST REFERRAL"],
    DocType.prior_auth_request: [
        "PRIOR AUTHORIZATION REQUEST", "PRE-AUTHORIZATION REQUEST",
        "REQUEST FOR PRIOR AUTHORIZATION",
    ],
    DocType.lab_order: ["LABORATORY ORDER", "DIAGNOSTIC ORDER", "LAB REQUISITION"],
    DocType.insurance_claim: ["HEALTH INSURANCE CLAIM FORM", "CLAIM FORM", "CMS-1500 CLAIM"],
    DocType.remittance_advice: [
        "EXPLANATION OF BENEFITS", "REMITTANCE ADVICE", "PAYMENT ADVICE",
    ],
    DocType.patient_intake_form: [
        "PATIENT INTAKE FORM", "NEW PATIENT REGISTRATION", "PATIENT INFORMATION",
    ],
}

DOC_TITLES_ES: dict[DocType, list[str]] = {
    DocType.referral_letter: ["REMISIÓN DEL PACIENTE"],
    DocType.prior_auth_request: ["SOLICITUD DE AUTORIZACIÓN PREVIA"],
    DocType.lab_order: ["ORDEN DE LABORATORIO"],
    DocType.insurance_claim: ["FORMULARIO DE RECLAMACIÓN"],
    DocType.remittance_advice: ["EXPLICACIÓN DE BENEFICIOS"],
    DocType.patient_intake_form: ["FORMULARIO DE ADMISIÓN"],
}


class Layout:
    """How a template positions a label/value pair.

    ``inline``  -- "Patient Name: Jane Doe" on one line (easiest for rules)
    ``below``   -- label on one line, value on the next (breaks same-line regex)
    ``grid``    -- bordered form cells, label small above value (breaks both)
    ``prose``   -- value embedded in a sentence (breaks anchors entirely)
    """

    INLINE = "inline"
    BELOW = "below"
    GRID = "grid"
    PROSE = "prose"


#: Template id -> (layout style, columns, label variant index, has letterhead).
#: 1-10 are the training layouts; 11-14 are held out for the gold test set and are
#: skewed toward the harder styles so the split actually tests generalisation.
TEMPLATES: dict[int, dict] = {
    1:  {"layout": Layout.INLINE, "cols": 1, "labels": 0, "letterhead": True,  "rule": True},
    2:  {"layout": Layout.INLINE, "cols": 2, "labels": 1, "letterhead": True,  "rule": False},
    3:  {"layout": Layout.BELOW,  "cols": 2, "labels": 0, "letterhead": False, "rule": True},
    4:  {"layout": Layout.INLINE, "cols": 1, "labels": 2, "letterhead": False, "rule": False},
    5:  {"layout": Layout.GRID,   "cols": 2, "labels": 1, "letterhead": True,  "rule": True},
    6:  {"layout": Layout.INLINE, "cols": 2, "labels": 3, "letterhead": True,  "rule": True},
    7:  {"layout": Layout.PROSE,  "cols": 1, "labels": 0, "letterhead": True,  "rule": False},
    8:  {"layout": Layout.BELOW,  "cols": 1, "labels": 4, "letterhead": False, "rule": True},
    9:  {"layout": Layout.GRID,   "cols": 3, "labels": 2, "letterhead": False, "rule": False},
    10: {"layout": Layout.INLINE, "cols": 1, "labels": 1, "letterhead": True,  "rule": True},
    # --- held out: gold test only -------------------------------------------------
    11: {"layout": Layout.GRID,   "cols": 3, "labels": 4, "letterhead": True,  "rule": False},
    12: {"layout": Layout.PROSE,  "cols": 1, "labels": 2, "letterhead": False, "rule": True},
    13: {"layout": Layout.BELOW,  "cols": 3, "labels": 3, "letterhead": True,  "rule": False},
    14: {"layout": Layout.GRID,   "cols": 2, "labels": 0, "letterhead": False, "rule": True},
}

TRAIN_TEMPLATES = list(range(1, 11))
HELDOUT_TEMPLATES = list(range(11, 15))

FONTS = [("Helvetica", "Helvetica-Bold"), ("Times-Roman", "Times-Bold"),
         ("Courier", "Courier-Bold")]


def label_for(field: str, template_id: int, lang: str) -> str:
    """The label wording this template uses for this field."""
    if lang == "es":
        variants = LABELS_ES.get(field, LABELS.get(field, [field]))
        return variants[TEMPLATES[template_id]["labels"] % len(variants)]
    variants = LABELS.get(field, [field])
    return variants[TEMPLATES[template_id]["labels"] % len(variants)]


class _Writer:
    """Thin cursor over a reportlab canvas that tracks vertical position."""

    def __init__(self, c: _canvas.Canvas, font: str, bold: str, size: float = 9.5):
        self.c, self.font, self.bold, self.size = c, font, bold, size
        self.y = PAGE_H - MARGIN

    def text(self, x: float, s: str, bold: bool = False, size: float | None = None):
        self.c.setFont(self.bold if bold else self.font, size or self.size)
        self.c.drawString(x, self.y, s)

    def text_at(self, x: float, y: float, s: str, bold: bool = False,
                size: float | None = None):
        self.c.setFont(self.bold if bold else self.font, size or self.size)
        self.c.drawString(x, y, s)

    def down(self, dy: float = 14):
        self.y -= dy

    def rule(self, pad: float = 4):
        self.c.setStrokeColor(colors.black)
        self.c.setLineWidth(0.7)
        self.c.line(MARGIN, self.y - pad, PAGE_W - MARGIN, self.y - pad)
        self.down(pad + 8)


def _draw_pair(w: _Writer, x: float, label: str, value: str, style: str,
               col_w: float) -> None:
    """Draw one label/value pair in the template's style, advancing the cursor."""
    if value is None:
        return
    if style == Layout.INLINE:
        w.text(x, f"{label}:", bold=True)
        w.c.setFont(w.font, w.size)
        w.c.drawString(x + max(w.c.stringWidth(f"{label}:", w.bold, w.size) + 6, 96), w.y, value)
        w.down()
    elif style == Layout.BELOW:
        w.text(x, f"{label}:", bold=True, size=w.size - 1.2)
        w.down(11)
        w.text(x, value)
        w.down(15)
    elif style == Layout.GRID:
        box_h = 26
        w.c.setStrokeColor(colors.grey)
        w.c.setLineWidth(0.5)
        w.c.rect(x, w.y - box_h + 10, col_w - 8, box_h, stroke=1, fill=0)
        w.text_at(x + 3, w.y + 1, label.upper(), bold=True, size=w.size - 2.5)
        w.text_at(x + 3, w.y - 10, value)
        w.down(box_h + 2)


def _service_line_table(w: _Writer, doc: GeneratedDoc, show_paid: bool) -> None:
    """The service-line table — the relationship task's source of truth."""
    if not doc.service_lines:
        return
    headers = ["CPT", "Description", "DOS", "Units", "Charge"]
    if show_paid:
        headers.append("Paid")
    xs = [MARGIN, MARGIN + 48, MARGIN + 230, MARGIN + 310, MARGIN + 352, MARGIN + 410]

    w.c.setFont(w.bold, w.size - 0.5)
    for x, h in zip(xs, headers):
        w.c.drawString(x, w.y, h)
    w.down(3)
    w.c.setLineWidth(0.6)
    w.c.line(MARGIN, w.y, PAGE_W - MARGIN, w.y)
    w.down(12)

    for sl in doc.service_lines:
        cells = [sl.procedure_code, sl.description[:34], sl.date_of_service,
                 sl.units, sl.charge]
        if show_paid:
            cells.append(sl.paid)
        w.c.setFont(w.font, w.size - 0.5)
        for x, cell in zip(xs, cells):
            w.c.drawString(x, w.y, str(cell))
        w.down(13)
    w.down(4)


def _letterhead(w: _Writer, doc: GeneratedDoc, tpl: dict) -> None:
    c = doc.content
    # Which org owns the letterhead depends on the document type. This is exactly the
    # signal that makes org-role assignment learnable but not trivially regexable.
    if doc.doc_type == DocType.remittance_advice:
        org, addr = c["payer_name"], c["servicing_address"]
    elif doc.doc_type in (DocType.referral_letter, DocType.prior_auth_request,
                          DocType.lab_order):
        org, addr = c["referring_org"], c["referring_address"]
    else:
        org, addr = c["servicing_org"], c["servicing_address"]

    w.text(MARGIN, org, bold=True, size=13)
    w.down(13)
    w.text(MARGIN, addr, size=8)
    w.down(10)
    w.text(MARGIN, f"Tel {c['referring_phone']}   Fax {c['fax_number']}", size=8)
    w.down(12)
    if tpl["rule"]:
        w.rule()


#: Fallback sentence per field, used to guarantee the prose layout prints every field
#: the ground truth claims. Written as sentences so values stay anchor-free.
_PROSE_SENTENCES_EN: dict[str, str] = {
    "patient_name": "The patient of record is {v}.",
    "patient_dob": "The patient was born on {v}.",
    "referring_provider_name": "This was submitted by {v}.",
    "referring_provider_npi": "The ordering provider is registered under NPI {v}.",
    "servicing_facility": "Services were rendered at {v}.",
    "payer_name": "Coverage is provided by {v}.",
    "member_id": "The subscriber policy on file is {v}.",
    "document_reference": "Please quote {v} in all correspondence.",
    "document_date": "This notice was issued on {v}.",
    "date_of_service": "The service was performed on {v}.",
    "diagnosis_code": "The coded diagnosis is {v}.",
    "procedure_code": "The procedure billed was coded {v}.",
    "total_charge": "The total amount billed came to {v}.",
    "amount_paid": "The plan has remitted {v} against this account.",
    "patient_responsibility": "The remaining balance owed by the patient is {v}.",
}

_PROSE_SENTENCES_ES: dict[str, str] = {
    "patient_name": "El paciente registrado es {v}.",
    "patient_dob": "El paciente nació el {v}.",
    "referring_provider_name": "Fue enviado por {v}.",
    "referring_provider_npi": "El proveedor está registrado con el NPI {v}.",
    "servicing_facility": "Los servicios se prestaron en {v}.",
    "payer_name": "La cobertura la proporciona {v}.",
    "member_id": "La póliza del asegurado es {v}.",
    "document_reference": "Por favor cite {v} en toda correspondencia.",
    "document_date": "Este aviso se emitió el {v}.",
    "date_of_service": "El servicio se realizó el {v}.",
    "diagnosis_code": "El diagnóstico codificado es {v}.",
    "procedure_code": "El procedimiento facturado se codificó {v}.",
    "total_charge": "El importe total facturado ascendió a {v}.",
    "amount_paid": "El plan ha pagado {v} de esta cuenta.",
    "patient_responsibility": "El saldo pendiente del paciente es {v}.",
}


def _prose_body(w: _Writer, doc: GeneratedDoc, fields: list[str]) -> None:
    """Embed values in sentences — no anchors, which is the point of this layout.

    Coverage is driven by the ground truth rather than by a hand-written script: after
    the narrative opening, any truth value that has not literally appeared yet gets its
    own sentence. A gold label for a field the page never prints is unlearnable and
    unscoreable, so this loop is what keeps prose templates honest.
    """
    c = doc.content
    es = doc.lang == "es"
    payment_doc = doc.doc_type == DocType.remittance_advice

    if es:
        if payment_doc:
            lines = [
                f"Estimado proveedor: adjuntamos la explicación de beneficios de "
                f"{c['patient_name']},",
                f"asegurado por {c['payer_name']} con póliza {c['member_id']}.",
                f"La atención se prestó en {c['servicing_org']} el {c['date_of_service']}.",
            ]
        else:
            lines = [
                f"Estimado colega: le remito a {c['patient_name']}, "
                f"nacido el {c['patient_dob']},",
                f"asegurado por {c['payer_name']} con póliza {c['member_id']}.",
                f"El diagnóstico es {c['diagnosis_desc']} ({c['diagnosis_code']}).",
                f"Solicito evaluación en {c['servicing_org']} el {c['date_of_service']}.",
                f"Atentamente, {c['provider_name']} (NPI {c['provider_npi']}).",
            ]
        lines.append(
            f"Referencia {c['document_reference']}, emitido {c['document_date']}."
        )
    else:
        if payment_doc:
            lines = [
                f"Dear Provider: enclosed is the explanation of benefits for "
                f"{c['patient_name']},",
                f"who is covered by {c['payer_name']} under policy {c['member_id']}.",
                f"Care was delivered at {c['servicing_org']} on {c['date_of_service']}.",
            ]
        else:
            lines = [
                f"Dear Colleague: I am referring {c['patient_name']}, "
                f"born {c['patient_dob']},",
                f"who is covered by {c['payer_name']} under policy {c['member_id']}.",
                f"The working diagnosis is {c['diagnosis_desc']} ({c['diagnosis_code']}).",
                f"Please evaluate at {c['servicing_org']} on {c['date_of_service']}.",
                f"Kind regards, {c['provider_name']} (NPI {c['provider_npi']}).",
            ]
        lines.append(
            f"Our reference {c['document_reference']} was issued {c['document_date']}."
        )

    # Guarantee every claimed field is actually printed somewhere on the page.
    sentences = _PROSE_SENTENCES_ES if es else _PROSE_SENTENCES_EN
    for fname, value in doc.truth.items():
        if not value or any(value in ln for ln in lines):
            continue
        template = sentences.get(fname)
        if template:
            lines.append(template.format(v=value))

    for line in lines:
        w.text(MARGIN, line)
        w.down(15)


def render_pdf(doc: GeneratedDoc, path: str, rng: random.Random | None = None) -> None:
    """Render ``doc`` to a PDF at ``path`` using its template."""
    rng = rng or random.Random(doc.doc_id)
    tpl = TEMPLATES[doc.template_id]
    font, bold = FONTS[doc.template_id % len(FONTS)]

    c = _canvas.Canvas(path, pagesize=LETTER)
    c.setTitle(doc.doc_id)
    w = _Writer(c, font, bold)
    content = doc.content

    if tpl["letterhead"]:
        _letterhead(w, doc, tpl)

    titles = DOC_TITLES_ES if doc.lang == "es" else DOC_TITLES
    title = rng.choice(titles.get(doc.doc_type, DOC_TITLES[doc.doc_type]))
    w.text(MARGIN, title, bold=True, size=12)
    w.down(18)
    if tpl["rule"] and not tpl["letterhead"]:
        w.rule()

    fields = [f for f in doc.truth if doc.truth[f] is not None]

    if tpl["layout"] == Layout.PROSE:
        _prose_body(w, doc, fields)
    else:
        cols = tpl["cols"]
        col_w = (PAGE_W - 2 * MARGIN) / cols
        start_y = w.y
        lowest = w.y
        per_col = (len(fields) + cols - 1) // cols
        for i, fname in enumerate(fields):
            col = i // per_col if per_col else 0
            if i % per_col == 0:
                w.y = start_y
            x = MARGIN + col * col_w
            _draw_pair(w, x, label_for(fname, doc.template_id, doc.lang),
                       doc.truth[fname], tpl["layout"], col_w)
            lowest = min(lowest, w.y)
        w.y = lowest
        w.down(6)

    if doc.service_lines:
        w.down(4)
        _service_line_table(
            w, doc, show_paid=doc.doc_type == DocType.remittance_advice
        )

    # Distractor block: numbers and names that look extractable but are not answers.
    w.down(6)
    if tpl["rule"]:
        w.rule()
    w.text(MARGIN, f"Account No: {content['account_number']}", size=8)
    w.down(11)
    w.text(MARGIN, f"Fax: {content['fax_number']}   Group: {content['group_number']}", size=8)
    w.down(11)
    w.text(MARGIN, f"Printed: {content['print_date']}   Contact: {content['contact_name']}",
           size=8)
    w.down(11)
    if doc.doc_type != DocType.remittance_advice:
        w.text(MARGIN, f"Patient Phone: {content['patient_phone']}", size=8)
        w.down(11)

    c.showPage()
    c.save()
