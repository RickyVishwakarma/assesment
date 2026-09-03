"""Samples one synthetic document: its content, and the ground truth that goes with it.

Two rules govern this module, and both exist to keep the benchmark honest:

1. **Truth is a surface form, not an idea.** ``truth["patient_dob"]`` stores the exact
   string that will be printed on the page (``"14-Mar-1980"``), not a canonical date.
   The span aligner and the token classifier both need to find that string in the OCR
   text, and gold labels that don't literally occur in the document are unusable.
2. **Every document carries distractors.** A page with exactly one 10-digit number and
   one date is not a document-understanding problem, it's a regex exercise. So each
   document also gets fax numbers, account numbers, print dates and secondary people
   that look extractable and are not the answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from ..schema import DocType, expected_fields
from . import pools

# --------------------------------------------------------------------------------------
# Surface formatting. A document picks one date style and one money style and mostly
# sticks to it, the way a real template would.
# --------------------------------------------------------------------------------------

_MONTH_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
_MONTH_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
             "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

DATE_STYLES = ["mdy_slash", "mdy_slash_short", "iso", "dmy_mon", "month_name", "dot"]
DATE_STYLES_ES = ["dmy_mon", "es_long", "iso", "mdy_slash"]


def format_date(d: date, style: str, lang: str = "en") -> str:
    """Render a date in one of the styles this corpus emits."""
    if style == "iso":
        return d.isoformat()
    if style == "mdy_slash":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "mdy_slash_short":
        return f"{d.month}/{d.day}/{str(d.year)[2:]}"
    if style == "dot":
        return f"{d.month:02d}.{d.day:02d}.{d.year}"
    if style == "dmy_mon":
        return f"{d.day:02d}-{_MONTH_EN[d.month - 1][:3]}-{d.year}"
    if style == "month_name":
        return f"{_MONTH_EN[d.month - 1]} {d.day}, {d.year}"
    if style == "es_long":
        return f"{d.day} de {_MONTH_ES[d.month - 1]} de {d.year}"
    raise ValueError(f"unknown date style: {style}")


MONEY_STYLES = ["dollar_comma", "plain_comma", "dollar_plain", "usd_comma"]


def format_money(amount: float, style: str) -> str:
    """Render a monetary amount in one of the styles this corpus emits."""
    if style == "dollar_comma":
        return f"${amount:,.2f}"
    if style == "plain_comma":
        return f"{amount:,.2f}"
    if style == "dollar_plain":
        return f"${amount:.2f}"
    if style == "usd_comma":
        return f"USD {amount:,.2f}"
    raise ValueError(f"unknown money style: {style}")


#: Reference-number prefix per document type.
REF_PREFIXES: dict[DocType, list[str]] = {
    DocType.referral_letter: ["REF", "RF"],
    DocType.prior_auth_request: ["AUTH", "PA"],
    DocType.lab_order: ["ORD", "LAB"],
    DocType.insurance_claim: ["CLM", "CL"],
    DocType.remittance_advice: ["EOB", "RA"],
    DocType.patient_intake_form: ["INT", "PT"],
}


@dataclass
class ServiceLineData:
    procedure_code: str
    description: str
    date_of_service: str
    units: str
    charge: str
    paid: str


@dataclass
class GeneratedDoc:
    """A sampled document: everything the renderer needs, plus the ground truth."""

    doc_id: str
    doc_type: DocType
    template_id: int
    lang: str
    content: dict
    truth: dict[str, str | None]
    service_lines: list[ServiceLineData] = field(default_factory=list)
    org_roles: dict[str, str | None] = field(default_factory=dict)
    date_style: str = "mdy_slash"
    money_style: str = "dollar_comma"


def sample_document(
    doc_id: str,
    doc_type: DocType,
    template_id: int,
    rng: random.Random,
    lang: str = "en",
) -> GeneratedDoc:
    """Sample one document of the given type and template."""
    date_style = rng.choice(DATE_STYLES_ES if lang == "es" else DATE_STYLES)
    money_style = rng.choice(MONEY_STYLES)
    fdate = lambda d: format_date(d, date_style, lang)  # noqa: E731
    fmoney = lambda a: format_money(a, money_style)     # noqa: E731

    # ---- people -----------------------------------------------------------------
    patient = pools.make_person(rng, lang)
    patient_name = f"{patient['first']} {patient['last']}"

    provider = pools.make_person(rng, "en")
    credential = rng.choice(pools.PROVIDER_CREDENTIALS)
    # Provider names appear with credentials on the page; the truth is the printed form.
    provider_name = f"Dr. {provider['first']} {provider['last']}, {credential}"
    if rng.random() < 0.4:
        provider_name = f"{provider['first']} {provider['last']}, {credential}"

    # A second person who is *not* the answer to any field.
    contact = pools.make_person(rng, lang)
    contact_name = f"{contact['first']} {contact['last']}"

    # ---- organisations ----------------------------------------------------------
    referring_org = pools.make_referring_practice(rng)
    servicing_org = pools.make_servicing_facility(rng)
    payer = rng.choice(pools.PAYERS)

    # ---- dates ------------------------------------------------------------------
    today = date(2025, 1, 1) + timedelta(days=rng.randint(0, 500))
    dob = date(
        rng.randint(1938, 2015), rng.randint(1, 12), rng.randint(1, 28)
    )
    service_date = today - timedelta(days=rng.randint(0, 45))
    print_date = today + timedelta(days=rng.randint(0, 3))  # distractor

    # DOBs are sometimes printed in a different style than the rest of the page.
    dob_style = rng.choice(DATE_STYLES) if rng.random() < 0.15 else date_style

    # ---- clinical ---------------------------------------------------------------
    icd, icd_desc, cpt_options = rng.choice(pools.DIAGNOSES)
    primary_cpt = rng.choice(cpt_options)

    # ---- identifiers ------------------------------------------------------------
    npi = pools.make_npi(rng)
    member_id = pools.make_member_id(rng)
    ref_no = pools.make_reference(rng, rng.choice(REF_PREFIXES[doc_type]))
    fax_number = pools.make_decoy_number(rng)      # invalid-checksum decoy
    account_number = pools.make_decoy_number(rng)  # invalid-checksum decoy
    group_number = f"GRP{rng.randint(10000, 99999)}"

    # ---- service lines and money ------------------------------------------------
    service_lines: list[ServiceLineData] = []
    total_charge = amount_paid = patient_resp = 0.0
    if doc_type in (DocType.insurance_claim, DocType.remittance_advice):
        n_lines = rng.randint(1, 4)
        cpts = rng.sample(cpt_options, min(n_lines, len(cpt_options)))
        while len(cpts) < n_lines:
            cpts.append(rng.choice(list(pools.CPT_CHARGES)))
        for cpt in cpts:
            lo, hi = pools.CPT_CHARGES.get(cpt, (50, 500))
            charge = round(rng.uniform(lo, hi), 2)
            rate = rng.uniform(0.55, 0.9)
            paid = round(charge * rate, 2)
            units = str(rng.randint(1, 2))
            line_dos = service_date - timedelta(days=rng.randint(0, 5))
            service_lines.append(
                ServiceLineData(
                    procedure_code=cpt,
                    description=pools.CPT_DESCRIPTIONS.get(cpt, "Medical service"),
                    date_of_service=fdate(line_dos),
                    units=units,
                    charge=fmoney(charge),
                    paid=fmoney(paid),
                )
            )
            total_charge += charge
            amount_paid += paid
        patient_resp = round(total_charge - amount_paid, 2)
        total_charge, amount_paid = round(total_charge, 2), round(amount_paid, 2)
    elif doc_type == DocType.prior_auth_request:
        lo, hi = pools.CPT_CHARGES.get(primary_cpt, (50, 500))
        total_charge = round(rng.uniform(lo, hi), 2)

    content = {
        "patient_name": patient_name,
        "patient_first": patient["first"],
        "patient_last": patient["last"],
        "patient_dob": format_date(dob, dob_style, lang),
        "patient_address": pools.make_address(rng),
        "patient_phone": pools.make_phone(rng),
        "patient_sex": rng.choice(["M", "F"]),
        "provider_name": provider_name,
        "provider_npi": npi,
        "contact_name": contact_name,
        "referring_org": referring_org,
        "referring_address": pools.make_address(rng),
        "referring_phone": pools.make_phone(rng),
        "servicing_org": servicing_org,
        "servicing_address": pools.make_address(rng),
        "payer_name": payer,
        "member_id": member_id,
        "group_number": group_number,
        "document_reference": ref_no,
        "document_date": fdate(today),
        "print_date": fdate(print_date),
        "date_of_service": fdate(service_date),
        "diagnosis_code": icd,
        "diagnosis_desc": icd_desc,
        "procedure_code": primary_cpt,
        "procedure_desc": pools.CPT_DESCRIPTIONS.get(primary_cpt, "Medical service"),
        "fax_number": fax_number,
        "account_number": account_number,
        "total_charge": fmoney(total_charge) if total_charge else None,
        "amount_paid": fmoney(amount_paid) if amount_paid else None,
        "patient_responsibility": fmoney(patient_resp) if patient_resp else None,
        "service_lines": service_lines,
    }

    # ---- ground truth: only the fields this document type actually carries -------
    all_truth = {
        "patient_name": content["patient_name"],
        "patient_dob": content["patient_dob"],
        "referring_provider_name": content["provider_name"],
        "referring_provider_npi": content["provider_npi"],
        "servicing_facility": content["servicing_org"],
        "payer_name": content["payer_name"],
        "member_id": content["member_id"],
        "document_reference": content["document_reference"],
        "document_date": content["document_date"],
        "date_of_service": content["date_of_service"],
        "diagnosis_code": content["diagnosis_code"],
        "procedure_code": content["procedure_code"],
        "total_charge": content["total_charge"],
        "amount_paid": content["amount_paid"],
        "patient_responsibility": content["patient_responsibility"],
    }
    wanted = expected_fields(doc_type)
    truth = {k: v for k, v in all_truth.items() if k in wanted}

    org_roles = {
        "referring_org": referring_org,
        "servicing_org": servicing_org,
        "payer_org": payer if "payer_name" in wanted else None,
    }

    return GeneratedDoc(
        doc_id=doc_id,
        doc_type=doc_type,
        template_id=template_id,
        lang=lang,
        content=content,
        truth=truth,
        service_lines=service_lines,
        org_roles=org_roles,
        date_style=date_style,
        money_style=money_style,
    )
