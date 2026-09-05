"""Apply adjudications for the 80 tier-2 gold fields.

**These decisions were made by Claude, not by a human.** They are recorded as
``claude_adjudicated`` and the evaluator will NOT report the split as `human_verified`.
The assignment asks for a manually verified gold set; this unblocks scoring without
pretending that requirement has been met. `scripts/review_gold.py` still performs the
human pass, and `--from-adjudication` pre-loads these as defaults so a reviewer confirms
or overrides rather than deciding cold.

Decision rule, applied uniformly and without ever seeing any model's prediction (so it
cannot tilt the comparison toward an approach):

``keep``
    OCR introduced only punctuation, spacing, accent or unambiguous single-character
    noise (``Y.CF59 1565308`` for ``YCF591565308``; ``RS51.9`` for ``R51.9``). The value
    is recoverable by a careful reader, so a correct extractor should return it and the
    generator's value stays as gold.

``ocr``
    The document genuinely says something different and the OCR reading is the honest
    one. Gold becomes what the page actually says.

``unreadable``
    The value was truncated or destroyed past recovery (``Brookfield Orthopedic
    Associlat``, ``2802  8405 - 31`` for a date). Excluded from scoring, because grading
    any approach on it measures the scanner rather than the extractor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# doc suffix -> field -> decision. "keep" | "unreadable" | explicit replacement string.
DECISIONS: dict[str, dict[str, str]] = {
    "0006-referral_letter-t13": {"member_id": "keep"},
    "0012-referral_letter-t11": {"document_date": "unreadable",
                                 "servicing_facility": "unreadable"},
    "0014-lab_order-t13": {"diagnosis_code": "keep"},
    "0015-insurance_claim-t14": {"referring_provider_name": "keep",
                                 "servicing_facility": "keep"},
    "0016-remittance_advice-t11": {"servicing_facility": "keep"},
    "0020-lab_order-t11": {"document_date": "keep"},
    "0025-prior_auth_request-t12": {"member_id": "keep"},
    "0028-remittance_advice-t11": {"document_date": "keep",
                                   "servicing_facility": "keep"},
    "0031-prior_auth_request-t14": {"diagnosis_code": "keep"},
    "0032-lab_order-t11": {"servicing_facility": "keep"},
    "0036-referral_letter-t11": {"patient_name": "keep", "payer_name": "keep",
                                 "referring_provider_name": "keep"},
    "0038-lab_order-t13": {"diagnosis_code": "unreadable"},
    "0039-insurance_claim-t14": {"document_reference": "keep"},
    "0042-referral_letter-t13": {"document_reference": "keep", "member_id": "keep",
                                 "patient_dob": "07-Sep-1982"},
    "0045-insurance_claim-t12": {"document_date": "keep", "patient_dob": "keep",
                                 "patient_name": "keep"},
    "0047-patient_intake_form-t14": {"member_id": "keep", "patient_name": "keep"},
    "0048-referral_letter-t11": {"diagnosis_code": "keep", "payer_name": "keep",
                                 "servicing_facility": "keep"},
    "0050-lab_order-t13": {"diagnosis_code": "keep", "document_date": "keep",
                           "patient_dob": "unreadable", "patient_name": "keep",
                           "procedure_code": "unreadable",
                           "referring_provider_name": "keep",
                           "referring_provider_npi": "unreadable",
                           "servicing_facility": "keep"},
    "0052-remittance_advice-t11": {"servicing_facility": "keep"},
    "0053-patient_intake_form-t12": {"member_id": "keep"},
    "0055-prior_auth_request-t14": {"document_date": "keep"},
    "0056-lab_order-t11": {"date_of_service": "unreadable",
                           "diagnosis_code": "unreadable",
                           "patient_dob": "unreadable",
                           "servicing_facility": "unreadable"},
    "0059-patient_intake_form-t14": {"patient_dob": "keep"},
    "0060-referral_letter-t11": {"document_reference": "keep", "member_id": "keep",
                                 "patient_dob": "keep",
                                 "referring_provider_name": "keep"},
    "0061-prior_auth_request-t12": {"patient_dob": "unreadable",
                                    "patient_name": "keep"},
    "0064-remittance_advice-t11": {"document_date": "unreadable",
                                   "patient_responsibility": "unreadable",
                                   "servicing_facility": "unreadable"},
    "0066-referral_letter-t13": {"diagnosis_code": "unreadable",
                                 "referring_provider_name": "keep",
                                 "referring_provider_npi": "keep",
                                 "servicing_facility": "keep"},
    "0068-lab_order-t11": {"servicing_facility": "keep"},
    "0069-insurance_claim-t12": {"document_date": "keep",
                                 "servicing_facility": "keep"},
    "0070-remittance_advice-t13": {"member_id": "keep", "patient_name": "keep",
                                   "patient_responsibility": "keep",
                                   "total_charge": "keep"},
    "0072-referral_letter-t11": {"document_date": "unreadable",
                                 "servicing_facility": "unreadable"},
    "0074-lab_order-t13": {"referring_provider_name": "keep"},
    "0075-insurance_claim-t14": {"payer_name": "keep"},
    "0078-referral_letter-t13": {"document_reference": "keep",
                                 "referring_provider_name": "keep"},
    "0079-prior_auth_request-t14": {"patient_name": "keep",
                                    "servicing_facility": "keep"},
    "0080-lab_order-t11": {"diagnosis_code": "unreadable",
                           "servicing_facility": "unreadable"},
    "0081-insurance_claim-t12": {"patient_dob": "unreadable", "patient_name": "keep"},
    "0083-patient_intake_form-t14": {"document_date": "keep"},
    "0084-referral_letter-t11": {"servicing_facility": "unreadable"},
    "0088-remittance_advice-t11": {"servicing_facility": "keep"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/gold/gold_synth.jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    kept = replaced = unreadable = untouched = 0
    applied_docs = 0

    for record in records:
        suffix = record["doc_id"].replace("gold_synth-", "")
        decisions = DECISIONS.get(suffix)
        if not decisions:
            continue
        applied_docs += 1
        log: list[str] = [c for c in record.get("corrections", [])
                          if not c.endswith("FLAGGED_UNRESOLVED")]

        for field, decision in decisions.items():
            if field not in record["fields"]:
                continue
            original = record["fields"][field]
            if decision == "keep":
                kept += 1
                log.append(f"{field}: kept generator value (OCR noise only)")
            elif decision == "unreadable":
                unreadable += 1
                record["fields"][field] = None
                log.append(f"{field}: UNREADABLE - destroyed by OCR, excluded")
            else:
                replaced += 1
                record["fields"][field] = decision
                log.append(f"{field}: {original!r} -> {decision!r} (document differs)")

        record["corrections"] = log
        # Deliberately NOT "verified": True. These are machine adjudications.
        record["verified"] = False
        record["adjudicated_by"] = "claude"
        record["provenance"] = record.get("provenance", "synthetic_unseen_template")

    total = kept + replaced + unreadable
    print(f"documents touched : {applied_docs}")
    print(f"fields adjudicated: {total}")
    print(f"  kept generator value : {kept}")
    print(f"  replaced with OCR    : {replaced}")
    print(f"  marked unreadable    : {unreadable}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    with args.gold.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = args.gold.parent / "gold_synth_verification.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "corrected": replaced,
        "unreadable": unreadable,
        "accepted": kept,
        "interactive": False,
        "adjudicated_by": "claude",
        "note": (
            "Tier-2 fields adjudicated by Claude, not a human. The split is reported as "
            "claude_adjudicated, never human_verified. Re-run scripts/review_gold.py to "
            "perform the human pass."
        ),
    })
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.gold} and {summary_path}")
    print("Split will report as 'claude_adjudicated' - NOT 'human_verified'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
