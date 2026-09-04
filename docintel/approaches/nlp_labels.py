"""Label vocabulary for the rules approach, observed in the TRAINING split only.

Provenance matters here, so it is worth stating plainly.

An earlier version of :mod:`docintel.approaches.nlp` imported ``LABELS`` directly from
the document *generator*. That is a methodological leak: it handed the rules engine
perfect foreknowledge of every synonym the generator would ever emit, including wording
that only ever appears on the held-out templates. No real rules engineer has that, and it
would have quietly inflated the rules approach's score on exactly the held-out documents
the evaluation exists to measure.

This module is instead built by scanning the text of the *train* and *val* documents and
keeping only label strings actually observed there -- which is what an engineer building
from a labelled training sample would end up with. Five variants that occur solely on
held-out templates (``PAID AMT``, ``PT RESP``, ``Payment``, ``Saldo``, ``You Owe``) are
therefore absent, and the rules approach genuinely does not recognise them. That missing
coverage is a real finding about the brittleness of dictionaries, not a bug to patch.

Regenerate with ``python scripts/build_label_dict.py`` after changing the corpus.
"""

#: field -> label strings seen in training documents
TRAIN_LABELS: dict[str, list[str]] = {
    "amount_paid": [
        "Amount Paid",
        "Monto Pagado",
        "Pagado",
        "Paid"
    ],
    "date_of_service": [
        "DATE OF SVC",
        "DOS",
        "Date of Service",
        "F. Servicio",
        "Fecha de Servicio",
        "Service Date",
        "Svc Date"
    ],
    "diagnosis_code": [
        "Código CIE-10",
        "DX",
        "Diagnosis",
        "Diagnosis Code",
        "Diagnóstico",
        "Dx Code",
        "ICD-10"
    ],
    "document_date": [
        "DATE",
        "Date",
        "Date Issued",
        "Document Date",
        "Fecha",
        "Fecha del Documento",
        "Issued"
    ],
    "document_reference": [
        "Control #",
        "Document ID",
        "N.º de Referencia",
        "REF",
        "Ref No.",
        "Reference #",
        "Referencia"
    ],
    "member_id": [
        "Insurance ID",
        "MEMBER NO",
        "Member ID",
        "N.º de Miembro",
        "Policy #",
        "Póliza",
        "Subscriber ID"
    ],
    "patient_dob": [
        "BIRTHDATE",
        "Birth Date",
        "D.O.B.",
        "DOB",
        "Date of Birth",
        "F. Nac.",
        "Fecha de Nacimiento"
    ],
    "patient_name": [
        "Name of Patient",
        "Nombre del Paciente",
        "PATIENT",
        "PT NAME",
        "Paciente",
        "Patient",
        "Patient Name"
    ],
    "patient_responsibility": [
        "Patient Responsibility",
        "Responsabilidad del Paciente"
    ],
    "payer_name": [
        "Aseguradora",
        "Health Plan",
        "Insurance",
        "Insurance Carrier",
        "PAYER",
        "Payer",
        "Seguro"
    ],
    "procedure_code": [
        "CPT",
        "CPT Code",
        "Código CPT",
        "PROC",
        "Procedimiento",
        "Procedure",
        "Procedure Code"
    ],
    "referring_provider_name": [
        "Médico Remitente",
        "Ordering Physician",
        "Proveedor",
        "Provider",
        "REFERRING PHYSICIAN",
        "Referred By",
        "Referring Provider"
    ],
    "referring_provider_npi": [
        "NPI",
        "NPI #",
        "NPI del Proveedor",
        "National Provider ID",
        "Provider NPI"
    ],
    "servicing_facility": [
        "Centro",
        "Centro de Servicio",
        "FACILITY NAME",
        "Facility",
        "Performing Facility",
        "Service Location",
        "Servicing Facility"
    ],
    "total_charge": [
        "Amount Billed",
        "CHARGES",
        "Cargo Total",
        "Total",
        "Total Charge",
        "Total Charges"
    ]
}
