"""Synthetic entity pools for the document generator.

Everything here is invented. No real patient, provider, NPI or member ID appears in
this corpus — which is the point: a healthcare document benchmark you can publish.

The pools are deliberately *confusable*. Several facilities share a city name, several
payers share the "Blue" prefix, and provider surnames are reused across the patient
pool. If the pools were trivially separable, the extraction task would be easy for the
wrong reason and every approach would score ~100.
"""

from __future__ import annotations

import random

from ..normalize import npi_check_digit

# --------------------------------------------------------------------------------------
# People. Surnames overlap deliberately between the patient and provider pools so that
# "which person is the patient and which is the referring provider?" is a real decision
# rather than a lookup.
# --------------------------------------------------------------------------------------

FIRST_NAMES_EN = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
    "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Kimberly",
    "Andrew", "Emily", "Kenneth", "Donna", "Joshua", "Michelle", "Kevin", "Carol",
]

FIRST_NAMES_ES = [
    "José", "María", "Carlos", "Ana", "Luis", "Carmen", "Miguel", "Rosa",
    "Juan", "Isabel", "Antonio", "Lucía", "Francisco", "Elena", "Jorge", "Sofía",
    "Manuel", "Patricia", "Ricardo", "Gabriela",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
]

PROVIDER_CREDENTIALS = ["MD", "DO", "NP", "PA-C", "MD, FACP", "M.D.", "D.O."]

# --------------------------------------------------------------------------------------
# Organisations. Three roles must be told apart on a single page: the referring practice,
# the servicing facility, and the payer. Clinic and facility names overlap in style on
# purpose so role assignment cannot be solved by keyword matching alone.
# --------------------------------------------------------------------------------------

CITIES = [
    "Cedar Park", "Northside", "Lakeside", "Summit", "Riverbend", "Fairview",
    "Oakhurst", "Westgate", "Brookfield", "Highland", "Stone Creek",
    "Pinehurst", "Maple Grove", "Kingsport", "Ashford", "Granite Bay",
]

REFERRING_PRACTICE_SUFFIXES = [
    "Family Medicine", "Family Practice", "Primary Care Associates",
    "Internal Medicine Group", "Medical Associates", "Community Health Partners",
    "Pediatric Associates", "Family Health Center",
]

SERVICING_FACILITY_SUFFIXES = [
    "Radiology, P.C.", "Imaging Center", "Orthopedic Associates, LLC",
    "General Hospital", "Regional Medical Center", "Diagnostic Laboratory",
    "Surgical Center", "Cardiology Institute", "Sports Medicine Clinic",
    "Neurology Specialists, P.A.",
]

PAYERS = [
    "Aetna", "Cigna Healthcare", "UnitedHealthcare", "Humana", "Kaiser Permanente",
    "Anthem Blue Cross", "Blue Cross Blue Shield of Texas", "Blue Shield of California",
    "Molina Healthcare", "Ambetter Health", "Oscar Health Plan", "WellCare",
    "Medicare Part B", "Medicaid Managed Care", "TRICARE East", "Bright HealthCare",
    "Highmark Blue Shield", "Premera Blue Cross",
]

# --------------------------------------------------------------------------------------
# Clinical codes. Real code *formats* with plausible descriptions, paired so that the
# diagnosis and the procedure on a document make clinical sense together.
# --------------------------------------------------------------------------------------

#: (icd10, description, [plausible CPT codes for that diagnosis])
DIAGNOSES: list[tuple[str, str, list[str]]] = [
    ("M54.51", "Vertebrogenic low back pain", ["72148", "97110", "99213"]),
    ("M17.11", "Unilateral primary osteoarthritis, right knee", ["73721", "20610", "29881"]),
    ("E11.9", "Type 2 diabetes mellitus without complications", ["83036", "80053", "99214"]),
    ("I10", "Essential (primary) hypertension", ["93000", "80053", "99213"]),
    ("J45.909", "Unspecified asthma, uncomplicated", ["94010", "71046", "99213"]),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis", ["43235", "99214"]),
    ("N39.0", "Urinary tract infection, site not specified", ["81001", "87086", "99213"]),
    ("G43.909", "Migraine, unspecified, not intractable", ["70450", "99214"]),
    ("R07.9", "Chest pain, unspecified", ["93000", "71046", "80053"]),
    ("S83.511A", "Sprain of anterior cruciate ligament of right knee", ["73721", "97110"]),
    ("E78.5", "Hyperlipidemia, unspecified", ["80061", "99213"]),
    ("F41.1", "Generalized anxiety disorder", ["90834", "99214"]),
    ("M25.561", "Pain in right knee", ["73562", "20610"]),
    ("R10.9", "Unspecified abdominal pain", ["76700", "74177", "80053"]),
    ("J06.9", "Acute upper respiratory infection, unspecified", ["87880", "99213"]),
    ("E03.9", "Hypothyroidism, unspecified", ["84443", "99213"]),
    ("H52.13", "Myopia, bilateral", ["92014", "92015"]),
    ("L20.9", "Atopic dermatitis, unspecified", ["11100", "99213"]),
    ("M79.641", "Pain in right hand", ["73130", "97110"]),
    ("R51.9", "Headache, unspecified", ["70450", "70553", "99214"]),
]

CPT_DESCRIPTIONS: dict[str, str] = {
    "20610": "Arthrocentesis, aspiration and/or injection, major joint",
    "29881": "Arthroscopy, knee, surgical, with meniscectomy",
    "36415": "Collection of venous blood by venipuncture",
    "43235": "Esophagogastroduodenoscopy, diagnostic",
    "70450": "CT head/brain without contrast material",
    "70553": "MRI brain with and without contrast",
    "71046": "Radiologic examination, chest, 2 views",
    "72148": "MRI lumbar spine without contrast material",
    "73130": "Radiologic examination, hand, 3 or more views",
    "73562": "Radiologic examination, knee, 3 views",
    "73721": "MRI any joint of lower extremity without contrast",
    "74177": "CT abdomen and pelvis with contrast material",
    "76700": "Ultrasound, abdominal, real time, complete",
    "80053": "Comprehensive metabolic panel",
    "80061": "Lipid panel",
    "81001": "Urinalysis, automated, with microscopy",
    "83036": "Hemoglobin A1C",
    "84443": "Thyroid stimulating hormone (TSH)",
    "87086": "Culture, bacterial, urine, quantitative",
    "87880": "Infectious agent detection, Streptococcus, group A",
    "90834": "Psychotherapy, 45 minutes with patient",
    "92014": "Ophthalmological services, established patient",
    "92015": "Determination of refractive state",
    "93000": "Electrocardiogram, routine ECG with interpretation",
    "94010": "Spirometry, with graphic record",
    "97110": "Therapeutic exercises, each 15 minutes",
    "99213": "Office/outpatient visit, established patient, low complexity",
    "99214": "Office/outpatient visit, established patient, moderate complexity",
}

#: Typical allowed charge per CPT, used to make amounts internally consistent.
CPT_CHARGES: dict[str, tuple[float, float]] = {
    "20610": (180, 340), "29881": (2800, 5200), "36415": (12, 28),
    "43235": (900, 1800), "70450": (620, 1400), "70553": (1400, 2900),
    "71046": (95, 260), "72148": (980, 2200), "73130": (85, 190),
    "73562": (90, 210), "73721": (1100, 2400), "74177": (1200, 2600),
    "76700": (280, 620), "80053": (45, 120), "80061": (38, 95),
    "81001": (18, 45), "83036": (32, 78), "84443": (40, 98),
    "87086": (35, 85), "87880": (28, 60), "90834": (120, 240),
    "92014": (110, 230), "92015": (45, 90), "93000": (60, 150),
    "94010": (70, 165), "97110": (55, 130), "99213": (110, 220),
    "99214": (165, 320),
}

STREETS = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Park Blvd", "Washington Ave",
    "Lincoln Way", "Sunset Blvd", "River Rd", "Highland Ave", "Mill St", "Elm St",
]

STATES = ["TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "AZ", "WA"]


def make_npi(rng: random.Random) -> str:
    """A 10-digit NPI with a valid CMS check digit.

    Generating *valid* NPIs matters: it lets the rules approach use the checksum to
    reject decoys, and it means a system that ignores the checksum is measurably
    leaving precision on the table.
    """
    first_nine = "".join(str(rng.randint(0, 9)) for _ in range(9))
    return first_nine + str(npi_check_digit(first_nine))


def make_decoy_number(rng: random.Random) -> str:
    """A 10-digit number that is *not* a valid NPI — a fax or account number.

    These are scattered through documents as distractors.
    """
    while True:
        first_nine = "".join(str(rng.randint(0, 9)) for _ in range(9))
        correct = npi_check_digit(first_nine)
        wrong = (correct + rng.randint(1, 9)) % 10
        if wrong != correct:
            return first_nine + str(wrong)


def make_person(rng: random.Random, lang: str = "en") -> dict:
    firsts = FIRST_NAMES_ES if lang == "es" else FIRST_NAMES_EN
    return {"first": rng.choice(firsts), "last": rng.choice(LAST_NAMES)}


def make_address(rng: random.Random) -> str:
    return (
        f"{rng.randint(100, 9899)} {rng.choice(STREETS)}, "
        f"{rng.choice(CITIES)}, {rng.choice(STATES)} {rng.randint(10000, 99999)}"
    )


def make_phone(rng: random.Random) -> str:
    return f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"


def make_member_id(rng: random.Random) -> str:
    """Payer member IDs vary in shape between carriers — so these do too."""
    style = rng.randint(0, 3)
    digits = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))  # noqa: E731
    letters = lambda n: "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))  # noqa: E731
    if style == 0:
        return letters(3) + digits(9)
    if style == 1:
        return digits(3) + "-" + digits(2) + "-" + digits(4)
    if style == 2:
        return letters(1) + digits(8) + letters(1)
    return digits(11)


def make_reference(rng: random.Random, prefix: str) -> str:
    """A claim/auth/order number with a type-specific prefix."""
    digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([6, 7, 8])))
    sep = rng.choice(["-", "", "-"])
    return f"{prefix}{sep}{digits}"


def make_referring_practice(rng: random.Random) -> str:
    return f"{rng.choice(CITIES)} {rng.choice(REFERRING_PRACTICE_SUFFIXES)}"


def make_servicing_facility(rng: random.Random) -> str:
    return f"{rng.choice(CITIES)} {rng.choice(SERVICING_FACILITY_SUFFIXES)}"
