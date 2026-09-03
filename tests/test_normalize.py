"""Tests for the shared normalisers.

These matter more than typical unit tests: every reported metric depends on these
functions agreeing about when two values are the same thing.
"""

import pytest

from docintel.normalize import (
    AMBIGUOUS_DATE_HITS,
    appears_in_text,
    is_valid_npi,
    normalize_amount,
    normalize_code,
    normalize_date,
    normalize_field,
    normalize_org,
    normalize_person,
    normalize_reference,
    npi_check_digit,
    values_match,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("03/14/2025", "2025-03-14"),
        ("3/14/25", "2025-03-14"),
        ("2025-03-14", "2025-03-14"),
        ("14-Mar-2025", "2025-03-14"),
        ("March 14, 2025", "2025-03-14"),
        ("14 March 2025", "2025-03-14"),
        ("Mar. 14, 2025", "2025-03-14"),
        ("14/03/2025", "2025-03-14"),          # day > 12 -> unambiguous day-first
        ("14 de marzo de 2025", "2025-03-14"),  # Spanish
        ("DOS: 03.14.2025", "2025-03-14"),      # extractor captured the anchor too
        ("", None),
        ("not a date", None),
        ("02/30/2025", None),                   # invalid calendar date
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize(
    "prose",
    ["patient 5 of 12", "see page 3", "refill 2 times", "unit 4", "N/A"],
)
def test_date_parser_does_not_invent_dates_from_prose(prose):
    """A lenient date parser silently manufactures scoring matches. It must not."""
    assert normalize_date(prose) is None


def test_two_digit_year_pivot():
    assert normalize_date("01/15/62") == "1962-01-15"   # a DOB
    assert normalize_date("01/15/25") == "2025-01-15"   # a service date


def test_ambiguous_dates_are_counted_not_hidden():
    """US ordering is an assumption; the report must be able to quantify its reach."""
    before = len(AMBIGUOUS_DATE_HITS)
    normalize_date("04/05/2025")  # both readings valid
    assert len(AMBIGUOUS_DATE_HITS) == before + 1
    normalize_date("14/05/2025")  # day > 12, no ambiguity
    assert len(AMBIGUOUS_DATE_HITS) == before + 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$1,234.56", "1234.56"),
        ("1234.56", "1234.56"),
        ("USD 1,234.56", "1234.56"),
        ("1,234.5", "1234.50"),
        ("$0.00", "0.00"),
        ("(1,234.56)", "-1234.56"),   # accounting negative
        ("1.234,56", "1234.56"),      # European separators
        ("Total: $89", "89.00"),
        ("", None),
        ("N/A", None),
    ],
)
def test_normalize_amount(raw, expected):
    assert normalize_amount(raw) == expected


def test_amounts_are_exact_not_floats():
    """Money must never round-trip through binary floating point."""
    assert normalize_amount("$0.10") == "0.10"
    assert normalize_amount("$1,000,000.05") == "1000000.05"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Dr. Jane Doe, MD", "jane doe"),
        ("JANE DOE", "jane doe"),
        ("Doe, Jane", "jane doe"),
        ("Jane  Doe", "jane doe"),
        ("Jane Doe M.D.", "jane doe"),
        ("José Álvarez", "jose alvarez"),
        ("", None),
    ],
)
def test_normalize_person(raw, expected):
    assert normalize_person(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mercy General Hospital, Inc.", "mercy general hospital"),
        ("MERCY GENERAL HOSPITAL", "mercy general hospital"),
        ("Mercy General Hospital LLC", "mercy general hospital"),
        ("Blue Cross Blue Shield", "blue cross blue shield"),
        ("Northside Radiology, P.C.", "northside radiology"),
        ("", None),
    ],
)
def test_normalize_org(raw, expected):
    assert normalize_org(raw) == expected


def test_reference_and_code_folding():
    assert normalize_reference("ABC-123 456") == "ABC123456"
    assert normalize_reference("abc123") == "ABC123"
    # ICD-10 with and without the dot must fold together.
    assert normalize_code("A01.1") == normalize_code("A011") == "A011"
    assert normalize_code("99213") == "99213"


def test_values_match_uses_field_kind():
    assert values_match("document_date", "03/14/2025", "2025-03-14")
    assert values_match("total_charge", "$1,234.56", "1234.56")
    assert values_match("payer_name", "Aetna Inc.", "AETNA")
    assert not values_match("patient_dob", "03/14/2025", "03/15/2025")
    # Both missing counts as agreement; one missing does not.
    assert values_match("member_id", None, None)
    assert not values_match("member_id", "X1", None)


def test_normalize_field_dispatches_by_kind():
    assert normalize_field("patient_dob", "March 14, 2025") == "2025-03-14"
    assert normalize_field("amount_paid", "$12.00") == "12.00"
    assert normalize_field("patient_name", "Dr. Jane Doe") == "jane doe"
    assert normalize_field("unknown_field", "  Some  Value ") == "some value"
    assert normalize_field("patient_name", None) is None


class TestNpiChecksum:
    """The NPI check digit is a free precision win for the rules approach."""

    def test_known_valid_npis(self):
        # Check digits computed per the CMS Luhn-with-80840-prefix spec.
        for nine in ("123456789", "987654321", "100000000"):
            cd = npi_check_digit(nine)
            assert is_valid_npi(nine + str(cd))

    def test_rejects_wrong_check_digit(self):
        nine = "123456789"
        wrong = (npi_check_digit(nine) + 1) % 10
        assert not is_valid_npi(nine + str(wrong))

    def test_rejects_wrong_length(self):
        assert not is_valid_npi("12345")
        assert not is_valid_npi("")
        assert not is_valid_npi(None)

    def test_accepts_formatted_input(self):
        nine = "123456789"
        npi = nine + str(npi_check_digit(nine))
        assert is_valid_npi(f"{npi[:3]}-{npi[3:6]}-{npi[6:]}")


class TestHallucinationDetection:
    text = "Patient: Jane Doe\nDOB: 03/14/1980\nTotal Charge: $1,234.56\nPayer: Aetna"

    def test_present_values_pass(self):
        assert appears_in_text("Jane Doe", self.text)
        assert appears_in_text("Aetna", self.text)

    def test_reformatted_values_are_not_flagged(self):
        """Reformatting is not invention — the detector must not cry wolf."""
        assert appears_in_text("1234.56", self.text)
        assert appears_in_text("1980-03-14", self.text, field="patient_dob")

    def test_invented_values_are_flagged(self):
        assert not appears_in_text("John Smith", self.text)
        assert not appears_in_text("Cigna", self.text)

    def test_empty_value_is_not_a_hallucination(self):
        assert appears_in_text(None, self.text)
        assert appears_in_text("", self.text)
