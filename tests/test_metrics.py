"""Tests for the scorer.

Every number in the comparison report comes from this module, so the properties pinned
here are the ones a reader would most reasonably doubt: that a wrong answer is punished
on both precision and recall, that correctly-empty fields are not silently counted as
wins, that hallucination detection does not fire on mere reformatting, and that the
bootstrap resamples documents rather than fields.
"""

import pytest

from docintel.eval.metrics import (
    PRF,
    aggregate_by_field,
    bootstrap_ci,
    doctype_metrics,
    hallucination_rate,
    macro_f1,
    micro_average,
    paired_bootstrap_pvalue,
    score_document,
    service_line_prf,
    slice_outcomes,
)
from docintel.schema import (
    DocType,
    ExtractedField,
    ExtractionResult,
    ServiceLine,
)


def make_prediction(fields: dict, doc_id="d1", doc_type=DocType.lab_order):
    return ExtractionResult(
        doc_id=doc_id,
        approach="test",
        doc_type=doc_type,
        fields={k: (ExtractedField(value=v) if v else None) for k, v in fields.items()},
    )


TEXT = (
    "Patient: Jane Doe\nDOB: 03/14/1980\nFacility: Mercy General Hospital\n"
    "Order #: ORD-123456\nProcedure: 80053\nDiagnosis: E11.9\nProvider: Dr. John Roe\n"
    "NPI: 1234567893\nDate: 05/01/2025\nService Date: 04/28/2025"
)

GOLD = {
    "patient_name": "Jane Doe",
    "patient_dob": "03/14/1980",
    "servicing_facility": "Mercy General Hospital",
    "document_reference": "ORD-123456",
    "procedure_code": "80053",
    "diagnosis_code": "E11.9",
    "referring_provider_name": "Dr. John Roe",
    "referring_provider_npi": "1234567893",
    "document_date": "05/01/2025",
    "date_of_service": "04/28/2025",
}


class TestScoring:
    def test_perfect_prediction(self):
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(GOLD), TEXT)
        prf = micro_average(outcomes)
        assert prf.f1 == 1.0 and prf.fp == 0 and prf.fn == 0

    def test_reformatted_value_still_counts_as_correct(self):
        """Normalisation is what keeps the comparison about understanding, not format."""
        pred = dict(GOLD, patient_dob="1980-03-14", document_date="May 1, 2025")
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(pred), TEXT)
        assert micro_average(outcomes).f1 == 1.0
        # ...but the strict metric must still see the difference.
        assert sum(o.exact for o in outcomes) < len(outcomes)

    def test_wrong_value_costs_both_precision_and_recall(self):
        pred = dict(GOLD, patient_name="John Smith")
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(pred), TEXT)
        table = aggregate_by_field(outcomes)
        assert table["patient_name"].fp == 1
        assert table["patient_name"].fn == 1
        assert table["patient_name"].f1 == 0.0

    def test_missing_value_is_recall_loss_only(self):
        pred = dict(GOLD, patient_name=None)
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(pred), TEXT)
        table = aggregate_by_field(outcomes)
        assert table["patient_name"].fn == 1 and table["patient_name"].fp == 0

    def test_only_fields_the_doctype_carries_are_scored(self):
        """A lab order has no amount_paid; nobody is rewarded or punished for it."""
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(GOLD), TEXT)
        assert "amount_paid" not in {o.field for o in outcomes}
        assert "patient_name" in {o.field for o in outcomes}


class TestHallucination:
    def test_invented_value_is_flagged(self):
        pred = dict(GOLD, patient_name="Cigna Corporation")
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(pred), TEXT)
        flagged = [o for o in outcomes if o.hallucinated]
        assert [o.field for o in flagged] == ["patient_name"]
        assert hallucination_rate(outcomes) > 0

    def test_reformatting_is_not_a_hallucination(self):
        """The detector must not cry wolf on a correctly reformatted date."""
        pred = dict(GOLD, patient_dob="1980-03-14")
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(pred), TEXT)
        assert not any(o.hallucinated for o in outcomes)

    def test_extractive_output_can_never_hallucinate(self):
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(GOLD), TEXT)
        assert hallucination_rate(outcomes) == 0.0


class TestPRF:
    def test_arithmetic(self):
        prf = PRF(tp=8, fp=2, fn=4, tn=6)
        assert prf.precision == pytest.approx(0.8)
        assert prf.recall == pytest.approx(8 / 12)
        assert prf.f1 == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))
        assert prf.accuracy == pytest.approx(14 / 20)

    def test_empty_is_zero_not_a_crash(self):
        prf = PRF()
        assert prf.precision == prf.recall == prf.f1 == prf.accuracy == 0.0

    def test_macro_equals_micro_when_every_field_has_equal_support(self):
        """With one document each field has support 1, so the two coincide."""
        outcomes = score_document("d1", GOLD, DocType.lab_order,
                                  make_prediction(dict(GOLD, patient_name="X")), TEXT)
        assert macro_f1(outcomes) == pytest.approx(micro_average(outcomes).f1)

    def test_macro_diverges_from_micro_under_support_imbalance(self):
        """Macro weights a rare field as heavily as a common one; micro does not.

        Built by failing one field on every document while the others always succeed,
        so the failing field's support differs from the rest.
        """
        outcomes = []
        for i in range(5):
            doc_id = f"d{i}"
            pred = dict(GOLD, patient_name="Wrong Person")
            rows = score_document(doc_id, GOLD, DocType.lab_order,
                                  make_prediction(pred, doc_id=doc_id), TEXT)
            for row in rows:
                row.doc_id = doc_id
            # Drop most fields from all but the first document, so supports differ.
            if i > 0:
                rows = [r for r in rows if r.field in ("patient_name", "patient_dob")]
            outcomes.extend(rows)
        assert macro_f1(outcomes) != pytest.approx(micro_average(outcomes).f1)


class TestDocType:
    def test_accuracy_and_confusion(self):
        pairs = [("lab_order", "lab_order"), ("lab_order", "referral_letter"),
                 ("referral_letter", "referral_letter"), ("referral_letter", "referral_letter")]
        m = doctype_metrics(pairs)
        assert m["accuracy"] == pytest.approx(0.75)
        assert m["confusion"]["lab_order"]["referral_letter"] == 1
        assert m["per_class"]["lab_order"]["recall"] == pytest.approx(0.5)


class TestServiceLines:
    def test_repeated_codes_are_multiset_matched(self):
        """A claim legitimately repeats a CPT across dates; a set would forgive a miss."""
        gold = [
            ServiceLine(procedure_code="99213", date_of_service="01/01/2025", charge="$10.00"),
            ServiceLine(procedure_code="99213", date_of_service="01/02/2025", charge="$20.00"),
        ]
        pred_one = [gold[0]]
        prf = service_line_prf(gold, pred_one)
        assert prf.tp == 1 and prf.fn == 1

    def test_exact_match(self):
        gold = [ServiceLine(procedure_code="80053", date_of_service="01/01/2025", charge="$50.00")]
        pred = [ServiceLine(procedure_code="80053", date_of_service="2025-01-01", charge="50.00")]
        prf = service_line_prf(gold, pred)
        assert prf.tp == 1 and prf.fp == 0 and prf.fn == 0


class TestUncertainty:
    def _two_docs(self):
        good = score_document("d1", GOLD, DocType.lab_order, make_prediction(GOLD), TEXT)
        bad_pred = {k: None for k in GOLD}
        bad = score_document("d2", GOLD, DocType.lab_order,
                             make_prediction(bad_pred, doc_id="d2"), TEXT)
        for o in bad:
            o.doc_id = "d2"
        return good + bad

    def test_bootstrap_interval_brackets_the_point_estimate(self):
        outcomes = self._two_docs()
        point, low, high = bootstrap_ci(outcomes, resamples=200)
        assert low <= point <= high
        assert 0.0 <= low and high <= 1.0

    def test_bootstrap_resamples_documents_not_fields(self):
        """Field-level resampling would manufacture significance that is not there."""
        outcomes = self._two_docs()
        _, low, high = bootstrap_ci(outcomes, resamples=300)
        # With only two documents (one perfect, one empty) the interval must be wide.
        assert high - low > 0.3

    def test_identical_systems_are_not_significantly_different(self):
        outcomes = self._two_docs()
        p = paired_bootstrap_pvalue(outcomes, outcomes, resamples=200)
        assert p > 0.05


def test_slicing_groups_by_key():
    outcomes = score_document(
        "d1", GOLD, DocType.lab_order, make_prediction(GOLD), TEXT,
        slice_keys={"condition": "scanned"},
    )
    buckets = slice_outcomes(outcomes, "condition")
    assert set(buckets) == {"scanned"}
