"""Tests for span alignment and BIO encode/decode.

Alignment sits between the LLM teacher and the small-model student: a value that cannot
be located in the text cannot become a training label. These tests pin the behaviour of
each rung of the cascade, and — just as importantly — pin what must *not* align, since a
matcher that is too eager produces confidently wrong training labels.
"""

import pytest

from docintel.align import (
    AlignmentReport,
    align_fields,
    bio_to_values,
    find_span,
    pick_best_value,
    spans_to_bio,
)
from docintel.schema import OcrDocument, Word


def make_doc(text: str) -> OcrDocument:
    words, pos = [], 0
    for line_no, line in enumerate(text.split("\n")):
        for token in line.split(" "):
            if token:
                start = text.index(token, pos)
                words.append(Word(text=token, start=start, end=start + len(token)))
                pos = start + len(token)
    return OcrDocument(doc_id="t", text=text, words=words)


class TestCascade:
    text = "Patient Name: Jane Doe\nPayer: Aetna Inc.\nMember ID: 003-32-2453"

    def test_exact_match(self):
        a = find_span(self.text, "Jane Doe", "patient_name")
        assert a.ok and a.method == "exact"
        assert self.text[a.start:a.end] == "Jane Doe"

    def test_whitespace_flexible_crosses_a_line_break(self):
        """Layout wrapping must not cost us a training label."""
        text = "Facility:\nCedar Park Imaging\nCenter closed"
        a = find_span(text, "Cedar Park Imaging Center", "servicing_facility")
        assert a.ok and a.method == "whitespace"
        assert text[a.start:a.end] == "Cedar Park Imaging\nCenter"

    def test_compact_match_survives_separator_damage(self):
        a = find_span(self.text, "003 32 2453", "member_id")
        assert a.ok and a.method == "compact"
        assert self.text[a.start:a.end] == "003-32-2453"

    def test_fuzzy_match_survives_ocr_characters(self):
        text = "Facility: Riverbend Cardiology lnstitute"   # OCR turned I into l
        a = find_span(text, "Riverbend Cardiology Institute", "servicing_facility")
        assert a.ok and a.method == "fuzzy"
        assert a.score >= 88

    def test_unrelated_value_does_not_align(self):
        """An over-eager matcher manufactures confidently wrong labels."""
        a = find_span(self.text, "Kaiser Permanente", "payer_name")
        assert not a.ok
        assert a.method == "none"

    def test_missing_and_empty_inputs(self):
        assert not find_span("", "x", "f").ok
        assert not find_span("some text", "", "f").ok

    def test_occurrences_are_counted(self):
        text = "Aetna ... Payer: Aetna"
        a = find_span(text, "Aetna", "payer_name")
        assert a.occurrences == 2


class TestOverlapResolution:
    def test_two_fields_cannot_claim_the_same_characters(self):
        """A token carries exactly one BIO label, so spans must not overlap."""
        doc = make_doc("Provider: Dr. Jane Doe MD")
        result = align_fields(doc, {
            "referring_provider_name": "Dr. Jane Doe MD",
            "patient_name": "Jane Doe",
        })
        aligned = [a for a in result.values() if a.ok]
        for i, a in enumerate(aligned):
            for b in aligned[i + 1:]:
                assert not (a.start < b.end and b.start < a.end), "spans overlap"

    def test_exact_beats_fuzzy_when_they_collide(self):
        doc = make_doc("Payer: Aetna")
        result = align_fields(doc, {"payer_name": "Aetna", "servicing_facility": "Aetna"})
        kept = [a for a in result.values() if a.ok]
        assert len(kept) == 1


class TestReport:
    def test_report_tracks_rate_and_methods(self):
        doc = make_doc("Patient: Jane Doe\nPayer: Aetna")
        report = AlignmentReport()
        align_fields(doc, {
            "patient_name": "Jane Doe",
            "payer_name": "Aetna",
            "member_id": "NOT-PRESENT-ANYWHERE",
        }, report)
        assert report.total == 3
        assert report.aligned == 2
        assert report.rate == pytest.approx(2 / 3)
        assert report.field_rates()["member_id"] == 0.0


class TestBio:
    def test_round_trip_through_bio(self):
        text = "Patient Name: Jane Doe"
        offsets = [(0, 7), (8, 12), (12, 13), (14, 18), (19, 22)]
        alignments = align_fields(make_doc(text), {"patient_name": "Jane Doe"})
        labels = spans_to_bio(text, alignments, offsets)
        assert labels == ["O", "O", "O", "B-patient_name", "I-patient_name"]
        decoded = bio_to_values(text, labels, offsets)
        assert decoded["patient_name"][0][0] == "Jane Doe"

    def test_special_tokens_are_ignored(self):
        text = "Jane Doe"
        offsets = [(0, 0), (0, 4), (5, 8), (0, 0)]
        alignments = align_fields(make_doc(text), {"patient_name": "Jane Doe"})
        labels = spans_to_bio(text, alignments, offsets)
        assert labels[0] == "O" and labels[-1] == "O"
        assert labels[1].startswith("B-")

    def test_stray_i_tag_starts_an_entity_rather_than_being_dropped(self):
        """Boundaries are often right even when the B/I prefix is wrong."""
        text = "Aetna Inc"
        offsets = [(0, 5), (6, 9)]
        decoded = bio_to_values(text, ["I-payer_name", "I-payer_name"], offsets)
        assert decoded["payer_name"][0][0] == "Aetna Inc"

    def test_adjacent_entities_of_different_fields_split(self):
        text = "Jane Doe Aetna"
        offsets = [(0, 4), (5, 8), (9, 14)]
        decoded = bio_to_values(
            text, ["B-patient_name", "I-patient_name", "B-payer_name"], offsets
        )
        assert decoded["patient_name"][0][0] == "Jane Doe"
        assert decoded["payer_name"][0][0] == "Aetna"


class TestPickBestValue:
    def test_longest_wins_for_names_and_orgs(self):
        """Partial extraction is the dominant failure; prefer the fuller span."""
        candidates = [("Mercy General", 0, 13), ("Mercy General Hospital", 0, 22)]
        assert pick_best_value(candidates, "servicing_facility") == "Mercy General Hospital"

    def test_first_wins_for_amounts_and_dates(self):
        candidates = [("$100.00", 0, 7), ("$999999.00", 20, 30)]
        assert pick_best_value(candidates, "total_charge") == "$100.00"

    def test_empty_gives_none(self):
        assert pick_best_value([], "patient_name") is None
