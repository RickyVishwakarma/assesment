"""Invariants the synthetic corpus must hold, or the benchmark is not measuring anything.

The two tests that matter most here are:

* :func:`test_every_truth_value_is_printed_on_the_page` — a gold label for a value the
  document never contains is unlearnable and unscoreable. This caught a real bug where
  the prose templates silently omitted three fields.
* :func:`test_train_and_heldout_templates_are_disjoint` — the anti-inflation control.
  If these sets ever overlap, every reported generalisation number becomes meaningless.
"""

import random

import fitz
import pytest

from docintel.gen.document import format_date, format_money, sample_document
from docintel.gen.pools import make_decoy_number, make_npi
from docintel.gen.render import (
    HELDOUT_TEMPLATES,
    TEMPLATES,
    TRAIN_TEMPLATES,
    render_pdf,
)
from docintel.normalize import is_valid_npi, normalize_date, normalize_amount
from docintel.schema import DOC_TYPES, expected_fields


@pytest.fixture(scope="module")
def rng():
    return random.Random(1234)


def test_train_and_heldout_templates_are_disjoint():
    """The core anti-inflation control of the whole project."""
    assert set(TRAIN_TEMPLATES).isdisjoint(HELDOUT_TEMPLATES)
    assert set(TRAIN_TEMPLATES) | set(HELDOUT_TEMPLATES) == set(TEMPLATES)
    assert len(HELDOUT_TEMPLATES) >= 4


def test_every_truth_value_is_printed_on_the_page(tmp_path, rng):
    """Ground truth must literally occur in the rendered document, for every template."""
    total = hits = 0
    misses = []
    for doc_type in DOC_TYPES:
        for template_id in TEMPLATES:
            for k in range(2):
                lang = "es" if (template_id + k) % 9 == 0 else "en"
                doc = sample_document(
                    f"{doc_type.value}-{template_id}-{k}", doc_type, template_id,
                    rng, lang=lang,
                )
                path = tmp_path / "d.pdf"
                render_pdf(doc, str(path), rng)
                text = fitz.open(str(path))[0].get_text()
                for fname, value in doc.truth.items():
                    if value is None:
                        continue
                    total += 1
                    if value in text:
                        hits += 1
                    else:
                        misses.append((template_id, doc_type.value, fname, value))
    assert total > 500, "smoke corpus too small to be meaningful"
    assert hits == total, f"{len(misses)} truth values absent from page: {misses[:5]}"


def test_truth_only_contains_fields_the_doctype_carries(rng):
    """A lab order must not claim an ``amount_paid``."""
    for doc_type in DOC_TYPES:
        doc = sample_document("x", doc_type, 1, rng)
        assert set(doc.truth) <= expected_fields(doc_type)
        assert set(doc.truth), f"{doc_type} produced no fields"


def test_generated_npis_pass_the_checksum_and_decoys_fail(rng):
    """Decoy numbers exist so the rules approach's checksum is a real discriminator."""
    for _ in range(200):
        assert is_valid_npi(make_npi(rng))
        assert not is_valid_npi(make_decoy_number(rng))


def test_documents_contain_distractors(tmp_path, rng):
    """Every page carries a decoy 10-digit number that is not the NPI."""
    doc = sample_document("d1", DOC_TYPES[0], 1, rng)
    path = tmp_path / "d.pdf"
    render_pdf(doc, str(path), rng)
    text = fitz.open(str(path))[0].get_text()
    assert doc.content["fax_number"] in text
    assert doc.content["account_number"] in text
    assert not is_valid_npi(doc.content["fax_number"])


def test_rendered_surface_forms_round_trip_through_the_normalisers(rng):
    """Whatever style a template prints, the normaliser must recover the value."""
    from datetime import date

    d = date(1980, 3, 14)
    for style in ["mdy_slash", "mdy_slash_short", "iso", "dmy_mon", "month_name", "dot"]:
        assert normalize_date(format_date(d, style)) == "1980-03-14", style
    assert normalize_date(format_date(d, "es_long", "es")) == "1980-03-14"
    for style in ["dollar_comma", "plain_comma", "dollar_plain", "usd_comma"]:
        assert normalize_amount(format_money(1234.5, style)) == "1234.50", style


def test_seeded_generation_is_reproducible():
    """Same seed, same corpus — otherwise results are not comparable across runs."""
    a = sample_document("id", DOC_TYPES[0], 1, random.Random(99))
    b = sample_document("id", DOC_TYPES[0], 1, random.Random(99))
    assert a.truth == b.truth
    assert a.content["provider_npi"] == b.content["provider_npi"]


def test_column_gaps_survive_into_the_text(tmp_path, rng):
    """A two-column row must not collapse into a single-space phrase.

    Regression test for a real bug: with a single space separator, the row
    ``Patient Name:   Reference #:`` was indistinguishable from ordinary prose, and
    both the LLM and the rules extractor merged the two columns' values into one field
    (``"Elizabeth Rivera : ORD-4322856"`` as a patient name). The wide gap is the only
    evidence a column boundary existed, so it has to reach the text.
    """
    from docintel.ocr import read_clean

    # Template 3 is a two-column label-above-value layout.
    doc = sample_document("cols", DOC_TYPES[2], 3, rng)
    path = tmp_path / "cols.pdf"
    render_pdf(doc, str(path), rng)
    ocr = read_clean(str(path), "cols")

    assert "   " in ocr.text, "no column gap preserved in a two-column template"
    # And the invariant that everything downstream depends on still holds.
    for word in ocr.words:
        assert ocr.text[word.start:word.end] == word.text


def test_word_offsets_index_into_text_for_every_template(tmp_path, rng):
    """``text[w.start:w.end] == w.text`` must hold everywhere, gaps or not."""
    from docintel.ocr import read_clean

    for template_id in (1, 3, 5, 7, 11, 14):
        doc = sample_document(f"off{template_id}", DOC_TYPES[0], template_id, rng)
        path = tmp_path / f"o{template_id}.pdf"
        render_pdf(doc, str(path), rng)
        ocr = read_clean(str(path), f"off{template_id}")
        assert ocr.words, f"template {template_id} produced no words"
        for word in ocr.words:
            assert ocr.text[word.start:word.end] == word.text, template_id
