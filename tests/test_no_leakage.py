"""Guards against the evaluation quietly measuring the wrong thing.

Every test here protects a claim the comparison report makes. If one fails, some number
in that report is no longer honest.
"""

import json
from pathlib import Path

CORPUS = Path("data/corpus")

#: Label wording that occurs only on held-out templates. The rules approach must not
#: know these — it never saw a training document containing them.
HELD_OUT_ONLY = {"PAID AMT", "PT RESP", "Payment", "Saldo", "You Owe"}


def test_rules_dictionary_does_not_import_from_the_generator():
    """The rules engine must not read the generator's own label list.

    Importing ``LABELS`` from :mod:`docintel.gen.render` would give the dictionary
    perfect coverage of wording it could never have seen, inflating its held-out score.
    """
    source = Path("docintel/approaches/nlp.py").read_text(encoding="utf-8")
    assert "from ..gen.render import" not in source
    assert "gen.render" not in source


def test_rules_dictionary_excludes_held_out_only_wording():
    from docintel.approaches.nlp import LABEL_DICT

    present = {v for variants in LABEL_DICT.values() for v in variants}
    assert not (HELD_OUT_ONLY & present), (
        f"rules dictionary knows held-out-only labels: {HELD_OUT_ONLY & present}"
    )


def test_gold_and_training_documents_share_no_template():
    """The anti-inflation control, checked against the corpus actually on disk."""
    manifest = CORPUS / "manifest.jsonl"
    if not manifest.exists():
        return  # corpus not built in this environment
    train, gold = set(), set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        target = train if record["split"] in ("train", "val") else gold
        target.add(record["template_id"])
    assert train and gold
    assert train.isdisjoint(gold), f"template leak: {train & gold}"


def test_gold_documents_are_never_used_for_training():
    """No gold or demo document may appear in the silver training data."""
    silver_dir = Path("data/silver")
    if not silver_dir.exists():
        return
    for path in silver_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            doc_id = json.loads(line)["doc_id"]
            assert not doc_id.startswith(("gold_synth", "demo")), (
                f"{doc_id} from a held-out split leaked into training data {path}"
            )


def test_gold_source_label_reflects_actual_verification():
    """The report must never claim a human verified something they did not.

    ``review_gold.py --auto-only`` writes gold records whose tier-2 fields are flagged
    but unresolved, with ``verified: false``. An earlier version of the runner stamped
    every record present in the gold file as ``human_verified`` regardless, which would
    have put an unearned claim of manual verification into the comparison report.
    """
    import json as _json
    from pathlib import Path as _Path

    gold = _Path("data/gold/gold_synth.jsonl")
    if not gold.exists():
        return

    records = [
        _json.loads(line)
        for line in gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source = _Path("docintel/eval/runner.py").read_text(encoding="utf-8")
    assert 'match.get("verified")' in source, (
        "runner must consult each record's own verified flag"
    )

    # And the label must actually differ for unverified records.
    if records and not any(r.get("verified") for r in records):
        assert "auto_verified_only" in source


def test_frontier_tier_reports_a_nonzero_cost():
    """The paid tier must not report itself as free.

    ``ApproachProfile.cost_per_doc_usd`` defaults to 0.0, and the frontier profile
    originally never overrode it. That made the one approach with a real marginal cost
    appear to cost nothing, inverting the comparison the report exists to make.
    """
    from pathlib import Path as _Path

    from docintel.approaches.llm_frontier import COST_BASIS, mean_cost_per_doc

    if not _Path("data/frontier/annotations.json").exists():
        return
    assert mean_cost_per_doc() > 0.0
    assert "estimated" in COST_BASIS.lower(), (
        "the cost basis must state that it is estimated, not metered"
    )
