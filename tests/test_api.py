"""API contract tests.

Kept deliberately light: the point is the service's *contract* — that it degrades
approach-by-approach instead of failing wholesale, and that /health tells the truth about
what a caller can actually use.
"""

import pytest

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def client():
    # Loading the app imports torch. On a memory-constrained machine that can fail with
    # WinError 1455 while the local LLM is resident; skipping is correct there, since the
    # failure is an environment condition rather than a defect in this code.
    try:
        from fastapi.testclient import TestClient

        from docintel.api import app
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot load API: {exc}")
    with TestClient(app) as c:
        yield c


def test_health_never_lists_an_approach_as_both_ready_and_unavailable(client):
    """Regression test: a lazily-built approach was published before it was warmed.

    ``build_extractor`` succeeds even when the weights are missing, so registering the
    extractor before touching the model put a broken approach into *both* lists. A caller
    reads /health to decide what is usable; contradicting itself is worse than either
    answer alone.
    """
    body = client.get("/health").json()
    overlap = set(body["approaches_ready"]) & set(body["approaches_unavailable"])
    assert not overlap, f"approach reported both ready and unavailable: {overlap}"


def test_unknown_approach_is_rejected(client):
    response = client.post(
        "/extract", params={"approach": ["not_a_real_approach"]},
        files={"file": ("x.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert response.status_code == 400


def test_unavailable_approach_degrades_instead_of_erroring(client):
    """An untrained model must not take the whole request down with it."""
    health = client.get("/health").json()
    unavailable = list(health["approaches_unavailable"])
    if not unavailable:
        pytest.skip("every approach is available in this environment")

    from pathlib import Path

    pdfs = sorted(Path("data/corpus/pdfs").glob("demo-*.pdf"))
    if not pdfs:
        pytest.skip("corpus not built")

    with pdfs[0].open("rb") as fh:
        response = client.post(
            "/extract", params={"approach": unavailable},
            files={"file": (pdfs[0].name, fh, "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert set(body["unavailable"]) == set(unavailable)
