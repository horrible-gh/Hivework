"""Permanent, model-free regression net for converge's deterministic guards."""
from __future__ import annotations

import pytest

from tests.converge_corpus import CASES, run_case

# Every stamp a guard/facet may ADD. A negative case asserts none of these appear, so a
# silent guard that quietly fires is caught. Keep in sync with the guards in converge.py.
_KNOWN_STAMPS = {
    "dropped_peer",
    "data_premise_refuted",
    "data_unstamped",
    "field_provenance_repointed",
    "field_provenance_confirmed",
    "http_datasource_provenance_repointed",
    "http_datasource_provenance_confirmed",
    "unrefuted_peer",
    "counterfactual_incomplete",
    "trace_ungrounded",
    "low_confidence",
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_converge_guard_corpus(case):
    try:
        result = run_case(case)
    except Exception as exc:  # guards are explicitly never-raise
        pytest.fail(f"{case.id} raised {type(exc).__name__}: {exc}")

    assert result.converged is case.expect["converged"]
    assert (result.attributed_defect or {}).get("file") == case.expect["attributed_file"]

    stamp = case.expect.get("stamp")
    causal_check = result.causal_check or {}
    if stamp:
        assert stamp in causal_check
    else:
        assert _KNOWN_STAMPS.isdisjoint(causal_check)


def test_every_guard_has_firing_and_negative_cases():
    by_guard = {}
    for case in CASES:
        coverage = by_guard.setdefault(case.guard, {"firing": False, "negative": False})
        if case.expect.get("stamp"):
            coverage["firing"] = True
        else:
            coverage["negative"] = True

    assert set(by_guard) == {
        "field_provenance",
        "premise_refuted",
        "http_datasource",
        "dropped_peer",
        "data_stamp",
        "counterfactual",
        "trace_grounding",
        "sufficiency",
    }
    assert all(item["firing"] and item["negative"] for item in by_guard.values())
