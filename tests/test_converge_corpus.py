"""Permanent, model-free regression net for converge's deterministic guards."""
from __future__ import annotations

from unittest import mock

import pytest

import hive.converge as C
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
    "design_change_preserved",
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


def _plumbing_abstain_cases():
    """Forced winning-path plumbing-decoy cases that abstain via a readable producer."""
    return [c for c in CASES if c.guard == "http_datasource"
            and c.expect.get("stamp") is None and c.producer_files]


def test_plumbing_decoy_family_present():
    """R0001 #4: the gate must be exercised by a FAMILY of getter/store/connection variants,
    not a single shape — and at least one must be CAPTURED from a real run (R0001 #2)."""
    cases = _plumbing_abstain_cases()
    assert len(cases) >= 3, "expected a family of plumbing-decoy abstain cases"
    assert any(c.captured for c in cases), "expected ≥1 decoy captured from a real run"


def test_plumbing_abstain_is_gate_driven_red_on_removal():
    """R0001 #3: each forced plumbing-decoy abstain GREEN must flip to a re-point when the
    ``_looks_like_datasource`` gate is disabled — direct proof the abstain is the GATE's doing
    (gate無력화 → RED), not an artifact of the fixture. With the gate intact the attribution
    stays on the correct dispose_group handler and no re-point stamp is added; with the gate
    forced to treat plumbing as a datasource the attribution is hijacked onto the decoy."""
    cases = _plumbing_abstain_cases()
    assert cases
    for case in cases:
        intact = run_case(case)
        assert (intact.attributed_defect or {}).get("file") == case.expect["attributed_file"]
        assert "http_datasource_provenance_repointed" not in (intact.causal_check or {})

        with mock.patch.object(C, "_looks_like_datasource", return_value=True):
            removed = run_case(case)
        assert "http_datasource_provenance_repointed" in (removed.causal_check or {}), (
            f"{case.id}: disabling the gate must re-point onto the decoy — otherwise the "
            f"GREEN abstain is not actually produced by the gate")
        assert (removed.attributed_defect or {}).get("file") != case.expect["attributed_file"]
