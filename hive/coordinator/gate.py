"""L-03 sufficiency judgement — two opposite-direction predicates:

1. ``gate_load_bearing`` (slot-level, STRICT): promote a candidate to a formal
   gap only if the answer would change the queen's search OR converge's verdict
   (FR-3 minimal-query). An expected-carveout candidate is promoted
   unconditionally (FR-4: true expected is design_change fuel, never dropped).
2. ``ready`` (session-level, LOOSE): pass as soon as the queen *can run*
   (FR-7) — every load-bearing gap resolved AND a minimal seed — OR a hard cap
   forces a seal. Both ``ready`` and ``sealed`` hand off (FR-8); there is no
   punt/abstain state.

State-machine OWNERSHIP is P-01 (W2); this module supplies the transition
predicates and the seal side-effect L-06 then consumes.
"""
from __future__ import annotations

from hive.coordinator.model import Gap, GapState, TAU_LB


def gate_load_bearing(candidates: list[dict], seed_draft: dict,
                      manifest: dict | None = None) -> list[Gap]:
    """Filter slot candidates to formal load-bearing gaps (all load_bearing=True,
    P-01 §8 invariant)."""
    gaps: list[Gap] = []
    for c in candidates:
        if c.get("expected_carveout"):
            gaps.append(_promote(c))                     # FR-4: unconditional
            continue
        if _assess_load_bearing(c, seed_draft) >= TAU_LB:
            gaps.append(_promote(c))
        # else: answer would not change the outcome → don't ask (FR-3 minimal-query)
    return gaps


def ready(gap_state: GapState) -> str:
    """Evaluate sufficiency. Mutates gap_state.status and returns it.

    Order is fixed (L-03 §4.2): hard cap FIRST (budget/loop runaway guard), then
    the loose pass, else keep collecting. Returns 'sealed'|'ready'|'collecting'.
    """
    caps = gap_state.caps or {}
    rounds = caps.get("rounds_left")
    budget = caps.get("budget_left")
    hardcap = (rounds is not None and rounds <= 0) or \
              (budget is not None and budget <= 0)
    if hardcap:
        seal(gap_state)
        gap_state.status = "sealed"
        return "sealed"                                  # cap termination also hands off
    if _all_resolved(gap_state.gaps) and _seed_min_sufficient(gap_state):
        gap_state.status = "ready"
        return "ready"
    gap_state.status = "collecting"
    return "collecting"


def seal(gap_state: GapState) -> None:
    """Cap-reached side-effect: every still-open gap → skipped (no re-open,
    P-01 §3), with provenance so the skip is never silent (L-06 consumes it)."""
    for g in gap_state.gaps:
        if g.status == "open":
            g.status = "skipped"
            g.provenance = {**(g.provenance or {}), "reason": "cap_reached"}


# ── §2.2 assess_load_bearing — "does this answer change the outcome?" ──────────
def _assess_load_bearing(c: dict, seed_draft: dict) -> float:
    if _slot_already_known(c, seed_draft):
        return 0.0                                       # already in the seed → not a gap
    # W1: no live targeted read; use the fork's outcome estimate. Borderline
    # verification (D-01 §7 read) is a W2 refinement — conservative keep is the
    # threshold itself, and the cap guarantees termination regardless.
    return max(0.0, min(1.0, float(c.get("load_bearing_hint") or 0.0)))


def _slot_already_known(c: dict, seed_draft: dict) -> bool:
    ctx = (seed_draft or {}).get("caller_supplied_context", "") or ""
    slot = (c.get("slot") or "").strip().lower()
    return bool(slot) and slot in ctx.lower()


def _all_resolved(gaps: list[Gap]) -> bool:
    return all(g.status in ("answered", "skipped")
               for g in gaps if g.load_bearing)


def _seed_min_sufficient(gap_state: GapState) -> bool:
    # Loose: "the queen can run" — a non-empty symptom context. expected is a
    # bonus, not required (best-effort, FR-8).
    return bool((gap_state.symptom_raw or "").strip())


def _promote(c: dict) -> Gap:
    return Gap(
        id=c.get("id", ""),
        slot=c.get("slot", ""),
        load_bearing=True,
        expected_carveout=bool(c.get("expected_carveout")),
        kind=c.get("kind", "context"),
        format="free",                                   # W1 is always free-form
        salience=float(c.get("salience") or 0.0),
        status="open",
        provenance=dict(c.get("provenance") or {}),
    )
