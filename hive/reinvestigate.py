"""Reaction #3 — route a specify ``needs_reinvestigation`` to ONE cheap re-run.

[hive.py] used to dead-end NR: log the termination and return, with no consumer
(M013 §1 diagnosis — "이름만 재조사인 막다른 길"). This module is the missing
consumer. It does NOT re-decompose or re-run the whole loop — that would defeat
Hive's cost reason to exist (M013 §3, an escalation "사다리" the user rejected). It
reads the STRUCTURED reason the gates stamped (``spec['reinvestigation']`` — Step A)
and the per-axis coverage tag (``verdict['coverage']`` — B2/#5) and decides the
CHEAPEST re-entry that could add NEW evidence:

    reason_code                                          → action
    ───────────────────────────────────────────────────────────────────────
    seed_target_uncovered / anchor_not_grounded /        → re_retrieve  (narrow, thin axes)
        deferred_root_cause
    ineffective / inconclusive                           → re_converge  (re-stitch, node out)
    stale_anchor / legacy_coerce / author_declared       → terminate   (no new evidence)

The honesty gate (M013 §3 #3 / #5): a re-run is proposed ONLY when the coverage tag
says new evidence is REACHABLE — an axis came back thin, so a wider re-fetch may
recover it. When the evidence was already SUFFICIENT, re-fetching the same scope
buys nothing, so the plan is ``terminate``: an honest NR, never a busy-loop. This
keeps the user's "새 증거 없으면 정직한 종료" a fact (from the coverage data), not a
guess.

The decision here is PURE and deterministic (free). The live re-run it proposes is
the paid, bounded (one-shot) step the orchestrator fires under approval — this
module only routes; it never calls a model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hive.specify import (
    RI_ANCHOR_NOT_GROUNDED,
    RI_CONVERGE_LOCUS_UNCOVERED,
    RI_DATASOURCE_REGRESSION,
    RI_DEFERRED_ROOT_CAUSE,
    RI_INCONCLUSIVE,
    RI_INEFFECTIVE,
    RI_SEED_TARGET_UNCOVERED,
    RI_VERIFY_INCONSISTENT,
)

logger = logging.getLogger("hive.reinvestigate")

ACTION_RE_RETRIEVE = "re_retrieve"
ACTION_RE_CONVERGE = "re_converge"
ACTION_RE_AUTHOR = "re_author"
ACTION_TERMINATE = "terminate"

# A reason whose gap is MISSING/UNGROUNDED evidence — a narrow re-retrieve of the
# starved axis can recover it (only worth firing where an axis actually came back thin).
# deferred_root_cause joins these: the punted substantive fix's axis was never grounded
# as an edit, so re-fetching that thin axis is the cheapest path to the missing evidence.
# datasource_regression joins too: the edit read the wrong table, so the real source has
# to be re-retrieved (N176) — not a causal-stitch error, so re_retrieve, not re_converge.
_RETRIEVE_REASONS = frozenset(
    {RI_SEED_TARGET_UNCOVERED, RI_ANCHOR_NOT_GROUNDED, RI_DEFERRED_ROOT_CAUSE,
     RI_DATASOURCE_REGRESSION})
# A reason whose gap is a causal/effectiveness CONTRADICTION — the evidence is present
# but the stitch was wrong; re-converge with the refuted node excluded (M013 §2 table).
_CONVERGE_REASONS = frozenset({RI_INEFFECTIVE, RI_INCONCLUSIVE})
# A reason whose gap is AUTHORING, not evidence: converge already declared N independent
# loci (the evidence is present and the stitch is right), but the author under-produced —
# fewer edits than loci. Re-fetching or re-stitching adds nothing; the cheap fix is to
# RE-AUTHOR the SAME honey (the per-locus contract now drives one edit per declared locus).
# No model evidence gap, so this never re-fetches — it re-runs specify on the same evidence.
_REAUTHOR_REASONS = frozenset({
    RI_CONVERGE_LOCUS_UNCOVERED,
    RI_VERIFY_INCONSISTENT,
})
# Everything else (stale_anchor → specify-local re-anchor, not a re-investigate;
# legacy_coerce / author_declared → no machine-routable evidence gap) → honest terminate.


@dataclass
class ReinvestPlan:
    """The routed decision for one NR spec. ``axis_ids`` are the re-run's candidates."""

    action: str
    reason_code: str
    axis_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    # M035 ⑥→④ feedback: loci specify REFUTED against live code that the re-converge must
    # EXCLUDE from its candidate set, so the re-stitch lands on a different (e.g. FE)
    # located fragment instead of re-crowning the disproven node. Empty for every other
    # route; only ``re_converge`` driven by a specify refutation populates it.
    exclude_loci: list[dict[str, Any]] = field(default_factory=list)

    @property
    def will_rerun(self) -> bool:
        return self.action in (ACTION_RE_RETRIEVE, ACTION_RE_CONVERGE, ACTION_RE_AUTHOR)


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip().strip("/").lower()


def _same_locus(a: str, b: str) -> bool:
    """Path-segment-aligned equality/suffix match (handles abs↔rel, basename-degrade)."""
    a, b = _norm_path(a), _norm_path(b)
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def _redirect_target_exists(verdicts: list[dict[str, Any]],
                            refuted_loci: list[dict[str, Any]]) -> bool:
    """True when a LOCATED verdict sits at a file NOT among the refuted loci — i.e. a
    non-refuted candidate exists for the re-converge to redirect the attribution onto."""
    refuted_files = [_norm_path(x.get("file", "")) for x in refuted_loci if x.get("file")]
    for v in verdicts:
        vd = v.get("verdict") or {}
        if not vd.get("located"):
            continue
        f = vd.get("file", "")
        if f and not any(_same_locus(f, rf) for rf in refuted_files):
            return True
    return False


def _thin_axes(verdicts: list[dict[str, Any]]) -> list[str]:
    """Axes whose local FIND came back thin (B2 tag) — re-fetch may add new evidence."""
    return [str(v.get("axis_id") or "?") for v in verdicts
            if (v.get("coverage") or {}).get("thin")]


def _located(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in verdicts if (v.get("verdict") or {}).get("located")]


def plan_reinvestigation(spec: dict[str, Any],
                         verdicts: list[dict[str, Any]] | None = None) -> ReinvestPlan:
    """Decide the one cheap re-entry for an NR spec — pure, deterministic, free.

    Returns a :class:`ReinvestPlan`. ``terminate`` means an honest NR (no cheaper
    re-run could add evidence); ``re_retrieve``/``re_converge`` name the bounded
    re-run the orchestrator may fire (one-shot, approval-gated).
    """
    verdicts = verdicts or []
    ri = spec.get("reinvestigation") or {}
    reason = str(ri.get("reason_code") or "")

    if spec.get("termination") != "needs_reinvestigation":
        return ReinvestPlan(ACTION_TERMINATE, reason,
                            rationale="spec is not needs_reinvestigation — nothing to route")

    # ── M035 ⑥→④ feedback (priority route): specify REFUTED the honey's attributed locus
    # against live code (no bug there) and stamped it in ``reinvestigation.refuted_loci``.
    # Without this, such an NR routes by its base reason — most commonly ``author_declared``
    # → terminate — i.e. the punt the operator's first principle forbids. The refutation is
    # a LEAD: when a LOCATED verdict at a DIFFERENT file still exists (e.g. the FE render
    # locus a sibling axis localised), re-converge with the refuted loc/i EXCLUDED so the
    # stitch must land on the surviving candidate. This is the cross-stage analog of
    # converge's own redirect re-stitch — it closes the loop instead of handing it back.
    refuted_loci = [x for x in (ri.get("refuted_loci") or [])
                    if isinstance(x, dict) and x.get("file")]
    if refuted_loci:
        names = [str(x.get("file")) for x in refuted_loci]
        if _redirect_target_exists(verdicts, refuted_loci):
            return ReinvestPlan(
                ACTION_RE_CONVERGE, reason or "locus_refuted",
                axis_ids=names, exclude_loci=refuted_loci,
                rationale=(f"specify refuted the honey's locus/loci {names} against live "
                           "code, and a located candidate at a DIFFERENT file remains — "
                           "re-converge with the refuted loc/i EXCLUDED so the stitch lands "
                           "on it (the ⑥→④ feedback edge, not a dead-end punt)"))
        return ReinvestPlan(
            ACTION_TERMINATE, reason or "locus_refuted",
            rationale=(f"specify refuted the honey's locus/loci {names} but NO other located "
                       "candidate exists to redirect to — honest NR (no reachable evidence)"))

    if reason in _RETRIEVE_REASONS:
        thin = _thin_axes(verdicts)
        if thin:
            return ReinvestPlan(
                ACTION_RE_RETRIEVE, reason, axis_ids=thin,
                rationale=(f"{reason}: axes {thin} came back thin — a narrow re-retrieve "
                           "of those may recover new evidence"))
        return ReinvestPlan(
            ACTION_TERMINATE, reason,
            rationale=(f"{reason}: no axis came back thin — the evidence was sufficient, so "
                       "re-fetching the same scope adds nothing (honest NR)"))

    if reason in _REAUTHOR_REASONS:
        if reason == RI_VERIFY_INCONSISTENT:
            missing = (spec.get("verify_consistency") or {}).get(
                "missing_test_edit_ids") or []
            return ReinvestPlan(
                ACTION_RE_AUTHOR, reason,
                axis_ids=[str(edit_id) for edit_id in missing],
                rationale=(f"{reason}: verify referenced missing edit ids {missing} — "
                           "re-author the SAME honey with internally consistent edits[] "
                           "and verify.test_edit_ids; no new evidence is needed"))
        cov = spec.get("converge_coverage") or {}
        loci = cov.get("loci") or []
        return ReinvestPlan(
            ACTION_RE_AUTHOR, reason,
            axis_ids=[str(t) for t in loci],
            rationale=(f"{reason}: converge declared {len(loci)} independent loci but the "
                       "author under-covered — re-author the SAME honey (evidence already "
                       "present; one edit per locus per the contract), no re-fetch"))

    if reason in _CONVERGE_REASONS:
        loc = _located(verdicts)
        if len(loc) >= 2:
            return ReinvestPlan(
                ACTION_RE_CONVERGE, reason,
                axis_ids=[str(v.get("axis_id") or "?") for v in loc],
                rationale=(f"{reason}: {len(loc)} located verdict(s) — re-converge with the "
                           "refuted node excluded (re-stitch, not re-fetch)"))
        return ReinvestPlan(
            ACTION_TERMINATE, reason,
            rationale=(f"{reason}: fewer than 2 located verdict(s) — nothing to re-stitch "
                       "(honest NR)"))

    return ReinvestPlan(
        ACTION_TERMINATE, reason,
        rationale=(f"{reason or 'unknown'}: no cheap re-run adds evidence — re-anchor is "
                   "specify-local and the remaining reasons are not routable"))


def log_reinvestigation_plan(spec: dict[str, Any],
                             verdicts: list[dict[str, Any]] | None = None,
                             log: logging.Logger = logger) -> ReinvestPlan:
    """Compute and LOG the routed plan at an NR exit (free) — surfaces what the dead-end
    used to swallow. The live re-run is fired separately under approval (one-shot)."""
    plan = plan_reinvestigation(spec, verdicts)
    if plan.will_rerun:
        log.info("reinvestigation: reason=%s → PLAN %s on axes %s — %s "
                 "(live re-run gated by cfg.reinvestigation.live)",
                 plan.reason_code or "?", plan.action, plan.axis_ids, plan.rationale)
    else:
        log.info("reinvestigation: reason=%s → terminate honestly — %s",
                 plan.reason_code or "?", plan.rationale)
    return plan


def run_reinvestigation_loop(spec: dict[str, Any], result: dict[str, Any], *, cfg,
                             rerun, respecify, read_honey,
                             log_plan=log_reinvestigation_plan,
                             log: logging.Logger = logger):
    """Bounded live re-run loop for an NR spec (M013 reaction #3 orchestration).

    I/O is injected as callables so the loop is pure and unit-testable:
      ``rerun(plan, result) -> result | None``  — execute the routed re-entry; writes the
          re-grounded honey as a side effect. ``None`` = nothing was re-run.
      ``respecify() -> spec``                   — re-run specify on the current honey.
      ``read_honey() -> str``                   — current honey text (for the change-guard).
      ``log_plan(spec, verdicts) -> ReinvestPlan``

    The plan is logged ONCE even when live is off (the free diagnostic the dead-end used
    to swallow). When ``cfg.reinvestigation.live`` is on it loops up to
    ``cfg.reinvestigation.max_rounds`` times, stopping early when: the plan says terminate,
    a re-run adds nothing (``None``), the re-grounded honey is UNCHANGED (e.g. re_converge
    re-grounds the same code → identical result → paying specify again buys nothing), or
    specify clears NR. So ``max_rounds`` is a true ceiling, never a busy-loop. Returns the
    final ``(spec, result)``.
    """
    if spec.get("termination") != "needs_reinvestigation":
        return spec, result
    plan = log_plan(spec, result.get("verdicts"))
    rounds = max(1, int(getattr(cfg.reinvestigation, "max_rounds", 1)))
    rnd = 0
    while (getattr(cfg.reinvestigation, "live", False) and plan.will_rerun
           and rnd < rounds
           and spec.get("termination") == "needs_reinvestigation"):
        rnd += 1
        # Re-author: the evidence is already present (converge declared the loci) — the gap
        # is the author under-covering, not missing/contradicted evidence. Re-run specify on
        # the SAME honey (the per-locus contract drives full coverage); do NOT re-fetch and
        # do NOT apply the honey-unchanged guard (the honey is unchanged BY DESIGN here).
        if plan.action == ACTION_RE_AUTHOR:
            log.info("reinvestigation live: re-authoring the same honey (round %d/%d) — "
                     "converge declared %d loci, author under-covered",
                     rnd, rounds, len(plan.axis_ids))
            spec = respecify()
            log.info("reinvestigation live: round %d/%d → termination=%s",
                     rnd, rounds, spec.get("termination", "?"))
            plan = log_plan(spec, result.get("verdicts"))
            continue
        prev_honey = read_honey()
        new_result = rerun(plan, result)
        if new_result is None:
            break
        result = new_result
        if read_honey() == prev_honey:
            log.info("reinvestigation live: re-grounded honey unchanged (round %d/%d) — "
                     "nothing new to specify, stopping", rnd, rounds)
            break
        log.info("reinvestigation live: re-running specify on the re-grounded honey "
                 "(round %d/%d)", rnd, rounds)
        spec = respecify()
        log.info("reinvestigation live: round %d/%d → termination=%s",
                 rnd, rounds, spec.get("termination", "?"))
        plan = log_plan(spec, result.get("verdicts"))
    return spec, result
