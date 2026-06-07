"""Investigate pipeline — the cheap (M004) path that replaces swarm fan-out.

    decompose (queen, 1 call)
      -> for each axis:  bridge (free) -> retrieve (free local FIND) -> JUDGE
      -> verdict report

This is the redesign's spine wired end-to-end: one queen decomposition, then a
zero-cost local retrieval per axis (no open-ended drone), then a budgeted JUDGE
verdict (``hive.judge``). The queen→retrieve seam is the ``hive.searchplan``
bridge; the JUDGE budget (calls/axis, axes/run) comes from ``cfg.judge``.

Cost shape (credit/usage billing, [[hivework-worker-cost-shift]]):
  - decompose: 1 queen call,
  - retrieve:  0 (local ripgrep + read + git),
  - judge:     ≤ ``max_calls_per_axis`` per judged axis, over ≤ ``max_axes`` axes.

Only the *judged* axes spend. We judge the leaf axes (no unmet ``depends_on``)
up to ``max_axes`` — synthesis/dependent axes are not localisation targets and
are skipped (a deterministic, free gate; smarter routing is a later lever).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from hive.converge import run_converge
from hive.decompose import run_decompose
from hive.fanout import ReinforceBudget, reinforce_thin_axis
from hive.judge import run_judge_votes
from hive.retriever import SearchPlan, _ripgrep, retrieve, retrieve_followup
from hive.searchplan import (
    coverage_risk, extract_doc_topics, extract_globs, extract_keywords,
    task_to_searchplan, is_visibility_symptom, with_visibility_probe,
)

logger = logging.getLogger("hive.investigate")

# Header for the honey section that lists the seed's own explicitly-named edit
# targets (Defect 2). specify parses this section to GROUND those files' live text
# and to enforce that none is silently dropped — keep the literal in sync with
# ``hive.specify.SEED_TARGET_SECTION`` (imported from here).
SEED_TARGET_SECTION = "## Seed-specified edit targets"

# Header for the machine-parseable list of loci converge attributed when a scenario has
# MULTIPLE INDEPENDENT defects (N179): the primary attributed_defect PLUS each
# additional_defect. Emitted ONLY in that multi-locus case (a single-defect convergence
# produces no such section). specify's converge-coverage gate reads it back to refuse a
# ready_to_apply spec that authored an edit for only SOME of the independent loci. Keep
# the literal in sync with ``hive.specify.CONVERGE_TARGET_SECTION`` (imported from here).
CONVERGE_TARGET_SECTION = "## Converge-attributed edit targets"

# Optional channel for the requester's own words — the direct message/hints the
# caller (a chat operator, or a FlowGate rejection note) supplies alongside the
# seed. It is OPT-IN (the ``--comment`` flag); when absent nothing changes. We fold
# it into the seed text so every downstream stage that already reads the seed
# (decompose/judge/assemble) and the local honey (which embeds the seed verbatim →
# specify) sees it — no signature changes. The header sentence carries the only
# guardrail needed: the requester's stated INTENT and VALUES are authoritative
# requirements, but any claim about WHERE code lives is still verified against live
# files (the edit-spec contract already mandates byte-for-byte re-anchoring), so a
# comment can resolve ambiguous direction without ever standing in for code grounding.
CALLER_CONTEXT_SECTION = "## Caller-supplied context (requester's direct input)"


def format_caller_context(comments: list[str] | None) -> str:
    """Render opt-in requester comments into a labelled seed section (or "").

    Returns a leading-newline block ready to append to the seed text, or an empty
    string when no comments were supplied (so callers can append unconditionally).
    Each comment is listed verbatim and order-preserving; blank/whitespace-only
    entries are dropped.
    """
    items = [c.strip() for c in (comments or []) if c and c.strip()]
    if not items:
        return ""
    lines = [
        "",
        CALLER_CONTEXT_SECTION,
        "",
        "Direct input from the requester. Treat the stated INTENT and VALUES as "
        "authoritative requirements (what the change must achieve). Any claim about "
        "WHERE code lives is a HINT — verify it against the live files, never anchor "
        "on the requester's prose alone.",
        "",
    ]
    lines += [f"- {c}" for c in items]
    lines.append("")
    return "\n".join(lines)

# Diagnostic seeds ask the pipeline to TRACE/MAP/EXPLAIN a path or behaviour — the
# deliverable is the answer (the converged call path), NOT an edit. Forcing such a
# seed toward an edit is exactly what drove N169 into the needs_reinvestigation
# loop (the author rightly could not author an edit for "trace the call path").
# Fix seeds ask to CHANGE code. The classifier is deterministic, free, and only a
# HINT to the honey/author — it never blocks authoring (a diagnostic seed that also
# names concrete fix targets still gets them authored).
_DIAGNOSTIC_VERBS = (
    "trace", "map ", "locate", "find where", "where is", "where does", "why ",
    "investigate", "diagnose", "audit", "understand", "explain", "identify",
    "figure out", "root cause", "root-cause", "call path", "call-path", "usage map",
)
_FIX_VERBS = (
    "fix", "change", "add ", "remove", "delete", "update", "implement", "replace",
    "refactor", "rename", "correct", "patch", "make it", "should be", "must be",
    "set ", "wire", "introduce", "ensure",
)


def classify_seed_kind(seed_text: str) -> str:
    """Classify a seed as ``"diagnostic"`` or ``"fix"`` (deterministic, free).

    Heuristic on the seed's leading lines (where the task verb lives): a strong
    diagnostic verb with NO strong fix verb ⇒ diagnostic; otherwise fix (the
    conservative default that preserves today's edit-oriented behaviour). Only a
    HINT — it never gates authoring.
    """
    head = "\n".join((seed_text or "").splitlines()[:8]).lower()
    has_fix = any(v in head for v in _FIX_VERBS)
    has_diag = any(v in head for v in _DIAGNOSTIC_VERBS)
    if has_diag and not has_fix:
        return "diagnostic"
    return "fix"


def _leaf_axes(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Axes with no dependencies — the parallel evidence-collection leaves.

    Dependent axes (synthesis/decision/verification) consume other axes' outputs;
    they are not bug-localisation targets, so they get no JUDGE call.
    """
    return [t for t in tasks if not (t.get("depends_on") or [])]


# A repo-relative path token the seed names AS A CONCRETE FILE (has an extension,
# no wildcard) — e.g. ``client/src/.../DocWorkflow.vue``. Directory scopes and
# ``**`` globs are not concrete-file anchors.
_CONCRETE_FILE_RE = re.compile(r"\.[A-Za-z0-9]{1,6}$")


def _seed_relevance(task: dict[str, Any], seed_basenames: set[str],
                    seed_kw: set[str]) -> int:
    """Score a decompose axis by how much it matches the SEED (free, deterministic).

    The queen fans out blind to the answer and routinely scatters a single-line
    change across a dozen unrelated axes (T891: a CSS class add drew SQL-drop and
    getter-reactivity axes). Truncation at ``max_axes`` is by POSITION, so a
    rabbit-hole axis can survive while the seed's own target is cut. Ranking by
    seed-relevance before truncation floats the on-topic axes up and lets the
    off-topic ones sink past the cap. Signal: the axis scopes a file the seed
    named (strong), plus how many of the seed's grep keywords it reuses.
    """
    sp = task.get("search_plan") or {}
    glob_blob = " ".join(str(g) for g in (sp.get("file_globs") or [])).lower()
    text = (str(task.get("title", "")) + " " + str(task.get("brief", ""))).lower()
    kws = {str(k).lower() for k in (sp.get("keywords") or [])}
    score = 0
    if any(bn in glob_blob or bn in text for bn in seed_basenames):
        score += 3
    score += len(kws & seed_kw)
    return score


def _prioritize_axes(leaves: list[dict[str, Any]], seed_text: str,
                     *, max_keywords: int = 14) -> list[dict[str, Any]]:
    """Re-order leaves by seed-relevance and inject the seed's own target axis.

    Two deterministic, zero-cost guards against decompose non-determinism
    (the T891 bottleneck — the engine fixes downstream of judge cannot help when
    the investigation never locates the seed's named spot):

      (a) PRIORITISE — sort the queen's leaves by :func:`_seed_relevance` so the
          axes that match the seed survive the ``max_axes`` cap and the scattered
          rabbit-hole axes sink past it (stable: ties keep the queen's order).
      (b) INJECT — when the seed names a CONCRETE file (``Foo.vue``, not just a
          directory), prepend a ``SEED_ANCHOR`` axis scoped to exactly that file
          with the seed's own keywords, so a scattered decompose can never skip
          the seed's target. It rides at the front, guaranteed past the cap.

    Pure text extraction (reuses the searchplan bridge) — no model call, never
    raises. When the seed names no concrete file, (b) is skipped and only (a)
    applies; ranking still needs only the seed's keywords.
    """
    seed_files = [g for g in extract_globs(seed_text)
                  if "*" not in g and _CONCRETE_FILE_RE.search(g)]
    seed_kw = extract_keywords(seed_text)
    seed_kw_set = {k.lower() for k in seed_kw}
    seed_basenames = {os.path.basename(g).lower() for g in seed_files}

    ranked = sorted(
        leaves,
        key=lambda t: _seed_relevance(t, seed_basenames, seed_kw_set),
        reverse=True)

    if not seed_files:
        return ranked

    seed_axis = {
        "id": "SEED_ANCHOR",
        "title": "seed-named target(s): "
                 + ", ".join(os.path.basename(g) for g in seed_files),
        "brief": ("Investigate the exact file/location the seed names for this "
                  "change. Deterministically injected so a scattered decompose "
                  "cannot skip the seed's own target."),
        "depends_on": [],
        "search_plan": {
            "keywords": seed_kw[:max_keywords],
            "file_globs": seed_files,
            "doc_topics": extract_doc_topics(seed_text),
        },
    }
    return [seed_axis] + ranked


# A repo-relative ``path.ext`` optionally followed by ``:line`` / ``:lo-hi`` as the
# seed writes it — used to honour an explicit line the seed already pinned.
_SEED_CITE_RE = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)(?::(\d+)(?:-(\d+))?)?")

# An edit target must be DESIGNATED, not merely mentioned. A seed routinely lists
# files in an orientation/context map ("Backend lives in X.py (verify)") or in
# investigative prose ("Trace the endpoint in Y.py") that the author must NOT treat
# as an edit site — doing so makes the seed-coverage gate force edits onto files the
# investigation (converge) ruled out, contradicting the honey's own conclusion. N177:
# the seed's ``[System context]`` file map ("SQL queries live in queries.json … head
# API: workflow_head_routes.py (verify) … FE view-state: workflowViewState.ts") became
# FIVE mandatory BE edit targets — including the queries.json the SAME seed says "Do
# NOT author an edit there" — so apply could never go ready and the author thrashed on
# bogus targets instead of converge's single FE attribution. A file qualifies as an
# edit target only when its mention carries an EDIT-INTENT cue (or the seed pinned an
# explicit ``:line`` on it) AND no DO-NOT-EDIT cue rules it out.
_EDIT_INTENT_RE = re.compile(
    r"\[\s*edit|\bedits?\b|\bedited\b|\bediting\b|\bfix(?:es|ed|ing)?\b|\breplace\b|"
    r"\brewrite\b|\bmodif(?:y|ies|ied)\b|\bchange\b|\bauthor\b|\bupdate\b|"
    r"수정|편집|고쳐|교정|바꿔",
    re.IGNORECASE)
# A do-not-edit cue rules a seed-named file OUT of edits[]. The verb list includes
# ``use`` (T906: "Do NOT use `list_routes.py` as the primary fix site" — a prohibition
# the earlier list missed, so the file was wrongly handed to the seed-coverage gate as a
# mandatory target even though the same seed's §6 named it off-path). ``off-path`` and the
# Korean off-path / do-not-use idioms (오답 경로 / 사용하지 마 / 쓰지 마) are caught the
# same way, since a "this is the wrong/decoy path" mention is never an edit designation.
_DO_NOT_EDIT_RE = re.compile(
    r"do\s+not\s+(?:author|edit|modif|produce|touch|change|rewrite|use)|"
    r"don'?t\s+(?:edit|touch|modify|change|use)|no-?op|ruled\s+out|stop\s+re-?examin|"
    r"leave\s+(?:it\s+)?unchanged|off-?path|"
    r"건드리지\s*마|수정하지\s*마|편집하지\s*마|손대지\s*마|만지지\s*마|"
    r"사용하지\s*마|쓰지\s*마|오답\s*경로",
    re.IGNORECASE)


def _seed_target_designation(seed_text: str, rel: str) -> tuple[bool, bool]:
    """``(designated_for_edit, ruled_out)`` for a seed-named file via line-scoped cues.

    Scans each seed line that MENTIONS the file (by path or basename): a line with an
    edit-intent cue designates it for editing; a line with a do-not-edit cue rules it
    out. A file mentioned only in orientation / investigation prose ("(verify)",
    "Trace …") is neither — so the seed-coverage gate does not force it into edits[].
    Pure-local, free, never raises.
    """
    base = os.path.basename(rel)
    designated = ruled_out = False
    for line in seed_text.splitlines():
        if rel not in line and base not in line:
            continue
        if _DO_NOT_EDIT_RE.search(line):
            ruled_out = True
        if _EDIT_INTENT_RE.search(line):
            designated = True
    return designated, ruled_out


def seed_edit_targets(seed_text: str, code_root: str | None,
                      docs_root: str | None = None,
                      *, max_targets: int = 8) -> list[dict[str, Any]]:
    """Resolve the concrete files the seed NAMES into groundable ``file:line`` targets.

    The seed routinely pins exact edit sites (``[Edit 1] server/sql/queries/queries.json
    get_pending_head_by_group → …``). When investigate fails to independently re-locate
    one (Defect 1), that user-provided target must NOT vanish: we lift it here so the
    honey can present it as an AUTHOR target with live ground truth, regardless of the
    judge's verdicts. The structural mirror of ``_prioritize_axes``'s SEED_ANCHOR
    injection, lifted from the axis layer up to the honey/grounding layer.

    For each named concrete file (extension, no wildcard):
      * honour an explicit ``:line`` the seed already wrote; else
      * grep the seed's keywords inside the file and pick the line with the most
        distinct keyword hits (ties → lowest line) as the representative anchor.

    A file that cannot be found on disk (code tree then docs tree) is skipped — we
    only surface targets we can actually ground. Pure-local, free, never raises.
    """
    if not code_root:
        return []
    seed_files = [g for g in extract_globs(seed_text)
                  if "*" not in g and _CONCRETE_FILE_RE.search(g)]
    if not seed_files:
        return []
    explicit: dict[str, tuple[int, int]] = {}      # file the seed pinned a line on
    for m in _SEED_CITE_RE.finditer(seed_text):
        if not m.group(2):
            continue
        rel = m.group(1).replace("\\", "/").lstrip("/")
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        explicit.setdefault(rel, (lo, hi))

    seed_kw = extract_keywords(seed_text)
    roots = [code_root] + ([docs_root] if docs_root else [])
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for g in seed_files:
        rel = g.replace("\\", "/").lstrip("/")
        if rel in seen:
            continue
        seen.add(rel)
        root = next((r for r in roots
                     if r and os.path.isfile(os.path.join(r, rel))), None)
        if root is None:
            continue
        # An explicit user-written ``:line`` is itself an edit designation.
        ln_lo = ln_hi = None
        for cited, (lo, hi) in explicit.items():
            if (rel == cited or rel.endswith("/" + cited) or cited.endswith("/" + rel)
                    or os.path.basename(cited) == os.path.basename(rel)):
                ln_lo, ln_hi = lo, hi
                break
        # Qualify the file: only a file the seed DESIGNATES for editing (an edit-intent
        # cue near its mention, or an explicit pinned line) becomes a binding target —
        # never one that is merely mapped in orientation prose or explicitly ruled out
        # (N177's [System context] map / "do NOT author an edit there" queries.json).
        designated, ruled_out = _seed_target_designation(seed_text, rel)
        if ruled_out or not (designated or ln_lo is not None):
            continue
        # Prose "(around) lines N-M" near the file mention (the seed writes the
        # range as prose, not path:line — e.g. "spec.ts\nAround lines 299-322").
        if ln_lo is None:
            idx = seed_text.find(os.path.basename(rel))
            if idx >= 0:
                pm = re.search(r"lines?\s+(\d+)(?:\s*-\s*(\d+))?",
                               seed_text[idx: idx + 200], re.IGNORECASE)
                if pm:
                    ln_lo = int(pm.group(1))
                    ln_hi = int(pm.group(2)) if pm.group(2) else ln_lo
        if ln_lo is None and seed_kw:
            kws_at: dict[int, set[str]] = {}
            for kw in seed_kw:
                for h in _ripgrep(kw, [rel], root):
                    kws_at.setdefault(h["line"], set()).add(kw)
            if kws_at:
                best = sorted(kws_at.keys(),
                              key=lambda l: (len(kws_at[l]), -l), reverse=True)[0]
                ln_lo = ln_hi = best
        if ln_lo is None:
            continue   # nothing groundable to cite
        targets.append({"file": rel, "lines": f"{ln_lo}-{ln_hi}"})
        if len(targets) >= max_targets:
            break
    return targets


def _apply_call_budget(judged: list[dict[str, Any]], votes_cfg: int, max_calls: int,
                       budget: int) -> tuple[list[dict[str, Any]], int]:
    """Fit the run under ``max_total_calls`` — the one-number judge budget cap.

    Returns ``(judged, effective_votes)`` such that the WORST-CASE judge calls,
    ``len(judged) × effective_votes × max_calls``, is ≤ ``budget``. Pure and
    deterministic. Strategy (M010 budget control): REDUCE VOTES first so axis
    coverage is preserved and only the voting depth shrinks; only when even a
    single vote across all axes overflows do we trim axes (and force one vote).
    ``budget <= 0`` ⇒ unlimited, unchanged (today's behaviour). The ceiling is
    worst-case (every vote assumed to spend ``max_calls``); actual runs land at or
    under it because the re-judge does not always fire.
    """
    votes_cfg = max(1, int(votes_cfg))
    max_calls = max(1, int(max_calls))
    budget = int(budget or 0)
    if budget <= 0 or not judged:
        return judged, votes_cfg
    if len(judged) * max_calls > budget:
        keep = max(1, budget // max_calls)
        return judged[:keep], 1
    affordable = budget // (len(judged) * max_calls)
    return judged, max(1, min(votes_cfg, affordable))


def _converge_fragments(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each axis's best-of-N union into one converge fragment per locus.

    converge stitches per-axis LOCATED fragments into one path, so the best-of-N
    union must reach it as fragments, not collapse to the representative verdict
    (M010 §5: union → causal gate, never majority — a 1/N locus is a candidate the
    gate vets, not noise a vote count discards). For a located axis we emit one
    fragment per distinct candidate ``(file, lines)``; an unlocated axis passes
    through once so its refute reason stays as context. At ``votes_per_axis=1``
    every located axis has exactly one candidate, so this yields one fragment per
    axis — byte-identical to the pre-voting converge input.
    """
    out: list[dict[str, Any]] = []
    for v in verdicts:
        cands = v.get("candidates") or []
        if cands:
            for c in cands:
                out.append({
                    "axis_id": v.get("axis_id", "?"),
                    "title": v.get("title", ""),
                    "verdict": {"located": True, "file": c.get("file", ""),
                                "lines": c.get("lines", ""),
                                "reason": c.get("reason", "")},
                })
        else:
            out.append(v)
    return out


def run_investigate(
    *,
    seed_text: str,
    recipe_path: str | None,
    code_root: str,
    docs_root: str | None,
    output_path: str,
    cfg,
    ledger=None,
    provider_kwargs: dict | None = None,
    default_globs: list[str] | None = None,
    k: int = 6,
    top_files: int = 8,
    blame_files: int = 3,
) -> dict[str, Any]:
    """Run decompose → (bridge → retrieve → judge)* → verdict report.

    Returns a result dict ``{axes_judged, verdicts:[...], report_path}``. Never
    runs the expensive fan-out path. The JUDGE is the only spend point and is
    capped by ``cfg.judge`` (``max_axes`` axes × ``max_calls_per_axis`` calls).
    """
    queen = cfg.queen
    judge_role = cfg.role("judge")
    pk = dict(provider_kwargs or {})

    # ── ① decompose (queen, 1 call) — now also emits per-axis search_plan.
    # Stable, pipeline-agnostic stage marker for external watchdogs: the mode→
    # pipeline routing means a "create" task runs here (no "STAGE ② fan-out"), so
    # monitors grep ``[HIVE_STAGE]`` (self-describing) rather than a run-only label.
    logger.info("[HIVE_STAGE] pipeline=investigate stage=1 name=decompose")
    logger.info("① decompose (queen %s/%s)", queen.provider, queen.model)
    decompose_result = run_decompose(
        seed_text=seed_text, recipe_path=recipe_path, codebase_root=code_root,
        model=queen.model, provider=queen.provider, ledger=ledger,
        provider_kwargs=pk,
    )
    tasks = decompose_result.get("tasks", []) or []
    leaves = _leaf_axes(tasks)
    # depends_on pruning is silent by default, yet in EDIT mode the queen routinely
    # makes the very edit-target axes (BE_EDIT, FE_TEST_EDIT…) depend on the
    # investigation axes, so the seed's own targets get no judge call at all
    # (T892: 7 of 12 axes pruned here, BE_EDIT among them). Surface the dropped
    # axes so the miss is visible; the SEED_ANCHOR injection + seed_edit_targets
    # grounding are what actually recover the seed's targets downstream.
    nonleaf = [t for t in tasks if (t.get("depends_on") or [])]
    if nonleaf:
        logger.info("decompose: %d non-leaf (dependent) axes not judged: %s",
                    len(nonleaf),
                    [t.get("id") or t.get("name") or "?" for t in nonleaf])
    # Deterministic, free guard against decompose non-determinism (T891): rank the
    # leaves by seed-relevance and inject the seed's own named target as a front
    # axis, BEFORE the position-based max_axes truncation — so a scattered queen
    # cannot bury or skip the spot the seed explicitly points at.
    leaves = _prioritize_axes(leaves, seed_text)
    judged = leaves[: cfg.judge.max_axes]
    if judged and judged[0].get("id") == "SEED_ANCHOR":
        logger.info("seed-anchor: injected front axis for seed-named target(s) %s",
                    judged[0]["search_plan"]["file_globs"])
    logger.info("decompose → %d axes (%d leaf, judging %d, ceiling max_axes=%d)",
                len(tasks), len(leaves), len(judged), cfg.judge.max_axes)
    # Truncation is a correctness risk, not just a cost note: leaf axes past the
    # ceiling are dropped by position (no priority ordering), so a decisive
    # grep-once axis can be silently cut (N164: css_rules). Surface which axes
    # got dropped at WARNING so the operator can raise max_axes or re-scope.
    if len(leaves) > cfg.judge.max_axes:
        dropped = [t.get("id") or t.get("name") or "?" for t in leaves[cfg.judge.max_axes:]]
        logger.warning(
            "max_axes ceiling (%d) < leaf axes (%d): DROPPING %d un-judged axes %s "
            "— a decisive axis may be among them; raise judge.max_axes in hive.config.json",
            cfg.judge.max_axes, len(leaves), len(dropped), dropped)

    # ── Budget cap (max_total_calls): one hard ceiling on TOTAL judge model calls
    # this run, honoured by reducing votes first (then axes). Computed AFTER the
    # axis set is fixed and BEFORE the fan-out so every axis votes the same amount
    # and the worst-case spend is bounded up front (not discovered mid-run).
    votes_cfg = max(1, int(getattr(cfg.judge, "votes_per_axis", 1)))
    max_calls_cfg = max(1, int(getattr(cfg.judge, "max_calls_per_axis", 2)))
    # Voting runs single-shot per vote (run_judge_votes drops the re-judge when
    # votes>1), so the ACTUAL per-vote cost is 1 call; max_calls_cfg only governs
    # the votes=1 single-judgment path. Budget on the real per-vote cost so the cap
    # neither over-throttles voting nor under-counts a votes=1 re-judge.
    per_vote_calls = 1 if votes_cfg > 1 else max_calls_cfg
    budget = int(getattr(cfg.judge, "max_total_calls", 0) or 0)
    judged, effective_votes = _apply_call_budget(judged, votes_cfg, per_vote_calls, budget)
    if budget > 0 and effective_votes != votes_cfg:
        logger.info("max_total_calls=%d budget: votes_per_axis %d→%d over %d axes "
                    "(worst-case ≤ %d judge calls = %d × %d × %d)",
                    budget, votes_cfg, effective_votes, len(judged),
                    len(judged) * effective_votes * per_vote_calls,
                    len(judged), effective_votes, per_vote_calls)

    # ── ②..③ per axis: bridge → local retrieve (free) → JUDGE (budgeted).
    # The axes are INDEPENDENT: each judges only its own retrieve bundle against the
    # shared, read-only ``seed_files`` — nothing here reads another axis's verdict
    # (cross-axis synthesis happens once, later, in ④ converge). So the per-axis
    # retrieve+judge work is run CONCURRENTLY over a bounded pool
    # (``cfg.judge.max_parallel``), overlapping the judge's network round-trips
    # instead of serialising N×~20s of them. Results are reassembled in the original
    # ``judged`` order, so ``verdicts``/``bundles`` (and thus converge's input) are
    # identical to the sequential path — only wall-clock changes.
    #
    # Seed-named, on-disk edit targets — passed to every axis's judge so a verdict
    # citing one is grounded even when that axis's own retrieve didn't window it
    # (Defect 4a: seed files aren't guaranteed in every per-axis bundle).
    seed_files = {t["file"] for t in seed_edit_targets(seed_text, code_root, docs_root)}
    # Pre-resolve each judged axis's search plan (pure, cheap) so the docs=(none)
    # confound (N165) is surfaced ONCE here, before the fan-out — not racily (and
    # possibly multiple times) from inside concurrent workers.
    plans = [task_to_searchplan(task, default_globs=default_globs) for task in judged]
    # Visibility-class symptom (N176): a "not visible / disabled / not rendered" report
    # is produced by the component template's conditional-render branch, not the data
    # layer the brief is usually worded around. Deterministically add the template
    # directives (v-if/v-show/…) to every plan so the branch is retrieved wherever a
    # front-end file is in scope — a no-op where none is (never a fabricated finding).
    if is_visibility_symptom(seed_text):
        plans = [with_visibility_probe(sp) for sp in plans]
        logger.info("visibility-class symptom detected — added template conditional-"
                    "render probe (v-if/v-show/v-for/:disabled) to %d axis plan(s)",
                    len(plans))
    if docs_root is None:
        # docs=(none) confound (N165): the queen produced doc_topics for an axis but
        # no docs tree was supplied, so the entire design-doc channel is silently
        # skipped and any doc-targeting glob degrades into a code-tree search. Surface
        # it once at WARNING — a missing --docs is an invocation bug, not a result.
        for sp in plans:
            if sp.doc_topics:
                logger.warning(
                    "docs_root not supplied (--docs) but axes carry doc_topics "
                    "(first: [%s] topics=%s): the design-doc channel is DISABLED and "
                    "doc-targeted globs fall back to the code tree. Pass --docs <dir> "
                    "to enable design retrieval.", sp.axis_id, sp.doc_topics)
                break

    def _investigate_axis(idx, task, sp):
        """Retrieve (free, local) then JUDGE one axis. Independent of other axes;
        returns ``(idx, verdict_entry, bundle_to_keep)`` for order-preserving merge."""
        symptom = str(task.get("brief") or task.get("title") or sp.axis_id)
        logger.info("[HIVE_STAGE] pipeline=investigate stage=2 name=retrieve axis=%s",
                    sp.axis_id)
        logger.info("② retrieve [%s] keywords=%d globs=%d (local, free)",
                    sp.axis_id, len(sp.keywords), len(sp.file_globs))
        _retr_t0 = time.monotonic()
        bundle = retrieve(sp, code_root, docs_root, k=k,
                          top_files=top_files, blame_files=blame_files)
        _retr_dt = time.monotonic() - _retr_t0
        st = bundle.get("stats", {})
        logger.info("   FIND: %s hits → %s snippets, %s call-chain",
                    st.get("raw_hits"), st.get("snippets"), st.get("call_chain"))
        # Register this LOCAL (free, deterministic) retrieve in the ledger so stage ②
        # shows up in the configured DB alongside the billed model calls (provider=
        # 'local', mechanism='ripgrep'; cost aggregate untouched).
        if ledger is not None:
            ledger.record_local(
                stage="retrieve", axis_id=sp.axis_id, mechanism="ripgrep",
                detail=f"hits={st.get('raw_hits')} snippets={st.get('snippets')} "
                       f"call_chain={st.get('call_chain')}",
                latency_s=_retr_dt)
        gv = st.get("glob_validation", {})
        if gv.get("dropped_empty") or gv.get("dropped_overbroad"):
            logger.info("   glob-guard: kept=%s dropped_empty=%s dropped_overbroad=%s",
                        gv.get("kept"), gv.get("dropped_empty"),
                        gv.get("dropped_overbroad"))
        # (B2/#5) Confirm the queen's coverage_risk self-doubt against what the FIND
        # actually retrieved: only a flagged AND empty axis is "starved". The tag
        # rides the verdict so converge/specify/reaction can route a retrieval gap
        # apart from a reasoning gap — free, deterministic, no extra call.
        coverage = _axis_coverage(coverage_risk(task) == "thin", st)
        if coverage["flagged"] or coverage["thin"]:
            note = ("REINFORCE (queen-flagged ∧ empty FIND)"
                    if coverage["needs_reinforcement"]
                    else "thin (unflagged)" if coverage["thin"]
                    else "flagged but FIND not empty (sufficient)")
            logger.info("   coverage: flagged=%s thin=%s → %s",
                        coverage["flagged"], coverage["thin"], note)

        # (B3) Capped scout reinforcement: ONLY for a needs_reinforcement axis (queen
        # flagged AND FIND empty) and ONLY when cfg.reinforce.enabled. A few quality
        # scouts (roles.scout) dig for evidence the blind grep missed; their cited files
        # are windowed LOCALLY (free) and merged into the bundle so the judge sees them.
        # Default off → this is a no-op and the un-reinforced path is unchanged.
        if coverage["needs_reinforcement"] and getattr(cfg.reinforce, "enabled", False):
            need = reinforce_thin_axis(task, sp, seed_text, code_root, cfg=cfg,
                                       budget=reinforce_budget, ledger=ledger,
                                       provider_kwargs=pk)
            if need is not None:
                fu = retrieve_followup(need, code_root, k=k)
                if fu and (fu.get("seeds") or fu.get("call_chain")):
                    bundle = {
                        **bundle,
                        "code_snippets": (list(bundle.get("code_snippets") or [])
                                          + list(fu.get("seeds") or [])),
                        "call_chain": (list(bundle.get("call_chain") or [])
                                       + list(fu.get("call_chain") or [])),
                    }
                    logger.info("   reinforce[%s]: merged +%d snippet(s), +%d call-chain",
                                sp.axis_id, len(fu.get("seeds") or []),
                                len(fu.get("call_chain") or []))

        votes = effective_votes
        logger.info("[HIVE_STAGE] pipeline=investigate stage=3 name=judge axis=%s",
                    sp.axis_id)
        logger.info("③ JUDGE [%s] (%s/%s, ≤%d calls × %d vote(s))", sp.axis_id,
                    judge_role.provider, judge_role.model,
                    per_vote_calls, votes)
        jr = run_judge_votes(
            votes=votes,
            plan_bundle=bundle, symptom=symptom, axis_globs=sp.file_globs,
            code_root=code_root, provider=judge_role.provider,
            model=judge_role.model, judge_cfg=cfg.judge, ledger=ledger,
            provider_kwargs=pk, k=k, seed_files=seed_files,
            seed_axis=(sp.axis_id == "SEED_ANCHOR"),
        )
        v = jr["verdict"]
        # Best-of-N tally (M010 §5): surface how the votes split across files so the
        # union feeding converge is auditable — not to RANK candidates by count (a
        # 1/N rare hit is kept on purpose; converge's causal gate decides, not votes).
        if jr.get("votes", 1) > 1:
            tally = jr.get("file_tally") or {}
            tally_s = ", ".join(f"{os.path.basename(f) or '∅'}×{c}"
                                for f, c in tally.items()) or "none located"
            logger.info("   votes: located %d/%d  candidates: %s",
                        jr.get("located_votes", 0), jr["votes"], tally_s)
        logger.info("   verdict: located=%s %s:%s — %s",
                    v.located, v.file, v.lines, v.reason)
        # Keep the merged bundle (first-pass + any follow-up) so converge sees the
        # same call-chain the judge did, not just the first-pass windows.
        fu = jr.get("followup_bundle")
        if fu:
            keep_bundle = {
                **bundle,
                "code_snippets": (list(bundle.get("code_snippets") or [])
                                  + list(fu.get("seeds") or [])),
                "call_chain": (list(bundle.get("call_chain") or [])
                               + list(fu.get("call_chain") or [])),
            }
        else:
            keep_bundle = bundle
        verdict_entry = {
            "axis_id": sp.axis_id,
            "title": task.get("title", ""),
            "search_plan": {"keywords": sp.keywords, "file_globs": sp.file_globs,
                            "doc_topics": sp.doc_topics},
            "calls_made": jr["calls_made"],
            "verdict": {"located": v.located, "file": v.file, "lines": v.lines,
                        "reason": v.reason},
            # Best-of-N union: every distinct located locus across the votes. The
            # representative ``verdict`` above is one of these (the most-voted file);
            # the full set is expanded into converge fragments so the causal gate —
            # not a vote count — decides which survive. At votes_per_axis=1 this is
            # exactly the one representative locus (or empty when unlocated).
            "votes": {"n": jr.get("votes", 1), "located": jr.get("located_votes", 0)},
            "candidates": [{"file": c.file, "lines": c.lines, "reason": c.reason}
                           for c in jr.get("candidates", [])],
            # (B2/#5) sufficiency tag: was this axis's evidence thin, and did the queen
            # flag it? Lets the honey/reaction tell a retrieval gap from a reasoning gap.
            "coverage": coverage,
        }
        return idx, verdict_entry, keep_bundle

    # Bounded concurrency: overlap the judge round-trips without unbounded fan-out
    # (and stay within the judge provider's rate budget). max_parallel=1 degrades to
    # a single-worker pool — i.e. today's sequential behaviour — with no code branch.
    max_workers = max(1, min(int(getattr(cfg.judge, "max_parallel", 2)), len(judged)))
    # ``bundles`` is kept parallel to ``verdicts`` (was discarded after judge) so the
    # ④ converge stage can pool the cross-axis call-chain hops it needs.
    slots: list[Any] = [None] * len(judged)
    # (B3) Shared per-run ceiling on reinforcement (scout) calls — drained across axes by
    # the thread pool. None when reinforcement is disabled, so the gate is a cheap
    # attribute check and no budget object is allocated on the common (off) path.
    reinforce_budget = (ReinforceBudget(cfg.reinforce.max_total_calls)
                        if getattr(cfg.reinforce, "enabled", False) else None)
    if judged:
        logger.info("②..③ judging %d axes (max_parallel=%d)", len(judged), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_investigate_axis, i, task, plans[i])
                       for i, task in enumerate(judged)]
            for fut in as_completed(futures):
                idx, verdict_entry, keep_bundle = fut.result()
                slots[idx] = (verdict_entry, keep_bundle)
    # Reassemble in the original judged order so the verdicts/bundles pairing (and
    # therefore converge's input) is identical to the sequential path.
    verdicts: list[dict[str, Any]] = [s[0] for s in slots if s is not None]
    bundles: list[dict[str, Any]] = [s[1] for s in slots if s is not None]

    # ── ④ converge (the reconcile step the cheap path was missing): stitch the
    # scattered per-axis verdicts into ONE executed call path and attribute the
    # defect to one node. ONE tool-OFF call, and only when ≥2 axes located (nothing
    # to stitch otherwise → free skip). This is what turns N169's "5 fragments on 5
    # files, 0 edits" into a single attributable target for specify.
    located_n = sum(1 for v in verdicts if v["verdict"]["located"])
    seed_kind = classify_seed_kind(seed_text)
    converge_dict: dict[str, Any] | None = None
    if located_n >= 2:
        conv_role = cfg.role("converge")
        # Read-only DB connection for this codebase (if configured) — lets converge
        # resolve an UNDECIDABLE causal check by reading the deciding row from the live
        # DB instead of guessing. None (the common case) → converge keeps its static
        # path. db_for_codebase never raises.
        db_conn = cfg.db_for_codebase(code_root)
        logger.info("[HIVE_STAGE] pipeline=investigate stage=4 name=converge")
        logger.info("④ converge (%s/%s) — stitch %d located verdict(s) into one path%s",
                    conv_role.provider, conv_role.model, located_n,
                    f" (DB data-state read available: {db_conn.kind})" if db_conn else "")
        # Feed converge the EXPANDED best-of-N union (one fragment per distinct
        # located locus), not the per-axis representatives — the causal gate vets
        # the wider candidate net. The ≥2 gate above stays on AXES located (not
        # fragment count), so a single noisy axis splitting into two files does not
        # by itself trigger converge.
        split_cfg = cfg.converge_split
        cres = run_converge(
            seed_text=seed_text, verdicts=_converge_fragments(verdicts),
            bundles=bundles,
            provider=conv_role.provider, model=conv_role.model, code_root=code_root,
            ledger=ledger, provider_kwargs=pk, k=k, max_hops=2, db_conn=db_conn,
            split_enabled=split_cfg.enabled, split_max_loci=split_cfg.max_loci,
            split_provider=split_cfg.provider, split_model=split_cfg.model)
        converge_dict = cres.as_dict()
        cc = cres.causal_check or {}
        if cres.converged and cres.attributed_defect:
            logger.info("   converged → defect at %s:%s (causal: consistent)",
                        cres.attributed_defect.get("file"),
                        cres.attributed_defect.get("lines"))
        elif cres.attributed_defect and cc.get("verdict") in (
                "contradicted", "undecidable", "unverified"):
            logger.info("   not converged → causal check %s for suspected %s:%s "
                        "(reachable, not a verified cause — routed to "
                        "reinvestigation/data-state)", cc.get("verdict"),
                        cres.attributed_defect.get("file"),
                        cres.attributed_defect.get("lines"))
        elif cres.missing_link:
            logger.info("   not converged → missing link: %s",
                        cres.missing_link.get("between"))
    else:
        logger.info("converge: skipped (%d located verdict(s) < 2 — nothing to stitch)",
                    located_n)

    result = {
        "seed_chars": len(seed_text),
        "axes_total": len(tasks),
        "axes_judged": len(verdicts),
        "max_axes": cfg.judge.max_axes,
        "max_calls_per_axis": cfg.judge.max_calls_per_axis,
        "seed_kind": seed_kind,
        "converge": converge_dict,
        "verdicts": verdicts,
    }
    _write_report(result, output_path)
    result["report_path"] = output_path
    return result


def _render_data_state_lines(converge: dict[str, Any]) -> list[str]:
    """Render the live-DB rows converge actually read into honey lines (or []).

    N173: when a read-only DB connection is configured and the converger's causal
    check depended on stored row state, the pipeline executes the read and the rows
    come back on ``converge["data_state_block"]``. We PASTE that block verbatim so the
    report is grounded on REAL data — the explicit antidote to converge fabricating a
    value (``result_doc_id = 'doc123'``) to reach a verdict. When a read was attempted
    but returned nothing/failed (``data_state_attempted`` set, not ``data_state_backed``)
    we say so HONESTLY rather than letting it look like a read never happened.
    """
    if not converge.get("data_state_attempted"):
        return []
    block = (converge.get("data_state_block") or "").strip()
    backed = converge.get("data_state_backed")
    if backed and block:
        return [
            "#### Live DB data confirmed (rows READ from the configured database — FACT, "
            "not assumed; the verdict above is data-backed)",
            "",
            "```",
            block,
            "```",
            "",
        ]
    # Attempted but no usable rows — be explicit so no assumed value fills the gap.
    return [
        "#### Live DB read attempted — NO usable rows returned",
        "",
        "A read-only DB connection is configured and the converger named the rows to "
        "read, but the read returned nothing / failed (see below). The verdict is NOT "
        "data-backed — do NOT substitute an assumed value; treat the data state as "
        "unconfirmed (needs a real fixture / runtime check).",
        "",
        "```",
        block or "(no query was executed)",
        "```",
        "",
    ]


def _render_converge_section(converge: dict[str, Any] | None,
                             seed_kind: str) -> list[str]:
    """Render the ④ converge result into honey lines (the START-HERE section).

    A successful convergence gives specify ONE stitched path + ONE attributed
    defect — replacing the "N scattered loci, you figure it out" framing that
    drove N169 to 0 edits. A failed convergence names the missing link so a
    needs_reinvestigation cites the specific hop rather than re-asking blind.
    Returns [] when converge was skipped (<2 located) so the honey is unchanged.
    """
    if not converge:
        return []
    out: list[str] = []
    winning = [node for node in (converge.get("winning_path") or [])
               if isinstance(node, dict) and node.get("file") and node.get("url")]
    if winning:
        out += [
            "## Winning HTTP request path (deterministic grounding)",
            "",
            "The following nodes were derived from the mounted router registration "
            "order and return-value call chain. Specify gates consume the structured "
            "markers below; edits/tests outside this path are off-path.",
            "",
        ]
        for node in winning:
            payload = json.dumps(node, ensure_ascii=True, sort_keys=True)
            out.append(f"<!-- hive-winning-http-path: {payload} -->")
            symbol = f" — {node.get('symbol')}" if node.get("symbol") else ""
            out.append(
                f"- {node.get('verb', 'GET')} `{node.get('url', '')}` "
                f"[{node.get('role', '?')}] "
                f"{node.get('file', '')}:{node.get('lines', '')}{symbol}"
            )
        out.append("")

    ad_any = converge.get("attributed_defect")
    cc_any = converge.get("causal_check") or {}
    if isinstance(ad_any, dict) and ad_any.get("file"):
        attribution = {
            "file": ad_any.get("file", ""),
            "lines": ad_any.get("lines", ""),
            "node": ad_any.get("node", ""),
            "converged": bool(converge.get("converged")),
            "causal_verdict": cc_any.get("verdict", ""),
        }
        out += [
            "<!-- hive-converge-attribution: "
            + json.dumps(attribution, ensure_ascii=True, sort_keys=True)
            + " -->",
            "",
        ]

    if converge.get("converged") and converge.get("attributed_defect"):
        ad = converge["attributed_defect"]
        extra = [d for d in (converge.get("additional_defects") or [])
                 if isinstance(d, dict) and (d.get("file") or d.get("node"))]
        if extra:
            # Multi-locus (N179): the scenario has SEVERAL independent defects. Do NOT
            # tell the author the other loci are mere context — they are SEPARATE edit
            # sites (rendered in full below + a machine list specify's coverage gate reads).
            out += [
                "## Converged — MULTIPLE INDEPENDENT defects (START HERE)",
                "",
                "The converge stage stitched the executed path and found this scenario "
                "is NOT one defect: it has SEVERAL independent failures, each in "
                "different code and each needing its OWN fix. Treat the primary node "
                "below as ONE target and the 'Additional independent defects' section as "
                "the OTHER edit sites this scenario requires — author (or explicitly "
                "defer with a reason) an edit for EACH. Shipping only the primary is a "
                "half-fix.",
                "",
            ]
        else:
            out += [
                "## Converged call path (the single executed path — START HERE)",
                "",
                "The independent axes below each located ONE fragment of what is really "
                "a SINGLE call path. The converge stage stitched them into the one path "
                "that actually executes for this scenario and attributed the defect to "
                "ONE node. **Convergence SUCCEEDED** — treat the node below as the "
                "primary target; the per-axis localisations further down are corroborating "
                "context for it, not separate edit sites.",
                "",
            ]
        path = converge.get("path") or []
        if path:
            out.append("Executed path:")
            for i, n in enumerate(path, 1):
                sym = f" — {n['symbol']}" if n.get("symbol") else ""
                loc = f"{n.get('file', '')}:{n.get('lines', '')}".strip(":")
                out.append(f"{i}. [{n.get('node', '?')}] {loc}{sym}")
            out.append("")
        flag = " (⚠ attributed file not in evidence — re-confirm it exists)" \
            if ad.get("ungrounded") else ""
        out += [
            "### Primary edit target — attributed defect",
            f"- location: {ad.get('file', '')}:{ad.get('lines', '')}{flag}",
            f"- node: {ad.get('node', '?')}",
            f"- why this is the defect: {ad.get('why', '')}",
            "",
        ]
        # Causal check (N170): the attribution passed the cause→symptom check.
        # Surface the data-state assumptions + trace so the author can confirm the
        # fix matches the verified failing condition (not just a reachable node).
        cc = converge.get("causal_check") or {}
        if cc.get("data_state_assumptions") or cc.get("trace"):
            out += ["#### Causal verification (cause→symptom — CONSISTENT)"]
            for a in cc.get("data_state_assumptions") or []:
                out.append(f"- assumes: {a}")
            if cc.get("trace"):
                out.append(f"- trace: {cc['trace']}")
            out.append("")
        # PASTE the real rows the read returned (N173) — the grounding for the verdict.
        out += _render_data_state_lines(converge)
        # Multi-locus (N179): render EACH additional independent defect as its own edit
        # target, then a machine-parseable list (primary + each extra) that specify's
        # converge-coverage gate reads back to refuse a ready spec covering only some.
        if extra:
            out += ["### Additional independent defects — each needs its OWN edit", ""]
            for d in extra:
                flag = " (⚠ attributed file not in evidence — re-confirm it exists)" \
                    if d.get("ungrounded") else ""
                loc = f"{d.get('file', '')}:{d.get('lines', '')}".strip(":")
                out.append(f"- {loc}{flag} [{d.get('node', '?')}] — {d.get('why', '')}")
            out.append("")
            out += [CONVERGE_TARGET_SECTION + " (author or explicitly defer EACH)", ""]
            out.append(f"- {ad.get('file', '')}:{ad.get('lines', '')}")
            for d in extra:
                out.append(f"- {d.get('file', '')}:{d.get('lines', '')}")
            out.append("")
        if seed_kind == "diagnostic":
            out += [
                "> **This seed is DIAGNOSTIC (trace/map), not a change request.** The "
                "converged call path above IS the deliverable. Report it as the "
                "completed investigation. Author an edit ONLY if the seed also names a "
                "concrete fix; otherwise returning 0 edits with this path as the "
                "finding is CORRECT — do NOT return needs_reinvestigation.",
                "",
            ]
        else:
            out += [
                "> Convergence succeeded and the executed path is established: author "
                "the MINIMAL edit at the attributed defect above (re-anchoring from "
                "live code per the contract). The path is no longer ambiguous, so do NOT "
                "loop back merely to re-locate it. BUT the `why this is the defect` above "
                "is converge's reasoning — CONFIRM the claimed mechanism against the live "
                "source before you anchor: if the live code does NOT actually exhibit it "
                "(the signature/argument is fine, the branch/value already reads as "
                "intended), the attribution is wrong — defer that edit (do not force one "
                "at an already-correct locus) and say why, rather than authoring a phantom "
                "fix.",
                "",
            ]
        return out

    cc = converge.get("causal_check") or {}
    cv = cc.get("verdict")
    ad = converge.get("attributed_defect")

    # ── Dropped-peer domain guard (N180): converge attributed a defect and certified its
    # cause→symptom check ``consistent`` — but ON A LIVE DB READ (a data-state fact) while
    # silently dropping a competing located hypothesis that names a RENDER/binding-layer
    # mechanism. A DB read proves rows EXIST, never that a render symptom is resolved, so
    # the verdict was certified on the wrong evidence domain and the dropped peer may be
    # the real cause. The converge guard demoted it to not-converged and stamped the peer;
    # render it as a re-examine target (NOT a ready edit) and route to reinvestigation.
    dp = cc.get("dropped_peer")
    if dp and ad:
        aloc = f"{ad.get('file', '')}:{ad.get('lines', '')}".strip(":")
        ploc = f"{dp.get('file', '')}:{dp.get('lines', '')}".strip(":")
        out += [
            "## Convergence certified on DATA but dropped a competing hypothesis — re-examine",
            "",
            f"The converge stage attributed the defect to `{aloc}` ({ad.get('node', '?')}) "
            f"and certified its cause→symptom check **consistent on a live DB read** (a "
            f"data-state fact). BUT a competing located hypothesis at `{ploc}` (axis "
            f"{dp.get('axis_id', '?')}) was left off the path and NEVER refuted. A DB read "
            f"can prove rows EXIST; it CANNOT prove a RENDER / binding-layer symptom is "
            f"resolved — so this verdict is certified on the WRONG evidence domain, and "
            f"`{ploc}` may be the real cause (e.g. a response key the front-end reads under "
            f"a different name, so adding rows / a UNION upstream leaves the screen empty). "
            f"**Do NOT author a ready edit at `{aloc}` on the strength of this convergence.**",
            "",
            f"- dropped (unrefuted) hypothesis: {ploc} — {dp.get('reason', '')}",
            "",
            "> Return **needs_reinvestigation** naming BOTH the data-certified node "
            f"`{aloc}` AND the unrefuted competing locus `{ploc}`. The re-stitch must "
            "either causally REFUTE the competing locus against LIVE CODE (trace the "
            "producer's emitted key → the consumer's read of it and show they AGREE), or "
            "attribute / author the fix THERE — not ship the data-only fix alone. If the "
            "seed names concrete edit targets (see below), author those.",
            "",
        ]
        out += _render_data_state_lines(converge)
        return out

    # ── Data-stamp gate (M017 lever 2): converge attributed a defect and ruled its
    # cause→symptom check ``consistent``, but FLAGGED it data_dependent (the ruling rests
    # on a stored row/field value) while NO live DB read backed it — it ruled on an
    # ASSUMED value. A read-only DB IS configured, so this is recoverable: route back to
    # name and READ the deciding row, then re-rule on fact. NOT a ready edit target.
    if cc.get("data_unstamped") and ad:
        loc = f"{ad.get('file', '')}:{ad.get('lines', '')}".strip(":")
        out += [
            "## Convergence ruled CONSISTENT on an UNREAD stored value — confirm with data",
            "",
            f"The converge stage attributed the defect to `{loc}` ({ad.get('node', '?')}) "
            f"and ruled its cause→symptom check **consistent**, but flagged the ruling as "
            f"**data-dependent** — its correctness rests on a STORED row/field value (which "
            f"row is selected, a status/id a field holds, whether a row exists) — while "
            f"**no live DB read backed it**. The verdict was ruled on an ASSUMED value, not "
            f"a confirmed one. A read-only database IS configured for this codebase, so the "
            f"deciding row CAN be read. **Do NOT author a ready edit at `{loc}` on the "
            f"strength of this unconfirmed convergence.**",
            "",
        ]
        for a in cc.get("data_state_assumptions") or []:
            out.append(f"- assumed (UNREAD) data state: {a}")
        if cc.get("trace"):
            out.append(f"- causal trace: {cc['trace']}")
        out.append("")
        out += _render_data_state_lines(converge)
        out += [
            "> Return **needs_reinvestigation** so the converge stage NAMES the exact rows "
            "that decide this verdict (table + row selector from the scenario key + the "
            "deciding column) and READS them from the live DB — then re-rules consistent or "
            "contradicted on the REAL values. Do NOT ship the fix on the assumed value, and "
            "do NOT defer it as needs_runtime: the data IS readable here. If the seed names "
            "concrete edit targets (see below), author those.",
            "",
        ]
        return out

    # ── Causal failure (N170): the converger reached a node on the executed path
    # but the cause→symptom check did NOT confirm it produces the symptom. This is
    # NOT a primary edit target — emitting it as one is exactly the N170 defect
    # (a reachable-but-innocent ORDER BY clause authored into a wrong edit). Route
    # it to reinvestigation (contradicted) or data-state confirmation (undecidable).
    if cv in ("contradicted", "undecidable", "unverified") and ad:
        loc = f"{ad.get('file', '')}:{ad.get('lines', '')}".strip(":")
        out += [
            "## Convergence reached a node but the CAUSAL CHECK did not confirm it",
            "",
            f"The converge stage stitched the executed path and SUSPECTED "
            f"`{loc}` ({ad.get('node', '?')}), but its cause→symptom check came back "
            f"**{cv}** — reachability alone, not a verified cause. **Do NOT author an "
            f"edit at this node on the strength of convergence.** It is a suspected "
            f"locus to re-examine, not an attributed defect.",
            "",
        ]
        for a in cc.get("data_state_assumptions") or []:
            out.append(f"- assumed data state: {a}")
        if cc.get("trace"):
            out.append(f"- causal trace: {cc['trace']}")
        out.append("")
        # If a live read ran, PASTE its rows (or honestly note it returned nothing) so a
        # contradiction/undecidable is grounded on real data, never on an assumed value.
        out += _render_data_state_lines(converge)
        if cv == "contradicted":
            out += [
                "> The suspected code is reachable but, under the only data state the "
                "scenario allows, it CANNOT produce the reported symptom (the cause "
                "contradicts the symptom). The real defect is elsewhere: return "
                "needs_reinvestigation NAMING this contradiction (which node was "
                "suspected and why it cannot be the cause), or — if the seed names "
                "concrete edit targets (see the seed-specified targets section "
                "below) — author only those, not this node.",
                "",
            ]
        else:  # undecidable / unverified
            needs = cc.get("need_data_state") or []
            out += [
                "> Whether this node is the defect depends on stored row/field state "
                "that static evidence cannot determine. Do NOT guess an edit: put the "
                "direction in deferred[] with reason=\"needs_runtime\", naming the "
                "exact row/field state required (e.g. \"which result_doc_id the target "
                "row carries, what review status it holds\"), and set "
                "termination=needs_runtime. Author only seed-named targets "
                "(see the seed-specified targets section below) if present.",
                "",
            ]
            if needs:
                out.append("Data state / fixture required to decide:")
                out += [f"- {n}" for n in needs]
                out.append("")
        return out

    ml = converge.get("missing_link")
    if ml:
        between = " ↔ ".join(ml.get("between") or []) or "two adjacent nodes"
        need = ml.get("need") or {}
        out += [
            "## Convergence incomplete — missing link",
            "",
            f"The converge stage could not connect **{between}** because the needed "
            "callee/symbol is not present in the retrieved evidence. The located "
            "fragments below are real but do not yet form one executed path.",
            "",
            f"- missing between: {between}",
            f"- need symbols: {', '.join(need.get('symbols') or []) or '(none)'}",
            f"- need greps: {', '.join(need.get('greps') or []) or '(none)'}",
            "",
            "> The missing link above means the call path cannot be traced from static "
            "evidence alone. Do NOT author an edit for either unconnected node — "
            "such an edit is ungrounded speculation on a path that has not been verified.",
            "> - If the link is a RUNTIME question (which code path actually executes "
            "for this request, which loader key is passed, which row is the active "
            "head): emit termination=needs_runtime with a deferred[] entry "
            "(reason: \"needs_runtime\") naming the exact runtime fact that would "
            "resolve this (e.g. \"which key get_pending passes to the handler\").",
            "> - If the missing symbol is likely statically resolvable from code: "
            "emit needs_reinvestigation naming this specific missing hop — not a "
            "blank re-investigation.",
            "> NEVER emit both a deferred[anchor_not_grounded] AND an edit for the "
            "same file.",
            "",
        ]
    return out


def _axis_coverage(flagged: bool, stats: dict[str, Any]) -> dict[str, Any]:
    """The (flag ∧ thin) sufficiency tag for one axis (B2/#5) — free, deterministic.

    ``flagged`` is the queen's coverage_risk self-doubt (searchplan.coverage_risk);
    ``thin`` is what the local FIND ACTUALLY retrieved for the axis. The pipeline only
    treats an axis as starved when BOTH hold (queen unsure AND evidence empty) — the
    cheap, confirmable signal that a thin envelope existed BEFORE specify pays for it.
    ``thin`` alone (no flag) or a flag on a well-retrieved axis are both left untouched.

    The tag rides the verdict downstream so the reaction side can separate a RETRIEVAL
    gap (thin → re-fetch may help) from a REASONING gap (sufficient → honest defer, no
    point re-fetching): #5 closes "did the queen give enough?" with data, not a guess.
    """
    gv = stats.get("glob_validation", {}) or {}
    all_globs_empty = bool(gv.get("dropped_empty")) and not gv.get("kept")
    thin = (stats.get("snippets", 0) == 0
            or stats.get("raw_hits", 0) == 0
            or all_globs_empty)
    return {
        "flagged": bool(flagged),
        "thin": bool(thin),
        "sufficient": not thin,
        "needs_reinforcement": bool(flagged) and bool(thin),
        "raw_hits": stats.get("raw_hits", 0),
        "snippets": stats.get("snippets", 0),
    }


def _coverage_phrase(coverage: dict[str, Any] | None) -> str:
    """One honey-ready clause describing an axis's evidence sufficiency (C/#5), or "".

    Renders the verdict's coverage tag into prose the specify author (and a future
    reaction pass) reads to decide WHY an axis is unlocated: a retrieval gap (the FIND
    came back empty → a wider re-retrieve may recover it) versus a reasoning gap (the
    evidence WAS retrieved → re-fetching buys nothing, defer honestly). Empty when no
    tag is present (older verdicts) so the honey is unchanged for them.
    """
    if not isinstance(coverage, dict):
        return ""
    if coverage.get("needs_reinforcement"):
        return (" — evidence THIN (queen flagged this axis AND the local FIND came back "
                "empty): a wider re-retrieve may recover it before deferring")
    if coverage.get("thin"):
        return (" — evidence thin (the local FIND came back empty): a wider re-retrieve "
                "may recover it")
    return (" — evidence sufficient (the FIND retrieved code here): retrieval was NOT the "
            "gap, so defer honestly — re-fetching the same scope will not help")


def render_local_honey(result: dict[str, Any], seed_text: str,
                       code_root: str | None = None,
                       docs_root: str | None = None) -> str:
    """Render investigate verdicts into a honey-shaped markdown — LOCAL, free.

    This is the seam that lets the cheap path feed ``specify``: the swarm pipeline
    pays an ``assemble`` model call to synthesise a honey, but ``specify`` only
    consumes the honey as free prose — it re-anchors against LIVE code and trusts
    nothing the honey quotes (``recipes/edit_spec_contract_v1.md`` cardinal rule).
    So the honey just has to carry two things the verdicts already hold: the
    REQUESTED CHANGE (the seed) and the GROUNDED LOCATIONS (judge file:lines +
    reason). We template those deterministically — no model, no ``assemble`` call.

    Located verdicts become grounded LOCALISATIONS (evidence), not per-axis edit
    imperatives; unlocated/downgraded ones are listed as "no confident
    localisation" so the specify author neither fabricates an edit there nor
    silently drops the axis.

    Why localisations are framed as evidence, not as one fix-direction each
    (T891): the fan-out axes investigate the SAME requested change from different
    angles. When they locate DIFFERENT loci, an earlier rendering printed "apply
    the requested change above at this location" under EVERY axis — turning
    corroborating localisations into N competing edit imperatives. The author then
    followed a localisation that contradicted the seed's tightly-scoped directive
    (anchored a v-for :class instead of the named placeholder div). So the seed's
    stated scope is made BINDING and given precedence over any single localisation,
    and same-file loci are grouped so convergence is visible without fabricating.
    """
    verdicts = result.get("verdicts", []) or []
    located = [v for v in verdicts if v.get("verdict", {}).get("located")]
    unlocated = [v for v in verdicts if not v.get("verdict", {}).get("located")]
    # The seed's OWN explicitly-named edit targets, grounded independently of the
    # judge's verdicts (Defect 2): a user-provided file:line must become an AUTHOR
    # target even when investigate failed to re-locate it on its own.
    seed_targets = seed_edit_targets(seed_text, code_root, docs_root)

    seed_kind = result.get("seed_kind", "fix")
    converge = result.get("converge")
    out: list[str] = [
        "# Hivework honey (local — rendered from investigate verdicts)",
        "",
        "- source: cheap path (decompose → retrieve(local) → judge → converge), no assemble call",
        f"- axes judged: {result.get('axes_judged', 0)}/{result.get('axes_total', 0)}; "
        f"located: {len(located)}; seed-kind: {seed_kind}",
        "",
        "## Requested change / reported symptom",
        "",
        seed_text.strip(),
        "",
    ]
    # ④ converge section (START HERE): the stitched single path + attributed defect,
    # or the named missing link. Empty when converge was skipped (<2 located).
    out += _render_converge_section(converge, seed_kind)
    out += [
        "## Grounded localisations (investigation evidence — NOT a list of edit sites)",
        "",
        "Independent investigation axes located the code below relevant to the "
        "Requested change above. They TRIANGULATE the relevant code — each is "
        "EVIDENCE, not an instruction to edit at that line. Author the MINIMAL "
        "edit(s) that satisfy the Requested change, treating its stated scope as "
        "BINDING:",
        "",
        "- When the Requested change names a specific element / anchor / file to "
        "change — or names something NOT to touch — that scope OVERRIDES any "
        "localisation below that points elsewhere: a conflicting localisation is "
        "context, not a target.",
        "- Several axes may converge on ONE locus (a strong signal) or land on "
        "DIFFERENT loci (they cover different angles — most are corroborating "
        "context, not all are edit sites). Do NOT author one edit per localisation.",
        "- Per the edit-spec contract, RE-OPEN each file and lift `anchor_old` from "
        "the CURRENT text byte-for-byte — the line ranges are the judge's grounding, "
        "not authoritative anchors.",
        "",
    ]
    if located:
        # Group by file so same-file loci sit together and convergence is visible.
        by_file: dict[str, list[dict[str, Any]]] = {}
        for v in located:
            by_file.setdefault(v.get("verdict", {}).get("file", ""), []).append(v)
        converged = [f for f, vs in by_file.items() if f and len(vs) > 1]
        if converged:
            out += ["### Convergence (≥2 axes on the same file — a stronger prior)"]
            for f in converged:
                axes = ", ".join(v.get("axis_id", "?") for v in by_file[f])
                loci = "; ".join(v.get("verdict", {}).get("lines", "") for v in by_file[f])
                out.append(f"- {f}: axes [{axes}] at lines {loci} "
                           "(confirm which locus the Requested change's scope names)")
            out.append("")
        for f, vs in by_file.items():
            for v in vs:
                vd = v.get("verdict", {})
                out += [
                    f"### {v.get('axis_id', '?')} — {v.get('title', '')}".rstrip(" —"),
                    f"- location: {vd.get('file', '')}:{vd.get('lines', '')}",
                    f"- why relevant: {vd.get('reason', '')}",
                    "",
                ]
    else:
        out += ["_No axis produced a grounded localisation. specify should defer "
                "rather than fabricate an edit._", ""]

    # Seed-specified edit targets — the user named these exact files, so they are
    # AUTHOR targets (not "context", not "do NOT edit"), grounded below regardless
    # of whether any judge axis located them (Defect 2 / T892). specify lifts the
    # live text at each cited file:line and enforces that none is silently dropped.
    if seed_targets:
        out += [
            SEED_TARGET_SECTION + " (the user named these files explicitly — AUTHOR them)",
            "",
            "The Requested change names these exact files as edit targets. They are "
            "NOT optional and NOT mere context: author the seed's specified change at "
            "each, lifting `anchor_old` from the live text in the \"Anchor ground "
            "truth\" block below. If you genuinely cannot express one as an edit, you "
            "MUST defer it with a reason that NAMES the file and states exactly what "
            "grounding was missing — never drop a seed-named target silently.",
            "",
        ]
        out += [f"- {t['file']}:{t['lines']}" for t in seed_targets]
        out.append("")

    if unlocated:
        out += ["## Axes without a confident localisation (do NOT fabricate edits here)", ""]
        if seed_targets:
            out.append("(Files under “Seed-specified edit targets” above remain AUTHOR "
                       "targets — the prohibition here applies only to these speculative "
                       "axis loci, not to a seed-named file.)")
            out.append("")
        if any(v.get("coverage") for v in unlocated):
            out.append("Each axis below carries an EVIDENCE note (C/#5): \"thin\" means the "
                       "local FIND retrieved nothing, so a wider re-retrieve may recover it; "
                       "\"sufficient\" means the code WAS retrieved and the axis still did not "
                       "localise — a reasoning gap, so re-fetching the same scope will not "
                       "help and an honest defer is correct.")
            out.append("")
        for v in unlocated:
            vd = v.get("verdict", {})
            reason = vd.get("reason") or "not located"
            out.append(f"- {v.get('axis_id', '?')} — {v.get('title', '')}: {reason}"
                       f"{_coverage_phrase(v.get('coverage'))}")
        out.append("")

    return "\n".join(out)


def _rebuild_bundles(verdicts: list[dict[str, Any]], code_root: str | None,
                     docs_root: str | None) -> list[dict[str, Any]]:
    """Re-derive a retrieve bundle per verdict from its stored ``search_plan`` — LOCAL,
    free. Lets a re-run feed converge the SAME evidence shape without threading the
    original (large) bundles through the orchestrator."""
    bundles: list[dict[str, Any]] = []
    for v in verdicts:
        spd = v.get("search_plan") or {}
        sp = SearchPlan(axis_id=str(v.get("axis_id") or "?"),
                        keywords=list(spd.get("keywords") or []),
                        file_globs=list(spd.get("file_globs") or []),
                        doc_topics=list(spd.get("doc_topics") or []))
        bundles.append(retrieve(sp, code_root or ".", docs_root))
    return bundles


def _norm_locus(p: str) -> str:
    return (p or "").replace("\\", "/").strip().strip("/").lower()


def _locus_aligns(a: str, b: str) -> bool:
    """Path-segment-aligned equality/suffix match (handles abs↔rel, basename-degrade)."""
    a, b = _norm_locus(a), _norm_locus(b)
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def _rerun_converge(result: dict[str, Any], seed_text: str, *, code_root: str | None,
                    docs_root: str | None, cfg, ledger, provider_kwargs,
                    honey_out: str | None,
                    exclude_loci: list[dict[str, Any]] | None = None
                    ) -> dict[str, Any] | None:
    """Reaction #3 (ineffective/inconclusive): re-stitch on LIVE-re-grounded evidence.

    Re-derives each axis's bundle from the current source (free local retrieve) and re-runs
    ``converge`` so its causal gate rules on TODAY's code, not the first pass's compacted
    snippets (the N177 live-grounding lever, applied at the reaction edge). Updates
    ``result['converge']``, re-renders the local honey to ``honey_out``, and returns the
    updated result — or ``None`` when there is nothing to re-stitch (<2 located).

    ``exclude_loci`` (M035 ⑥→④): loci specify refuted against live code. They are DROPPED
    from the verdicts before re-converging so the re-stitch cannot re-crown the disproven
    node and must attribute to a surviving candidate (the FE render locus). The drop is by
    path-aligned file match; a verdict on any other file is kept."""
    verdicts = list(result.get("verdicts") or [])
    if exclude_loci:
        ex_files = [_norm_locus(x.get("file", "")) for x in exclude_loci if x.get("file")]
        kept = [v for v in verdicts
                if not any(_locus_aligns((v.get("verdict") or {}).get("file", ""), ef)
                           for ef in ex_files)]
        dropped = len(verdicts) - len(kept)
        if dropped:
            logger.info("reinvestigation live: re-converge EXCLUDING %d refuted locus/loci "
                        "%s (specify ⑥→④ feedback)", dropped,
                        [x.get("file") for x in exclude_loci])
            verdicts = kept
    located = [v for v in verdicts if (v.get("verdict") or {}).get("located")]
    if len(located) < 2:
        logger.info("reinvestigation live: <2 located verdict(s) on re-run — "
                    "nothing to re-stitch (honest NR stands)")
        return None
    conv_role = cfg.role("converge")
    db_conn = cfg.db_for_codebase(code_root)
    bundles = _rebuild_bundles(verdicts, code_root, docs_root)
    logger.info("reinvestigation live: re-converge %d located verdict(s) on re-grounded "
                "evidence (%s/%s)", len(located), conv_role.provider, conv_role.model)
    pk = dict(provider_kwargs or {})
    cres = run_converge(
        seed_text=seed_text, verdicts=_converge_fragments(verdicts), bundles=bundles,
        provider=conv_role.provider, model=conv_role.model, code_root=code_root,
        ledger=ledger, provider_kwargs=pk, k=6, max_hops=2, db_conn=db_conn)
    result["converge"] = cres.as_dict()
    if honey_out:
        with open(honey_out, "w", encoding="utf-8") as f:
            f.write(render_local_honey(result, seed_text, code_root, docs_root))
        logger.info("reinvestigation live: re-rendered honey → %s", honey_out)
    return result


def rerun_reinvestigation(plan, result: dict[str, Any], *, seed_text: str,
                          code_root: str | None, docs_root: str | None, cfg,
                          ledger=None, provider_kwargs: dict | None = None,
                          honey_out: str | None = None) -> dict[str, Any] | None:
    """Execute ONE cheap re-run for a routed NR plan — gated, bounded (M013 reaction #3).

    Gated by ``cfg.reinvestigation.live`` (default off): when off this is a no-op and the
    plan stands as an honest NR (already logged for free). When on, it performs the single
    re-entry the routing brain chose and returns an updated ``result`` (honey re-rendered
    to ``honey_out``) for the caller to re-run specify ONCE — or ``None`` (no re-run, do not
    re-spend on specify).

    ``re_converge`` is wired (live re-grounding + re-stitch). ``re_retrieve`` re-judge is
    intentionally NOT fired here: it needs the per-axis judge core extracted into a reusable
    entry, and re-rendering the honey WITHOUT changed verdicts would spend a specify call for
    nothing. So re_retrieve logs that the plan stands and returns None — no wasted spend.
    """
    if not getattr(cfg.reinvestigation, "live", False):
        return None
    # Compare on the stable action strings (avoid importing reinvestigate → no cycle risk).
    if plan.action == "re_converge":
        return _rerun_converge(result, seed_text, code_root=code_root, docs_root=docs_root,
                               cfg=cfg, ledger=ledger, provider_kwargs=provider_kwargs,
                               honey_out=honey_out,
                               exclude_loci=getattr(plan, "exclude_loci", None))
    if plan.action == "re_retrieve":
        logger.info("reinvestigation live: re_retrieve re-judge not yet wired (needs the "
                    "per-axis judge core extracted) — plan stands on axes %s, no spend",
                    plan.axis_ids)
        return None
    return None


def _write_report(result: dict[str, Any], output_path: str) -> None:
    """Write the verdict report as JSON, plus a sibling markdown summary table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_path = os.path.splitext(output_path)[0] + ".md"
    lines = [
        "# Investigation verdicts (cheap path: decompose → retrieve → judge)",
        "",
        f"- axes total: {result['axes_total']}, judged: {result['axes_judged']} "
        f"(cap {result['max_axes']})",
        "",
        "| axis | located | file:lines | calls | reason |",
        "|---|---|---|---|---|",
    ]
    for v in result["verdicts"]:
        vd = v["verdict"]
        loc = "✅" if vd["located"] else "—"
        where = f"{vd['file']}:{vd['lines']}" if vd["file"] else ""
        reason = (vd["reason"] or "").replace("|", "\\|")[:80]
        lines.append(f"| {v['axis_id']} | {loc} | {where} | {v['calls_made']} | {reason} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
