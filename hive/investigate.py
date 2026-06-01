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
from typing import Any

from hive.decompose import run_decompose
from hive.judge import run_judge
from hive.retriever import retrieve
from hive.searchplan import task_to_searchplan

logger = logging.getLogger("hive.investigate")


def _leaf_axes(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Axes with no dependencies — the parallel evidence-collection leaves.

    Dependent axes (synthesis/decision/verification) consume other axes' outputs;
    they are not bug-localisation targets, so they get no JUDGE call.
    """
    return [t for t in tasks if not (t.get("depends_on") or [])]


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
    judged = leaves[: cfg.judge.max_axes]
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

    # ── ②..③ per axis: bridge → local retrieve (free) → JUDGE (budgeted).
    verdicts: list[dict[str, Any]] = []
    warned_no_docs = False
    for task in judged:
        sp = task_to_searchplan(task, default_globs=default_globs)
        symptom = str(task.get("brief") or task.get("title") or sp.axis_id)
        # docs=(none) confound (N165): the queen produced doc_topics for an axis
        # but no docs tree was supplied, so the entire design-doc channel is
        # silently skipped and any doc-targeting glob degrades into a code-tree
        # search. Surface it once at WARNING — a missing --docs is an invocation
        # bug, not a localisation result.
        if docs_root is None and sp.doc_topics and not warned_no_docs:
            logger.warning(
                "docs_root not supplied (--docs) but axes carry doc_topics "
                "(first: [%s] topics=%s): the design-doc channel is DISABLED and "
                "doc-targeted globs fall back to the code tree. Pass --docs <dir> "
                "to enable design retrieval.", sp.axis_id, sp.doc_topics)
            warned_no_docs = True
        logger.info("[HIVE_STAGE] pipeline=investigate stage=2 name=retrieve axis=%s",
                    sp.axis_id)
        logger.info("② retrieve [%s] keywords=%d globs=%d (local, free)",
                    sp.axis_id, len(sp.keywords), len(sp.file_globs))
        bundle = retrieve(sp, code_root, docs_root, k=k,
                          top_files=top_files, blame_files=blame_files)
        st = bundle.get("stats", {})
        logger.info("   FIND: %s hits → %s snippets, %s call-chain",
                    st.get("raw_hits"), st.get("snippets"), st.get("call_chain"))
        gv = st.get("glob_validation", {})
        if gv.get("dropped_empty") or gv.get("dropped_overbroad"):
            logger.info("   glob-guard: kept=%s dropped_empty=%s dropped_overbroad=%s",
                        gv.get("kept"), gv.get("dropped_empty"),
                        gv.get("dropped_overbroad"))

        logger.info("[HIVE_STAGE] pipeline=investigate stage=3 name=judge axis=%s",
                    sp.axis_id)
        logger.info("③ JUDGE [%s] (%s/%s, ≤%d calls)", sp.axis_id,
                    judge_role.provider, judge_role.model,
                    cfg.judge.max_calls_per_axis)
        jr = run_judge(
            plan_bundle=bundle, symptom=symptom, axis_globs=sp.file_globs,
            code_root=code_root, provider=judge_role.provider,
            model=judge_role.model, judge_cfg=cfg.judge, ledger=ledger,
            provider_kwargs=pk, k=k,
        )
        v = jr["verdict"]
        logger.info("   verdict: located=%s %s:%s — %s",
                    v.located, v.file, v.lines, v.reason)
        verdicts.append({
            "axis_id": sp.axis_id,
            "title": task.get("title", ""),
            "search_plan": {"keywords": sp.keywords, "file_globs": sp.file_globs,
                            "doc_topics": sp.doc_topics},
            "calls_made": jr["calls_made"],
            "verdict": {"located": v.located, "file": v.file, "lines": v.lines,
                        "reason": v.reason},
        })

    result = {
        "seed_chars": len(seed_text),
        "axes_total": len(tasks),
        "axes_judged": len(verdicts),
        "max_axes": cfg.judge.max_axes,
        "max_calls_per_axis": cfg.judge.max_calls_per_axis,
        "verdicts": verdicts,
    }
    _write_report(result, output_path)
    result["report_path"] = output_path
    return result


def render_local_honey(result: dict[str, Any], seed_text: str) -> str:
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

    out: list[str] = [
        "# Hivework honey (local — rendered from investigate verdicts)",
        "",
        "- source: cheap path (decompose → retrieve(local) → judge), no assemble call",
        f"- axes judged: {result.get('axes_judged', 0)}/{result.get('axes_total', 0)}; "
        f"located: {len(located)}",
        "",
        "## Requested change / reported symptom",
        "",
        seed_text.strip(),
        "",
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

    if unlocated:
        out += ["## Axes without a confident localisation (do NOT fabricate edits here)", ""]
        for v in unlocated:
            vd = v.get("verdict", {})
            reason = vd.get("reason") or "not located"
            out.append(f"- {v.get('axis_id', '?')} — {v.get('title', '')}: {reason}")
        out.append("")

    return "\n".join(out)


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
