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
    logger.info("decompose → %d axes (%d leaf, judging %d, cap max_axes=%d)",
                len(tasks), len(leaves), len(judged), cfg.judge.max_axes)

    # ── ②..③ per axis: bridge → local retrieve (free) → JUDGE (budgeted).
    verdicts: list[dict[str, Any]] = []
    for task in judged:
        sp = task_to_searchplan(task, default_globs=default_globs)
        symptom = str(task.get("brief") or task.get("title") or sp.axis_id)
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

    Located verdicts become fix-direction sections; unlocated/downgraded ones are
    listed as "no confident localisation" so the specify author neither fabricates
    an edit there nor silently drops the axis.
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
        "## Fix directions (grounded localisations)",
        "",
        "Each section is a judge-confirmed location for the requested change. "
        "Per the edit-spec contract, RE-OPEN each file and lift `anchor_old` from "
        "the CURRENT text byte-for-byte — the line ranges below are the judge's "
        "grounding, not authoritative anchors.",
        "",
    ]
    if located:
        for v in located:
            vd = v.get("verdict", {})
            out += [
                f"### {v.get('axis_id', '?')} — {v.get('title', '')}".rstrip(" —"),
                f"- target: {vd.get('file', '')}:{vd.get('lines', '')}",
                f"- grounding / reason: {vd.get('reason', '')}",
                "- fix direction: apply the requested change above at this location "
                "(author the concrete edit/new content per the contract).",
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
