"""Apply stage — renders an edit-spec into a human-facing proposal (propose only).

Pipeline position (fix-extension):

  investigate (fan-out)  →  merge (honey)  →  specify (edit-spec)  →  APPLY (this)

apply is the third and final box of the fix loop. Unlike specify it calls NO
worker: it is a pure, deterministic local pass over the edit-spec JSON (the SSOT
authored by specify) and the LIVE codebase. It re-opens each target file, checks
that the spec's ``anchor_old`` still exists there exactly once, renders the
``anchor_old → replacement_new`` change as a unified diff, and emits a proposal.

Key invariants (mirrored from recipes/edit_spec_contract_v1.md):
  - Stage-1 is PROPOSE ONLY. apply NEVER writes to the target codebase. The PM
    applies; auto-apply behind ``gate.apply`` is a later promotion, not now.
  - The JSON edit-spec is the SSOT; the unified diff is a DERIVED view rendered
    here from ``anchor_old``/``replacement_new`` — apply does not invent edits.
  - Anchors are re-verified against LIVE code at apply time. Even an edit specify
    marked ``verified`` can have drifted (live code changed since specify ran), so
    apply re-locates the anchor and refuses to present a drifted edit as ready.

An edit is "applicable" only when its ``anchor_old`` is found EXACTLY ONCE in the
current file. Zero matches (missing / already-applied) or multiple matches
(ambiguous) make the edit non-applicable and drop the overall verdict to
not-ready — the same conservative posture specify takes with stale anchors.
"""

import difflib
import json
import logging
import os
from typing import Any

logger = logging.getLogger("hive.apply")

_VALID_TERMINATION = {"ready_to_apply", "needs_reinvestigation", "needs_pm"}

# Per-edit applicability statuses (only "applicable" can contribute to a ready verdict).
APPLICABLE = "applicable"
ANCHOR_MISSING = "anchor_missing"
ANCHOR_AMBIGUOUS = "anchor_ambiguous"
ALREADY_APPLIED = "already_applied"
FILE_MISSING = "file_missing"
NO_CHANGE = "no_change"


def load_spec(spec_path: str) -> dict[str, Any]:
    """Load the edit-spec JSON (the SSOT produced by specify)."""
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_unified_diff(rel_path: str, original: str, modified: str) -> str:
    """Render a git-style unified diff from ``original`` to ``modified``.

    Labels use ``a/<rel_path>`` and ``b/<rel_path>`` so the output reads like a
    real patch. This is the DERIVED human view; the spec's text fields are SSOT.
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=3,
    )
    return "".join(diff)


def evaluate_edit(edit: dict[str, Any], codebase_root: str) -> dict[str, Any]:
    """Re-verify one edit against LIVE code and render its diff.

    Returns a result dict describing applicability. The edit is ``applicable``
    only when ``anchor_old`` occurs exactly once in the current file; the diff is
    rendered for that single, unambiguous replacement.
    """
    edit_id = edit.get("id", "?")
    rel_path = edit.get("file", "")
    anchor_old = edit.get("anchor_old", "")
    replacement_new = edit.get("replacement_new", "")

    result: dict[str, Any] = {
        "id": edit_id,
        "file": rel_path,
        "confidence": edit.get("confidence", "?"),
        "anchor_status_claimed": edit.get("anchor_status", "?"),
        "status": ANCHOR_MISSING,
        "applicable": False,
        "diff": "",
        "messages": [],
    }

    if not rel_path:
        result["status"] = FILE_MISSING
        result["messages"].append("edit has no 'file' field")
        return result

    abs_path = os.path.join(codebase_root, rel_path)
    if not os.path.isfile(abs_path):
        result["status"] = FILE_MISSING
        result["messages"].append(f"file not found under codebase root: {rel_path}")
        return result

    with open(abs_path, "r", encoding="utf-8") as f:
        file_text = f.read()

    if anchor_old == replacement_new:
        result["status"] = NO_CHANGE
        result["messages"].append("anchor_old equals replacement_new — no-op edit")
        return result

    occurrences = file_text.count(anchor_old) if anchor_old else 0

    if occurrences == 1:
        modified = file_text.replace(anchor_old, replacement_new, 1)
        result["status"] = APPLICABLE
        result["applicable"] = True
        result["diff"] = render_unified_diff(rel_path, file_text, modified)
        if str(edit.get("anchor_status", "")).lower() != "verified":
            # Live matches, but specify did not mark it verified — surface, don't block.
            result["messages"].append(
                f"anchor located in live code but spec anchor_status="
                f"{edit.get('anchor_status')!r}")
        return result

    if occurrences > 1:
        result["status"] = ANCHOR_AMBIGUOUS
        result["messages"].append(
            f"anchor_old occurs {occurrences}x in {rel_path} — not unique, "
            "cannot target a single edit")
        return result

    # Zero occurrences. Distinguish "already applied" from "drifted/missing".
    if replacement_new and file_text.count(replacement_new) >= 1:
        result["status"] = ALREADY_APPLIED
        result["messages"].append(
            "anchor_old absent but replacement_new already present — edit looks "
            "already applied")
    else:
        result["status"] = ANCHOR_MISSING
        msg = f"anchor_old not found in live {rel_path}"
        if str(edit.get("anchor_status", "")).lower() == "verified":
            msg += " — DRIFT: spec marked it 'verified' but live code has changed"
        result["messages"].append(msg)
    return result


def build_proposal(spec: dict[str, Any], codebase_root: str) -> dict[str, Any]:
    """Evaluate every edit against live code and decide the overall verdict.

    The proposal is ``ready`` only when ALL of these hold:
      - termination == "ready_to_apply"
      - there is at least one edit
      - every edit is APPLICABLE (anchor unique in live code right now)

    Anything else (deferred-only specs, drifted anchors, ambiguous anchors,
    non-ready termination) yields ``ready = False`` with explicit reasons.
    """
    termination = spec.get("termination")
    edits = spec.get("edits") if isinstance(spec.get("edits"), list) else []
    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []
    gate = spec.get("gate") if isinstance(spec.get("gate"), dict) else {}

    edit_results = [evaluate_edit(e, codebase_root) for e in edits if isinstance(e, dict)]
    n_applicable = sum(1 for r in edit_results if r["applicable"])

    reasons: list[str] = []
    if termination != "ready_to_apply":
        reasons.append(f"termination is {termination!r}, not 'ready_to_apply'")
    if not edit_results:
        reasons.append("spec contains no edits (nothing to apply)")
    for r in edit_results:
        if not r["applicable"]:
            reasons.append(f"{r['id']} ({r['file']}): {r['status']}")

    # Stage-1 safety net: gate.apply must never drive an actual write here. apply
    # only ever proposes, but we still flag a spec that tried to flip the switch.
    if gate.get("apply") is True:
        reasons.append("gate.apply was True — ignored (Stage-1 is propose-only)")

    ready = (
        termination == "ready_to_apply"
        and bool(edit_results)
        and all(r["applicable"] for r in edit_results)
    )

    return {
        "ready": ready,
        "not_ready_reasons": reasons,
        "codebase_root": os.path.abspath(codebase_root),
        "source_honey": spec.get("source_honey", ""),
        "termination": termination,
        "edits": edit_results,
        "deferred": deferred,
        "gate_commands": list(gate.get("commands") or []),
        "n_edits": len(edit_results),
        "n_applicable": n_applicable,
    }


def render_proposal_markdown(proposal: dict[str, Any]) -> str:
    """Render the human-facing apply proposal (diffs + gate + verdict)."""
    lines: list[str] = []
    lines.append("# Apply proposal — Stage-1 (propose only; nothing was written)")
    lines.append("")
    lines.append(f"- codebase_root: `{proposal['codebase_root']}`")
    if proposal.get("source_honey"):
        lines.append(f"- source_honey: `{proposal['source_honey']}`")
    lines.append(f"- spec termination: `{proposal['termination']}`")
    lines.append(f"- edits: {proposal['n_edits']} "
                 f"({proposal['n_applicable']} applicable)")
    lines.append(f"- deferred: {len(proposal['deferred'])}")
    lines.append("")

    if proposal["ready"]:
        lines.append("## ✅ VERDICT: READY TO APPLY")
        lines.append("")
        lines.append("Every edit's anchor was re-verified unique in live code. "
                     "Review the diffs below, then apply by hand and run the gate.")
    else:
        lines.append("## ⛔ VERDICT: NOT READY")
        lines.append("")
        lines.append("Do not apply. Reasons:")
        for reason in proposal["not_ready_reasons"]:
            lines.append(f"- {reason}")
    lines.append("")

    if proposal["gate_commands"]:
        lines.append("## Gate — run these AFTER you apply the edits")
        lines.append("")
        for cmd in proposal["gate_commands"]:
            lines.append(f"- `{cmd}`")
        lines.append("")

    lines.append("## Edits")
    lines.append("")
    if not proposal["edits"]:
        lines.append("_(none)_")
        lines.append("")
    for r in proposal["edits"]:
        flag = "✔" if r["applicable"] else "✗"
        lines.append(f"### {flag} {r['id']} — `{r['file']}`")
        lines.append(f"- status: `{r['status']}` | confidence: "
                     f"`{r['confidence']}` | anchor_status (spec): "
                     f"`{r['anchor_status_claimed']}`")
        for msg in r["messages"]:
            lines.append(f"- ⚠ {msg}")
        lines.append("")
        if r["diff"]:
            lines.append("```diff")
            lines.append(r["diff"].rstrip("\n"))
            lines.append("```")
            lines.append("")

    if proposal["deferred"]:
        lines.append("## Deferred (stay as investigation/surface — not edits)")
        lines.append("")
        for d in proposal["deferred"]:
            if isinstance(d, dict):
                lines.append(f"- **{d.get('reason', '?')}**: "
                             f"{d.get('issue', '')} "
                             f"(stays_as: {d.get('stays_as', '?')})")
            else:
                lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)


def run_apply(
    spec_path: str,
    codebase_root: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run the apply stage: edit-spec JSON + live code → proposal (propose only).

    Loads the SSOT edit-spec, re-verifies each anchor against the live codebase,
    renders unified diffs, and writes a human-facing proposal markdown. Returns
    the proposal dict. NEVER writes to the target codebase.

    Args:
        spec_path: Path to the edit-spec JSON produced by specify.
        codebase_root: Live codebase root. Defaults to the spec's ``codebase_root``.
        output_path: Where to write the proposal markdown. If None, nothing is
            written to disk and the caller uses the returned dict.

    Raises:
        FileNotFoundError: if the spec file is missing.
        ValueError: if codebase_root cannot be resolved (not given and absent
            from the spec).
    """
    spec = load_spec(spec_path)

    root = codebase_root or spec.get("codebase_root")
    if not root:
        raise ValueError(
            "codebase_root not provided and not present in the edit-spec; "
            "cannot re-verify anchors against live code")
    if not os.path.isdir(root):
        logger.warning("apply: codebase_root %r is not a directory — anchor "
                       "re-verification will fail for every edit", root)

    term = spec.get("termination")
    if term is not None and term not in _VALID_TERMINATION:
        logger.warning("apply: spec termination is invalid: %r", term)

    proposal = build_proposal(spec, root)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(render_proposal_markdown(proposal))
        logger.info("Apply proposal written to %s", output_path)

    verdict = "READY" if proposal["ready"] else "NOT READY"
    logger.info("Apply: %s — %d/%d edits applicable, termination=%s",
                verdict, proposal["n_applicable"], proposal["n_edits"],
                proposal["termination"])
    if not proposal["ready"]:
        for reason in proposal["not_ready_reasons"]:
            logger.info("  not-ready: %s", reason)
    return proposal
