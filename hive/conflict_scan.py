"""Conflict scanner — detects inter-comb conflicts.

Compares parsed comb dicts and detects:
  (a) root_cause_signal disagreement — different axes point to different root causes
  (b) termination divergence — one axis says "resolved", another "needs_pm" etc.
  (c) unresolved conditional reachability — axis has reachable=conditional and isn't closed

Each conflict is a dict with:
  {
    "type": "root_cause_mismatch" | "termination_divergence" | "unresolved_conditional",
    "axis_a": str,
    "axis_b": str | None,
    "detail": str,
    "evidence_a": str,
    "evidence_b": str | None,
  }
"""

from typing import Any


def scan_conflicts(combs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan a list of parsed comb dicts for inter-comb conflicts.

    Args:
        combs: List of parsed comb JSON dicts (each has axis_id, root_cause_signal,
               termination, findings, etc.)

    Returns:
        List of conflict dicts. Empty if no conflicts.
    """
    conflicts = []
    conflicts.extend(_check_root_cause_mismatch(combs))
    conflicts.extend(_check_termination_divergence(combs))
    conflicts.extend(_check_unresolved_conditional(combs))
    return conflicts


def _check_root_cause_mismatch(combs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect root_cause_signal disagreement between axes.

    Two axes that both claim root_cause_signal (non-null) but point to different
    file:line locations are in conflict.
    """
    conflicts = []
    axes_with_root = []
    for c in combs:
        sig = c.get("root_cause_signal")
        if sig and sig.lower() != "null" and sig.strip():
            axes_with_root.append((c.get("axis_id", "?"), sig))

    for i in range(len(axes_with_root)):
        for j in range(i + 1, len(axes_with_root)):
            aid_a, sig_a = axes_with_root[i]
            aid_b, sig_b = axes_with_root[j]
            # Normalize: strip whitespace, lowercase for comparison
            if _signals_disagree(sig_a, sig_b):
                conflicts.append({
                    "type": "root_cause_mismatch",
                    "axis_a": aid_a,
                    "axis_b": aid_b,
                    "detail": f"Axis {aid_a} and {aid_b} both claim root_cause_signal but differ",
                    "evidence_a": sig_a,
                    "evidence_b": sig_b,
                })
    return conflicts


def _signals_disagree(sig_a: str, sig_b: str) -> bool:
    """Determine if two root_cause_signal strings meaningfully disagree.

    A real root_cause conflict means two axes point at the SAME locus but draw
    DIFFERENT conclusions. In a MECE fan-out the axes deliberately investigate
    different facets (FE / BE / CSS / predicate …), so merely having different
    signals is the NORMAL case, not a conflict — flagging those produced a flood of
    spurious conflicts (N163: 47 from 12 axes).

    So they disagree only when BOTH cite file:line refs AND those refs OVERLAP (same
    locus) AND the signals are not substring-equal (different conclusion about that
    locus). If a shared locus cannot be established — either side has no file:line
    ref, or the refs do not overlap — the axes are about different things and are NOT
    in conflict. (Tradeoff: a rare prose-only "same bug, different words" conflict may
    be missed; termination_divergence and unresolved_conditional still run.)
    """
    a_norm = sig_a.strip().lower()
    b_norm = sig_b.strip().lower()
    if a_norm == b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return False
    a_refs = _extract_file_line_refs(a_norm)
    b_refs = _extract_file_line_refs(b_norm)
    if not a_refs or not b_refs:
        return False  # no shared locus can be established → complementary, not a conflict
    return bool(a_refs & b_refs)  # same locus + different conclusion → genuine disagreement


def _extract_file_line_refs(text: str) -> set[str]:
    """Extract file:line-like references from a signal string."""
    import re
    # Match patterns like "inbox_routes.py:843" or "queries.json:128"
    return set(re.findall(r'[\w./\\]+\.(?:py|json|js|ts|vue|sql|md):\d+', text))


def _check_termination_divergence(combs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect termination divergence between axes.

    Conflict = one axis says "resolved" while another says "needs_pm", "needs_runtime",
    or "needs_external" for what appears to be the same issue (shared cross_refs or
    overlapping root_cause_signal).
    """
    conflicts = []
    resolved_axes = []
    unresolved_axes = []
    for c in combs:
        term = c.get("termination", "").strip().lower()
        aid = c.get("axis_id", "?")
        if term == "resolved":
            resolved_axes.append((aid, c))
        elif term in ("needs_pm", "needs_runtime", "needs_external"):
            unresolved_axes.append((aid, c, term))

    for aid_r, comb_r in resolved_axes:
        for aid_u, comb_u, term_u in unresolved_axes:
            # Check if they share cross_refs or related root_cause
            if _axes_related(comb_r, comb_u):
                conflicts.append({
                    "type": "termination_divergence",
                    "axis_a": aid_r,
                    "axis_b": aid_u,
                    "detail": (f"Axis {aid_r} terminated as 'resolved' but "
                               f"axis {aid_u} terminated as '{term_u}' — "
                               f"potentially conflicting conclusions"),
                    "evidence_a": comb_r.get("root_cause_signal", "(none)"),
                    "evidence_b": comb_u.get("root_cause_signal", "(none)"),
                })

    return conflicts


def _axes_related(comb_a: dict, comb_b: dict) -> bool:
    """Check if two combs are related (cross-reference each other or share signals)."""
    aid_a = comb_a.get("axis_id", "")
    aid_b = comb_b.get("axis_id", "")

    # Direct cross-reference
    refs_a = comb_a.get("cross_refs", []) or []
    refs_b = comb_b.get("cross_refs", []) or []
    if aid_b in refs_a or aid_a in refs_b:
        return True

    # Shared root_cause_signal file references
    sig_a = comb_a.get("root_cause_signal", "") or ""
    sig_b = comb_b.get("root_cause_signal", "") or ""
    if sig_a and sig_b:
        refs_sa = _extract_file_line_refs(sig_a.lower())
        refs_sb = _extract_file_line_refs(sig_b.lower())
        if refs_sa & refs_sb:
            return True

    return False


def _check_unresolved_conditional(combs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect axes with reachable=conditional findings that aren't resolved.

    An axis that has any finding with reachable containing "conditional" AND
    termination != "resolved" needs re-investigation.
    """
    conflicts = []
    for c in combs:
        term = c.get("termination", "").strip().lower()
        if term == "resolved":
            continue
        aid = c.get("axis_id", "?")
        findings = c.get("findings", []) or []
        for f in findings:
            reach = str(f.get("reachable", "")).lower()
            if "conditional" in reach:
                conflicts.append({
                    "type": "unresolved_conditional",
                    "axis_a": aid,
                    "axis_b": None,
                    "detail": (f"Axis {aid} has reachable=conditional finding "
                               f"not closed (termination={term})"),
                    "evidence_a": f.get("claim", "(no claim)"),
                    "evidence_b": None,
                })
                break  # One per axis is enough
    return conflicts
