"""L-02 1-stage decompose engine — message → context-unit information-gap slot
candidates. A dedicated LIGHTWEIGHT FORK of decompose (CON-2): it shares only the
plumbing (``_llm`` harness, structured-output parser, manifest injection) and
runs a context-axis reasoning prompt, NEVER the queen's word/code-axis profile.

The LLM proposes slots; this module's logic is the deterministic post-validation
that does not trust it: grounding (anti-hallucination), the code-axis-leak guard
(CON-2), the salience gate, dedup, and a non-silent MAX_SLOTS truncation. Empty
output is normal (E2), not failure: it means "no context gap to fill" and lets
L-03 ready() pass immediately (AC-3).
"""
from __future__ import annotations

from typing import Any

from hive.coordinator import _llm
from hive.coordinator.model import (TAU_SALIENCE, MAX_SLOTS, WD_G, WD_O, WD_U,
                                    FORK_PROFILE)

_EXPECTED_KIND = "expected"


def build_decompose_fork_prompt(message: str, manifest: dict | None = None) -> str:
    man = ""
    if manifest:
        man = ("\nPROJECT MANIFEST (screens/components/doc-types you may reference"
               " — do NOT invent):\n" + str(manifest) + "\n")
    return f"""You decompose a user's casual request into CONTEXT-UNIT information
gaps: the things the user has NOT yet said that, if answered, would change how the
request is handled. You are NOT the queen — do NOT propose which symbols/files to
read or any investigation plan. Find what to ASK the user, not where to dig.

For each unsaid, outcome-changing gap, output one slot:
- slot: the missing context (a meaning unit, in plain language)
- kind: one of expected|context|scope|constraint|env
- what_is_unsaid: why this is a gap (what the user has not stated)
- changes_outcome_score: 0..1, your estimate that the answer changes the outcome
- targets_symbol_or_file: true if this is really a "read symbol/file X" directive
- is_single_token_lexical: true if this is a word-level code-search axis
- proposes_investigation: true if this is a "where to dig" plan

Set the last three honestly — a deterministic guard drops any slot with any of
them true (that is queen territory, not coordinator).
{man}
═══════════════ USER MESSAGE ═══════════════
{message}
════════════════════════════════════════════

Respond with ONLY this JSON: {{"slots":[{{"slot":"...","kind":"context",
"what_is_unsaid":"...","changes_outcome_score":0.0,"targets_symbol_or_file":false,
"is_single_token_lexical":false,"proposes_investigation":false}}]}}
Start with `{{`, end with `}}`. No prose.
"""


def decompose(message: str, manifest: dict | None = None, *,
              model: str = "claude-sonnet-4.5", provider: str = "copilot",
              ledger=None, provider_kwargs: dict | None = None,
              timeout: int = 300, cwd: str | None = None) -> list[dict]:
    """Return slot candidates (all status=open), or [] (normal: no gap / failure).

    Candidates carry only the gap *fields* (slot/kind/status/expected_carveout/
    evidence); the surface question/format/options and the load_bearing AUTHORITY
    are filled by L-03/D-02 when promoting a candidate to a formal gap.
    """
    if not message or not message.strip():
        return []                                       # E1
    raw = _llm.structured_call(
        build_decompose_fork_prompt(message, manifest), stage="coordinator",
        axis_id="decompose", model=model, provider=provider, ledger=ledger,
        provider_kwargs=provider_kwargs, timeout=timeout, cwd=cwd)
    if raw is None:
        return []                                       # E6: fabricate nothing

    candidates: list[dict] = []
    for i, s in enumerate(raw.get("slots") or []):
        if not _is_grounded(s, message):                # E3 hallucination guard
            continue
        if _is_code_axis_directive(s):                  # E5 / CON-2 leak guard
            continue
        kind = _classify_kind(s)
        candidates.append({
            "id": f"slot-{i}",
            "slot": _normalize_slot(s.get("slot", "")),
            "kind": kind,
            "expected_carveout": (kind == _EXPECTED_KIND),
            "load_bearing_hint": _clamp01(s.get("changes_outcome_score")),
            "salience": _score_salience(s, message, manifest),
            "status": "open",
            "evidence_absent": s.get("what_is_unsaid", ""),
            "provenance": {"source": "L-02", "profile": FORK_PROFILE},
        })

    kept = [c for c in candidates if c["salience"] >= TAU_SALIENCE]
    kept = _dedup_by_slot(kept)
    kept.sort(key=lambda c: c["salience"], reverse=True)
    if len(kept) > MAX_SLOTS:                            # E7 non-silent truncation
        dropped = len(kept) - MAX_SLOTS
        kept = kept[:MAX_SLOTS]
        kept[-1]["provenance"]["truncated"] = dropped
    return kept


# ── §2.4 code-axis leak guard (CON-2 deterministic) ───────────────────────────
def _is_code_axis_directive(s: dict) -> bool:
    return bool(s.get("targets_symbol_or_file")
                or s.get("is_single_token_lexical")
                or s.get("proposes_investigation"))


# ── §2.3 salience (grounding/unsaid-dominant; denom is a const weight sum) ─────
def _score_salience(s: dict, message: str, manifest: dict | None) -> float:
    slot = (s.get("slot") or "")
    grounding = 1.0 if slot and _overlaps(slot, message) else 0.0
    outcome = _clamp01(s.get("changes_outcome_score"))
    unsaid = 1.0 if not _overlaps(slot, message) else 0.0   # already-said ≠ a gap
    return max(0.0, min(1.0, (WD_G * grounding + WD_O * outcome + WD_U * unsaid)
                        / (WD_G + WD_O + WD_U)))


def _is_grounded(s: dict, message: str) -> bool:
    slot = (s.get("slot") or "").strip()
    if not slot:
        return False
    # Grounded if the slot shares meaningful tokens with the message OR the LLM
    # flagged a concrete unsaid rationale (both anchor it to this request).
    return _overlaps(slot, message) or bool((s.get("what_is_unsaid") or "").strip())


def _overlaps(text: str, message: str) -> bool:
    toks = {t for t in _words(text) if len(t) >= 3}
    msg = set(_words(message))
    return bool(toks & msg)


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " "
                               for c in (text or "")).split() if w]


def _classify_kind(s: dict) -> str:
    kind = (s.get("kind") or "context").strip().lower()
    return kind if kind in ("expected", "context", "scope", "constraint", "env") \
        else "context"


def _normalize_slot(slot: str) -> str:
    return (slot or "").strip()


def _dedup_by_slot(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = r["slot"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0
