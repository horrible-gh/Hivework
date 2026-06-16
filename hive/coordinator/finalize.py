"""L-06 best-effort termination + P-03 handoff section.

Consumes a ``ready``/``sealed`` gap_state and ALWAYS produces a seed (best-effort,
FR-8): even an almost-empty state hands off on the symptom alone (NFR-4: a
correctable wrong answer beats an uncorrectable non-answer). There is no
punt/abstain return path — ``finalize`` is total.

Two structural invariants are enforced here:
- ``assert_no_banned_vocab`` (L-06 §2.3, CON-4): the ``need_`` prefix family and
  its semantic work-arounds (awaiting/blocked/punt/abstain) are eradicated;
  "insufficiency" is expressed ONLY via gap.status + provenance.
- ``assert_single_terminal`` (L-06 §2.4, FR-8): the only terminal is HANDOFF.

The handoff section reuses the EXACT header the pipeline already reads
(``hive.investigate.CALLER_CONTEXT_SECTION``); a sync guard test pins the two
strings equal so they cannot drift.
"""
from __future__ import annotations

from hive.coordinator.model import (GapState, BANNED_PREFIXES, BANNED_SEMANTICS,
                                    ALLOWED_STATUS_VOCAB, ALLOWED_PROVENANCE_KEYS,
                                    TERMINAL_SET)

# Kept byte-identical to hive.investigate.CALLER_CONTEXT_SECTION (test-pinned) so
# the coordinator writes the very section the existing pipeline already folds in.
CALLER_CONTEXT_SECTION = "## Caller-supplied context (requester's direct input)"


class CoordinatorVocabError(AssertionError):
    """A banned (need_/awaiting/blocked/punt/abstain) lexeme reached an artifact."""


def finalize(gap_state: GapState) -> str:
    """Return the enriched seed (seed_base + Caller-supplied context section).
    Total: every input path returns a seed; no NO_ANSWER/PUNT path exists."""
    assert gap_state.status in ("ready", "sealed"), \
        f"finalize requires ready|sealed, got {gap_state.status!r}"
    section = build_caller_context(gap_state)
    seed = _attach_section(gap_state.seed_base, section)
    assert_no_banned_vocab(_structural_lexemes(gap_state))   # §2.3 (CON-4)
    assert_single_terminal()                                 # §2.4 (FR-8)
    return seed


def build_caller_context(gs: GapState) -> str:
    """P-03 §2 section. answered load-bearing → context; skipped load-bearing →
    provenance.skipped_slots only; expected omitted (not blanked) if unextracted."""
    lines = ["", CALLER_CONTEXT_SECTION, ""]

    exp = gs.expected or []
    if len(exp) == 1:
        lines.append(f"- expected: {exp[0].get('expected')}")
    elif len(exp) > 1:
        lines.append("- expected:")
        lines += [f"  - {e.get('expected')}" for e in exp]
    # else: omit the field entirely — a false expected is worse than none (FR-4)

    if (gs.symptom_raw or "").strip():
        lines.append(f"- symptom: {_one_line(gs.symptom_raw)}")

    answered = [(g.slot, g.answer) for g in gs.gaps
                if g.load_bearing and g.status == "answered"]
    if answered:
        lines.append("- context:")
        lines += [f"  - {slot}: {ans}" for slot, ans in answered]

    skipped = [g.slot for g in gs.gaps
               if g.load_bearing and g.status == "skipped"]
    lines.append("- provenance:")
    lines.append(f"    interactive: {str(bool(gs.interactive_flag)).lower()}")
    lines.append(f"    sealed: {str(gs.status == 'sealed').lower()}")
    lines.append(f"    skipped_slots: [{', '.join(skipped)}]")
    lines.append("")
    return "\n".join(lines)


def assert_no_banned_vocab(lexemes: list[str]) -> None:
    """L-06 §2.3: no need_/needs_ prefix, no awaiting/blocked/punt/abstain
    work-around. Applied to STRUCTURAL lexemes (status values, provenance keys,
    field labels) — NOT the user's free-text symptom/answers."""
    for raw in lexemes:
        t = str(raw).lower()
        if t.startswith(BANNED_PREFIXES):
            raise CoordinatorVocabError(f"banned prefix in artifact: {raw!r}")
        for pat in BANNED_SEMANTICS:
            if pat in t:
                raise CoordinatorVocabError(f"banned semantic in artifact: {raw!r}")


def assert_single_terminal() -> str:
    """L-06 §2.4: the engine's only terminal is HANDOFF (no abstain/punt node)."""
    assert tuple(TERMINAL_SET) == ("HANDOFF",), "terminal set drifted"
    return "HANDOFF"


def _structural_lexemes(gs: GapState) -> list[str]:
    """The controlled vocabulary the engine emits: gap status values, provenance
    keys, and the section field labels. Drift here (a future code change) is what
    the lint catches; user content is excluded by design (LINT_SCOPE)."""
    out: list[str] = ["expected", "symptom", "context", "provenance",
                      "interactive", "sealed", "skipped_slots"]
    for g in gs.gaps:
        if g.status not in ALLOWED_STATUS_VOCAB:
            raise CoordinatorVocabError(
                f"gap.status not in allowed vocab: {g.status!r}")
        out.append(g.status)
        out += list((g.provenance or {}).keys())
    for k in ("interactive", "sealed", "skipped_slots"):
        if k not in ALLOWED_PROVENANCE_KEYS:
            raise CoordinatorVocabError(f"provenance key not allowed: {k!r}")
    return out


def _attach_section(seed_base: str, section: str) -> str:
    base = seed_base or ""
    return f"{base}\n{section}" if section else base


def _one_line(text: str, cap: int = 280) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= cap else s[:cap - 1].rstrip() + "…"
