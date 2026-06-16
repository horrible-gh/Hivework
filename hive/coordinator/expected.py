"""L-01 expected extraction — casual symptom → true ``expected`` (one line, or a
multi-axis list). The W0 piece: zero coordinator dependency, reused as the
coordinator's expected slot but also runnable standalone.

The extraction is an LLM call (Sonnet-tier, language-agnostic — no ko/ja/en
regex); this module's *logic* is the deterministic post-validation that does not
trust the LLM: the anti-fabrication ladder (§2.2), the grounding-dominant
confidence score (§2.3), the refutability filter, and empty-on-failure
termination (E1/E6). Fabricating a false expected is worse than none (FR-4).
"""
from __future__ import annotations

from typing import Any

from hive.coordinator import _llm
from hive.coordinator.model import TAU_EMIT, MAX_AXES, W_G, W_S, W_P

_NEG = "not: "   # negation marker for the weakest (always-refutable) expected


def build_expected_prompt(symptom_text: str, manifest: dict | None = None) -> str:
    """Dedicated extraction prompt. Asks for axes with an observed phrase + the
    user's intent, NOT a guessed fix (the coordinator is not the queen)."""
    man = ""
    if manifest:
        man = ("\nPROJECT MANIFEST (closed sets you may use to resolve a "
               "complement — do NOT invent options):\n" + str(manifest) + "\n")
    return f"""You extract the TRUE EXPECTED behaviour a user implies in a casual
bug/feature complaint. You do NOT propose a fix and you do NOT investigate code.

For each distinct anomaly the user reports, output one axis:
- axis_label: short tag for the anomaly
- observed_phrase: the WRONG state, quoted/derived from the user's own words
- intent: {{ "states_target": true|false, "target_state": "<the state the user
  says it SHOULD be, ONLY if they stated it; else null>" }}

Rules:
- NEVER fabricate an expected. If the user did not state the target, leave
  target_state null — a deterministic guard will derive a refutable fallback.
- Ground every observed_phrase in the user's text. Do not add anomalies they
  did not report.
{man}
═══════════════ USER SYMPTOM ═══════════════
{symptom_text}
════════════════════════════════════════════

Respond with ONLY this JSON: {{"axes":[{{"axis_label":"...","observed_phrase":
"...","intent":{{"states_target":false,"target_state":null}}}}]}}
Start with `{{`, end with `}}`. No prose.
"""


def extract_expected(symptom_text: str, manifest: dict | None = None, *,
                     model: str = "claude-sonnet-4.5", provider: str = "copilot",
                     ledger=None, provider_kwargs: dict | None = None,
                     timeout: int = 300, cwd: str | None = None) -> list[dict]:
    """Return a ranked list of expected axes (≤ MAX_AXES), or [] (never fabricate).

    [] means "no true expected could be grounded" — a normal best-effort outcome,
    not an error; downstream proceeds without expected (FR-8).
    """
    if not symptom_text or not symptom_text.strip():
        return []                                   # E1: blank input, no LLM call
    raw = _llm.structured_call(
        build_expected_prompt(symptom_text, manifest), stage="coordinator",
        axis_id="expected", model=model, provider=provider, ledger=ledger,
        provider_kwargs=provider_kwargs, timeout=timeout, cwd=cwd)
    if raw is None:
        return []                                   # E6: schema unrecoverable

    candidates: list[dict] = []
    for c in (raw.get("axes") or []):
        observed = _normalize_state(c.get("observed_phrase", ""), manifest)
        expected = _derive_expected(observed, c.get("intent") or {}, manifest)
        if expected is None:                        # ladder bottomed out → drop axis
            continue
        conf = _score_confidence(observed, expected, symptom_text)
        candidates.append({
            "axis": c.get("axis_label", ""),
            "observed": observed,
            "expected": expected,
            "polarity": "anomaly",
            "refutable": _is_refutable(observed, expected),
            "confidence": conf,
        })

    kept = [c for c in candidates if c["refutable"] and c["confidence"] >= TAU_EMIT]
    kept = _dedup_by_axis(kept)
    kept.sort(key=lambda c: c["confidence"], reverse=True)
    return kept[:MAX_AXES]                           # [] if none survive


# ── §2.2 anti-fabrication ladder ──────────────────────────────────────────────
def _derive_expected(observed: str, intent: dict, manifest: dict | None):
    """Strongest available grounding wins; fall to a refutable negation; else drop.
    Never invents a value (FR-4 / P-03 §3)."""
    if intent.get("states_target") and (intent.get("target_state") or "").strip():
        return intent["target_state"].strip()                       # rung 1 (strongest)
    if manifest and _has_unique_complement(manifest, observed):
        return _complement(manifest, observed)                      # rung 2
    if observed:
        return _NEG + observed                                      # rung 3 (always refutable)
    return None                                                     # nothing grounds → drop


def _score_confidence(observed: str, expected: str, symptom: str) -> float:
    grounding = 1.0 if observed and observed.lower() in symptom.lower() else 0.0
    specificity = 0.5 if str(expected).startswith(_NEG) else 1.0
    support = min(1.0, len((observed or "").split()) / max(1, len(symptom.split())))
    return max(0.0, min(1.0, (W_G * grounding + W_S * specificity + W_P * support)
                        / (W_G + W_S + W_P)))                       # E5: denom is a const sum


def _is_refutable(observed: str, expected: str) -> bool:
    # A negation is always refutable; a concrete value is refutable iff it differs
    # from the observed (it makes a falsifiable claim about the right state).
    return bool(expected) and (str(expected).startswith(_NEG) or expected != observed)


def _dedup_by_axis(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["axis"] or r["expected"]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _normalize_state(phrase: str, manifest: dict | None) -> str:
    return (phrase or "").strip()


def _has_unique_complement(manifest: dict, observed: str) -> bool:
    comp = (manifest or {}).get("complements") or {}
    return observed in comp


def _complement(manifest: dict, observed: str) -> str:
    return (manifest or {}).get("complements", {}).get(observed, "")
