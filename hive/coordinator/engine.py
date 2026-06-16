"""Coordinator orchestration (D-01 §5 data flow) — the W1 non-interactive 1-shot.

    seed → symptom → [L-01 expected ‖ L-02 decompose] → L-03 gate → ready()
         → L-06 finalize → enriched seed (handoff)

W1 has no ask-loop (P-01 ping-pong is W2), so the engine makes ONE pass and
best-effort seals: rounds_left starts at 0 for the non-interactive track, which
drives ready() down the hard-cap → seal → sealed → handoff path. The coordinator
thus never blocks; it adds the true expected (L-01) plus a transparent record of
what it could not fill (provenance.skipped_slots), and the queen runs as before.

Reuses the shared LLM plumbing (CON-2) and the ledger; persists the gap-state
(DB-01) for audit / future resume. The only output is the Caller-supplied context
section — the queen's schema is never touched (CON-1, D-01 §9).
"""
from __future__ import annotations

import logging
import re
import uuid as _uuid
from typing import Any

from hive.coordinator.decompose_fork import decompose
from hive.coordinator.expected import extract_expected
from hive.coordinator.finalize import finalize
from hive.coordinator.gate import gate_load_bearing, ready
from hive.coordinator.model import GapState, ROUNDS_CAP

logger = logging.getLogger("hive.coordinator")

_FRONTMATTER = re.compile(r"^\s*---\n.*?\n---\n", re.DOTALL)


def run_coordinator(seed_text: str, *, codebase_root: str | None = None,
                    recipe_path: str | None = None,
                    model: str = "claude-sonnet-4.5", provider: str = "copilot",
                    ledger=None, provider_kwargs: dict | None = None,
                    timeout: int = 600, interactive: bool = False,
                    manifest: dict | None = None, store=None) -> dict[str, Any]:
    """Run the W1 coordinator over ``seed_text`` and return:

        {"enriched_seed": str,   # seed + Caller-supplied context section
         "uuid": str, "status": "ready"|"sealed",
         "expected": [...], "provenance": {...}, "gap_state": {...}}

    ``enriched_seed`` is what the caller feeds to ``run_decompose`` in place of the
    raw seed.
    """
    symptom = _extract_symptom(seed_text)
    gs = GapState(
        uuid=_uuid.uuid4().hex,
        track="interactive" if interactive else "api",
        interactive_flag=interactive,
        symptom_raw=symptom,
        seed_base=seed_text or "",
    )
    # W1: no interactive rounds available (asking is W2) → seal on the single pass.
    gs.caps["rounds_left"] = ROUNDS_CAP if interactive else 0

    common = dict(model=model, provider=provider, ledger=ledger,
                  provider_kwargs=provider_kwargs, timeout=timeout, cwd=codebase_root)

    gs.expected = extract_expected(symptom, manifest, **common)        # L-01 (W0)
    candidates = decompose(symptom, manifest, **common)                # L-02
    gs.gaps = gate_load_bearing(candidates, gs.seed_draft, manifest)   # L-03 gate

    status = ready(gs)        # W1 non-interactive → 'sealed' via hard cap
    # (W2 would loop here while status == "collecting", emitting questions.)
    logger.info("coordinator: status=%s expected=%d gaps=%d",
                status, len(gs.expected), len(gs.gaps))

    enriched = finalize(gs)                                            # L-06 handoff

    if store is not None:
        try:
            store.save(gs)
        except Exception as e:  # persistence is best-effort (DB-01) — never block
            logger.warning("coordinator: gap-state save failed (non-fatal): %s", e)

    skipped = [g.slot for g in gs.gaps if g.load_bearing and g.status == "skipped"]
    return {
        "enriched_seed": enriched,
        "uuid": gs.uuid,
        "status": gs.status,
        "expected": gs.expected,
        "provenance": {
            "interactive": gs.interactive_flag,
            "sealed": gs.status == "sealed",
            "skipped_slots": skipped,
        },
        "gap_state": gs.to_dict(),
    }


def _extract_symptom(seed_text: str) -> str:
    """The coordinator's symptom = the user's prose. Strip a leading YAML
    frontmatter block if present so L-01/L-02 reason over the body, not metadata."""
    if not seed_text:
        return ""
    return _FRONTMATTER.sub("", seed_text, count=1).strip()
