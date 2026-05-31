"""Specify stage — lowers a honey's prose fix directions into a precise edit-spec.

Pipeline position (fix-extension):

  investigate (fan-out)  →  merge (honey)  →  SPECIFY (this)  →  apply (propose-only)

Unlike investigate, specify is NOT a swarm. A single consistent author takes the
assembled honey (whose fix directions are written as prose) plus the LIVE target
codebase and lowers each direction into a concrete ``anchor_old → replacement_new``
edit. Code edits must be internally coherent, so this stage is never fan-out.

Key invariants (mirrored from recipes/edit_spec_contract_v1.md):
  - Anchors come from LIVE code, never from the honey — the honey may be stale.
    The author records anchor_status (verified | stale | not_found) as feedback.
  - The spec defines its own boundary: a direction that cannot be expressed as an
    edit goes to ``deferred[]``, it is not forced into an edit.
  - Stage-1 safety: gate.apply is ALWAYS false here. specify proposes; the PM (or
    a later promotion) applies. specify never writes to the target codebase.
  - The JSON edit-spec is the SSOT; the human-facing unified diff is a DERIVED view
    rendered later by hive/apply.py — specify does not author the diff.

The author's role prompt is the contract file itself, loaded at runtime so the
contract stays the single source of authoring rules (no duplicated prompt here).
"""

import json
import logging
import os
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker

logger = logging.getLogger("hive.specify")

# The authoring contract doubles as the specify author's role/system prompt.
_DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "recipes", "edit_spec_contract_v1.md"
)

# Structural expectations for the emitted edit-spec JSON.
_REQUIRED_KEYS = ("edits", "deferred", "gate", "termination")
_VALID_TERMINATION = {"ready_to_apply", "needs_reinvestigation", "needs_pm"}
_STALE_STATUSES = {"stale", "not_found"}


def load_contract(contract_path: str | None = None) -> str:
    """Load the edit-spec authoring contract (the specify author's role prompt)."""
    path = contract_path or _DEFAULT_CONTRACT_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_specify_prompt(honey_text: str, contract_text: str, codebase_root: str) -> str:
    """Build the full prompt for the single specify author.

    The contract is the role/system prompt; the honey is the input to lower; the
    codebase_root tells the author where the LIVE code is. Anchors must be lifted
    from there byte-for-byte, never copied from the (possibly stale) honey.
    """
    return f"""{contract_text}

[Codebase root — read LIVE files from here]
{codebase_root}

[Input honey — lower each fix direction into the edit-spec contracted above]
Re-open every file you touch under the codebase root and lift `anchor_old` from
the CURRENT text byte-for-byte. Do NOT trust code quoted in the honey below.

{honey_text}
"""


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Enforce Stage-1 invariants and reconcile internal inconsistencies.

    - gate.apply is ALWAYS forced false here (specify proposes only).
    - A spec containing a stale/not_found edit is not ready: if the author still
      claimed ready_to_apply, override it to needs_reinvestigation rather than
      presenting an unverified anchor as applicable.
    """
    gate = spec.get("gate")
    if not isinstance(gate, dict):
        gate = {}
        spec["gate"] = gate
    if gate.get("apply") is not False:
        logger.warning("specify: gate.apply was %r — forcing false (Stage-1 safety)",
                       gate.get("apply"))
        gate["apply"] = False

    edits = spec.get("edits")
    edits = edits if isinstance(edits, list) else []
    unverified = [e.get("id", "?") for e in edits
                  if isinstance(e, dict)
                  and str(e.get("anchor_status", "")).lower() in _STALE_STATUSES]
    if unverified and spec.get("termination") == "ready_to_apply":
        logger.warning("specify: edits %s are stale/not_found but termination=ready_to_apply"
                       " — overriding to needs_reinvestigation", unverified)
        spec["termination"] = "needs_reinvestigation"
    return spec


def _validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return a list of structural problems (empty = ok). Non-fatal; caller decides."""
    problems: list[str] = []
    for key in _REQUIRED_KEYS:
        if key not in spec:
            problems.append(f"missing required key: {key}")
    term = spec.get("termination")
    if term is not None and term not in _VALID_TERMINATION:
        problems.append(f"invalid termination: {term!r}")
    if "edits" in spec and not isinstance(spec["edits"], list):
        problems.append("edits is not a list")
    if "deferred" in spec and not isinstance(spec["deferred"], list):
        problems.append("deferred is not a list")
    return problems


def run_specify(
    honey_path: str,
    codebase_root: str,
    output_path: str,
    contract_path: str | None = None,
    model: str = "gpt-5-mini",
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run the specify stage: honey + live code → edit-spec JSON.

    Calls a single author worker, extracts the first complete JSON object from its
    stdout, enforces Stage-1 invariants, writes the spec to ``output_path`` as the
    SSOT JSON, and returns the parsed dict.

    Raises:
        ValueError: if the author produced no parseable JSON object.
        FileNotFoundError: if the honey or contract file is missing.
    """
    with open(honey_path, "r", encoding="utf-8") as f:
        honey_text = f.read()
    contract_text = load_contract(contract_path)
    prompt = build_specify_prompt(honey_text, contract_text, codebase_root)

    logger.info("Running specify author (single, not fan-out)...")
    logger.debug("Prompt length: %d chars", len(prompt))

    wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=600,
                     **(provider_kwargs or {}))
    if ledger is not None:
        ledger.record_call("specify", "specify", provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "")

    spec = extract_first_json(wr.stdout)  # raises ValueError if no JSON found
    spec = _normalize_spec(spec)

    problems = _validate_spec(spec)
    if problems:
        logger.warning("specify: edit-spec has structural problems: %s", "; ".join(problems))

    # Record provenance so a derived diff / reconcile loop can trace this back.
    spec.setdefault("source_honey", honey_path)
    spec.setdefault("codebase_root", os.path.abspath(codebase_root))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    n_edits = len(spec.get("edits") or [])
    n_deferred = len(spec.get("deferred") or [])
    logger.info("Edit-spec written to %s (%d edits, %d deferred, termination=%s)",
                output_path, n_edits, n_deferred, spec.get("termination", "?"))
    return spec
