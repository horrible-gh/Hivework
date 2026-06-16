"""Shared LLM plumbing for the coordinator (CON-2: plumbing is shared, the
reasoning prompt/profile is NOT). Mirrors the decompose call shape — same
``call_worker`` harness, ``extract_first_json`` parser, and ledger recording —
so the only thing a coordinator stage owns is its prompt and post-validation.

``structured_call`` returns the parsed JSON dict, or ``None`` on any failure
(non-zero exit, blank output, or unrecoverable schema violation after RETRY_MAX).
Callers treat ``None`` as SCHEMA_INVALID and fall back to ``[]`` (best-effort,
never fabricate — L-01/L-02 E6).
"""
from __future__ import annotations

from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.coordinator.model import RETRY_MAX


def structured_call(prompt: str, *, stage: str, axis_id: str, model: str,
                    provider: str, ledger=None, provider_kwargs: dict | None = None,
                    timeout: int = 300, cwd: str | None = None,
                    retries: int = RETRY_MAX) -> dict[str, Any] | None:
    """Run one Sonnet-tier structured-output call with bounded retry.

    Returns the parsed dict, or None when the call fails or the output never
    decodes to JSON within ``retries`` extra attempts.
    """
    for attempt in range(retries + 1):
        call_id = ledger.begin_call(stage, axis_id, provider, model, prompt) \
            if ledger is not None else None
        try:
            wr = call_worker(
                provider, model, prompt, cwd=cwd, timeout=timeout,
                on_start=(lambda: ledger.mark_running(call_id))
                if (ledger is not None and call_id is not None) else None,
                **(provider_kwargs or {}),
            )
        except Exception as e:  # timeout / provider error — record + maybe retry
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            if attempt < retries:
                continue
            return None
        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout, latency_s=wr.latency_s,
                               ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)
        if wr.exit_code != 0 or not (wr.stdout or "").strip():
            if attempt < retries:
                continue
            return None
        try:
            return extract_first_json(wr.stdout)
        except ValueError:
            if attempt < retries:
                continue
            return None
    return None
