#!/usr/bin/env python3
"""Regression tests for the per-provider billing model (hivework.0013, R0001).

Two billing paradigms must stay separate:
  - credit-billed (copilot)  = calls × credits_per_call × credit_usd, token-free
  - token-billed (deepinfra/openai) = real-token-aware USD via the price table

The headline acceptance criterion is that run459 — 6 openai/Qwen swarm calls
carrying 1,174,222 real tokens + 4 credit-billed copilot calls — converges to
~$0.12 once the Qwen output price is the real $0.10/Mtok (not the stale $0.60).

Run with: python -m pytest perf/metrics/test_emit_billing.py
       or: python perf/metrics/test_emit_billing.py   (no pytest needed)
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emit  # noqa: E402

# run459's real ledger shape: (provider, model, in_chars, out_chars, est, real)
_RUN459_CALLS = [
    ("queen", "copilot", "gpt-5-mini", 11127, 6013, 4284, None),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4164, 2137, 1575, 48659),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4104, 3118, 1805, 45774),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4154, 2762, 1728, 47749),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4128, 2609, 1684, 107884),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4110, 897, 1251, 336845),
    ("swarm", "openai", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4116, 2675, 1697, 587311),
    ("queen", "copilot", "gpt-5-mini", 4691, 3229, 1979, None),
    ("queen", "copilot", "gpt-5-mini", 4689, 3022, 1927, None),
    ("assemble", "copilot", "gpt-5-mini", 23776, 11179, 8738, None),
]


def _make_ledger(calls, run_id=459):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, ts TEXT, seed TEXT, "
                 "work_type TEXT, codebase TEXT, model_queen TEXT, model_swarm TEXT, "
                 "axes_n INTEGER, rounds INTEGER, conflicts_n INTEGER, remaining_n INTEGER, "
                 "parse_errs INTEGER, total_in_chars INTEGER, total_out_chars INTEGER, "
                 "total_est_tokens INTEGER, total_real_tokens INTEGER, elapsed_s REAL, "
                 "honey_path TEXT, status TEXT)")
    conn.execute("CREATE TABLE worker_calls (id INTEGER PRIMARY KEY, run_id INTEGER, "
                 "stage TEXT, axis_id TEXT, provider TEXT, model TEXT, in_chars INTEGER, "
                 "out_chars INTEGER, est_tokens INTEGER, real_tokens INTEGER, latency_s REAL, "
                 "comb_path TEXT, ok INTEGER, err TEXT, status TEXT, started_at TEXT)")
    conn.execute("INSERT INTO runs (id, ts, model_queen, model_swarm, axes_n, status) "
                 "VALUES (?,?,?,?,?,?)",
                 (run_id, "2026-06-19T09:05:38+00:00", "gpt-5-mini",
                  "Qwen/Qwen3-235B-A22B-Instruct-2507", 8, "done"))
    for stage, prov, model, ic, oc, est, real in calls:
        conn.execute("INSERT INTO worker_calls (run_id, stage, axis_id, provider, model, "
                     "in_chars, out_chars, est_tokens, real_tokens, latency_s, ok) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (run_id, stage, "T1", prov, model, ic, oc, est, real, 1.0, 1))
    conn.commit()
    conn.close()
    return path


def test_run459_converges_to_12_cents():
    path = _make_ledger(_RUN459_CALLS)
    try:
        rec = emit.build_record(path, 459)
    finally:
        os.unlink(path)
    by_prov = rec["cost"]["by_provider"]
    total = sum(p["usd"] for p in by_prov.values())
    assert abs(total - 0.12) < 0.005, f"expected ~$0.12, got ${total:.5f}"


def test_copilot_credit_billed_not_token():
    """copilot pays per-call credits (gpt-5-mini = 0×), never token estimate."""
    path = _make_ledger(_RUN459_CALLS)
    try:
        rec = emit.build_record(path, 459)
    finally:
        os.unlink(path)
    cop = rec["cost"]["by_provider"]["copilot"]
    assert cop["calls"] == 4
    assert cop["credits"] == 0          # gpt-5-mini is a 0× (included) model
    assert cop["usd"] == 0.0            # credit-billed → no token charge


def test_credit_multiplier_drives_usd():
    """A non-zero per-call multiplier yields calls × credits × $0.01."""
    calls = [("queen", "copilot", "claude-sonnet-4.5", 100, 100, 50, None)] * 3
    path = _make_ledger(calls)
    try:
        rec = emit.build_record(path, 459)
    finally:
        os.unlink(path)
    cop = rec["cost"]["by_provider"]["copilot"]
    assert cop["calls"] == 3
    assert cop["credits"] == 3          # 3 calls × 1× (claude-sonnet) = 3 credits
    assert abs(cop["usd"] - 0.03) < 1e-9  # 3 credits × $0.01


def test_deepinfra_token_billed_real_tokens():
    """deepinfra USD scales with real_tokens, not the char/4 estimate."""
    path = _make_ledger(_RUN459_CALLS)
    try:
        rec = emit.build_record(path, 459)
    finally:
        os.unlink(path)
    di = rec["cost"]["by_provider"]["deepinfra"]
    assert di["calls"] == 6
    assert di["credits"] == 0
    # 1.17M real tokens at the corrected Qwen price → ~$0.117, far above the
    # est-token (9740 tok) charge that would be a fraction of a cent.
    assert 0.10 < di["usd"] < 0.13


def test_axes_split_separates_credit_and_dollar():
    """The two paradigms never bleed: a credit provider has 0 token-USD and a
    token provider has 0 credits."""
    path = _make_ledger(_RUN459_CALLS)
    try:
        rec = emit.build_record(path, 459)
    finally:
        os.unlink(path)
    cop = rec["cost"]["by_provider"]["copilot"]
    di = rec["cost"]["by_provider"]["deepinfra"]
    assert cop["credits"] >= 0 and cop["usd"] == 0.0
    assert di["credits"] == 0 and di["usd"] > 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
