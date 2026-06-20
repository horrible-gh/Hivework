#!/usr/bin/env python3
"""Regression tests for the per-axis breakdown projection (hivework.0029, R0001).

The "측정 불가" bug: emit.build_record's per-axis ``axes[]`` array filtered
worker_calls on a STALE positive stage allowlist ``{swarm, fanout, reinforce}``.
The investigate pipeline mines under stage ``retrieve`` and records the converge
refutation lenses under ``lens`` (historically ``swarm``), so the old filter
dropped every comb-bearing retrieve axis and surfaced only the comb-less lenses
(or nothing) — ``funnel.comb_fired > 0`` while ``Σ axes.fired == 0``, i.e. the
per-axis yield was unmeasurable.

The fix projects axes over the SOURCE-MINING stages (a negative exclude set, so a
renamed mining stage is kept by default) and drops ``lens:*`` axes by id. These
tests pin: lenses excluded, retrieve axes present, and Σ axes.fired consistent
with the funnel's comb count — for both the investigate and the fanout shapes.

Run with: python -m pytest perf/metrics/test_emit_axes.py
       or: python perf/metrics/test_emit_axes.py   (no pytest needed)
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emit  # noqa: E402

# An investigate run shaped like run477: decompose (queen) → retrieve (the 3
# comb-bearing mining axes) → judge (5-vote, no comb) → converge → lens
# refutations recorded under the legacy stage 'swarm' (comb-less). Each tuple is
# (stage, axis_id, comb_path).
_INVESTIGATE_CALLS = [
    ("queen", "decompose", ""),
    ("retrieve", "design_ssot", "reads=2 rows=True"),
    ("retrieve", "be_route_lookup", "reads=1 rows=True"),
    ("retrieve", "frontend_submit_ui", "reads=3 rows=True"),
    ("judge", "design_ssot", ""),
    ("judge", "be_route_lookup", ""),
    ("judge", "frontend_submit_ui", ""),
    ("converge", "converge", ""),
    ("swarm", "lens:datasource-liveness", ""),
    ("swarm", "lens:omission", ""),
    ("swarm", "lens:reproduction", ""),
    ("swarm", "lens:wiring", ""),
    ("assemble", "assemble", ""),
]

# A fanout run shaped like run466: mining axes under the legacy 'swarm' stage
# (these DO carry combs and ARE real axes), a reinforce scout, plus reconcile
# (queen) and assemble synthesis that must NOT appear as mining axes.
_FANOUT_CALLS = [
    ("queen", "decompose", ""),
    ("swarm", "T1_design", "comb"),
    ("swarm", "T2_be_endpoint", "comb"),
    ("scout", "T2_be_endpoint#reinforce0", "comb"),
    ("queen", "RECONCILE_R1", "comb"),
    ("assemble", "assemble", ""),
]


def _make_ledger(calls, run_id=900, axes_n=3):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, ts TEXT, seed TEXT, "
                 "work_type TEXT, codebase TEXT, model_queen TEXT, model_swarm TEXT, "
                 "model_fanout TEXT, axes_n INTEGER, elapsed_s REAL, honey_path TEXT, "
                 "status TEXT)")
    conn.execute("CREATE TABLE worker_calls (id INTEGER PRIMARY KEY, run_id INTEGER, "
                 "stage TEXT, axis_id TEXT, provider TEXT, model TEXT, in_chars INTEGER, "
                 "out_chars INTEGER, est_tokens INTEGER, real_tokens INTEGER, latency_s REAL, "
                 "comb_path TEXT, ok INTEGER, err TEXT, status TEXT, started_at TEXT)")
    conn.execute("INSERT INTO runs (id, ts, work_type, model_queen, axes_n, status, honey_path) "
                 "VALUES (?,?,?,?,?,?,?)",
                 (run_id, "2026-06-20T00:00:00+00:00", "investigate", "gpt-5-mini",
                  axes_n, "done", "honey.md"))
    for stage, axis_id, comb in calls:
        conn.execute("INSERT INTO worker_calls (run_id, stage, axis_id, provider, model, "
                     "in_chars, out_chars, est_tokens, real_tokens, latency_s, comb_path, ok) "
                     "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (run_id, stage, axis_id, "local", "ripgrep", 10, 10, 5, None,
                      1.0, comb, 1))
    conn.commit()
    conn.close()
    return path


def _axes(calls, axes_n=3):
    path = _make_ledger(calls, axes_n=axes_n)
    try:
        return emit.build_record(path, 900)
    finally:
        os.unlink(path)


def test_investigate_axes_are_retrieve_not_lenses():
    """The per-axis array reports the 3 retrieve mining axes, never the lenses."""
    rec = _axes(_INVESTIGATE_CALLS)
    ids = [a["axis_id"] for a in rec["axes"]]
    assert ids == ["design_ssot", "be_route_lookup", "frontend_submit_ui"]
    assert not any(a["axis_id"].startswith("lens:") for a in rec["axes"])


def test_investigate_sigma_fired_matches_comb_fired():
    """The headline consistency rule: Σ axes.fired == funnel.comb_fired (was 0≠3
    before the fix — the comb-less lenses hid every fired retrieve axis)."""
    rec = _axes(_INVESTIGATE_CALLS)
    fired = sum(1 for a in rec["axes"] if a["fired"])
    assert fired == rec["funnel"]["comb_fired"] == 3


def test_orchestration_and_synthesis_stages_excluded():
    """decompose/judge/converge/assemble are never per-axis mining rows."""
    rec = _axes(_INVESTIGATE_CALLS)
    ids = {a["axis_id"] for a in rec["axes"]}
    assert ids.isdisjoint({"decompose", "converge", "assemble"})


def test_fanout_legacy_swarm_mining_axes_kept():
    """Legacy fanout runs mined under stage 'swarm' with REAL axis ids — those
    (and the reinforce scout) stay; reconcile/assemble synthesis drop out."""
    rec = _axes(_FANOUT_CALLS, axes_n=2)
    ids = [a["axis_id"] for a in rec["axes"]]
    assert ids == ["T1_design", "T2_be_endpoint", "T2_be_endpoint#reinforce0"]
    assert "RECONCILE_R1" not in ids and "assemble" not in ids
    assert all(a["fired"] for a in rec["axes"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
