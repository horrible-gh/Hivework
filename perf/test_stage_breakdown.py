#!/usr/bin/env python3
"""Unit tests for perf/stage_breakdown.py (hivework.0061 R0001 성능개선).

Pure-function tests with a synthetic ledger shape — no model calls, no live DB,
no paid spend. Verifies the critical-path model and the stage-share aggregate
that operationalise the serialization diagnosis (wall is serial-stage-bound, not
single-worker-bound; ~1/3 of wall is local residual not recorded as a call).
"""
from __future__ import annotations
import os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_breakdown as sb  # noqa: E402


def test_serial_stage_sums_parallel_stage_takes_max():
    # judge (parallel) has 3 calls of 10s each -> contributes max=10, NOT 30.
    # converge + specify (serial) contribute their full latency.
    sl = {"judge": [10.0, 10.0, 10.0], "converge": [40.0], "specify": [30.0]}
    wall = 100.0
    b = sb.compute_run_breakdown(sl, wall)
    # modelpath = max(judge)=10 + converge 40 + specify 30 = 80
    assert b["modelpath"] == 80.0
    assert b["max_lat"] == 40.0          # slowest single call is converge@40
    assert b["sum_lat"] == 100.0         # judge 3x10=30 + converge 40 + specify 30
    assert b["residual"] == 20.0         # 100 - 80 local orchestration
    assert b["n_calls"] == 5


def test_serialization_factor_is_wall_over_max_lat():
    sl = {"queen": [60.0]}
    b = sb.compute_run_breakdown(sl, 180.0)
    assert b["serialization_factor"] == 3.0   # 180 / 60


def test_zero_and_none_latencies_are_dropped():
    sl = {"db_read": [0.0, None], "judge": [None]}
    b = sb.compute_run_breakdown(sl, 50.0)
    # no positive latencies anywhere -> empty model, residual == wall
    assert b["max_lat"] == 0.0
    assert b["serialization_factor"] is None
    assert b["residual"] == 50.0
    assert b["n_calls"] == 0


def test_stage_shares_rank_and_sum_to_one():
    runs = [
        {"judge": [20.0, 20.0], "converge": [40.0]},   # judge 40, converge 40
        {"judge": [20.0], "specify": [40.0]},           # judge 20, specify 40
    ]
    shares = sb.stage_shares(runs)
    by = {r["stage"]: r for r in shares}
    assert by["judge"]["sum_s"] == 60.0 and by["judge"]["calls"] == 3
    assert by["converge"]["sum_s"] == 40.0
    assert by["specify"]["sum_s"] == 40.0
    assert abs(sum(r["share"] for r in shares) - 1.0) < 1e-9
    # descending by sum_s
    assert [r["sum_s"] for r in shares] == sorted((r["sum_s"] for r in shares), reverse=True)


def test_analyze_end_to_end_on_synthetic_db(tmp_path):
    db = tmp_path / "ledger.db"
    con = sqlite3.connect(db)
    con.execute("create table runs (id integer primary key, elapsed_s real)")
    con.execute("create table worker_calls (run_id integer, stage text, latency_s real)")
    # run 1: wall 300, slowest single call 60 -> factor 5.0
    con.execute("insert into runs values (1, 300.0)")
    for s, l in [("queen", 60.0), ("judge", 20.0), ("judge", 20.0), ("converge", 40.0)]:
        con.execute("insert into worker_calls values (1, ?, ?)", (s, l))
    # run 2 (smoke): wall 30 -> excluded by default min_wall=120
    con.execute("insert into runs values (2, 30.0)")
    con.execute("insert into worker_calls values (2, 'queen', 10.0)")
    con.commit(); con.close()

    res = sb.analyze(str(db), min_wall=120.0)
    assert res["cycles"] == 1                       # smoke run excluded
    assert res["median_serialization_factor"] == 5.0
    # modelpath = queen 60 + max(judge)=20 + converge 40 = 120; residual frac = (300-120)/300 = .6
    assert abs(res["median_modelpath_fit_err"] - 0.6) < 1e-9
    assert res["stage_shares"][0]["stage"] in {"queen", "converge"}  # both 60s/40s heavy


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
