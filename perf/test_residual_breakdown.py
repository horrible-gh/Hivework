#!/usr/bin/env python3
"""Unit tests for perf/residual_breakdown.py."""
from __future__ import annotations
import os, sqlite3, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residual_breakdown as rb  # noqa: E402


def test_local_parallel_rows_contribute_by_stage_critical_path(tmp_path):
    db = tmp_path / "ledger.db"
    con = sqlite3.connect(db)
    con.execute("create table runs (id integer primary key, elapsed_s real)")
    con.execute(
        "create table worker_calls (run_id integer, stage text, provider text, latency_s real)"
    )
    con.execute("insert into runs values (1, 130.0)")
    rows = [
        ("queen", "copilot", 50.0),
        ("judge", "copilot", 60.0),
        ("judge", "copilot", 60.0),
        ("judge_prep", "local", 5.0),
        ("judge_prep", "local", 7.0),
        ("axis_plan", "local", 3.0),
    ]
    for stage, provider, lat in rows:
        con.execute(
            "insert into worker_calls values (1, ?, ?, ?)", (stage, provider, lat)
        )
    con.commit()
    con.close()

    res = rb.breakdown(str(db), min_wall=0)
    run = res["per_run"][0]

    assert run["modelpath_s"] == 120.0
    assert run["residual_now_s"] == 10.0
    assert run["local_critical_attributed_s"] == 10.0
    assert run["serial_local_attributed_s"] == 3.0
    assert run["residual_before_instr_s"] == 20.0
    assert run["residual_explained_pct"] == 50.0
    assert run["local_critical_by_stage"] == {"axis_plan": 3.0, "judge_prep": 7.0}
