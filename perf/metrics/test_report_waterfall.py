#!/usr/bin/env python3
"""Regression tests for the per-run waterfall (hivework.0017, T0006 / CH0005).

The headline contract is an HONESTY rule the chat hammered on: a stage that
isn't instrumented yet renders as 미계측 (an honest blank), NEVER as 0%. The two
mean opposite things — 0% reads as "the signal died in this stage (this stage
cost us performance)", 미계측 reads as "we haven't measured this stage yet". If
the renderer ever paints an unmeasured cell as 0% the report lies about where the
pipeline leaks, so these tests pin that boundary.

Also covered: every run is a clickable panel (per-run breakdown the chat asked
for), the funnel comb counts surface as real numbers, and a future
``stage_golden`` block flows real % into the previously-blank cells.

Run with: python -m pytest perf/metrics/test_report_waterfall.py
       or: python perf/metrics/test_report_waterfall.py   (no pytest needed)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report  # noqa: E402

# A minimal run with golden scoring but NO per-stage instrumentation — the
# common case today. Intermediate stages must be 미계측, never 0%.
_RUN = {
    "run_id": "runT", "ts": "2026-06-20T00:00:00+09:00", "seed": "0082",
    "codebase": "FlowGate-dev", "models": {"queen": "gpt-5-mini", "swarm": "gpt-5-mini"},
    "funnel": {"axes_attempted": 6, "comb_fired": 7, "comb_shaped": 6,
               "conclusion_converted": 1, "submitted": 1, "passed": 0},
    "cost": {"by_provider": {"copilot": {"tokens": 100, "calls": 1, "credits": 0, "usd": 0.0}}},
    "cycle": {"fixes_landed": 0, "fixes_total": 0, "wall_clock_s": 10.0},
    "golden": {"seeded": 1, "recalled": 1, "false_positives": 0, "verified_fixed": 0,
               "per_bug": [{"id": "0082", "level": 3, "found": True, "fixed": False}]},
}

# Cells the renderer leaves blank should carry this honest glyph, never "0%".
_UNMEASURED = "미계측"


def _panel(run):
    return report.section_waterfall_panel(run, report.derive(run))


def test_unmeasured_stage_is_blank_not_zero_percent():
    """The CENTRAL rule: an un-instrumented stage renders 미계측, never 0%."""
    panel = _panel(_RUN)
    # The intermediate stages (decompose/retrieve/judge/converge) have no golden
    # instrumentation, so their golden cell must be the honest blank.
    assert _UNMEASURED in panel
    # No unmeasured span may ever contain a fabricated 0%.
    for span in re.findall(r'<span class="unmeasured"[^>]*>(.*?)</span>', panel):
        assert "0%" not in span, f"unmeasured cell faked a score: {span!r}"


def test_measured_funnel_counts_are_real_numbers():
    """The measured column shows the actual comb counts from the funnel."""
    panel = _panel(_RUN)
    # comb_fired=7 and comb_shaped=6 are real, bolded counts.
    assert "<b>7</b>" in panel and "<b>6</b>" in panel


def test_terminal_recall_is_scored_not_blank():
    """honey is the scoring point — recall is a real % there, not 미계측."""
    panel = _panel(_RUN)
    # 100% recall (1/1) appears as a scored cell tagged 채점지점.
    assert "100%" in panel and "채점지점" in panel


def test_coordinator_count_is_unmeasured_not_zero():
    """coordinator has no funnel counterpart → its count cell is blank, not 0."""
    panel = _panel(_RUN)
    # The first waterfall row (coordinator) must not render a literal count of 0.
    first_row = panel.split("</tr>")[1]  # [0] is the <thead> row
    assert "coordinator" in first_row
    assert _UNMEASURED in first_row


def test_stage_golden_block_flows_real_pct_into_blanks():
    """Forward contract: when instrumentation lands a stage_golden block, the
    previously-blank judge cell fills with a real %, not 미계측 (T0006:
    "앞으로는 각 신호마다 볼수있게 측정")."""
    run = dict(_RUN)
    run["stage_golden"] = {"judge": {"alive": 1, "of": 1,
                                     "per_bug": {"0082": 1}}}
    panel = _panel(run)
    # The judge stage now carries a measured 100% (1/1) rather than the blank.
    assert '<span class="ok">100%</span> <span class="muted">(1/1)</span>' in panel
    # And the per-signal matrix marks 0082 as 생존 at the judge column.
    assert "생존" in panel


def test_every_run_is_a_clickable_panel():
    """The tabbed section emits one radio + one tab + one panel per run, with the
    latest selected — the per-run breakdown CH0005 asked for."""
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(3)]
    derived = [report.derive(r) for r in runs]
    html = report.section_runs_tabbed(runs, derived)
    assert html.count('class="run-tab"') == 3
    assert html.count('class="run-panel"') == 3
    # latest (index 2) is checked by default
    assert 'id="rs-2" class="run-radio" checked' in html


def test_full_render_has_no_unmeasured_zero_percent():
    """End-to-end: across a realistic multi-run report, not a single unmeasured
    cell is painted as 0% (the report never lies about a leak)."""
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(4)]
    html = report.render(runs)
    for span in re.findall(r'<span class="unmeasured"[^>]*>(.*?)</span>', html):
        assert "0%" not in span


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
