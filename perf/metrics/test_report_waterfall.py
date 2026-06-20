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


# ── R0021-1: latest-N cap ───────────────────────────────────────────────────

def test_per_run_views_capped_to_latest_n():
    """With more runs than the limit, the per-run table/tabs show only the latest
    N — older runs drop out so the report stays readable (R0021-1)."""
    n_total = report.RECENT_RUNS_LIMIT + 6
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(n_total)]
    html = report.render(runs)
    # one tab per shown run, capped at the limit (not n_total)
    assert html.count('class="run-tab"') == report.RECENT_RUNS_LIMIT
    # the oldest runs are gone, the newest are present
    assert ">run0 " not in html and "run0<" not in html
    assert f"run{n_total - 1}" in html


def test_header_keeps_true_total_and_flags_truncation():
    """The header reports the TRUE total run count and says it truncated — silent
    truncation would read as 'this is everything' (R0021-1 honesty rule)."""
    n_total = report.RECENT_RUNS_LIMIT + 3
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(n_total)]
    html = report.render(runs)
    assert f"전체 {n_total}런" in html
    assert f"최신 {report.RECENT_RUNS_LIMIT}런만 표시" in html


def test_no_truncation_note_when_within_limit():
    """At or below the limit there is no truncation and no '최신 N런만' note."""
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(3)]
    html = report.render(runs)
    assert "전체 3런" in html
    assert "런만 표시" not in html


# ── R0021-2: cost = operator actual or 미계측, never the fabricated estimate ──

def test_cost_unmeasured_without_actual():
    """No ``cost.actual_usd`` → the run cost is 미계측, and the fabricated local
    estimate is NOT presented as the bottom-line cost (R0021-2)."""
    # _RUN's by_provider usd is 0.0 (fabricated). Without an actual, the summary
    # cost card must read 미계측, not a $ figure.
    d = report.derive(_RUN)
    assert d["usd"] is None
    summary = report.section_summary(_RUN, d)
    assert "미계측" in summary
    # the cost table bottom line is 미계측 too, and tells the operator how to fill it
    tbl = report.section_cost_table(_RUN, d)
    assert "미계측" in tbl and "--actual-usd" in tbl


def test_cost_actual_is_shown_when_entered():
    """When ``cost.actual_usd`` is present it becomes the displayed cost, with its
    source, across the summary card and cost table (R0021-2)."""
    run = dict(_RUN, cost={
        "by_provider": {"copilot": {"tokens": 100, "calls": 1, "credits": 0, "usd": 0.0}},
        "actual_usd": 0.42, "actual_source": "copilot dashboard 2026-06-20",
    })
    d = report.derive(run)
    assert d["usd"] == 0.42
    summary = report.section_summary(run, d)
    assert "$0.420" in summary
    tbl = report.section_cost_table(run, d)
    assert "$0.420" in tbl and "copilot dashboard 2026-06-20" in tbl


def test_cost_bars_degrade_to_unmeasured_note():
    """The cost-bars chart never fakes $0 bars: with no actual cost on any run it
    degrades to an honest 미계측 note (R0021-2)."""
    runs = [dict(_RUN, run_id=f"run{i}") for i in range(3)]
    derived = [report.derive(r) for r in runs]
    out = report.svg_cost_bars(runs, derived)
    assert "미계측" in out and "<rect" not in out


# ── R0025: chart labels — diagonal + trimmed (no overlap) ────────────────────

def test_short_run_id_keeps_runxxx_trims_solo():
    """runxxx ids stay verbatim; solo- arm-ids drop the redundant prefix to the
    recognisable core (R0001: 'sonnet45-0077 이런식으로 해도 잘 알아본다')."""
    assert report._short_run_id({"run_id": "run475"}) == "run475"
    assert report._short_run_id({"run_id": "solo-sonnet45-0077"}) == "sonnet45-0077"
    assert report._short_run_id({"run_id": "solo-gpt54mini-0082"}) == "gpt54mini-0082"


def test_run_label_does_not_duplicate_arm():
    """A solo run already encodes its arm in the (trimmed) id, so the label must
    not append it a second time (R0025)."""
    run = {"run_id": "solo-sonnet45-0077", "arm": "solo-sonnet45"}
    assert report._run_label(run) == "sonnet45-0077"


def test_chart_x_labels_are_diagonal_and_trimmed():
    """The trend chart draws each x label on a diagonal (rotate transform) using
    the trimmed id — never the long raw run_id horizontally (R0001 overlap fix)."""
    runs = [{"run_id": "solo-gpt54mini-0082", "arm": "solo-gpt54mini",
             "golden": {"seeded": 1, "recalled": 1}, "cost": {}, "cycle": {},
             "funnel": {}}]
    out = report.svg_lines(runs, [("재현율", lambda r: 1.0, "#54c7a3", True)])
    assert 'class="x-rot"' in out and "rotate(-32" in out
    # the trimmed core is shown, not the long raw id
    assert ">gpt54mini-0082<" in out and "solo-gpt54mini-0082" not in out


def test_cost_bars_x_labels_are_diagonal():
    """Cost bars share the same diagonal, trimmed axis as the trend chart."""
    runs = [{"run_id": "solo-sonnet45-0077",
             "cost": {"actual_usd": 0.01}, "cycle": {}, "golden": {}, "funnel": {}}]
    derived = [report.derive(r) for r in runs]
    out = report.svg_cost_bars(runs, derived)
    assert 'class="x-rot"' in out and "rotate(-32" in out
    assert ">sonnet45-0077<" in out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
