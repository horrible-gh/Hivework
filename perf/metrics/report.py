#!/usr/bin/env python3
"""Render a self-contained HTML performance report from ``runs.jsonl``.

Design constraints (agreed in CH0002):
  - Lightweight, local-first. NO FastAPI, NO template engine, NO server.
  - Stdlib only. Charts are inline <svg> (zero CDN, zero network) so the
    output is one self-contained .html openable by double-click (file://),
    even fully offline.
  - Input is assumed already produced: one JSON object per line (see SCHEMA.md).

The report answers the chat's three questions, in priority order:
  (1) result  — passed fixes (the north star) + golden-set accuracy
  (2) why     — the harvest funnel (where the pipeline collapsed this cycle)
  (3) cost    — per-provider spend and cost-per-yield / cost-per-fix

Usage:
    python report.py runs.jsonl -o report.html
    python report.py --demo            # render the bundled runs.sample.jsonl
    python report.py runs.jsonl --open # render then open in the browser
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from html import escape

# ─────────────────────────────────────────────────────────────────────────────
# Loading + derivation
# ─────────────────────────────────────────────────────────────────────────────

# The six funnel stages, top (widest) to bottom (narrowest). Counts are
# expected to decrease monotonically but we never assume it.
FUNNEL_STAGES = [
    ("axes_attempted", "축 시도"),
    ("comb_fired", "comb 발화"),
    ("comb_shaped", "comb-형태 (진짜 finding)"),
    ("conclusion_converted", "결론 전환"),
    ("submitted", "제출"),
    ("passed", "통과"),
]

# Honest blank. An UNMEASURED cell means "this signal isn't instrumented yet",
# which is the OPPOSITE of a measured 0% ("signal died here"). CH0005 is explicit:
# never paint 미계측 as 0% — that would let the report lie about where the
# pipeline leaks. So unmeasured cells render this glyph (grey), and only become a
# real number once instrumentation fills the slot.
UNMEASURED = "—"

# The canonical hive pipeline as a run-level waterfall (chat diagram, CH0005),
# top → bottom. Each stage carries:
#   funnel_key  — the measured comb count that survives to this stage, or None
#                 (coordinator has no funnel counterpart → 미계측, not a fake 0).
#   golden_attr — the per-stage golden-signal-survival slot. None for every
#                 intermediate stage today: no stage emits a "골든 축 살아있음"
#                 flag yet, so recall is only scored once at the end. Those cells
#                 render 미계측 (honest blank). The two terminal scoring points
#                 (honey=recall, apply=fixed) carry real numbers. When future
#                 instrumentation lands a ``stage_golden`` block (SCHEMA.md), the
#                 renderer fills real % into the blanks — same skeleton, data
#                 flows in. This is "앞으로는 각 신호마다 볼수있게 측정" (T0006).
WATERFALL_STAGES = [
    ("coordinator", "coordinator · 기대 추출 (선택)",  None,                   None),
    ("decompose",   "decompose · 축 분해",             "axes_attempted",       None),
    ("retrieve",    "retrieve/fan-out · 증거 수집",     "comb_fired",           None),
    ("judge",       "judge/gate · comb-형태 판정",      "comb_shaped",          None),
    ("converge",    "converge · 봉합 + 인과검증",        "conclusion_converted", None),
    ("honey",       "honey · 정본 조립 (재현율 채점)",   "submitted",            "recall"),
    ("apply",       "specify/apply · 수정 → 통과",       "passed",               "fixed"),
]


def load_runs(path):
    """Read a JSONL file into a list of run dicts, sorted by timestamp.

    Blank lines are skipped; a malformed line is reported to stderr and
    dropped rather than aborting the whole report.
    """
    runs = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{lineno}: skipping malformed line ({exc})",
                      file=sys.stderr)
    runs.sort(key=lambda r: str(r.get("ts", "")))
    return runs


def _num(d, *keys, default=0):
    """Safely walk nested dict keys, coercing the leaf to a number."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if isinstance(cur, (int, float)):
        return cur
    return default


def derive(run):
    """Compute the derived metrics the report needs (never stored in jsonl)."""
    axes = _num(run, "funnel", "axes_attempted")
    shaped = _num(run, "funnel", "comb_shaped")
    landed = _num(run, "cycle", "fixes_landed")
    total_fix = _num(run, "cycle", "fixes_total")

    usd = 0.0
    for prov in (run.get("cost", {}).get("by_provider", {}) or {}).values():
        usd += float(prov.get("usd", 0) or 0)

    golden = run.get("golden") or {}
    seeded = float(golden.get("seeded", 0) or 0)
    recalled = float(golden.get("recalled", 0) or 0)
    fp = float(golden.get("false_positives", 0) or 0)
    verified = float(golden.get("verified_fixed", 0) or 0)

    return {
        "yield": (shaped / axes) if axes else 0.0,
        "pass_rate": (landed / total_fix) if total_fix else 0.0,
        "usd": usd,
        "cost_per_finding": (usd / shaped) if shaped else None,
        "cost_per_fix": (usd / landed) if landed else None,
        "recall": (recalled / seeded) if seeded else None,
        "precision": (recalled / (recalled + fp)) if (recalled + fp) else None,
        "golden_fixed_rate": (verified / seeded) if seeded else None,
        "has_golden": bool(golden),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tiny inline-SVG chart helpers (no dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_usd(v):
    return f"${v:,.3f}" if v is not None else "—"


def _fmt_pct(v):
    return f"{v * 100:.0f}%" if v is not None else "—"


def svg_funnel(funnel):
    """Horizontal funnel: one bar per stage, width ∝ count, with drop-off %."""
    counts = [(label, int(_num({"_": funnel}, "_", key))) for key, label in FUNNEL_STAGES]
    top = max((c for _, c in counts), default=0) or 1
    row_h, gap, w, x0 = 34, 10, 560, 220
    rows = []
    prev = None
    for i, (label, c) in enumerate(counts):
        y = i * (row_h + gap)
        bw = max(2, int(w * c / top))
        # Conversion vs the previous stage — the number that exposes the leak.
        conv = ""
        if prev is not None:
            pct = (c / prev * 100) if prev else 0
            conv = f"{pct:.0f}%"
        prev = c
        rows.append(
            f'<text x="{x0 - 12}" y="{y + row_h * 0.66}" class="fn-lab">{escape(label)}</text>'
            f'<rect x="{x0}" y="{y}" width="{bw}" height="{row_h}" rx="4" class="fn-bar"/>'
            f'<text x="{x0 + bw + 8}" y="{y + row_h * 0.66}" class="fn-val">{c}'
            f'{f"  ({conv})" if conv else ""}</text>'
        )
    h = len(counts) * (row_h + gap)
    return f'<svg viewBox="0 0 {x0 + w + 90} {h}" class="chart" role="img">{"".join(rows)}</svg>'


def svg_lines(runs, series, height=200):
    """Multi-series line chart over runs. ``series`` = [(label, fn, color, is_pct)]."""
    n = len(runs)
    if n == 0:
        return '<p class="muted">데이터 없음</p>'
    w, h = 640, height
    pad_l, pad_b, pad_t, pad_r = 44, 28, 14, 14
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    # Each series is normalised independently against its own max so percentage
    # and dollar series share one frame without one flattening the other.
    def xpos(i):
        return pad_l + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    parts = []
    # horizontal gridlines
    for g in range(5):
        gy = pad_t + plot_h * g / 4
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" class="grid"/>')

    legend = []
    for label, fn, color, is_pct in series:
        vals = [fn(r) for r in runs]
        vmax = max([v for v in vals if v is not None] + [0.0001])
        pts = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            x = xpos(i)
            y = pad_t + plot_h * (1 - v / vmax)
            pts.append((x, y, v))
        if pts:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y, v in pts:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
            tip = _fmt_pct(pts[-1][2]) if is_pct else (_fmt_usd(pts[-1][2]))
            legend.append(f'<span class="lg"><i style="background:{color}"></i>{escape(label)} '
                          f'<b>{tip}</b></span>')

    # x labels (run ids)
    for i, r in enumerate(runs):
        x = xpos(i)
        parts.append(f'<text x="{x:.1f}" y="{h - 8}" class="x-lab">'
                     f'{escape(str(r.get("run_id", i)))}</text>')

    svg = f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(parts)}</svg>'
    return svg + f'<div class="legend">{"".join(legend)}</div>'


def svg_cost_bars(runs, height=180):
    """Grouped per-run stacked bars: copilot + deepinfra (+ any other) USD."""
    n = len(runs)
    if n == 0:
        return '<p class="muted">데이터 없음</p>'
    # collect provider set + palette
    palette = {"copilot": "#7c9cff", "deepinfra": "#54c7a3", "local": "#888"}
    extra = ["#d98cff", "#f0a85f", "#e06c75"]
    provs = []
    for r in runs:
        for p in (r.get("cost", {}).get("by_provider", {}) or {}):
            if p not in provs:
                provs.append(p)
    for p in provs:
        palette.setdefault(p, extra[provs.index(p) % len(extra)])

    w, h = 640, height
    pad_l, pad_b, pad_t, pad_r = 44, 28, 14, 14
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    totals = [sum(float(pp.get("usd", 0) or 0)
                  for pp in (r.get("cost", {}).get("by_provider", {}) or {}).values())
              for r in runs]
    vmax = max(totals + [0.0001])
    bw = min(48, plot_w / n * 0.6)
    parts = []
    for g in range(5):
        gy = pad_t + plot_h * g / 4
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" class="grid"/>')
    for i, r in enumerate(runs):
        cx = pad_l + plot_w * (i + 0.5) / n
        by = r.get("cost", {}).get("by_provider", {}) or {}
        y = pad_t + plot_h
        for p in provs:
            v = float((by.get(p) or {}).get("usd", 0) or 0)
            bh = plot_h * v / vmax
            y -= bh
            parts.append(f'<rect x="{cx - bw/2:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                         f'height="{bh:.1f}" fill="{palette[p]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t + plot_h + 16}" class="x-lab">'
                     f'{escape(str(r.get("run_id", i)))}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y - 4:.1f}" class="bar-val">'
                     f'{_fmt_usd(totals[i])}</text>')
    legend = "".join(f'<span class="lg"><i style="background:{palette[p]}"></i>{escape(p)}</span>'
                     for p in provs)
    svg = f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(parts)}</svg>'
    return svg + f'<div class="legend">{legend}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# HTML sections
# ─────────────────────────────────────────────────────────────────────────────

def card(value, label, sub=""):
    sub_html = f'<div class="card-sub">{escape(sub)}</div>' if sub else ""
    return (f'<div class="card"><div class="card-val">{value}</div>'
            f'<div class="card-lab">{escape(label)}</div>{sub_html}</div>')


def section_summary(latest, d):
    """North-star cards for the latest cycle, in priority order.

    R0014-D: the main accuracy metric is recall/precision (정확도), not yield.
    Yield is demoted to an auxiliary signal living in the harvest funnel section
    (section_funnel / section_axes) and no longer occupies a north-star card.
    """
    cyc = latest.get("cycle", {})
    cards = [
        card(f'{int(_num(cyc, "fixes_landed"))}/{int(_num(cyc, "fixes_total"))}',
             "통과한 수정 (북극성)", f'통과율 {_fmt_pct(d["pass_rate"])}'),
    ]
    if d["has_golden"]:
        g = latest["golden"]
        recalled = int(g.get("recalled", 0))
        seeded = int(g.get("seeded", 0))
        fp = int(g.get("false_positives", 0))
        cards.append(card(
            _fmt_pct(d["recall"]),
            "재현율 (메인)", f'{recalled}/{seeded} 찾음'))
        cards.append(card(
            _fmt_pct(d["precision"]),
            "정밀도 (메인)", f'헛다리 {fp}건'))
    cards.append(card(_fmt_usd(d["usd"]),
                      "런 비용", f'수율당 {_fmt_usd(d["cost_per_finding"])}'))
    return '<div class="cards">' + "".join(cards) + "</div>"


def section_golden(latest):
    g = latest.get("golden")
    if not g:
        return ""
    rows = []
    for b in g.get("per_bug", []):
        found = b.get("found")
        fixed = b.get("fixed")
        f_cls = "ok" if found else "no"
        x_cls = "ok" if fixed else "no"
        rows.append(
            f'<tr><td class="lvl">Lv{b.get("level", "?")}</td>'
            f'<td>{escape(str(b.get("id", "")))}</td>'
            f'<td class="{f_cls}">{"찾음" if found else "놓침"}</td>'
            f'<td class="{x_cls}">{"통과" if fixed else "—"}</td></tr>')
    seeded = int(g.get("seeded", 0))
    recalled = int(g.get("recalled", 0))
    fp = int(g.get("false_positives", 0))
    fixed = int(g.get("verified_fixed", 0))
    recall = recalled / seeded if seeded else None
    precision = recalled / (recalled + fp) if (recalled + fp) else None
    summary = (f'<div class="cards small">'
               + card(_fmt_pct(recall), "재현율 (꼼꼼함)", f'{recalled}/{seeded} 찾음')
               + card(_fmt_pct(precision), "정밀도 (믿을만함)", f'헛다리 {fp}건')
               + card(str(fixed), "고쳐서 통과", "verified-fixed")
               + '</div>')
    return (
        '<section><h2>정확도 — 골든셋 채점 (NR0004 5버그)</h2>'
        '<p class="muted">"심은 N개 중 X개 찾음 · 헛다리 Y번 · 고쳐서 통과 Z개." '
        '레벨은 난이도 사다리(1 순수로직 → 5 비동기/SSE).</p>'
        + summary
        + '<table class="grid-tbl"><thead><tr><th>레벨</th><th>버그</th>'
          '<th>탐지</th><th>수정 통과</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></section>')


def section_cost_table(latest):
    """Per-provider cost table with the credit axis and the $ axis split out (R0001).

    Credit-billed providers (copilot) show their credit consumption; token-billed
    providers (deepinfra) show tokens. USD is the common bottom line so the two
    billing paradigms remain comparable without conflating credits and tokens.
    """
    by_prov = (latest.get("cost", {}) or {}).get("by_provider", {}) or {}
    if not by_prov:
        return ""
    rows = []
    tot_credits = tot_usd = 0.0
    for prov in sorted(by_prov):
        v = by_prov[prov] or {}
        credits = float(v.get("credits", 0) or 0)
        usd = float(v.get("usd", 0) or 0)
        calls = int(v.get("calls", 0) or 0)
        tokens = int(v.get("tokens", 0) or 0)
        tot_credits += credits
        tot_usd += usd
        # Billing paradigm is inferred from which axis carries the charge.
        if credits > 0 or (usd == 0 and tokens and calls):
            model, basis = "크레딧", f'{calls}회 호출'
        else:
            model, basis = "토큰", f'{tokens:,} tok'
        rows.append(
            f'<tr><td>{escape(prov)}</td><td class="muted-cell">{model}</td>'
            f'<td>{basis}</td>'
            f'<td>{credits:,.2f}</td>'
            f'<td>{_fmt_usd(usd)}</td></tr>')
    rows.append(
        f'<tr class="tot"><td>합계</td><td></td><td></td>'
        f'<td>{tot_credits:,.2f}</td><td>{_fmt_usd(tot_usd)}</td></tr>')
    return (
        '<table class="grid-tbl"><thead><tr><th>provider</th><th>과금</th>'
        '<th>기준</th><th>크레딧 (1cr=$0.01)</th><th>$</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table>'
        '<p class="muted">크레딧계(코파일럿)=호출수×크레딧단가 · '
        '토큰계(deepinfra/openai호환)=실토큰×단가. 두 축은 분리 집계된다.</p>')


def section_axes(latest):
    axes = latest.get("axes") or []
    if not axes:
        return ""
    rows = []
    for a in axes:
        fired = a.get("fired")
        shaped = int(a.get("shaped", 0) or 0)
        cls = "ok" if shaped > 0 else ("no" if not fired else "zero")
        state = "수확" if shaped > 0 else ("미발화" if not fired else "0")
        rows.append(
            f'<tr class="{cls}"><td>{escape(str(a.get("axis_id", "")))}</td>'
            f'<td>{escape(str(a.get("label", "")))}</td>'
            f'<td>{int(a.get("findings", 0) or 0)}</td>'
            f'<td>{shaped}</td>'
            f'<td>{float(a.get("longest_chain_s", 0) or 0):.1f}s</td>'
            f'<td class="{cls}">{state}</td></tr>')
    return (
        '<section><h2>축별 분해 — 어디서 수확했나</h2>'
        '<table class="grid-tbl"><thead><tr><th>축</th><th>대상</th>'
        '<th>findings</th><th>comb-형태</th><th>최장 체인</th><th>상태</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table></section>')


def _run_label(run):
    """Short human label for a run in pickers/tables (arm name if it has one)."""
    arm = run.get("arm")
    base = str(run.get("run_id", "?"))
    return f"{base} · {arm}" if arm else base


def section_run_table(runs, derived_all):
    """Flat per-run comparison: one row per run so solo-5mini sits beside the
    hybrid runs and "후지냐 좋냐" is answerable at a glance (CH0005).

    This is the per-run breakdown the chat asked for — the rest of the report
    details only the latest cycle, so without this table runs are invisible to
    each other. Accuracy columns lead (R0014-D), cost trails.
    """
    rows = []
    for run, d in zip(runs, derived_all):
        cyc = run.get("cycle", {})
        rows.append(
            f'<tr><td>{escape(_run_label(run))}</td>'
            f'<td class="muted-cell">{escape(str(run.get("codebase", "")))}</td>'
            f'<td>{_fmt_pct(d["recall"])}</td>'
            f'<td>{_fmt_pct(d["precision"])}</td>'
            f'<td>{int(_num(cyc, "fixes_landed"))}/{int(_num(cyc, "fixes_total"))}</td>'
            f'<td>{_fmt_usd(d["usd"])}</td></tr>')
    return (
        '<section><h2>런별 비교 — 한 줄에 한 런</h2>'
        '<p class="muted">solo 단독 arm을 하이브리드 런 옆에 나란히. '
        '재현율·정밀도가 메인, 비용은 뒤(R0014-D).</p>'
        '<table class="grid-tbl"><thead><tr><th>런</th><th>대상</th>'
        '<th>재현율</th><th>정밀도</th><th>통과</th><th>런당 $</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></section>')


def _stage_count_cell(run, funnel_key):
    """Measured comb count surviving to a stage — or an honest blank when the
    stage has no funnel counterpart (never a fabricated 0)."""
    if funnel_key is None:
        return f'<span class="unmeasured" title="아직 계측 안 됨">{UNMEASURED} 미계측</span>'
    return f'<b>{int(_num(run.get("funnel", {}), funnel_key))}</b>'


def _stage_golden_cell(run, stage_key, golden_attr, d):
    """Per-stage golden-signal survival.

    Precedence: (1) a ``stage_golden`` block if future instrumentation filled it
    for this stage; (2) the two terminal scoring points (honey→recall,
    apply→fixed); (3) otherwise an honest 미계측 blank. CRITICALLY this never
    returns 0% for an unmeasured stage — 미계측 ≠ 0 (CH0005)."""
    sg = (run.get("stage_golden") or {}).get(stage_key)
    if isinstance(sg, dict) and sg.get("of"):
        alive, of = int(sg.get("alive", 0) or 0), int(sg["of"])
        rate = alive / of if of else None
        return f'<span class="ok">{_fmt_pct(rate)}</span> <span class="muted">({alive}/{of})</span>'
    if golden_attr == "recall" and d.get("recall") is not None:
        return f'<span class="ok">{_fmt_pct(d["recall"])}</span> <span class="muted">(채점지점)</span>'
    if golden_attr == "fixed" and d.get("golden_fixed_rate") is not None:
        return f'<span class="ok">{_fmt_pct(d["golden_fixed_rate"])}</span> <span class="muted">(채점지점)</span>'
    return f'<span class="unmeasured" title="아직 계측 안 됨 — 0%가 아님">{UNMEASURED} 미계측</span>'


def section_waterfall_panel(run, d):
    """One run's pipeline waterfall: where did the golden signal leak?

    Two columns per stage: the MEASURED comb count (real, from the funnel) and
    the per-stage golden survival (mostly 미계측 today — only the terminal
    scoring points carry a real number). Below it, a per-signal × per-stage
    matrix so each golden bug is tracked across the flow (T0006: "각 신호마다
    보여주되 현재 없는 부분은 없음같이 표현").
    """
    wf_rows = []
    for key, label, funnel_key, golden_attr in WATERFALL_STAGES:
        wf_rows.append(
            f'<tr><td class="wf-stage">{escape(label)}</td>'
            f'<td class="wf-count">{_stage_count_cell(run, funnel_key)}</td>'
            f'<td class="wf-gold">{_stage_golden_cell(run, key, golden_attr, d)}</td></tr>')
    waterfall = (
        '<table class="grid-tbl wf-tbl"><thead><tr><th>파이프라인 단계</th>'
        '<th>살아남은 comb (실측)</th><th>골든 신호 생존</th></tr></thead><tbody>'
        + "".join(wf_rows) + '</tbody></table>')

    # Per-signal × per-stage matrix. Only the honey (scored) column has data per
    # bug today; every earlier stage is an honest blank awaiting instrumentation.
    matrix = ""
    g = run.get("golden") or {}
    per_bug = g.get("per_bug") or []
    if per_bug:
        head = "".join(f'<th>{escape(lbl.split(" · ")[0])}</th>'
                       for _k, lbl, _f, _ga in WATERFALL_STAGES)
        m_rows = []
        sg = run.get("stage_golden") or {}
        for b in per_bug:
            cells = []
            for key, _lbl, _fk, golden_attr in WATERFALL_STAGES:
                per = (sg.get(key) or {}).get("per_bug") if isinstance(sg.get(key), dict) else None
                if isinstance(per, dict) and b.get("id") in per:
                    alive = per[b["id"]]
                    cells.append(f'<td class="{"ok" if alive else "no"}">'
                                 f'{"생존" if alive else "누락"}</td>')
                elif golden_attr == "recall":
                    found = b.get("found")
                    cells.append(f'<td class="{"ok" if found else "no"}">'
                                 f'{"찾음" if found else "놓침"}</td>')
                elif golden_attr == "fixed":
                    fixed = b.get("fixed")
                    cells.append(f'<td class="{"ok" if fixed else "no"}">'
                                 f'{"통과" if fixed else UNMEASURED}</td>')
                else:
                    cells.append(f'<td class="unmeasured">{UNMEASURED}</td>')
            m_rows.append(
                f'<tr><td class="lvl">Lv{b.get("level", "?")}</td>'
                f'<td>{escape(str(b.get("id", "")))}</td>' + "".join(cells) + '</tr>')
        matrix = (
            '<p class="muted" style="margin-top:14px">신호별 × 단계별 — 각 골든 버그가 '
            '어느 칸까지 살아남았나. 회색 칸은 0%가 아니라 <b>미계측</b>(아직 그 단계를 안 잼).</p>'
            '<div class="wf-matrix-scroll"><table class="grid-tbl wf-matrix">'
            f'<thead><tr><th>Lv</th><th>버그</th>{head}</tr></thead><tbody>'
            + "".join(m_rows) + '</tbody></table></div>')
    return waterfall + matrix


def section_runs_tabbed(runs, derived_all):
    """Clickable per-run waterfall. Zero-JS: radio inputs + ``:checked ~`` CSS
    toggle which panel shows, so the report stays a self-contained static file
    (CH0002 — no server, openable via file://). Latest run is selected by
    default. "런 하나 누르면 저 플로가 그려진다" (CH0005)."""
    if not runs:
        return ""
    n = len(runs)
    sel = n - 1  # latest run open by default
    inputs = "".join(
        f'<input type="radio" name="runsel" id="rs-{i}" class="run-radio"'
        f'{" checked" if i == sel else ""}>' for i in range(n))
    tabs = "".join(
        f'<label for="rs-{i}" class="run-tab">{escape(_run_label(r))}</label>'
        for i, r in enumerate(runs))
    panels = "".join(
        f'<div class="run-panel" id="rp-{i}">{section_waterfall_panel(r, d)}</div>'
        for i, (r, d) in enumerate(zip(runs, derived_all)))
    # Per-index toggle rules generated for the actual run count (zero-JS).
    css = "".join(
        f"#rs-{i}:checked~.run-panels #rp-{i}{{display:block}}"
        f"#rs-{i}:checked~.run-tabs label[for='rs-{i}']"
        f"{{background:var(--ok);color:#0f1115;border-color:var(--ok)}}" for i in range(n))
    return (
        '<section class="runs-tabbed"><h2>런별 워터폴 — 어느 칸에서 새는지</h2>'
        '<p class="muted">런을 누르면 그 런의 파이프라인(decompose→retrieve→judge→'
        'converge→honey→apply)이 열린다. 실측 칸은 진짜 comb 수, '
        '<b>회색 \'미계측\'</b>은 0%가 아니라 아직 그 칸을 안 잰 빈 슬롯이다 — '
        '계측이 박히면 그 자리에 진짜 %가 흘러든다(CH0005·T0006).</p>'
        f'<style>{css}</style>{inputs}'
        f'<div class="run-tabs">{tabs}</div>'
        f'<div class="run-panels">{panels}</div></section>')


def render(runs):
    if not runs:
        body = '<section><p class="muted">runs.jsonl 에 레코드가 없습니다.</p></section>'
        return PAGE.format(generated="", body=body)

    latest = runs[-1]
    d = derive(latest)
    derived_all = [derive(r) for r in runs]

    # Trend series (R0014-D): accuracy metrics lead — recall/precision/pass-rate.
    # Yield is demoted to the funnel section and no longer drawn on the main trend.
    trend = svg_lines(runs, [
        ("재현율", lambda r: derive(r)["recall"], "#54c7a3", True),
        ("정밀도", lambda r: derive(r)["precision"], "#d98cff", True),
        ("통과율", lambda r: derive(r)["pass_rate"], "#7c9cff", True),
    ])
    cost_trend = svg_lines(runs, [
        ("런당 $", lambda r: derive(r)["usd"], "#f0a85f", False),
    ], height=160)

    meta = (f'{escape(str(latest.get("run_id", "")))} · '
            f'{escape(str(latest.get("seed", "")))} · '
            f'{escape(str(latest.get("codebase", "")))} · '
            f'queen={escape(str(_num({"_": latest.get("models", {})}, "_", "queen") or latest.get("models", {}).get("queen", "")))} '
            f'swarm={escape(str(latest.get("models", {}).get("swarm", "")))}')

    body = (
        f'<header><h1>Hive 성능지표 레포트</h1>'
        f'<p class="meta">최신 사이클: {meta}<br>'
        f'전체 {len(runs)}런 · 마지막 ts {escape(str(latest.get("ts", "")))}</p></header>'
        + section_summary(latest, d)
        + section_run_table(runs, derived_all)
        + section_runs_tabbed(runs, derived_all)
        + '<section><h2>수확 퍼널 — 이번 사이클은 어디서 무너졌나</h2>'
        + f'<p class="muted">축 시도에서 통과까지. 괄호 안은 직전 단계 대비 전환율. '
        f'수확 수율(comb-형태/축) = <b>{_fmt_pct(d["yield"])}</b> '
        f'({int(_num(latest.get("funnel", {}), "comb_shaped"))} / '
        f'{int(_num(latest.get("funnel", {}), "axes_attempted"))} 축) — R0014-D로 메인 카드에서 이 퍼널 보조지표로 강등.</p>'
        + svg_funnel(latest.get("funnel", {})) + '</section>'
        + section_golden(latest)
        + '<section><h2>추세 — 런 누적</h2>'
        + '<p class="muted">재현율·정밀도·통과율(좌) / 각 시리즈는 자기 최대값 기준 정규화. (수율은 퍼널 섹션으로 이동 — R0014-D)</p>'
        + trend + '</section>'
        + '<section><h2>비용 분해 — provider별 (크레딧계 / 토큰계 분리)</h2>'
        + section_cost_table(latest)
        + svg_cost_bars(runs)
        + '<p class="muted">런당 총비용 추세:</p>' + cost_trend
        + f'<p class="muted">최신 런: 수율당 {_fmt_usd(d["cost_per_finding"])} · '
        f'수정당 {_fmt_usd(d["cost_per_fix"])}</p></section>'
        + section_axes(latest)
        + '<footer>perf/metrics/report.py · runs.jsonl 기반 · 자기완결 정적 HTML (서버·CDN 없음)</footer>'
    )
    return PAGE.format(generated=escape(str(latest.get("ts", ""))), body=body)


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hive 성능지표 레포트</title>
<style>
 :root {{ --bg:#0f1115; --panel:#181b22; --line:#2a2f3a; --ink:#e6e9ef; --muted:#8b93a3;
         --ok:#54c7a3; --no:#e06c75; --zero:#caa84a; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
        font:14px/1.5 -apple-system,Segoe UI,Roboto,'Malgun Gothic',sans-serif; }}
 .wrap {{ max-width:840px; margin:0 auto; padding:28px 20px 60px; }}
 h1 {{ font-size:22px; margin:0 0 4px; }}
 h2 {{ font-size:16px; margin:0 0 6px; border-left:3px solid var(--ok); padding-left:9px; }}
 header .meta, .meta {{ color:var(--muted); font-size:12.5px; margin:0; }}
 section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:16px 18px; margin:16px 0; }}
 .muted {{ color:var(--muted); font-size:12.5px; margin:4px 0 10px; }}
 .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin:14px 0; }}
 .cards.small .card {{ flex:1 1 120px; }}
 .card {{ flex:1 1 150px; background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:12px 14px; }}
 .card-val {{ font-size:22px; font-weight:700; }}
 .card-lab {{ color:var(--muted); font-size:12px; margin-top:2px; }}
 .card-sub {{ color:var(--muted); font-size:11.5px; margin-top:4px; opacity:.85; }}
 .chart {{ width:100%; height:auto; display:block; }}
 .grid {{ stroke:var(--line); stroke-width:1; }}
 .fn-bar {{ fill:var(--ok); opacity:.85; }}
 .fn-lab {{ fill:var(--muted); font-size:12px; text-anchor:end; }}
 .fn-val {{ fill:var(--ink); font-size:12px; }}
 .x-lab {{ fill:var(--muted); font-size:10.5px; text-anchor:middle; }}
 .bar-val {{ fill:var(--muted); font-size:10px; text-anchor:middle; }}
 .legend {{ display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }}
 .lg {{ color:var(--muted); font-size:12px; }}
 .lg i {{ display:inline-block; width:10px; height:10px; border-radius:2px;
         margin-right:5px; vertical-align:middle; }}
 .lg b {{ color:var(--ink); font-weight:600; }}
 table.grid-tbl {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }}
 .grid-tbl th {{ text-align:left; color:var(--muted); font-weight:600;
                border-bottom:1px solid var(--line); padding:6px 8px; }}
 .grid-tbl td {{ padding:6px 8px; border-bottom:1px solid var(--line); }}
 .grid-tbl td.ok, .grid-tbl .ok {{ color:var(--ok); }}
 .grid-tbl td.no, .grid-tbl .no {{ color:var(--no); }}
 .grid-tbl td.zero, .grid-tbl .zero {{ color:var(--zero); }}
 .grid-tbl td.muted-cell {{ color:var(--muted); }}
 .grid-tbl tr.tot td {{ font-weight:700; border-top:1px solid var(--line); }}
 .lvl {{ color:var(--muted); }}
 /* honest blank: 미계측 ≠ 0% (CH0005) — grey, never coloured like a score */
 .unmeasured {{ color:var(--muted); opacity:.7; font-style:italic; }}
 /* per-run waterfall: zero-JS radio tabs */
 .run-radio {{ position:absolute; opacity:0; pointer-events:none; }}
 .run-tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 4px; }}
 .run-tab {{ cursor:pointer; font-size:12px; color:var(--muted); padding:4px 10px;
            border:1px solid var(--line); border-radius:14px; background:var(--bg);
            user-select:none; }}
 .run-tab:hover {{ color:var(--ink); }}
 .run-panel {{ display:none; margin-top:10px; }}
 .wf-tbl td.wf-stage {{ font-weight:600; }}
 .wf-tbl td.wf-count {{ width:170px; }}
 .wf-tbl td.wf-gold {{ width:170px; }}
 .wf-matrix-scroll {{ overflow-x:auto; }}
 .wf-matrix th, .wf-matrix td {{ white-space:nowrap; font-size:12px; padding:5px 7px; }}
 footer {{ color:var(--muted); font-size:11.5px; text-align:center; margin-top:24px; }}
</style></head>
<body><div class="wrap">{body}</div></body></html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="runs.jsonl → 자기완결 HTML 성능지표 레포트")
    ap.add_argument("input", nargs="?", help="runs.jsonl 경로")
    ap.add_argument("-o", "--out", default=None, help="출력 HTML 경로 (기본: <input>.html)")
    ap.add_argument("--demo", action="store_true",
                    help="번들된 runs.sample.jsonl 로 렌더 (입력 생략 가능)")
    ap.add_argument("--open", action="store_true", dest="open_",
                    help="렌더 후 브라우저로 열기")
    args = ap.parse_args(argv)

    inp = args.input
    if args.demo or not inp:
        inp = os.path.join(here, "runs.sample.jsonl")
        if not inp or not os.path.exists(inp):
            ap.error("입력 jsonl 이 없습니다. 경로를 주거나 --demo 를 쓰세요.")

    if not os.path.exists(inp):
        ap.error(f"입력을 찾을 수 없음: {inp}")

    runs = load_runs(inp)
    html = render(runs)

    out = args.out or (os.path.splitext(inp)[0] + ".html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out}  ({len(runs)} runs)")

    if args.open_:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
