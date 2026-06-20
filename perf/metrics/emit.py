#!/usr/bin/env python3
"""Project one hive cycle from ``hive_ledger.db`` into a ``runs.jsonl`` line.

This closes the gap SCHEMA.md flagged as "후속 작업": the report renderer
(``report.py``) only *reads* ``runs.jsonl`` — nothing wrote it, so the input
had to be hand-authored each cycle (see hivework.default.0012.0005-NR §2). This
module is that missing emitter. It is the inverse of report.py: ledger → jsonl.

Design constraints (mirrors report.py, CH0002):
  - Stdlib only. Read-only SQLite (``mode=ro``). Non-fatal: any failure logs and
    returns/skips rather than breaking the caller (the hive run must never die
    because telemetry emit failed).
  - The field mapping is exactly SCHEMA.md's ``← ledger.…`` annotations. The few
    fields that are NOT a direct ledger projection are derived deterministically
    and documented inline below.

Cost (NR0005 §1): ``est_tokens`` (char/4 of the *final* stored prompt/output)
undercounts a tool-using swarm worker by 25–346× because it never sees the
multi-turn ``run_agent_loop`` context the model is actually billed for. So for
usage-billed providers that report ``real_tokens`` (DeepInfra), USD is computed
from ``real_tokens`` (reasoning-aware split); copilot (no usage reported) falls
back to the char/4 estimate. Prices live OUT of code (SCHEMA.md:45): loaded from
``prices.json`` next to this file if present, else the built-in fallback.

Usage:
    python emit.py --run-id 459 --workdir <run_workdir> --out runs.jsonl
    python emit.py --latest --out runs.jsonl          # newest run in the ledger
    python emit.py --run-id 459 --golden-json g.json   # splice a golden score in
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

# Built-in fallback price table — (input $/Mtok, output $/Mtok). Overridden by a
# prices.json sibling so the published repo ships no vendor-specific pricing in
# code (SCHEMA.md:45 "가격은 코드에 박지 않음"). Keep in sync with that file.
_FALLBACK_PRICES = {
    "openai/gpt-oss-120b": (0.039, 0.19),
    "gpt-5-mini": (0.25, 2.00),
    "claude-sonnet-4.5": (3.00, 15.00),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.071, 0.10),
}

# Built-in fallback billing model (R0001). Two paradigms: token-billed providers
# use the price table above; credit-billed providers (copilot) charge a flat
# credits_per_call × credit_usd per worker call and ignore tokens entirely.
# Overridden by prices.json's "_billing" block. Keep in sync with that file.
_FALLBACK_BILLING = {
    "credit_usd": 0.01,
    "default": "token",
    "providers": {
        "copilot": {"type": "credit", "credits_per_call": 0,
                    "by_model": {"gpt-5-mini": 0, "gpt-5": 0, "claude-sonnet-4.5": 1}},
    },
}

# Providers whose name the ledger records as "openai" but whose telemetry we
# surface under the SCHEMA's "deepinfra" cost bucket (same HTTP endpoint).
_PROVIDER_LABEL = {"openai": "deepinfra"}


def _load_pricing() -> tuple[dict, dict]:
    """Load ``(prices, billing)`` from prices.json, else the built-in fallbacks.

    Top-level keys are token unit prices ``model → (in, out)``; ``_``-prefixed
    keys (``_about``, ``_billing``) are config, not prices, and are split out.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.json")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return dict(_FALLBACK_PRICES), dict(_FALLBACK_BILLING)
    billing = raw.get("_billing") or dict(_FALLBACK_BILLING)
    prices = {k: tuple(v) for k, v in raw.items()
              if not k.startswith("_") and isinstance(v, (list, tuple))}
    return prices, billing


def _billing_for(provider: str, billing: dict) -> dict:
    """Resolve the billing rule for a ledger ``provider`` (falls back to default)."""
    provs = billing.get("providers", {}) or {}
    rule = provs.get(provider)
    if rule is None:
        return {"type": billing.get("default", "token")}
    return rule


def _credits_per_call(rule: dict, model: str | None) -> float:
    """Per-call credit multiplier for a credit-billed provider, model override first."""
    by_model = rule.get("by_model") or {}
    if model in by_model:
        return float(by_model[model] or 0)
    return float(rule.get("credits_per_call", 0) or 0)


def _call_usd(model: str, in_chars: int, out_chars: int, real_tokens, prices: dict) -> float:
    """Reasoning-aware cost for one worker call (NR0005 methodology).

    When the provider reports real usage, prompt tokens ≈ in_chars/4 are billed at
    the input rate and the remainder (completion + reasoning, possibly far larger
    than out_chars due to the agent loop) at the output rate. Without usage
    (copilot), fall back to a char/4 split of the stored prompt/output.
    """
    pin, pout = prices.get(model, (0.0, 0.0))
    if real_tokens is not None and real_tokens > 0:
        in_tok = min((in_chars or 0) / 4.0, real_tokens)
        out_tok = max(real_tokens - in_tok, 0)
    else:
        in_tok = (in_chars or 0) / 4.0
        out_tok = (out_chars or 0) / 4.0
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


def _shaped_by_axis(workdir: str | None) -> tuple[dict, int]:
    """Derive per-axis shaped counts from ``final_combs.json`` (findings present).

    Returns ``(by_axis, total)``. comb_shaped is the count of combs that carry a
    real finding — the ledger has no is_comb_shaped column (NR0005 §2.3), so this
    one field must come from the workdir. Returns ``({}, -1)`` when unavailable, so
    the caller can fall back to comb_fired.
    """
    if not workdir:
        return {}, -1
    path = os.path.join(workdir, "final_combs.json")
    try:
        with open(path, encoding="utf-8") as f:
            combs = json.load(f)
    except (OSError, ValueError):
        return {}, -1
    by_axis: dict = {}
    total = 0
    for c in combs if isinstance(combs, list) else []:
        if not isinstance(c, dict):
            continue
        findings = c.get("findings")
        n = len(findings) if isinstance(findings, list) else (1 if findings else 0)
        shaped = 1 if n else 0
        by_axis[c.get("axis_id")] = {"findings": n, "shaped": shaped}
        total += shaped
    return by_axis, total


def build_record(db_path: str, run_id: int, workdir: str | None = None,
                 golden: dict | None = None,
                 stage_golden: dict | None = None,
                 actual_usd: float | None = None,
                 actual_source: str | None = None) -> dict | None:
    """Project ledger run ``run_id`` (+ optional workdir/golden) into a SCHEMA record.

    ``actual_usd`` (R0021-2): the operator-entered billed cost read off the
    provider management screen. The local per-provider USD is a token/credit
    *estimate* that never reconciles with that screen (copilot premium-request
    multipliers are opaque), so the report treats this entered actual — not the
    estimate — as the bottom-line cost, and renders 미계측 when it is absent.
    """
    prices, billing = _load_pricing()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        print(f"emit: cannot open ledger {db_path}: {e}", file=sys.stderr)
        return None
    run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if run is None:
        print(f"emit: run {run_id} not found", file=sys.stderr)
        return None
    run = dict(run)
    calls = [dict(r) for r in conn.execute(
        "SELECT stage,axis_id,provider,model,in_chars,out_chars,est_tokens,"
        "real_tokens,latency_s,comb_path,ok FROM worker_calls WHERE run_id=? ORDER BY id",
        (run_id,)).fetchall()]
    conn.close()

    comb_fired = sum(1 for c in calls if c.get("comb_path"))
    shaped_by_axis, shaped_total = _shaped_by_axis(workdir)
    comb_shaped = shaped_total if shaped_total >= 0 else comb_fired

    # Lower funnel: the assembled honey is the single converged conclusion of an
    # investigate cycle. submitted = that honey; passed = landed fixes (0 unless a
    # full run/apply recorded them). Documented proxy (NR0005 §0).
    done = run.get("status") == "done" and bool(run.get("honey_path"))
    conclusion_converted = 1 if done else 0
    submitted = conclusion_converted
    fixes_landed = 0  # investigate-only; a future run/apply path can override

    # Per-axis breakdown from the source-mining swarm (+ reinforce) calls.
    axes_out = []
    seen = set()
    for c in calls:
        if c.get("stage") not in ("swarm", "fanout", "reinforce"):
            continue
        aid = c.get("axis_id") or "?"
        if aid in seen:
            continue
        seen.add(aid)
        lat = max((cc.get("latency_s") or 0.0) for cc in calls if cc.get("axis_id") == aid)
        sh = shaped_by_axis.get(aid, {})
        axes_out.append({
            "axis_id": aid,
            "label": aid,
            "fired": any(cc.get("comb_path") for cc in calls if cc.get("axis_id") == aid),
            "findings": sh.get("findings", 0),
            "shaped": sh.get("shaped", 0),
            "longest_chain_s": round(lat, 1),
        })

    # Cost: per-provider, split by billing paradigm (R0001). Token-billed
    # providers (deepinfra/openai) pay real-token-aware USD; credit-billed
    # providers (copilot) pay calls × credits_per_call × credit_usd (flat,
    # token-independent) — the credit axis and the $ axis stay separate.
    credit_usd = float(billing.get("credit_usd", 0.01) or 0.01)
    by_prov_tok = defaultdict(int)
    by_prov_credits = defaultdict(float)
    by_prov_usd = defaultdict(float)
    by_prov_calls = defaultdict(int)
    for c in calls:
        provider = c.get("provider") or "?"
        label = _PROVIDER_LABEL.get(provider, provider)
        rule = _billing_for(provider, billing)
        by_prov_tok[label] += c.get("est_tokens") or 0
        by_prov_calls[label] += 1
        if rule.get("type") == "credit":
            cr = _credits_per_call(rule, c.get("model"))
            by_prov_credits[label] += cr
            by_prov_usd[label] += cr * credit_usd
        else:
            by_prov_usd[label] += _call_usd(c.get("model"), c.get("in_chars"),
                                            c.get("out_chars"), c.get("real_tokens"), prices)
    cost = {"by_provider": {
        prov: {
            "tokens": by_prov_tok[prov],
            "calls": by_prov_calls[prov],
            "credits": round(by_prov_credits[prov], 3),
            "usd": round(by_prov_usd[prov], 5),
        }
        for prov in sorted(by_prov_tok)
    }}
    # R0021-2: operator-entered actual billed USD (management screen). This — not
    # the per-provider estimate above — is what the report shows as the cost.
    if actual_usd is not None:
        cost["actual_usd"] = round(float(actual_usd), 5)
        if actual_source:
            cost["actual_source"] = actual_source

    record = {
        "run_id": f"run{run['id']}",
        "ts": run.get("ts"),
        "seed": run.get("seed"),
        "work_type": run.get("work_type"),
        "codebase": os.path.basename(str(run.get("codebase") or "").rstrip("\\/")),
        "models": {"queen": run.get("model_queen"),
                   # B0001: column renamed model_swarm -> model_fanout; fall back to the
                   # old column so historical ledger DBs still emit a model here.
                   "fanout": run.get("model_fanout") or run.get("model_swarm")},
        "funnel": {
            "axes_attempted": run.get("axes_n") or 0,
            "comb_fired": comb_fired,
            "comb_shaped": comb_shaped,
            "conclusion_converted": conclusion_converted,
            "submitted": submitted,
            "passed": fixes_landed,
        },
        "axes": axes_out,
        "cost": cost,
        "cycle": {
            "fixes_landed": fixes_landed,
            "fixes_total": fixes_landed,
            "wall_clock_s": round(run.get("elapsed_s") or 0.0, 1),
        },
    }
    if golden:  # golden requires the external scorer (NR0004 5-bug set); omit if absent
        record["golden"] = golden
    # Per-stage golden-signal-survival slot (hivework.0017, T0006). No hive stage
    # emits this yet — recall is scored once at the end — so it is omitted today
    # and the report renders those waterfall cells as 미계측 (an honest blank,
    # never 0%). The passthrough is the forward contract: once instrumentation
    # produces a stage_golden block, it flows verbatim into the report's blanks.
    if stage_golden:
        record["stage_golden"] = stage_golden
    return record


def emit(db_path: str, run_id: int, out_path: str, workdir: str | None = None,
         golden: dict | None = None, stage_golden: dict | None = None,
         actual_usd: float | None = None, actual_source: str | None = None) -> bool:
    """Append one record to ``out_path`` (JSON Lines). Best-effort, non-fatal."""
    rec = build_record(db_path, run_id, workdir=workdir, golden=golden,
                       stage_golden=stage_golden, actual_usd=actual_usd,
                       actual_source=actual_source)
    if rec is None:
        return False
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"emit: cannot append to {out_path}: {e}", file=sys.stderr)
        return False
    return True


def _latest_run_id(db_path: str) -> int | None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        row = conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hive_ledger.db → runs.jsonl 한 줄 적재")
    ap.add_argument("--db", default="hive_ledger.db", help="ledger DB 경로")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", type=int, help="적재할 run id")
    g.add_argument("--latest", action="store_true", help="ledger의 최신 run")
    ap.add_argument("--workdir", default=None,
                    help="해당 run의 hive workdir (final_combs.json로 comb_shaped 파생)")
    ap.add_argument("--out", default="runs.jsonl", help="append 대상 runs.jsonl")
    ap.add_argument("--golden-json", default=None,
                    help="골든셋 채점 블록(JSON 파일) — 있으면 record.golden 에 삽입")
    ap.add_argument("--stage-golden-json", default=None,
                    help="단계별 골든 생존 블록(JSON 파일) — 있으면 record.stage_golden 에 삽입 "
                         "(T0006: 워터폴 미계측 칸을 실측 퍼센트로 채우는 적재 슬롯)")
    ap.add_argument("--actual-usd", type=float, default=None,
                    help="관리화면 실청구액(USD) — 있으면 record.cost.actual_usd 에 적재. "
                         "로컬 토큰추정 USD는 청구와 불일치하므로 이 값이 레포트의 비용으로 표시됨(R0021)")
    ap.add_argument("--actual-source", default=None,
                    help="실청구액 출처 메모(예: 'copilot dashboard 2026-06-20')")
    ap.add_argument("--print", action="store_true", dest="print_only",
                    help="append 하지 않고 record를 stdout으로만 출력")
    a = ap.parse_args(argv)

    run_id = _latest_run_id(a.db) if a.latest else a.run_id
    if run_id is None:
        ap.error("적재할 run을 찾지 못했습니다.")
    golden = None
    if a.golden_json:
        with open(a.golden_json, encoding="utf-8") as f:
            golden = json.load(f)
    stage_golden = None
    if a.stage_golden_json:
        with open(a.stage_golden_json, encoding="utf-8") as f:
            stage_golden = json.load(f)

    if a.print_only:
        rec = build_record(a.db, run_id, workdir=a.workdir, golden=golden,
                           stage_golden=stage_golden, actual_usd=a.actual_usd,
                           actual_source=a.actual_source)
        if rec is None:
            return 1
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return 0
    ok = emit(a.db, run_id, a.out, workdir=a.workdir, golden=golden,
              stage_golden=stage_golden, actual_usd=a.actual_usd,
              actual_source=a.actual_source)
    if ok:
        print(f"emit: run{run_id} → {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
