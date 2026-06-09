#!/usr/bin/env python3
"""Frozen-decompose judge A/B — isolates the JUDGE model/votes from queen variance.

queen is run ZERO times: hive.investigate.run_decompose is monkeypatched to return a
FROZEN axis set (perf/frozen_decompose.json, lifted from the core_recall=1.0 run, with
SEED_ANCHOR dropped so _prioritize_axes re-injects it deterministically — exactly the
original pipeline). Every rep therefore judges the SAME axes; the only variables are the
judge model and votes_per_axis. reinvestigation is OFF so judge runs exactly votes x axes.

  A: 120b @ votes=5   (profile frozen-judge-120b-v5)
  B: 20b  @ votes=7   (profile frozen-judge-20b-v7)   <- cost parity (120b ~1.38x/call)

5 reps each. Scored on judge_located golden recall (perf/score.py). Read-only target.
Cost is computed from REAL ledger tokens x deepinfra prices, not assumed.

    python perf/run_judge_frozen.py            # run (PAID — gpt-oss only, ~$0.07, no queen, no Qwen)
    python perf/run_judge_frozen.py --dry-run  # validate wiring (load cfg + patch), no model calls
"""
from __future__ import annotations
import argparse, json, os, sys, sqlite3
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
for _s in (sys.stdout, sys.stderr):
    rc = getattr(_s, "reconfigure", None)
    if rc: rc(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import score as scorer
import hive.investigate as inv
from hive.config import load_config
from hive.ledger import open_ledger
from hive import secrets as hive_secrets

REPS = 5
PLAN = [("frozen-judge-120b-v5", "120b"), ("frozen-judge-20b-v7", "20b"),
        ("frozen-judge-20b-v5", "20b5")]
# deepinfra $/Mtok (in, out), user-supplied
PRICE = {"120b": (0.039, 0.19), "20b": (0.030, 0.14), "20b5": (0.030, 0.14)}


def build_provider_kwargs(cfg) -> dict:
    """Mirror hive.py::build_provider_kwargs (that helper lives in the CLI script,
    not an importable submodule)."""
    kw: dict = {}
    if cfg.copilot.exe: kw["exe"] = cfg.copilot.exe
    if cfg.copilot.allow: kw["allow_flag"] = cfg.copilot.allow
    kw["read_only"] = cfg.copilot.read_only
    tok = cfg.copilot.token or (os.environ.get(cfg.copilot.token_env) if cfg.copilot.token_env else None)
    if tok: kw["copilot_token"] = tok
    if cfg.codex.exe: kw["codex_exe"] = cfg.codex.exe
    kw["codex_lock_timeout_sec"] = cfg.codex.lock_timeout_sec
    if cfg.openai.base_url: kw["base_url"] = cfg.openai.base_url
    if cfg.openai.api_key_env: kw["api_key_env"] = cfg.openai.api_key_env
    return kw


def install_frozen_decompose():
    with open(os.path.join(HERE, "frozen_decompose.json"), encoding="utf-8") as f:
        frozen = json.load(f)
    tasks = frozen["tasks"]
    def _frozen(**_kwargs):
        # identical object every call; run_investigate only reads ["tasks"]
        return {"tasks": [dict(t) for t in tasks]}
    inv.run_decompose = _frozen
    return len(tasks)


def judge_cost(db: str, label: str) -> dict:
    if not os.path.exists(db): return {"calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT in_chars, out_chars FROM worker_calls WHERE stage='judge'").fetchall()
    con.close()
    in_tok = sum((r[0] or 0) for r in rows) / 4.0
    out_tok = sum((r[1] or 0) for r in rows) / 4.0
    pin, pout = PRICE[label]
    usd = in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return {"calls": len(rows), "in_tok": round(in_tok), "out_tok": round(out_tok), "usd": round(usd, 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="run only profiles matching this (e.g. '20b' or a profile name)")
    ap.add_argument("--reps", type=int, default=REPS, help=f"reps per profile (default {REPS})")
    args = ap.parse_args()
    reps = max(1, int(args.reps))
    plan = [(p, l) for (p, l) in PLAN if not args.only or args.only == l or args.only == p]

    loaded = hive_secrets.load_secrets()  # pull ~/.hivework/.env into env (hive.py main does this)
    print(f"secrets: loaded {loaded or '(none — relying on real env)'}; "
          f"DEEPINFRA_TOKEN={'set' if os.environ.get('DEEPINFRA_TOKEN') else 'MISSING'}")
    n_axes = install_frozen_decompose()
    seed_path = "C:\\workspace\\projects\\Documents\\projects\\FlowGate\\410_tasks\\T901_prevent_duplicate_r_in_group_and_fix_toast.md"
    docs = "C:\\workspace\\projects\\Documents\\projects\\FlowGate\\210_design"
    cb = "C:\\workspace\\projects\\FlowGate-dev\\branches\\20260607"
    golden = json.load(open(os.path.join(HERE, "golden", "manifest.json"), encoding="utf-8"))
    seed_text = open(seed_path, encoding="utf-8").read()

    print(f"frozen decompose: {n_axes} queen axes (+SEED_ANCHOR re-injected) — queen runs 0 times")
    if args.dry_run:
        for prof, label in plan:
            cfg = load_config(profile=prof)
            print(f"  [{prof}] judge={cfg.role('judge').provider}/{cfg.role('judge').model} "
                  f"votes={cfg.judge.votes_per_axis} reinv_live={cfg.reinvestigation.live} "
                  f"ledger={os.path.basename(os.path.dirname(cfg.ledger.db_path))}")
        print("DRY RUN — config + patch validated, no model calls.")
        return

    summary = {}
    for prof, label in plan:
        cfg = load_config(profile=prof)
        pk = build_provider_kwargs(cfg)
        ldb = cfg.ledger.db_path
        os.makedirs(os.path.dirname(ldb), exist_ok=True)
        if os.path.exists(ldb):
            try: os.remove(ldb)
            except OSError: pass
        per_rep = []
        for rep in range(1, reps + 1):
            rep_dir = os.path.join(HERE, "results", prof, str(rep))
            os.makedirs(rep_dir, exist_ok=True)
            out = os.path.join(rep_dir, "verdict.json")
            ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
            ldg.start_run(seed=seed_path, codebase=cb,
                          model_queen=cfg.queen.model, model_swarm=cfg.role("judge").model)
            try:
                inv.run_investigate(
                    seed_text=seed_text, recipe_path=None, code_root=cb, docs_root=docs,
                    output_path=out, cfg=cfg, ledger=ldg, provider_kwargs=pk)
                ldg.finish_run(honey_path=out, status="done")
            except Exception as e:  # noqa: BLE001
                ldg.finish_run(status="failed")
                print(f"  [{prof} rep{rep}] FAILED: {type(e).__name__}: {e}")
            finally:
                ldg.close()
            sc = scorer.score_cell("judge", rep_dir, golden)
            per_rep.append(sc.get("core_recall"))
            print(f"[{prof} rep{rep}] core_recall={sc.get('core_recall')} core_hit={sc.get('core_hit')}")
        vals = [x for x in per_rep if x is not None]
        mean = round(sum(vals) / len(vals), 3) if vals else None
        cost = judge_cost(cfg.ledger.db_path, label)
        summary[prof] = {"model": label, "votes": cfg.judge.votes_per_axis,
                         "mean_core_recall": mean, "per_rep": per_rep,
                         "crashes": sum(1 for x in per_rep if x is None), "cost": cost}
        print(f"  => {prof}: mean_core={mean} per_rep={per_rep} "
              f"judge_calls={cost['calls']} usd=${cost['usd']}")

    # recall-per-dollar
    print("\n" + "=" * 64)
    for prof, s in summary.items():
        rpd = (s["mean_core_recall"] / s["cost"]["usd"]) if (s["mean_core_recall"] and s["cost"]["usd"]) else None
        s["recall_per_usd"] = round(rpd, 1) if rpd else None
        print(f"{prof:22} mean_core={s['mean_core_recall']}  ${s['cost']['usd']:.5f}  "
              f"recall/$={s['recall_per_usd']}")
    json.dump(summary, open(os.path.join(HERE, "results", "judge_frozen_summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nSummary -> perf/results/judge_frozen_summary.json")


if __name__ == "__main__":
    main()
