#!/usr/bin/env python3
"""L3 judge jury_size A/B — votes=5 (baseline) vs votes=3 (lever), same 120b model.

0062.0004-T (L3). Isolates jury_size from queen/model variance: queen is frozen
(perf/frozen_decompose.json), judge model held at openai/gpt-oss-120b in BOTH arms,
reinvestigation OFF. The ONLY variable is votes_per_axis. Because voting is
single-shot (hive/judge.py: votes>1 => deterministic axes x votes calls), v3 must
spend 40% fewer judge calls than v5; the A/B's job is to show core_recall does NOT
regress (R0001 hard gate: golden recall delta >= 0).

Grounded on the REAL, PRESENT FlowGate tree (C:\\workspace\\projects\\FlowGate) —
the perf sweep's original target FlowGate-dev/branches/20260607 is absent on this
host, but all three golden core files (process_service.py / ToastContainer.vue /
NewRequirementModal.vue) exist in the live tree, so judge_located recall scores.

    python perf/run_judge_l3_ab.py            # PAID (gpt-oss-120b only, ~$0.05)
    python perf/run_judge_l3_ab.py --reps 2   # fewer reps
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
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import score as scorer
import hive.investigate as inv
from hive.config import load_config
from hive.ledger import open_ledger
from hive import secrets as hive_secrets
from run_judge_frozen import build_provider_kwargs, install_frozen_decompose

PLAN = [("frozen-judge-120b-v5", 5), ("frozen-judge-120b-v3", 3)]
PRICE = (0.039, 0.19)  # deepinfra $/Mtok gpt-oss-120b (in, out)
CB = r"C:\workspace\projects\FlowGate"
DOCS = r"C:\workspace\projects\Documents\projects\FlowGate\210_design"
SEED = r"C:\workspace\projects\Documents\projects\FlowGate\410_tasks\T901_prevent_duplicate_r_in_group_and_fix_toast.md"


def judge_cost(db: str) -> dict:
    if not os.path.exists(db): return {"calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT in_chars, out_chars FROM worker_calls WHERE stage='judge'").fetchall()
    con.close()
    in_tok = sum((r[0] or 0) for r in rows) / 4.0
    out_tok = sum((r[1] or 0) for r in rows) / 4.0
    return {"calls": len(rows), "in_tok": round(in_tok), "out_tok": round(out_tok),
            "usd": round(in_tok/1e6*PRICE[0] + out_tok/1e6*PRICE[1], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    reps = max(1, args.reps)
    loaded = hive_secrets.load_secrets()
    print(f"secrets:{loaded}; DEEPINFRA_TOKEN={'set' if os.environ.get('DEEPINFRA_TOKEN') else 'MISSING'}")
    n_axes = install_frozen_decompose()
    golden = json.load(open(os.path.join(HERE, "golden", "manifest.json"), encoding="utf-8"))
    seed_text = open(SEED, encoding="utf-8").read()
    print(f"frozen axes={n_axes} (+SEED_ANCHOR), grounding={CB}")
    if args.dry_run:
        for prof, v in PLAN:
            c = load_config(profile=prof)
            print(f"  [{prof}] judge={c.role('judge').model} votes={c.judge.votes_per_axis} "
                  f"expect_calls~={n_axes}x{v}={n_axes*v}")
        print("DRY RUN ok"); return

    summary = {}
    for prof, v in PLAN:
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
            ldg.start_run(seed=SEED, codebase=CB, model_queen=cfg.queen.model,
                          model_fanout=cfg.role("judge").model)
            try:
                inv.run_investigate(seed_text=seed_text, recipe_path=None, code_root=CB,
                                    docs_root=DOCS, output_path=out, cfg=cfg, ledger=ldg,
                                    provider_kwargs=pk)
                ldg.finish_run(honey_path=out, status="done")
            except Exception as e:  # noqa: BLE001
                ldg.finish_run(status="failed")
                print(f"  [{prof} rep{rep}] FAILED: {type(e).__name__}: {e}")
            finally:
                ldg.close()
            sc = scorer.score_cell("judge", rep_dir, golden)
            per_rep.append(sc.get("core_recall"))
            print(f"[{prof} rep{rep}] core_recall={sc.get('core_recall')} hit={sc.get('core_hit')} located={sc.get('located')}")
        vals = [x for x in per_rep if x is not None]
        mean = round(sum(vals)/len(vals), 3) if vals else None
        cost = judge_cost(cfg.ledger.db_path)
        summary[prof] = {"votes": v, "mean_core_recall": mean, "per_rep": per_rep,
                         "crashes": sum(1 for x in per_rep if x is None), "cost": cost}
        print(f"  => {prof}: mean_core={mean} judge_calls={cost['calls']} usd=${cost['usd']}")

    print("\n" + "="*64)
    a = summary.get("frozen-judge-120b-v5", {}); b = summary.get("frozen-judge-120b-v3", {})
    ar, br = a.get("mean_core_recall"), b.get("mean_core_recall")
    ac, bc = a.get("cost", {}).get("calls", 0), b.get("cost", {}).get("calls", 0)
    if ac:
        print(f"v5 mean_core={ar} calls={ac} ${a.get('cost',{}).get('usd')}")
        print(f"v3 mean_core={br} calls={bc} ${b.get('cost',{}).get('usd')}")
        print(f"calls reduction: {round((ac-bc)/ac*100,1)}%  |  recall delta (v3-v5): "
              f"{None if ar is None or br is None else round(br-ar,3)}")
    json.dump(summary, open(os.path.join(HERE, "results", "judge_l3_ab_summary.json"), "w",
              encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Summary -> perf/results/judge_l3_ab_summary.json")


if __name__ == "__main__":
    main()
