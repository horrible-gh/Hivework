#!/usr/bin/env python3
"""judge cost-parity A/B: 120b @ max_total_calls=10  vs  20b @ max_total_calls=13.

Both hold queen=copilot/gpt-5-mini (stable upstream, no codex-decompose noise),
everything else baseline, investigate path (NO full-swarm run). 5 reps each.
Scored on judge_located golden recall (perf/score.py). Read-only on the target.

    python perf/run_judge_ab.py            # run the A/B (PAID — gpt-oss only, no Qwen)
    python perf/run_judge_ab.py --dry-run  # print the plan, touch nothing
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, sqlite3
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    rc = getattr(_s, "reconfigure", None)
    if rc: rc(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HIVE = os.path.join(REPO, "hive.py")
sys.path.insert(0, HERE)
import score as scorer

PROFILES = ["judge-120b-v10", "judge-20b-v13"]
REPS = 5


def _matrix():
    with open(os.path.join(HERE, "matrix.json"), encoding="utf-8") as f:
        return json.load(f)


def _ledger_tokens(db):
    if not os.path.exists(db): return (0, 0, 0)
    try:
        con = sqlite3.connect(db)
        rows = con.execute("SELECT stage, est_tokens, ok FROM worker_calls").fetchall()
        con.close()
    except Exception:
        return (0, 0, 0)
    judge_tok = sum(t or 0 for st, t, ok in rows if st == "judge")
    tot = sum(t or 0 for _, t, _ in rows)
    fails = sum(1 for *_, ok in rows if ok not in (1, None))
    return (judge_tok, tot, fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    m = _matrix()
    seed, docs, cb = m["seed"], m["docs"], m["target"]["codebase"]
    golden = json.load(open(os.path.join(HERE, "golden", "manifest.json"), encoding="utf-8"))

    plan = []
    for prof in PROFILES:
        for rep in range(1, REPS + 1):
            rep_dir = os.path.join(HERE, "results", prof, str(rep))
            out = os.path.join(rep_dir, "verdict.json")
            cmd = [sys.executable, HIVE, "--profile", prof, "investigate",
                   "--seed", seed, "--codebase", cb, "--docs", docs, "--out", out]
            plan.append((prof, rep, rep_dir, out, cmd))

    if args.dry_run:
        print(f"DRY RUN — {len(plan)} runs ({len(PROFILES)} profiles x {REPS} reps), investigate path, no write.")
        for prof, rep, *_ , cmd in plan:
            print(f"  {prof} rep{rep}: " + " ".join(cmd))
        return

    results = {p: [] for p in PROFILES}
    for prof in PROFILES:
        ldb = os.path.join(HERE, "results", prof, "ledger.db")
        if os.path.exists(ldb):
            try: os.remove(ldb)
            except OSError: pass

    for prof, rep, rep_dir, out, cmd in plan:
        os.makedirs(rep_dir, exist_ok=True)
        log = os.path.join(rep_dir, "pipeline.log")
        with open(log, "w", encoding="utf-8") as lf:
            lf.write("$ " + " ".join(cmd) + "\n\n"); lf.flush()
            try:
                p = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                                   text=True, timeout=1200)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                lf.write("\n[TIMEOUT]\n"); rc = -9
        sc = scorer.score_cell("judge", rep_dir, golden)
        results[prof].append({"rep": rep, "rc": rc, "score": sc})
        print(f"[{prof} rep{rep}] rc={rc} measurement={sc.get('measurement')} "
              f"core_recall={sc.get('core_recall')}")

    # roll-up
    print("\n" + "=" * 64)
    summ = {}
    for prof in PROFILES:
        runs = results[prof]
        crs = [r["score"]["core_recall"] for r in runs
               if r["score"].get("measurement") == "recall" and r["score"].get("core_recall") is not None]
        mean = round(sum(crs) / len(crs), 3) if crs else None
        jt, tt, fails = _ledger_tokens(os.path.join(HERE, "results", prof, "ledger.db"))
        summ[prof] = {"mean_core_recall": mean, "per_rep": [r["score"].get("core_recall") for r in runs],
                      "crashes": sum(1 for r in runs if r["score"].get("measurement") is None),
                      "judge_tokens": jt, "total_tokens": tt, "ledger_fails": fails}
        print(f"{prof:18} mean_core={mean}  per_rep={summ[prof]['per_rep']}  "
              f"crashes={summ[prof]['crashes']}  judge_tok={jt}  total_tok={tt}")
    with open(os.path.join(HERE, "results", "judge_ab_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    print("\nSummary -> perf/results/judge_ab_summary.json")


if __name__ == "__main__":
    main()
