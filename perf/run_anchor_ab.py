#!/usr/bin/env python3
"""writer-anchor lever A/B (ablation) — isolates HIVE_NO_WRITER_ANCHOR as the SOLE variable.

Motivation (0040 NR0003): run511 was the first FOUND in the 0082/swarm-OFF family, but the
T0006 writer-anchor lever fired 0 times there (no-op). With decompose stochasticity, an
uncommitted working tree, and several pre-existing levers all moving between run510(MISS) and
run511(FOUND), the FOUND cannot be attributed to the lever. This harness removes the
confounders so the ONLY thing that differs between the two arms is the lever:

    Arm A  "anchor-off"  -> HIVE_NO_WRITER_ANCHOR=1   (lever OFF, control)
    Arm B  "anchor-on"   -> HIVE_NO_WRITER_ANCHOR unset (lever ON, treatment)

Confounders pinned:
  C1 decompose drift  -> frozen decompose (monkeypatch inv.run_decompose to a fixed axis set),
                         queen runs 0 times. Pass --frozen <run510-form 0082 axis set>.
  C2 code variance    -> caller's responsibility: snapshot the working tree to ONE commit/stash
                         before running so both arms start from byte-identical source.
  C3 side levers       -> identical in both arms (this script touches ONLY HIVE_NO_WRITER_ANCHOR),
                         so any other lever becomes a constant, not a variable.

Decisive evidence beyond recall: per rep we also count `via=writer-anchor` injections in the
verdict, so an A/B where B's recall rises BUT the lever never fired tells us the win came from
elsewhere (exactly the run511 story). Scored on golden recall (perf/score.py). Read-only target.

    python perf/run_anchor_ab.py --dry-run        # validate wiring (cfg + frozen patch), no model calls
    python perf/run_anchor_ab.py --golden <0082 manifest> --frozen <0082 axes> \
        --codebase <0082 clone> --seed <0082 seed.md> --docs <design dir>   # PAID live A/B

For a real 0082 dispose-FK measurement you MUST supply 0082 artifacts (the defaults below point at
the T901 fixtures and only exercise wiring): a run510-form 0082 frozen axis set, the golden_0082
manifest, and the 0082 clone codebase. DEEPINFRA_TOKEN is read from ~/.hivework/.env.
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

# The two arms. label -> env value for HIVE_NO_WRITER_ANCHOR ("1" sets it, None unsets it).
ARMS = [("anchor-off", "1"), ("anchor-on", None)]
DEFAULT_REPS = 5
# deepinfra $/Mtok (in, out) for cost roll-up; override if the run510 preset uses other models.
PRICE_IN, PRICE_OUT = 0.030, 0.14


def _matrix() -> dict:
    with open(os.path.join(HERE, "matrix.json"), encoding="utf-8") as f:
        return json.load(f)


def build_provider_kwargs(cfg) -> dict:
    """Mirror hive.py::build_provider_kwargs (lives in the CLI script, not importable)."""
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


def install_frozen_decompose(frozen_path: str) -> int:
    """Pin the axis set (kills C1). queen runs 0 times — every rep judges the SAME axes."""
    with open(frozen_path, encoding="utf-8") as f:
        frozen = json.load(f)
    tasks = frozen["tasks"]
    def _frozen(**_kwargs):
        return {"tasks": [dict(t) for t in tasks]}
    inv.run_decompose = _frozen
    return len(tasks)


def _count_anchor_firings(obj) -> int:
    """Count `via == 'writer-anchor'` injections anywhere in the verdict/honey JSON.

    This is the decisive evidence: a recall lift in arm B with 0 firings means the lever
    did NOT cause the win (the run511 no-op pattern)."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("via") == "writer-anchor":
            n += 1
        for v in obj.values():
            n += _count_anchor_firings(v)
    elif isinstance(obj, list):
        for v in obj:
            n += _count_anchor_firings(v)
    return n


def anchor_firings(rep_dir: str) -> int:
    total = 0
    for name in ("verdict.json", "honey.edit_spec.json"):
        p = os.path.join(rep_dir, name)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    total += _count_anchor_firings(json.load(f))
            except Exception:
                pass
    honey = os.path.join(rep_dir, "honey.md")
    if os.path.exists(honey):
        try:
            with open(honey, encoding="utf-8") as f:
                total += f.read().count("writer-anchor")
        except Exception:
            pass
    return total


def arm_cost(db: str) -> dict:
    if not os.path.exists(db):
        return {"calls": 0, "in_tok": 0, "out_tok": 0, "usd": 0.0}
    con = sqlite3.connect(db)
    rows = con.execute("SELECT in_chars, out_chars FROM worker_calls").fetchall()
    con.close()
    in_tok = sum((r[0] or 0) for r in rows) / 4.0
    out_tok = sum((r[1] or 0) for r in rows) / 4.0
    usd = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
    return {"calls": len(rows), "in_tok": round(in_tok), "out_tok": round(out_tok), "usd": round(usd, 5)}


def _decision(a_mean, b_mean, b_firings) -> str:
    """0040 NR0003 §5 judgment matrix."""
    if a_mean is None or b_mean is None:
        return "INCONCLUSIVE — a crash arm produced no recall"
    if a_mean >= 0.999 and b_mean >= 0.999:
        return ("SHAPE NOT ISOLATED — both arms recall=1; the answer is already surfaced. "
                "Re-freeze from a stricter run510-form (under-anchored) axis set.")
    if b_mean > a_mean + 1e-9:
        if b_firings > 0:
            return "LEVER PROVEN — B>A and the lever fired; recall lift is attributable to writer-anchor."
        return ("LEVER NO-OP — B>A but the lever never fired; the lift came from elsewhere "
                "(DESIGN_SSOT / pre-existing FK scanner). run511's caveat confirmed.")
    return ("LEVER NO-OP — A≈B; toggling the lever off did not lower recall, so it is not the "
            "cause of detection in this shape.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profile", default="default",
                    help="run510 preset profile (default: 'default' = gpt-5-mini)")
    ap.add_argument("--frozen", default=os.path.join(HERE, "frozen_decompose.json"),
                    help="frozen axis-set JSON (kills decompose drift). For 0082 supply a run510-form set.")
    ap.add_argument("--golden", default=os.path.join(HERE, "golden", "manifest.json"),
                    help="golden manifest to score against (supply golden_0082 for the real run)")
    ap.add_argument("--seed", default=None, help="seed .md path (defaults to matrix.json seed)")
    ap.add_argument("--docs", default=None, help="design docs dir (defaults to matrix.json docs)")
    ap.add_argument("--codebase", default=None, help="target clone (defaults to matrix.json target.codebase)")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    args = ap.parse_args()
    reps = max(1, int(args.reps))

    m = _matrix()
    seed_path = args.seed or m["seed"]
    docs = args.docs or m["docs"]
    cb = args.codebase or m["target"]["codebase"]

    loaded = hive_secrets.load_secrets()
    print(f"secrets: loaded {loaded or '(none — relying on real env)'}; "
          f"DEEPINFRA_TOKEN={'set' if os.environ.get('DEEPINFRA_TOKEN') else 'MISSING'}")
    n_axes = install_frozen_decompose(args.frozen)
    golden = json.load(open(args.golden, encoding="utf-8"))
    print(f"frozen decompose: {n_axes} axes from {os.path.basename(args.frozen)} — queen runs 0 times")
    print(f"profile={args.profile}  seed={os.path.basename(seed_path)}  golden={os.path.basename(args.golden)}")
    print(f"arms: " + " | ".join(f"{lbl}(HIVE_NO_WRITER_ANCHOR={'1' if v else 'unset'})" for lbl, v in ARMS))

    if args.dry_run:
        cfg = load_config(profile=args.profile)
        print(f"  [cfg] queen={cfg.queen.model} judge={cfg.role('judge').provider}/{cfg.role('judge').model} "
              f"votes={cfg.judge.votes_per_axis} ledger={os.path.basename(os.path.dirname(cfg.ledger.db_path))}")
        if not os.path.exists(cb):
            print(f"  [warn] codebase does not exist: {cb} (real run needs a 0082 clone)")
        print(f"DRY RUN — cfg loaded + frozen patch installed ({n_axes} axes), no model calls. "
              f"{len(ARMS)} arms x {reps} reps = {len(ARMS)*reps} runs when live.")
        return

    seed_text = open(seed_path, encoding="utf-8").read()
    summary = {}
    for label, env_val in ARMS:
        cfg = load_config(profile=args.profile)
        pk = build_provider_kwargs(cfg)
        # per-arm isolated ledger
        ldb = os.path.join(HERE, "results", f"anchor-{label}", "ledger.db")
        os.makedirs(os.path.dirname(ldb), exist_ok=True)
        if os.path.exists(ldb):
            try: os.remove(ldb)
            except OSError: pass
        cfg.ledger.db_path = ldb

        # the SOLE variable
        if env_val is None:
            os.environ.pop("HIVE_NO_WRITER_ANCHOR", None)
        else:
            os.environ["HIVE_NO_WRITER_ANCHOR"] = env_val

        per_rep, firings = [], []
        for rep in range(1, reps + 1):
            rep_dir = os.path.join(HERE, "results", f"anchor-{label}", str(rep))
            os.makedirs(rep_dir, exist_ok=True)
            out = os.path.join(rep_dir, "verdict.json")
            ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
            ldg.start_run(seed=seed_path, codebase=cb,
                          model_queen=cfg.queen.model, model_fanout=cfg.role("judge").model)
            try:
                inv.run_investigate(
                    seed_text=seed_text, recipe_path=None, code_root=cb, docs_root=docs,
                    output_path=out, cfg=cfg, ledger=ldg, provider_kwargs=pk)
                ldg.finish_run(honey_path=out, status="done")
            except Exception as e:  # noqa: BLE001
                ldg.finish_run(status="failed")
                print(f"  [{label} rep{rep}] FAILED: {type(e).__name__}: {e}")
            finally:
                ldg.close()
            sc = scorer.score_cell("judge", rep_dir, golden)
            fired = anchor_firings(rep_dir)
            per_rep.append(sc.get("core_recall"))
            firings.append(fired)
            print(f"[{label} rep{rep}] core_recall={sc.get('core_recall')} "
                  f"core_hit={sc.get('core_hit')} anchor_fired={fired}")
        vals = [x for x in per_rep if x is not None]
        mean = round(sum(vals) / len(vals), 3) if vals else None
        summary[label] = {"env": f"HIVE_NO_WRITER_ANCHOR={'1' if env_val else 'unset'}",
                          "mean_core_recall": mean, "per_rep": per_rep,
                          "anchor_firings_per_rep": firings, "total_anchor_firings": sum(firings),
                          "crashes": sum(1 for x in per_rep if x is None),
                          "cost": arm_cost(cfg.ledger.db_path)}
        print(f"  => {label}: mean_core={mean} per_rep={per_rep} "
              f"anchor_fired={sum(firings)} usd=${summary[label]['cost']['usd']}")

    a, b = summary.get("anchor-off", {}), summary.get("anchor-on", {})
    verdict = _decision(a.get("mean_core_recall"), b.get("mean_core_recall"), b.get("total_anchor_firings", 0))
    summary["_decision"] = verdict
    print("\n" + "=" * 64)
    print(f"A anchor-off  mean_core={a.get('mean_core_recall')}  fired={a.get('total_anchor_firings')}")
    print(f"B anchor-on   mean_core={b.get('mean_core_recall')}  fired={b.get('total_anchor_firings')}")
    print(f"DECISION: {verdict}")
    json.dump(summary, open(os.path.join(HERE, "results", "anchor_ab_summary.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nSummary -> perf/results/anchor_ab_summary.json")


if __name__ == "__main__":
    main()
