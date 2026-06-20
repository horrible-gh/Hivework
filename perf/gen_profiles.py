#!/usr/bin/env python3
"""Generate one --profile config per sweep cell (+ a baseline and a smoke profile).

Each generated file lives at ``config/hive.config.perf-<id>.json`` and is the
CURRENT baseline config with EXACTLY ONE role swapped to the cell's provider/model
(OFAT). Path/stage flags are set per cell:

  - run-path cells (swarm/assemble)  -> ops.swarm_run.allow = true
  - scout cells                      -> stages.reinforce.enabled = true
  - every cell                       -> targets bound to the REVERTED branch, and a
                                        per-cell isolated ledger DB under perf/results/<id>/.

This is config generation only — it writes NOTHING to the target codebase and makes
NO model calls. Run it before the sweep (run_sweep.py calls it automatically).

    python perf/gen_profiles.py            # write all profiles
    python perf/gen_profiles.py --clean    # delete previously generated perf-* profiles
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(REPO, "config")
MATRIX_PATH = os.path.join(HERE, "matrix.json")

# Role name in the matrix -> the roles.<key> the config loader reads. (1:1 today,
# but kept explicit so a future rename of an internal role key is a one-line change.)
ROLE_KEY = {
    "queen": "queen", "swarm": "fanout", "scout": "scout", "judge": "judge",
    "converge": "converge", "assemble": "assemble", "specify": "specify",
    "review": "review",
}

# The HTTP-shape red-test harness recipe shipped with Hive (lever 7). Kept so the
# generated FlowGate target mirrors the default config's wiring.
HTTP_SHAPE_SETUP = os.path.join(REPO, "recipes", "flowgate_http_shape_harness.py")


def load_matrix() -> dict:
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def baseline_config(m: dict) -> dict:
    """The grouped-layout config every cell starts from: current baseline roles,
    the reverted-branch target binding, swarm OFF, reinforce OFF. Cells mutate a copy."""
    tgt = m["target"]
    roles = {k: dict(v) for k, v in m["baseline_roles"].items()}
    # specify keeps its author timeout/retries from the shipped default.
    roles["specify"].setdefault("timeout_sec", 600)
    roles["specify"].setdefault("retries", 1)
    roles["commit"] = {"provider": "copilot", "model": "gpt-5-mini"}

    branch = tgt["codebase"]
    return {
        "roles": roles,
        "stages": {
            "judge": {"votes_per_axis": 5, "max_parallel": 5, "max_axes": 10,
                      "max_calls_per_axis": 2, "max_total_calls": 40},
            "converge": {"split": {"enabled": True, "max_loci": 4}},
            "reinforce": {"enabled": False, "max_workers": 2, "max_total_calls": 4},
            "reinvestigation": {"live": True, "max_rounds": 2},
            "commit": {"filename_only_threshold": 20},
        },
        # Key the target by the branch LEAF folder name so the loader's
        # codebase->target match (key == leaf, case-insensitive) binds it, AND set
        # explicit `codebase` paths as a belt-and-braces second match key.
        "targets": {
            tgt["branch_leaf"]: {
                "db": {"kind": "sqlite", "path": tgt["db_path"], "codebase": branch},
                "tests": {
                    "command": ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                    "cwd": tgt.get("tests_cwd", "server"),
                    "timeout_sec": 120,
                    "codebase": branch,
                },
                "http_shape": {
                    "test_dir": "server/tests",
                    "app_fixture": "",
                    "setup_block_file": HTTP_SHAPE_SETUP,
                    "codebase": branch,
                },
            }
        },
        "ops": {
            "swarm_run": {"allow": False},
            "apply": {"backup_dir": ".apply_backups", "backup_ttl_hours": 168},
            "ledger": {"enabled": True, "db_path": "hive_ledger.db"},
        },
        "providers": {
            "copilot": {"exe": None, "allow": "--allow-all", "timeout_sec": 300,
                        "read_only": True},
            "codex": {"exe": None, "lock_timeout_sec": 1800},
            "openai": {"base_url": "https://api.deepinfra.com/v1/openai",
                       "api_key_env": "DEEPINFRA_TOKEN"},
        },
    }


def cell_config(m: dict, cell: dict) -> dict:
    cfg = baseline_config(m)
    rk = ROLE_KEY[cell["stage"]]
    # OFAT: swap exactly one role.
    cfg["roles"][rk] = {**cfg["roles"].get(rk, {}),
                        "provider": cell["provider"], "model": cell["model"]}
    if cell["path"] == "run":
        cfg["ops"]["swarm_run"]["allow"] = True
    if cell.get("needs_reinforce"):
        cfg["stages"]["reinforce"]["enabled"] = True
    # Isolate this cell's cost in its own ledger so spend is attributable per cell.
    cfg["ops"]["ledger"]["db_path"] = os.path.join(
        HERE, "results", cell["id"], "ledger.db")
    return cfg


def smoke_config(m: dict) -> dict:
    """All roles forced to openai/gpt-oss-20b — the cheapest wiring probe (req #5).

    Deliberately SLIM so it verifies plumbing, not coverage: judge votes drop to 1
    with a low total cap, converge-split off, no live re-investigation. One full
    investigate --specify then lands at roughly decompose(1) + judge(<=axes, cap 6)
    + converge(1) + specify(1) + review(1) ~= under 10 gpt-oss-20b calls.
    """
    cfg = baseline_config(m)
    sm = m["smoke_model"]
    for rk in cfg["roles"]:
        cfg["roles"][rk] = {**cfg["roles"][rk],
                            "provider": sm["provider"], "model": sm["model"]}
    cfg["stages"]["judge"] = {"votes_per_axis": 1, "max_parallel": 3, "max_axes": 4,
                              "max_calls_per_axis": 1, "max_total_calls": 6}
    cfg["stages"]["converge"] = {"split": {"enabled": False, "max_loci": 4}}
    cfg["stages"]["reinvestigation"] = {"live": False, "max_rounds": 1}
    cfg["ops"]["ledger"]["db_path"] = os.path.join(HERE, "results", "_smoke", "ledger.db")
    return cfg


def write_profile(name: str, cfg: dict) -> str:
    path = os.path.join(CONFIG_DIR, f"hive.config.{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def clean() -> int:
    n = 0
    for p in glob.glob(os.path.join(CONFIG_DIR, "hive.config.perf-*.json")):
        os.remove(p)
        n += 1
    sm = os.path.join(CONFIG_DIR, "hive.config.perf-smoke.json")
    if os.path.exists(sm):
        os.remove(sm)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate perf-sweep --profile configs")
    ap.add_argument("--clean", action="store_true", help="delete generated perf-* profiles")
    args = ap.parse_args()
    if args.clean:
        print(f"Removed {clean()} generated perf profile(s).")
        return
    m = load_matrix()
    written = []
    for cell in m["cells"]:
        name = f"perf-{cell['id']}"
        os.makedirs(os.path.join(HERE, "results", cell["id"]), exist_ok=True)
        written.append(write_profile(name, cell_config(m, cell)))
    os.makedirs(os.path.join(HERE, "results", "_smoke"), exist_ok=True)
    written.append(write_profile("perf-smoke", smoke_config(m)))
    print(f"Wrote {len(written)} profile(s) to {CONFIG_DIR}:")
    for p in written:
        print("  ", os.path.basename(p))


if __name__ == "__main__":
    main()
