"""Real-key trial (group 0073) — box-0 acceptance synthesis on the REAL FlowGate repo.

R0001/CH0002: the box/keymaster axes hit their pure-synthesis ceiling at 0072 L6b. The
user's next directive is NOT another abstract axis but a *real key cut in the workshop*
(``C:\\workspace\\projects\\Hivework-test``): run the engine end-to-end on an ACTUAL
project, dropping ts0006's "no model, no FlowGate" disk-SUT shell.

This harness drops the "no FlowGate" half: it drives the EXACT 0065 live path, but the
codebase root is the real ``Hivework-test/FlowGate/server`` tree and the acceptance
oracle grounds against a REAL FlowGate symbol (``modules/flow_gate/linter.py::
TITLE_MAX_LEN``, currently 100). A ``## 수용기준`` demands it equal 120; the spec's source
edit is the fix (100→120). box-0 synthesises the RED unit-value test, and
``verify.verify_red_green`` certifies it through a REAL pytest subprocess run inside the
workshop repo, then restores every touched file (non-destructive — net zero repo change).

    hive.config.load_config(target.acceptance)          # Gap C config binding (real root)
      → hive.acceptance_specify_kwargs(cfg, root)       # Gap C CLI→specify builder
      → specify._synthesize_acceptance_red_test(...)     # box-0 synthesis, grounded on real code
      → verify.verify_red_green(spec, runner)            # REAL pytest, REAL FlowGate tree

Run:  PYTHONPATH=<Hivework> PYTHONIOENCODING=utf-8 python perf/realkey_flowgate_0073.py
Exits non-zero on any miss (not grounded / no-bite / still-red / not restored).
"""
import json
import os
import sys
import tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from hive.config import load_config  # noqa: E402
from hive import specify  # noqa: E402
from hive import verify as verifymod  # noqa: E402
import hive  # noqa: E402  (acceptance_specify_kwargs lives on the CLI module hive.py)
import importlib.util as _ilu  # noqa: E402


def _load_hive_cli():
    path = os.path.join(_REPO, "hive.py")
    spec = _ilu.spec_from_file_location("hive_cli_entry_realkey", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The workshop's real FlowGate server tree (a genuine git repo with a real pytest suite).
_FLOWGATE = os.environ.get(
    "REALKEY_FLOWGATE_ROOT",
    r"C:\workspace\projects\Hivework-test\FlowGate\server",
)
_TARGET_REL = "modules/flow_gate/linter.py"
_TARGET = f"{_TARGET_REL}::TITLE_MAX_LEN"

# A design doc with an explicit unit_value oracle against the REAL symbol. read_acceptance_
# criteria needs BOTH id and prose (an oracle-only criterion is dropped), so we supply both.
_CRITERIA = (
    "# Feature design — raise the document title cap\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    f"  prose: {_TARGET} must equal 120 (the widened title length cap)\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    f"    target: {_TARGET}\n"
    "    must: equals\n"
    "    expected: 120\n"
)


def _write_config(root: str, criteria_text: str) -> str:
    """A hive config binding the real FlowGate root to a pytest runner + acceptance doc."""
    targets = {
        "flowgate": {
            "tests": {
                "command": ["python", "-m", "pytest", "-q"],
                "codebase": root,
                "timeout_sec": 300,
                # Real-repo friction the synthetic ts0006 (fresh temp dirs) never hit:
                # the unit_value gate loads the SAME real module across red→green, and a
                # same-size value flip (100↔120) within the .pyc mtime resolution leaves
                # Python reading STALE bytecode. Forbid bytecode writes so every run
                # recompiles the live source (paired with a pre-run .pyc purge below).
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},
            },
            "acceptance": {
                "codebase": root,
                "criteria_text": criteria_text,
                "test_dir": "tests",
            },
        }
    }
    fd, cfg_path = tempfile.mkstemp(prefix="realkey_0073_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


def _criteria(target: str, expected: int) -> str:
    """A ``## 수용기준`` design with an explicit unit_value oracle (id + prose both, so
    read_acceptance_criteria keeps it)."""
    return (
        "# Feature design — title cap scenario\n\n"
        "## 수용기준\n"
        "- id: AC1\n"
        f"  prose: {target} must equal {expected}\n"
        "  oracle:\n"
        "    kind: unit_value\n"
        f"    target: {target}\n"
        "    must: equals\n"
        f"    expected: {expected}\n"
    )


def _spec_with_fix(fix_new: int) -> dict:
    """A spec whose ONLY authored edit is the real source fix (TITLE_MAX_LEN 100→fix_new).
    box-0 appends the RED unit-value test + verify.red_test_node from the criteria text."""
    return {
        "edits": [{
            "id": "FIX_TITLE_CAP",
            "file": _TARGET_REL,
            "anchor_old": "TITLE_MAX_LEN = 100",
            "replacement_new": f"TITLE_MAX_LEN = {fix_new}",
            "rationale": f"build the feature: set the title cap to {fix_new}",
            "confidence": "high",
        }],
        "termination": "ready_to_apply",
        "verify": {},
    }


def _run_scenario(cli, root: str, *, target: str, expected: int, fix_new: int) -> dict:
    """Drive the full 0065 live path once against the REAL FlowGate tree and report the
    grounded/verdict/restored triple. Non-destructive: verify restores every touched file."""
    tgt_abs = os.path.join(root, _TARGET_REL)
    before_bytes = open(tgt_abs, "rb").read()
    # Purge any stale bytecode for the target so a prior scenario's value can't leak in
    # via __pycache__ (see the env note in _write_config).
    _pyc_dir = os.path.join(os.path.dirname(tgt_abs), "__pycache__")
    _stem = os.path.splitext(os.path.basename(tgt_abs))[0]
    if os.path.isdir(_pyc_dir):
        for _f in os.listdir(_pyc_dir):
            if _f.startswith(_stem + ".") and _f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(_pyc_dir, _f))
                except OSError:
                    pass

    cfg = load_config(path=_write_config(root, _criteria(target, expected)))
    kw = cli.acceptance_specify_kwargs(cfg, root)
    spec = _spec_with_fix(fix_new)
    spec = specify._synthesize_acceptance_red_test(
        spec, kw.get("acceptance_criteria_text"), root,
        setup_block=kw.get("acceptance_setup_block"),
        app_fixture=kw.get("acceptance_app_fixture"),
        test_dir=kw.get("acceptance_test_dir", "tests"))

    vblock = spec.get("verify") or {}
    node = vblock.get("red_test_node")
    grounded = bool(node)

    runner = cfg.test_runner_for_codebase(root)
    verdict = {"transition": "SKIPPED", "verified": False}
    if grounded:
        backup_root = os.path.join(tempfile.gettempdir(), "realkey_0073_backups")
        verdict = verifymod.verify_red_green(spec, root, runner, backup_root, ttl_hours=1)

    after_bytes = open(tgt_abs, "rb").read()
    return {
        "target": target, "expected": expected, "fix_new": fix_new,
        "grounded": grounded, "synth_node": node,
        "transition": verdict.get("transition"),
        "verified": bool(verdict.get("verified")),
        "red_status": (verdict.get("red") or {}).get("status"),
        "green_status": (verdict.get("green") or {}).get("status"),
        "source_restored": (before_bytes == after_bytes),
    }


def main() -> int:
    cli = _load_hive_cli()
    root = _FLOWGATE
    out: dict = {"codebase_root": root}

    if not os.path.isdir(root):
        out["error"] = f"workshop FlowGate root not found: {root}"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    # S1 — happy path: real symbol, expected(120) ≠ current(100), fix 100→120.
    #   EXPECT: grounded, red_to_green, verified, restored.
    s1 = _run_scenario(cli, root, target=_TARGET, expected=120, fix_new=120)
    # S2 — fail-closed on an UNGROUNDED symbol: box-0 must decline (no guessed test).
    #   EXPECT: grounded=False, no node, source untouched.
    s2 = _run_scenario(cli, root, target=f"{_TARGET_REL}::TITLE_MAX_LEN_NOPE",
                       expected=120, fix_new=120)
    # S3 — no-bite guard on real code: expected(100) == current(100) so the test passes
    #   WITHOUT the fix. EXPECT: grounded but transition=no_bite, verified=False, restored.
    s3 = _run_scenario(cli, root, target=_TARGET, expected=100, fix_new=120)

    out["scenarios"] = {"S1_happy": s1, "S2_ungrounded_decline": s2, "S3_no_bite": s3}
    out["checks"] = {
        "S1_red_to_green_verified_restored":
            s1["grounded"] and s1["verified"]
            and s1["transition"] == "red_to_green" and s1["source_restored"],
        "S2_declines_no_node_restored":
            (not s2["grounded"]) and s2["synth_node"] is None and s2["source_restored"],
        "S3_no_bite_guard_fires_restored":
            s3["grounded"] and (not s3["verified"])
            and s3["transition"] == "test_does_not_bite"
            and s3["red_status"] == "pass" and s3["source_restored"],
    }
    out["GO"] = all(out["checks"].values())
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
