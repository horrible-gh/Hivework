"""TS0006 end-to-end harness (group 0065) — proves box-0's LIVE wiring goes red→green.

Unlike TSR0010's scratchpad harness (which the AI review flagged as absent from the working
tree), this lives in the repo and is re-runnable:  PYTHONPATH=<repo> python perf/ts0006_e2e.py

It chains the EXACT 0065 live path on a deterministic disk SUT (no model, no FlowGate):

    hive.config.load_config(targets.<name>.acceptance)         # Gap C config binding
      → hive.acceptance_specify_kwargs(cfg, root)              # Gap C CLI→specify builder
      → specify._synthesize_acceptance_red_test(spec, **kw)    # box-0 synthesis (TR0008)
      → verify.verify_red_green(spec, runner)                  # real pytest subprocess

Scenario 1/2c (unit_value, harness-free): a module value starts wrong (RED), the source fix
lands (GREEN). Scenario 3 (ablation OFF): no binding + no design path → builder returns {} →
no red_test_node → synthesis no-op. Prints a JSON verdict; exits non-zero on any miss.
"""
import importlib.util
import json
import os
import sys
import tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from hive.config import load_config, RunnerConfig  # noqa: E402
from hive import specify  # noqa: E402
from hive import verify as verifymod  # noqa: E402


def _load_hive_cli():
    path = os.path.join(_REPO, "hive.py")
    spec = importlib.util.spec_from_file_location("hive_cli_entry_ts0006", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CRITERIA = (
    "# Feature design\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: app/calc.py::answer must equal 42\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::answer\n"
    "    must: equals\n"
    "    expected: 42\n"
)


def _make_sut() -> str:
    root = tempfile.mkdtemp(prefix="ts0006_sut_")
    os.makedirs(os.path.join(root, "app"), exist_ok=True)
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    # RED state: the module value is wrong until the fix lands.
    with open(os.path.join(root, "app", "calc.py"), "w", encoding="utf-8") as fh:
        fh.write("answer = 0\n")
    return root


def _write_config(root: str, *, with_binding: bool) -> str:
    targets: dict = {"sut": {"tests": {"command": ["python", "-m", "pytest", "-q"],
                                       "codebase": root, "timeout_sec": 120}}}
    if with_binding:
        targets["sut"]["acceptance"] = {"codebase": root, "criteria_text": _CRITERIA,
                                        "test_dir": "tests"}
    cfg_path = os.path.join(root, "hive.config.ts0006.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


def _base_spec() -> dict:
    """A spec whose ONLY edit is the source fix (answer 0→42). The acceptance pass appends
    the red-test edit + verify.red_test_node when criteria text is supplied."""
    return {
        "edits": [{"id": "FIX_ANSWER", "file": "app/calc.py",
                   "anchor_old": "answer = 0", "replacement_new": "answer = 42",
                   "rationale": "build the feature: answer is 42", "confidence": "high"}],
        "termination": "ready_to_apply", "verify": {},
    }


def main() -> int:
    cli = _load_hive_cli()
    results: dict = {}

    # ── Scenario 1/2c: ON (config binding) → builder feeds synth → red→green ──────
    root = _make_sut()
    cfg = load_config(path=_write_config(root, with_binding=True))
    kw = cli.acceptance_specify_kwargs(cfg, root)
    results["builder_kwargs_on"] = {k: (v[:40] + "…" if isinstance(v, str) and len(v) > 40
                                        else v) for k, v in kw.items()}
    spec = _base_spec()
    spec = specify._synthesize_acceptance_red_test(
        spec, kw.get("acceptance_criteria_text"), root,
        setup_block=kw.get("acceptance_setup_block"),
        app_fixture=kw.get("acceptance_app_fixture"),
        test_dir=kw.get("acceptance_test_dir", "tests"))
    node = (spec.get("verify") or {}).get("red_test_node")
    results["synth_node"] = node
    runner = cfg.test_runner_for_codebase(root)
    backup_root = os.path.join(root, ".apply_backups")
    verdict = verifymod.verify_red_green(spec, root, runner, backup_root, ttl_hours=1)
    results["scenario_1"] = {"transition": verdict.get("transition"),
                             "verified": verdict.get("verified"),
                             "red_status": (verdict.get("red") or {}).get("status"),
                             "green_status": (verdict.get("green") or {}).get("status")}

    # ── Scenario 3: ablation OFF (no binding, no design path) → builder {} → no-op ──
    root2 = _make_sut()
    cfg2 = load_config(path=_write_config(root2, with_binding=False))
    kw2 = cli.acceptance_specify_kwargs(cfg2, root2, None)
    spec2 = _base_spec()
    spec2 = specify._synthesize_acceptance_red_test(
        spec2, kw2.get("acceptance_criteria_text"), root2,
        test_dir=kw2.get("acceptance_test_dir", "tests"))
    results["scenario_3"] = {"builder_kwargs": kw2,
                             "red_test_node": (spec2.get("verify") or {}).get("red_test_node"),
                             "edits": [e["id"] for e in spec2["edits"]]}

    print(json.dumps(results, indent=2, ensure_ascii=False))

    ok = (results["scenario_1"]["transition"] == "red_to_green"
          and results["scenario_1"]["verified"] is True
          and kw2 == {}
          and results["scenario_3"]["red_test_node"] is None)
    print("\nTS0006 e2e:", "PASS (go)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
