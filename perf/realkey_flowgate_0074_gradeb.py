"""Real-key trial GRADE B (group 0074) — box-0 http_read acceptance on the REAL FlowGate.

R0001/NR0003 (0074): 0073 cut the FIRST real key but only on the harness-free GRADE A
path (``unit_value``, a module symbol). The GRADE B path (``http_read`` — an assertion on
a live HTTP payload) stayed deferred for one reason: box-0's http_read synthesis binds the
generated test to the target's OWN seeded ``TestClient`` fixture, and ``discover_app_fixture``
DECLINES on the real FlowGate tree because it defines MANY distinctly-named client fixtures
(``t330_client``, ``ts021_client``, ``t823_inbox_client`` …) — auto-binding to an arbitrary
one is unsafe. NR0003 조각 A: supply ONE faithful shared harness (a ``setup_block``) so box-0
can ground a REAL route.

This harness removes that deferral for the simplest real route: ``GET /api/v1/help``
(unauthenticated, no DB) whose payload carries ``"version": "v1"``. A ``## 수용기준`` http_read
criterion demands ``version == "v2"`` (the widened API version); the spec's only authored edit
is the source fix (``v1`` → ``v2``). box-0 synthesises the RED http_read test bound to the
supplied ``client`` harness, and ``verify.verify_red_green`` certifies it through a REAL pytest
subprocess run inside the workshop repo, then restores every touched file (non-destructive).

    config.AcceptanceConfig(setup_block=<shared TestClient harness>)  # NR0003 조각 A
      → hive.acceptance_specify_kwargs(cfg, root)                     # Gap C CLI→specify builder
      → specify._synthesize_acceptance_red_test(spec, **kw)           # box-0 http_read synthesis
      → verify.verify_red_green(spec, root, runner)                   # REAL pytest, REAL FlowGate

Triangulation (mirrors 0073 grade A):
  B1 happy        — expected "v2" != current "v1", fix v1->v2, WITH harness  => red_to_green.
  B2 needs-harness— SAME criterion, NO setup_block/app_fixture: discover_app_fixture sees the
                    real tree's many client fixtures, DECLINES => grounded=False, no node, no
                    write. This is the whole point of 조각 A: without a supplied harness the
                    grade-B lever stays a safe no-op.
  B3 no-bite      — expected "v1" == current "v1": the RED test passes WITHOUT the fix
                    => transition=test_does_not_bite (red=pass), source restored.

Run:  PYTHONPATH=<Hivework> PYTHONIOENCODING=utf-8 python perf/realkey_flowgate_0074_gradeb.py
Exits non-zero on any miss (not grounded / not-declined / still-red / not restored).
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
import importlib.util as _ilu  # noqa: E402


def _load_hive_cli():
    path = os.path.join(_REPO, "hive.py")
    spec = _ilu.spec_from_file_location("hive_cli_entry_realkey_gb", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The workshop's real FlowGate server tree (a genuine git repo with a real pytest suite).
_FLOWGATE = os.environ.get(
    "REALKEY_FLOWGATE_ROOT",
    r"C:\workspace\projects\Hivework-test\FlowGate\server",
)
_TARGET_REL = "modules/flow_gate/api/v1/help_routes.py"
_ROUTE = "/api/v1/help"

# A faithful shared TestClient harness (NR0003 조각 A). It mounts the REAL help router on a
# bare FastAPI app — /help is unauthenticated and touches no DB, so no seeding is needed. This
# is exactly the ``setup_block`` box-0 needs where auto-discovery declines on the real tree.
_SETUP_BLOCK = (
    "import pytest\n"
    "from fastapi import FastAPI\n"
    "from fastapi.testclient import TestClient\n"
    "from modules.flow_gate.api.v1 import help_routes\n\n\n"
    "@pytest.fixture\n"
    "def client():\n"
    "    app = FastAPI()\n"
    "    app.include_router(help_routes.router)\n"
    "    return TestClient(app)\n"
)


def _criteria(route: str, expected: str) -> str:
    """A ``## 수용기준`` design with an explicit http_read equals oracle (id + prose both, so
    read_acceptance_criteria keeps it)."""
    return (
        "# Feature design — widen the advertised API version\n\n"
        "## 수용기준\n"
        "- id: AC1\n"
        f"  prose: GET {route} must expose version {expected}\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    verb: get\n"
        f"    route: {route}\n"
        "    json_path: version\n"
        "    must: equals\n"
        f"    expected: {expected}\n"
    )


def _write_config(root: str, criteria_text: str, *, with_harness: bool) -> str:
    """A hive config binding the real FlowGate root to a pytest runner + acceptance doc.

    ``with_harness`` supplies the shared ``setup_block`` (grade-B enabler, 조각 A). Omitting it
    forces box-0 through ``discover_app_fixture``, which DECLINES on the real tree's many
    distinctly-named client fixtures — the B2 no-op proof."""
    acceptance = {
        "codebase": root,
        "criteria_text": criteria_text,
        "test_dir": "tests",
    }
    if with_harness:
        acceptance["setup_block"] = _SETUP_BLOCK
    targets = {
        "flowgate": {
            "tests": {
                "command": ["python", "-m", "pytest", "-q"],
                "codebase": root,
                "timeout_sec": 300,
                # Same real-repo friction 0073 hit: a same-byte-width value flip ("v1"<->"v2")
                # within the .pyc mtime resolution leaves Python reading STALE bytecode for the
                # imported help_routes module across red->green. Forbid bytecode writes (paired
                # with a pre-run .pyc purge below) so every run recompiles the live source.
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},
            },
            "acceptance": acceptance,
        }
    }
    fd, cfg_path = tempfile.mkstemp(prefix="realkey_0074_gb_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


def _spec_with_fix(fix_new: str) -> dict:
    """A spec whose ONLY authored edit is the real source fix ("version" v1->fix_new).
    box-0 appends the RED http_read test + verify.red_test_node from the criteria text."""
    return {
        "edits": [{
            "id": "FIX_API_VERSION",
            "file": _TARGET_REL,
            "anchor_old": '"version": "v1"',
            "replacement_new": f'"version": "{fix_new}"',
            "rationale": f"build the feature: advertise API version {fix_new}",
            "confidence": "high",
        }],
        "termination": "ready_to_apply",
        "verify": {},
    }


def _purge_pyc(tgt_abs: str) -> None:
    _pyc_dir = os.path.join(os.path.dirname(tgt_abs), "__pycache__")
    _stem = os.path.splitext(os.path.basename(tgt_abs))[0]
    if os.path.isdir(_pyc_dir):
        for _f in os.listdir(_pyc_dir):
            if _f.startswith(_stem + ".") and _f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(_pyc_dir, _f))
                except OSError:
                    pass


def _run_scenario(cli, root: str, *, expected: str, fix_new: str,
                  with_harness: bool) -> dict:
    """Drive the box-0 http_read live path once against the REAL FlowGate tree and report the
    grounded/verdict/restored triple. Non-destructive: verify restores every touched file."""
    tgt_abs = os.path.join(root, _TARGET_REL)
    before_bytes = open(tgt_abs, "rb").read()
    _purge_pyc(tgt_abs)

    cfg = load_config(path=_write_config(root, _criteria(_ROUTE, expected),
                                         with_harness=with_harness))
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
        backup_root = os.path.join(tempfile.gettempdir(), "realkey_0074_gb_backups")
        verdict = verifymod.verify_red_green(spec, root, runner, backup_root, ttl_hours=1)

    after_bytes = open(tgt_abs, "rb").read()
    return {
        "expected": expected, "fix_new": fix_new, "with_harness": with_harness,
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
    out: dict = {"codebase_root": root, "route": _ROUTE, "grade": "B (http_read)"}

    if not os.path.isdir(root):
        out["error"] = f"workshop FlowGate root not found: {root}"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    # B1 — happy path: real route, expected "v2" != current "v1", fix v1->v2, WITH harness.
    #   EXPECT: grounded, red_to_green, verified, restored.
    b1 = _run_scenario(cli, root, expected="v2", fix_new="v2", with_harness=True)
    # B2 — needs-harness decline: SAME criterion but NO setup_block → discover_app_fixture sees
    #   the real tree's many client fixtures and declines. EXPECT: grounded=False, no node,
    #   source untouched. (The whole justification for 조각 A.)
    b2 = _run_scenario(cli, root, expected="v2", fix_new="v2", with_harness=False)
    # B3 — no-bite guard on real code: expected "v1" == current "v1" so the RED test passes
    #   WITHOUT the fix. EXPECT: grounded but transition=test_does_not_bite, restored.
    b3 = _run_scenario(cli, root, expected="v1", fix_new="v2", with_harness=True)

    out["scenarios"] = {"B1_happy": b1, "B2_needs_harness_decline": b2, "B3_no_bite": b3}
    out["checks"] = {
        "B1_red_to_green_verified_restored":
            b1["grounded"] and b1["verified"]
            and b1["transition"] == "red_to_green" and b1["source_restored"],
        "B2_declines_without_harness_restored":
            (not b2["grounded"]) and b2["synth_node"] is None and b2["source_restored"],
        "B3_no_bite_guard_fires_restored":
            b3["grounded"] and (not b3["verified"])
            and b3["transition"] == "test_does_not_bite"
            and b3["red_status"] == "pass" and b3["source_restored"],
    }
    out["GO"] = all(out["checks"].values())
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
