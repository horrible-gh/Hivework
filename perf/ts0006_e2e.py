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


# box-1 (group 0066, level-2): a design with TWO criteria must yield TWO gates, both
# certified red→green through the real subprocess.
_CRITERIA_MULTI = (
    "# Feature design\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: app/calc.py::answer must equal 42\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::answer\n"
    "    must: equals\n"
    "    expected: 42\n"
    "- id: AC2\n"
    "  prose: app/calc.py::greeting must equal 'hi'\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    "    target: app/calc.py::greeting\n"
    "    must: equals\n"
    "    expected: hi\n"
)


def _make_sut(*, multi: bool = False) -> str:
    root = tempfile.mkdtemp(prefix="ts0006_sut_")
    os.makedirs(os.path.join(root, "app"), exist_ok=True)
    os.makedirs(os.path.join(root, "tests"), exist_ok=True)
    # RED state: the module value(s) are wrong until the fix lands.
    body = "answer = 0\n" + ("greeting = 'x'\n" if multi else "")
    with open(os.path.join(root, "app", "calc.py"), "w", encoding="utf-8") as fh:
        fh.write(body)
    return root


def _write_config(root: str, *, with_binding: bool, criteria_text: str = _CRITERIA) -> str:
    targets: dict = {"sut": {"tests": {"command": ["python", "-m", "pytest", "-q"],
                                       "codebase": root, "timeout_sec": 120}}}
    if with_binding:
        targets["sut"]["acceptance"] = {"codebase": root, "criteria_text": criteria_text,
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

    # ── Scenario 2: box-1 level-2 — TWO criteria → TWO gates, both red→green ─────
    rootm = _make_sut(multi=True)
    cfgm = load_config(path=_write_config(rootm, with_binding=True,
                                          criteria_text=_CRITERIA_MULTI))
    kwm = cli.acceptance_specify_kwargs(cfgm, rootm)
    specm = _base_spec()
    # Add the SECOND source fix so both gates can go green.
    specm["edits"].append({"id": "FIX_GREETING", "file": "app/calc.py",
                           "anchor_old": "greeting = 'x'", "replacement_new": "greeting = 'hi'",
                           "rationale": "build the feature: greeting is hi", "confidence": "high"})
    specm = specify._synthesize_acceptance_red_test(
        specm, kwm.get("acceptance_criteria_text"), rootm,
        test_dir=kwm.get("acceptance_test_dir", "tests"))
    vblockm = specm.get("verify") or {}
    runnerm = cfgm.test_runner_for_codebase(rootm)
    verdictm = verifymod.verify_red_green(
        specm, rootm, runnerm, os.path.join(rootm, ".apply_backups"), ttl_hours=1)
    results["scenario_2"] = {
        "transition": verdictm.get("transition"),
        "verified": verdictm.get("verified"),
        "gate_count": len(vblockm.get("red_test_nodes") or []),
        "red_all_fail": all(r.get("status") == "fail" for r in (verdictm.get("red_runs") or [])),
        "green_all_pass": all(g.get("passed") for g in (verdictm.get("green_runs") or [])),
    }

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

    # ── Scenario 4: box-2 level-3 — ONE criterion names TWO fields → ONE gate ─────
    # Field cross-validation (GAP-2): a single http_read criterion asserts `total` AND
    # `items` on one fetch. The gate stays SINGLE (no red_test_nodes — this is not a
    # multi-gate), the test body carries TWO field assertions, and the whole thing goes
    # red→green through the real pytest subprocess (RED: `total` missing; GREEN: fix adds it).
    root4 = tempfile.mkdtemp(prefix="ts0006_sut_mf_")
    os.makedirs(os.path.join(root4, "app"), exist_ok=True)
    os.makedirs(os.path.join(root4, "tests"), exist_ok=True)
    with open(os.path.join(root4, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 'def _rows():\n    return [1, 2, 3]\n'
                 'def _n():\n    return 3\n\n'
                 '@router.get("/summary")\n'
                 'def summary():\n'
                 '    return {"items": _rows()}\n')  # RED: no `total` yet
    with open(os.path.join(root4, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria4 = ("# Feature design\n\n## 수용기준\n"
                 "- id: AC1\n"
                 "  prose: GET /api/v1/summary must return non-empty `total` and `items`\n")
    cfg4 = load_config(path=_write_config(root4, with_binding=True, criteria_text=criteria4))
    kw4 = cli.acceptance_specify_kwargs(cfg4, root4)
    spec4 = {"edits": [{"id": "FIX_TOTAL", "file": "app/routes.py",
                        "anchor_old": 'return {"items": _rows()}',
                        "replacement_new": 'return {"total": _n(), "items": _rows()}',
                        "rationale": "build the feature: expose total", "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec4 = specify._synthesize_acceptance_red_test(
        spec4, kw4.get("acceptance_criteria_text"), root4,
        setup_block=kw4.get("acceptance_setup_block"),
        app_fixture=kw4.get("acceptance_app_fixture"),
        test_dir=kw4.get("acceptance_test_dir", "tests"))
    vblock4 = spec4.get("verify") or {}
    accept4 = [e for e in spec4["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    assert_field_count = accept4[0]["content"].count(" in payload") if accept4 else 0
    runner4 = cfg4.test_runner_for_codebase(root4)
    verdict4 = verifymod.verify_red_green(
        spec4, root4, runner4, os.path.join(root4, ".apply_backups"), ttl_hours=1)
    results["scenario_4"] = {
        "transition": verdict4.get("transition"),
        "verified": verdict4.get("verified"),
        "gate_count_plural": len(vblock4.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept4),
        "assert_field_count": assert_field_count,
        "red_status": (verdict4.get("red") or {}).get("status"),
        "green_status": (verdict4.get("green") or {}).get("status"),
    }

    # ── Scenario 5: box-3 level-4 — ONE criterion, POST→GET multi-step → ONE gate ──
    # Multi-step (GAP-3): a single criterion's explicit ``steps:`` oracle creates a project
    # (POST) then asserts it appears in the list (GET). The gate stays SINGLE (no
    # red_test_nodes), the test body issues BOTH calls in sequence, and it goes red→green
    # through the real pytest subprocess (RED: POST is a no-op so the list stays empty;
    # GREEN: the fix persists the project so `items[].name` is non-empty). A request body is
    # NEVER guessed — it is lifted verbatim from the author's oracle.
    root5 = tempfile.mkdtemp(prefix="ts0006_sut_ms_")
    os.makedirs(os.path.join(root5, "app"), exist_ok=True)
    os.makedirs(os.path.join(root5, "tests"), exist_ok=True)
    with open(os.path.join(root5, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 '_STORE = []\n\n'
                 '@router.post("/projects", status_code=201)\n'
                 'def create(project: dict):\n'
                 '    return {"ok": True}  # RED: not persisted\n\n'
                 '@router.get("/projects")\n'
                 'def list_projects():\n'
                 '    return {"items": _STORE}\n')
    with open(os.path.join(root5, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria5 = (
        "# Feature design\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: creating a project makes it appear in the list\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    steps:\n"
        "      - verb: post\n"
        "        route: /api/v1/projects\n"
        "        body: {name: \"alpha\"}\n"
        "        expect_status: 201\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items[].name\n"
        "            must: non_empty\n")
    cfg5 = load_config(path=_write_config(root5, with_binding=True, criteria_text=criteria5))
    kw5 = cli.acceptance_specify_kwargs(cfg5, root5)
    spec5 = {"edits": [{"id": "FIX_PERSIST", "file": "app/routes.py",
                        "anchor_old": '    return {"ok": True}  # RED: not persisted',
                        "replacement_new": '    _STORE.append(project)\n    return {"ok": True}',
                        "rationale": "build the feature: persist the created project",
                        "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec5 = specify._synthesize_acceptance_red_test(
        spec5, kw5.get("acceptance_criteria_text"), root5,
        setup_block=kw5.get("acceptance_setup_block"),
        app_fixture=kw5.get("acceptance_app_fixture"),
        test_dir=kw5.get("acceptance_test_dir", "tests"))
    vblock5 = spec5.get("verify") or {}
    accept5 = [e for e in spec5["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    body5 = accept5[0]["content"] if accept5 else ""
    runner5 = cfg5.test_runner_for_codebase(root5)
    verdict5 = verifymod.verify_red_green(
        spec5, root5, runner5, os.path.join(root5, ".apply_backups"), ttl_hours=1)
    results["scenario_5"] = {
        "transition": verdict5.get("transition"),
        "verified": verdict5.get("verified"),
        "gate_count_plural": len(vblock5.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept5),
        "post_call": body5.count(".post("),
        "get_call": body5.count(".get("),
        "step_count": body5.count(".post(") + body5.count(".get("),
        "red_status": (verdict5.get("red") or {}).get("status"),
        "green_status": (verdict5.get("green") or {}).get("status"),
    }

    # Multi-route (GAP-4): a single criterion's explicit ``reads:`` oracle asserts the created
    # project surfaces in TWO INDEPENDENT routes — the list (GET /projects) AND the summary
    # count (GET /dashboard/summary). There is NO ordering between them (unlike scenario_5's
    # POST→GET). The gate stays SINGLE (no red_test_nodes), the test body issues BOTH reads
    # into their own payloads, and it goes red→green through the real pytest subprocess (RED:
    # summary reads a SEPARATE empty store so `project_count` is missing/zero and `items` is
    # empty; GREEN: the fix shares one store so both reads reflect the seeded project).
    root6 = tempfile.mkdtemp(prefix="ts0006_sut_mr_")
    os.makedirs(os.path.join(root6, "app"), exist_ok=True)
    os.makedirs(os.path.join(root6, "tests"), exist_ok=True)
    with open(os.path.join(root6, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 '_STORE = [{"name": "seed"}]\n\n'
                 '@router.get("/projects")\n'
                 'def list_projects():\n'
                 '    return {"items": _STORE}\n\n'
                 '@router.get("/dashboard/summary")\n'
                 'def summary():\n'
                 '    return {}  # RED: count not surfaced\n')
    with open(os.path.join(root6, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria6 = (
        "# Feature design\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: a project reflects in both the list and the summary count\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    reads:\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items[].name\n"
        "            must: non_empty\n"
        "      - verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        asserts:\n"
        "          - json_path: project_count\n"
        "            must: exists\n")
    cfg6 = load_config(path=_write_config(root6, with_binding=True, criteria_text=criteria6))
    kw6 = cli.acceptance_specify_kwargs(cfg6, root6)
    spec6 = {"edits": [{"id": "FIX_SUMMARY", "file": "app/routes.py",
                        "anchor_old": '    return {}  # RED: count not surfaced',
                        "replacement_new": '    return {"project_count": len(_STORE)}',
                        "rationale": "build the feature: surface the project count",
                        "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec6 = specify._synthesize_acceptance_red_test(
        spec6, kw6.get("acceptance_criteria_text"), root6,
        setup_block=kw6.get("acceptance_setup_block"),
        app_fixture=kw6.get("acceptance_app_fixture"),
        test_dir=kw6.get("acceptance_test_dir", "tests"))
    vblock6 = spec6.get("verify") or {}
    accept6 = [e for e in spec6["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    body6 = accept6[0]["content"] if accept6 else ""
    runner6 = cfg6.test_runner_for_codebase(root6)
    verdict6 = verifymod.verify_red_green(
        spec6, root6, runner6, os.path.join(root6, ".apply_backups"), ttl_hours=1)
    results["scenario_6"] = {
        "transition": verdict6.get("transition"),
        "verified": verdict6.get("verified"),
        "gate_count_plural": len(vblock6.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept6),
        "get_call": body6.count(".get("),
        "post_call": body6.count(".post("),
        "route_count": body6.count(".get(") + body6.count(".post("),
        "payload_count": body6.count(".json()"),
        "red_status": (verdict6.get("red") or {}).get("status"),
        "green_status": (verdict6.get("green") or {}).get("status"),
    }

    # Relation (GAP-5, box-5 level-6): a single criterion's explicit relational assert
    # (``equals_len``) demands the summary's ``total`` field EQUAL the length of its ``items``
    # list — a cross-field invariant that compares one observed value to ANOTHER observed value
    # (not to a literal, the level-1..5 ceiling). The gate stays SINGLE (no red_test_nodes) and
    # goes red→green through the real pytest subprocess (RED: ``total`` is hardcoded 0 while
    # ``items`` has a seed, so ``0 != len(items)``; GREEN: the fix computes ``total`` from the
    # list). Triangulates against scenario_6: the assert body contains ``len(payload...)`` — a
    # value-to-value relation — where the route axis compared against a constant.
    root7 = tempfile.mkdtemp(prefix="ts0006_sut_rel_")
    os.makedirs(os.path.join(root7, "app"), exist_ok=True)
    os.makedirs(os.path.join(root7, "tests"), exist_ok=True)
    with open(os.path.join(root7, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 '_STORE = [{"name": "seed"}]\n\n'
                 '@router.get("/dashboard/summary")\n'
                 'def summary():\n'
                 '    return {"total": 0, "items": _STORE}  # RED: total != len(items)\n')
    with open(os.path.join(root7, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria7 = (
        "# Feature design\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: the summary total must equal the number of items\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    verb: get\n"
        "    route: /api/v1/dashboard/summary\n"
        "    asserts:\n"
        "      - json_path: total\n"
        "        must: equals_len\n"
        "        other_path: items\n")
    cfg7 = load_config(path=_write_config(root7, with_binding=True, criteria_text=criteria7))
    kw7 = cli.acceptance_specify_kwargs(cfg7, root7)
    spec7 = {"edits": [{"id": "FIX_TOTAL", "file": "app/routes.py",
                        "anchor_old": '    return {"total": 0, "items": _STORE}  # RED: total != len(items)',
                        "replacement_new": '    return {"total": len(_STORE), "items": _STORE}',
                        "rationale": "build the feature: compute total from the list",
                        "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec7 = specify._synthesize_acceptance_red_test(
        spec7, kw7.get("acceptance_criteria_text"), root7,
        setup_block=kw7.get("acceptance_setup_block"),
        app_fixture=kw7.get("acceptance_app_fixture"),
        test_dir=kw7.get("acceptance_test_dir", "tests"))
    vblock7 = spec7.get("verify") or {}
    accept7 = [e for e in spec7["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    body7 = accept7[0]["content"] if accept7 else ""
    runner7 = cfg7.test_runner_for_codebase(root7)
    verdict7 = verifymod.verify_red_green(
        spec7, root7, runner7, os.path.join(root7, ".apply_backups"), ttl_hours=1)
    results["scenario_7"] = {
        "transition": verdict7.get("transition"),
        "verified": verdict7.get("verified"),
        "gate_count_plural": len(vblock7.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept7),
        "get_call": body7.count(".get("),
        "post_call": body7.count(".post("),
        "len_call": body7.count("len("),          # >0 = value-to-value relation, not a literal
        "isinstance_list": body7.count("isinstance(") and "list)" in body7,
        "red_status": (verdict7.get("red") or {}).get("status"),
        "green_status": (verdict7.get("green") or {}).get("status"),
    }

    # ── Scenario 8: box-6 level-7 — ONE criterion, before/mutate/after DELTA → ONE gate ──
    # Delta (GAP-6): a single criterion's explicit ``delta:`` oracle reads the summary total
    # BEFORE, creates a project (POST), reads the total AFTER, and asserts it rose by exactly
    # one — the MAGNITUDE OF CHANGE, orthogonal to scenario_5's absolute post-state. The gate
    # stays SINGLE (no red_test_nodes) and goes red→green through the real pytest subprocess
    # (RED: POST is a no-op so the count never moves, ``1 != 1 + 1``; GREEN: the fix persists the
    # project so the count climbs by one). Triangulates against scenario_5: the body reads the
    # SAME route twice (get_call == 2) and the assert references ``before`` — a difference, not
    # an absolute. A request body / delta amount is NEVER guessed — lifted from the oracle.
    root8 = tempfile.mkdtemp(prefix="ts0006_sut_delta_")
    os.makedirs(os.path.join(root8, "app"), exist_ok=True)
    os.makedirs(os.path.join(root8, "tests"), exist_ok=True)
    with open(os.path.join(root8, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 '_STORE = [{"id": 1}]\n\n'
                 '@router.post("/projects", status_code=201)\n'
                 'def create(project: dict):\n'
                 '    return {"ok": True}  # RED: not persisted, count never moves\n\n'
                 '@router.get("/projects")\n'
                 'def list_projects():\n'
                 '    return {"items": _STORE}\n\n'
                 '@router.get("/dashboard/summary")\n'
                 'def summary():\n'
                 '    return {"total": len(_STORE)}\n')
    with open(os.path.join(root8, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria8 = (
        "# Feature design\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: creating a project raises the summary total by exactly one\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    delta:\n"
        "      observe:\n"
        "        verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        json_path: total\n"
        "      mutate:\n"
        "        verb: post\n"
        "        route: /api/v1/projects\n"
        "        body: {name: \"alpha\"}\n"
        "        expect_status: 201\n"
        "      must: increases_by\n"
        "      by: 1\n")
    cfg8 = load_config(path=_write_config(root8, with_binding=True, criteria_text=criteria8))
    kw8 = cli.acceptance_specify_kwargs(cfg8, root8)
    spec8 = {"edits": [{"id": "FIX_PERSIST", "file": "app/routes.py",
                        "anchor_old": '    return {"ok": True}  # RED: not persisted, count never moves',
                        "replacement_new": '    _STORE.append(project)\n    return {"ok": True}',
                        "rationale": "build the feature: persist the created project so the count moves",
                        "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec8 = specify._synthesize_acceptance_red_test(
        spec8, kw8.get("acceptance_criteria_text"), root8,
        setup_block=kw8.get("acceptance_setup_block"),
        app_fixture=kw8.get("acceptance_app_fixture"),
        test_dir=kw8.get("acceptance_test_dir", "tests"))
    vblock8 = spec8.get("verify") or {}
    accept8 = [e for e in spec8["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    body8 = accept8[0]["content"] if accept8 else ""
    runner8 = cfg8.test_runner_for_codebase(root8)
    verdict8 = verifymod.verify_red_green(
        spec8, root8, runner8, os.path.join(root8, ".apply_backups"), ttl_hours=1)
    results["scenario_8"] = {
        "transition": verdict8.get("transition"),
        "verified": verdict8.get("verified"),
        "gate_count_plural": len(vblock8.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept8),
        "post_call": body8.count(".post("),
        "get_call": body8.count(".get("),                # 2 = same route read before AND after
        "before_read": "before = r0.json()" in body8,
        "after_read": "after = r2.json()" in body8,
        "delta_assert": body8.count("== before['total'] + 1"),  # >0 = a difference, not absolute
        "red_status": (verdict8.get("red") or {}).get("status"),
        "green_status": (verdict8.get("green") or {}).get("status"),
    }

    # ── Scenario 9: box-5b level-6b — ONE criterion, a relation ACROSS two routes → ONE gate ──
    # Cross-route relation (GAP-5b): box-5 (relation) composed with box-4 (multi-route). A single
    # criterion's explicit ``reads`` oracle demands the summary route's ``project_count`` EQUAL the
    # length of the ``items`` list on a SEPARATE route (``other_read: 1``) — a value-to-value
    # invariant whose two sides live on DIFFERENT payloads (the level-6 ceiling compared within ONE
    # payload). The gate stays SINGLE (no red_test_nodes) and goes red→green through the real pytest
    # subprocess (RED: ``project_count`` is hardcoded 0 while ``items`` has a seed, so ``0 !=
    # len(items)``; GREEN: the fix computes the count from the shared store). Triangulates against
    # scenario_7: the body reads TWO routes (get_call == 2) and the relation crosses payloads
    # (``len(payload2[...]``), where the level-6 relation stayed inside one payload.
    root9 = tempfile.mkdtemp(prefix="ts0006_sut_xrel_")
    os.makedirs(os.path.join(root9, "app"), exist_ok=True)
    os.makedirs(os.path.join(root9, "tests"), exist_ok=True)
    with open(os.path.join(root9, "app", "routes.py"), "w", encoding="utf-8") as fh:
        fh.write('from fastapi import APIRouter\n'
                 'router = APIRouter(prefix="/api/v1")\n'
                 '_STORE = [{"name": "seed"}]\n\n'
                 '@router.get("/dashboard/summary")\n'
                 'def summary():\n'
                 '    return {"project_count": 0}  # RED: count != len(items)\n\n'
                 '@router.get("/projects")\n'
                 'def list_projects():\n'
                 '    return {"items": _STORE}\n')
    with open(os.path.join(root9, "tests", "conftest.py"), "w", encoding="utf-8") as fh:
        fh.write('import pytest\n'
                 'from fastapi.testclient import TestClient\n\n'
                 '@pytest.fixture\n'
                 'def client():\n'
                 '    from app.routes import router\n'
                 '    from fastapi import FastAPI\n'
                 '    app = FastAPI(); app.include_router(router)\n'
                 '    return TestClient(app)\n')
    criteria9 = (
        "# Feature design\n\n## 수용기준\n"
        "- id: AC1\n"
        "  prose: the summary project_count must equal the number of listed projects\n"
        "  oracle:\n"
        "    kind: http_read\n"
        "    reads:\n"
        "      - verb: get\n"
        "        route: /api/v1/dashboard/summary\n"
        "        asserts:\n"
        "          - json_path: project_count\n"
        "            must: equals_len\n"
        "            other_path: items\n"
        "            other_read: 1\n"
        "      - verb: get\n"
        "        route: /api/v1/projects\n"
        "        asserts:\n"
        "          - json_path: items\n"
        "            must: non_empty\n")
    cfg9 = load_config(path=_write_config(root9, with_binding=True, criteria_text=criteria9))
    kw9 = cli.acceptance_specify_kwargs(cfg9, root9)
    spec9 = {"edits": [{"id": "FIX_XREL", "file": "app/routes.py",
                        "anchor_old": '    return {"project_count": 0}  # RED: count != len(items)',
                        "replacement_new": '    return {"project_count": len(_STORE)}',
                        "rationale": "build the feature: compute the count from the shared store",
                        "confidence": "high"}],
             "termination": "ready_to_apply", "verify": {}}
    spec9 = specify._synthesize_acceptance_red_test(
        spec9, kw9.get("acceptance_criteria_text"), root9,
        setup_block=kw9.get("acceptance_setup_block"),
        app_fixture=kw9.get("acceptance_app_fixture"),
        test_dir=kw9.get("acceptance_test_dir", "tests"))
    vblock9 = spec9.get("verify") or {}
    accept9 = [e for e in spec9["edits"] if e["id"].startswith("ACCEPTANCE_RED")]
    body9 = accept9[0]["content"] if accept9 else ""
    runner9 = cfg9.test_runner_for_codebase(root9)
    verdict9 = verifymod.verify_red_green(
        spec9, root9, runner9, os.path.join(root9, ".apply_backups"), ttl_hours=1)
    results["scenario_9"] = {
        "transition": verdict9.get("transition"),
        "verified": verdict9.get("verified"),
        "gate_count_plural": len(vblock9.get("red_test_nodes") or []),  # 0 = single gate
        "acceptance_files": len(accept9),
        "get_call": body9.count(".get("),                # 2 = two independent routes
        "post_call": body9.count(".post("),
        "payload_count": body9.count(".json()"),         # 2 = each route its own payload
        "cross_relation": "len(payload2['items'])" in body9,   # relation crosses payloads
        "relation_fn": body9.count("def test_acceptance_multiroute_relation"),  # distinct stem
        "red_status": (verdict9.get("red") or {}).get("status"),
        "green_status": (verdict9.get("green") or {}).get("status"),
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    ok = (results["scenario_1"]["transition"] == "red_to_green"
          and results["scenario_1"]["verified"] is True
          and results["scenario_2"]["transition"] == "red_to_green"
          and results["scenario_2"]["verified"] is True
          and results["scenario_2"]["gate_count"] == 2
          and results["scenario_2"]["red_all_fail"] is True
          and results["scenario_2"]["green_all_pass"] is True
          and kw2 == {}
          and results["scenario_3"]["red_test_node"] is None
          and results["scenario_4"]["transition"] == "red_to_green"
          and results["scenario_4"]["verified"] is True
          and results["scenario_4"]["gate_count_plural"] == 0
          and results["scenario_4"]["acceptance_files"] == 1
          and results["scenario_4"]["assert_field_count"] == 2
          and results["scenario_5"]["transition"] == "red_to_green"
          and results["scenario_5"]["verified"] is True
          and results["scenario_5"]["gate_count_plural"] == 0
          and results["scenario_5"]["acceptance_files"] == 1
          and results["scenario_5"]["post_call"] == 1
          and results["scenario_5"]["get_call"] == 1
          and results["scenario_5"]["step_count"] == 2
          and results["scenario_6"]["transition"] == "red_to_green"
          and results["scenario_6"]["verified"] is True
          and results["scenario_6"]["gate_count_plural"] == 0
          and results["scenario_6"]["acceptance_files"] == 1
          and results["scenario_6"]["get_call"] == 2
          and results["scenario_6"]["post_call"] == 0
          and results["scenario_6"]["route_count"] == 2
          and results["scenario_6"]["payload_count"] == 2
          and results["scenario_7"]["transition"] == "red_to_green"
          and results["scenario_7"]["verified"] is True
          and results["scenario_7"]["gate_count_plural"] == 0
          and results["scenario_7"]["acceptance_files"] == 1
          and results["scenario_7"]["get_call"] == 1
          and results["scenario_7"]["post_call"] == 0
          and results["scenario_7"]["len_call"] >= 1
          and results["scenario_8"]["transition"] == "red_to_green"
          and results["scenario_8"]["verified"] is True
          and results["scenario_8"]["gate_count_plural"] == 0
          and results["scenario_8"]["acceptance_files"] == 1
          and results["scenario_8"]["post_call"] == 1
          and results["scenario_8"]["get_call"] == 2
          and results["scenario_8"]["before_read"] is True
          and results["scenario_8"]["after_read"] is True
          and results["scenario_8"]["delta_assert"] >= 1
          and results["scenario_9"]["transition"] == "red_to_green"
          and results["scenario_9"]["verified"] is True
          and results["scenario_9"]["gate_count_plural"] == 0
          and results["scenario_9"]["acceptance_files"] == 1
          and results["scenario_9"]["get_call"] == 2
          and results["scenario_9"]["post_call"] == 0
          and results["scenario_9"]["payload_count"] == 2
          and results["scenario_9"]["cross_relation"] is True
          and results["scenario_9"]["relation_fn"] == 1)
    print("\nTS0006 e2e:", "PASS (go)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
