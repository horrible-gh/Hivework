"""Regression tests for group 0065 (hivework.default.0065): the CLI→config→run_specify and
recipe-selection boundaries that were the box-0 driver's MISSING live wiring.

NR0003 (0065) root cause: box-0 (group 0064) shipped the acceptance-synthesis engine AND the
recipe/axis driver functions GREEN at the unit level, but NONE of them was reachable from a live
run:

  - Gap C (the ★): ``run_specify()`` was wired for ``_synthesize_acceptance_red_test`` (TR0008),
    but every CLI specify entry built its ``specify_kwargs`` from http_shape/write_sink/
    overwrite_race ONLY — there was no ``acceptance_specify_kwargs`` builder, so
    ``acceptance_criteria_text`` was always None and the pass was a permanent no-op live.
  - Gap B: ``enforce_feature_axis_order`` was never called by ``run_decompose``, so the feature
    recipe never forced the acceptance axis to the head of step 0.
  - Gap A: ``select_recipe`` had zero call sites — ``--recipe`` was required, so the seed-based
    auto-classifier never engaged.

These lock the three boundaries: a real ``targets.<name>.acceptance`` block flows through
``load_config`` → ``acceptance_for_codebase`` → ``acceptance_specify_kwargs``; ``run_decompose``
injects the acceptance axis ONLY for the feature recipe (the ablation contrast — bug recipe is
byte-for-byte unchanged); and ``resolve_recipe_path`` auto-selects the card from the seed while an
explicit ``--recipe`` still overrides. Deterministic, zero model cost.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.config import load_config, AcceptanceConfig
from hive import decompose as dec
from hive.providers import WorkerResult


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_hive_cli():
    """Load the top-level ``hive.py`` CLI module to reach ``acceptance_specify_kwargs`` /
    ``resolve_recipe_path`` — by file path (not ``import hive``), exactly like
    test_overwrite_race_cli_plumbing (hive.py's top level is import-cheap by design, B0001)."""
    path = os.path.join(_REPO, "hive.py")
    spec = importlib.util.spec_from_file_location("hive_cli_entry_0065", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A minimal design body carrying the box-0 marker section. The kwargs builder forwards the TEXT
# verbatim (it does not parse); the parse is run_specify's job, covered by test_acceptance_synth.
_DESIGN_TEXT = (
    "# Feature design\n\n"
    "## 수용기준\n"
    "- AC-1 oracle{kind: http_read, verb: GET, route: /api/v1/projects, "
    "json_path: projects[].modules, must: non_empty}\n"
)
_HARNESS = "import pytest\n\n@pytest.fixture\ndef client():\n    ...\n"


def _make_codebase() -> str:
    return tempfile.mkdtemp()


def _write_config(root: str, *, with_binding: bool, with_design: bool = True,
                  with_harness: bool = True) -> str:
    """Write a config JSON binding (or not) an ``acceptance`` block to ``root``."""
    targets: dict = {"flowgate": {}}
    if with_binding:
        block: dict = {"codebase": root, "test_dir": "server/tests"}
        if with_design:
            block["criteria_text"] = _DESIGN_TEXT
        if with_harness:
            block["app_fixture"] = "client"
            block["setup_block"] = _HARNESS
        targets["flowgate"]["acceptance"] = block
    cfg_path = os.path.join(root, "hive.config.test.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


# ── Gap C: config schema ──────────────────────────────────────────────────────
class TestAcceptanceConfigSchema(unittest.TestCase):
    """load_config parses targets.<name>.acceptance and resolves it per-codebase."""

    def test_parses_and_resolves_binding(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True))
        ac = cfg.acceptance_for_codebase(root)
        self.assertIsInstance(ac, AcceptanceConfig)
        self.assertEqual(ac.resolve_criteria_text(root), _DESIGN_TEXT)
        self.assertEqual(ac.resolve_setup_block(root), _HARNESS)
        self.assertEqual(ac.app_fixture, "client")
        self.assertEqual(ac.test_dir, "server/tests")

    def test_no_binding_resolves_none(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=False))
        self.assertIsNone(cfg.acceptance_for_codebase(root))

    def test_design_file_read_at_resolve_time(self):
        root = _make_codebase()
        design = os.path.join(root, "design.md")
        with open(design, "w", encoding="utf-8") as fh:
            fh.write(_DESIGN_TEXT)
        ac = AcceptanceConfig(design_file="design.md")  # relative → resolves under root
        self.assertEqual(ac.resolve_criteria_text(root), _DESIGN_TEXT)


# ── Gap C: the CLI→run_specify plumbing builder ───────────────────────────────
class TestAcceptanceSpecifyKwargs(unittest.TestCase):
    def setUp(self):
        self.cli = _load_hive_cli()

    def test_forwards_criteria_and_harness_from_config(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True))
        kw = self.cli.acceptance_specify_kwargs(cfg, root)
        self.assertEqual(kw.get("acceptance_criteria_text"), _DESIGN_TEXT)
        self.assertEqual(kw.get("acceptance_app_fixture"), "client")
        self.assertEqual(kw.get("acceptance_setup_block"), _HARNESS)
        self.assertEqual(kw.get("acceptance_test_dir"), "server/tests")

    def test_design_path_arg_overrides_config(self):
        # A per-run --acceptance-design path wins over the config binding's criteria text.
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True))
        per_run = os.path.join(root, "per_run_design.md")
        per_run_text = "## 수용기준\n- AC-9 prose: GET /api/v1/x must return non-empty `y`\n"
        with open(per_run, "w", encoding="utf-8") as fh:
            fh.write(per_run_text)
        kw = self.cli.acceptance_specify_kwargs(cfg, root, per_run)
        self.assertEqual(kw.get("acceptance_criteria_text"), per_run_text)
        # harness fields still come from the config binding
        self.assertEqual(kw.get("acceptance_app_fixture"), "client")

    def test_noop_without_criteria(self):
        # ★ load-bearing: with no binding AND no design path there is NO criteria text, so the
        # builder returns {} — synthesis stays a no-op (the live behaviour before 0065). This is
        # the exact starvation NR0003 Gap C identified; if this returns kwargs, the box-0 pass
        # would fire blind.
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=False))
        self.assertEqual(self.cli.acceptance_specify_kwargs(cfg, root, None), {})

    def test_design_path_alone_arms_without_config_binding(self):
        # A --acceptance-design path arms synthesis even with no config binding (harness then
        # falls back to auto-discovery inside synthesis).
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=False))
        design = os.path.join(root, "d.md")
        with open(design, "w", encoding="utf-8") as fh:
            fh.write(_DESIGN_TEXT)
        kw = self.cli.acceptance_specify_kwargs(cfg, root, design)
        self.assertEqual(kw.get("acceptance_criteria_text"), _DESIGN_TEXT)
        self.assertNotIn("acceptance_app_fixture", kw)  # no binding → no harness fields


# ── Gap B: run_decompose drives enforce_feature_axis_order ─────────────────────
class TestRunDecomposeAxisEnforcement(unittest.TestCase):
    """run_decompose injects the acceptance axis for the FEATURE recipe, not the bug recipe."""

    def _run(self, recipe_path):
        def fake(provider, model, prompt, cwd=None, timeout=300, on_start=None, **kw):
            return WorkerResult(
                stdout='{"tasks": [{"id": "A"}], "steps": [["A"]]}',
                stderr="", exit_code=0, latency_s=1.0)
        cwd = tempfile.mkdtemp()
        old = os.getcwd()
        os.chdir(cwd)  # run_decompose writes decompose_raw_last.txt to cwd
        try:
            with mock.patch.object(dec, "call_worker", fake):
                return dec.run_decompose(seed_text="add a new feature",
                                         recipe_path=recipe_path)
        finally:
            os.chdir(old)

    def test_feature_recipe_injects_acceptance_axis_at_step_head(self):
        out = self._run("recipes/recipe_code_feature.md")
        ids = [t["id"] for t in out["tasks"]]
        self.assertIn(dec.ACCEPTANCE_AXIS_ID, ids)
        self.assertEqual(ids[0], dec.ACCEPTANCE_AXIS_ID, "axis must be forced to the FRONT")
        self.assertEqual(out["steps"][0][0], dec.ACCEPTANCE_AXIS_ID)

    def test_bug_recipe_leaves_decomposition_untouched(self):
        # ★ ablation contrast: the bug recipe (the default) must NOT inject the axis — existing
        # bug flows are byte-for-byte unchanged. If this fires, 0065's wiring is not conservative.
        out = self._run("recipes/recipe_code_bug.md")
        ids = [t["id"] for t in out["tasks"]]
        self.assertNotIn(dec.ACCEPTANCE_AXIS_ID, ids)
        self.assertEqual(ids, ["A"])

    def test_recipe_id_from_path(self):
        self.assertEqual(dec._recipe_id_from_path("a/b/recipe_code_feature.md"),
                         "recipe_code_feature")
        self.assertEqual(dec._recipe_id_from_path("x/recipe_code_bug.md"), "recipe_code_bug")
        self.assertIsNone(dec._recipe_id_from_path(None))


# ── Gap A: recipe auto-selection from the seed ────────────────────────────────
class TestResolveRecipePath(unittest.TestCase):
    def setUp(self):
        self.cli = _load_hive_cli()

    def test_explicit_recipe_wins(self):
        self.assertEqual(
            self.cli.resolve_recipe_path("/custom/recipe.md", "add a new feature"),
            "/custom/recipe.md")

    def test_feature_seed_selects_feature_card(self):
        path = self.cli.resolve_recipe_path(None, "새 기능 추가: 사용자 프로필 페이지 구현")
        self.assertTrue(path.endswith("recipe_code_feature.md"),
                        f"feature seed must map to the feature card, got {path}")
        self.assertTrue(os.path.exists(path))

    def test_bug_seed_selects_bug_card(self):
        path = self.cli.resolve_recipe_path(None, "500 error when saving the document")
        self.assertTrue(path.endswith("recipe_code_bug.md"),
                        f"bug seed must map to the bug card (safe default), got {path}")
        self.assertTrue(os.path.exists(path))


# ── Static guard: no specify entry may forget acceptance ──────────────────────
class TestAllSpecifyEntrypointsArmAcceptance(unittest.TestCase):
    """Every specify entry point that arms lever L3 must also arm box-0 acceptance (0065)."""

    def test_every_overwrite_race_call_is_paired_with_acceptance(self):
        with open(os.path.join(_REPO, "hive.py"), encoding="utf-8") as fh:
            src = fh.read()
        orc_n = src.count("specify_kwargs.update(overwrite_race_specify_kwargs(")
        acc_n = src.count("specify_kwargs.update(acceptance_specify_kwargs(")
        self.assertGreaterEqual(orc_n, 4, "expected the four known specify entry points")
        self.assertEqual(
            acc_n, orc_n,
            "every specify entry point that arms lever L3 must also arm box-0 acceptance "
            "(group 0065) — an overwrite_race_specify_kwargs call without a sibling "
            "acceptance_specify_kwargs re-opens the live starvation gap NR0003 found")


if __name__ == "__main__":
    unittest.main()
