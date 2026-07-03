"""max_iterations wiring: config → provider_kwargs → HTTP agent loop (R0001 0077).

The runaway-history cost lives in ``hive/http_tools.py run_agent_loop`` — every
round resends the whole transcript, so a drone that spins to the round ceiling
dominates a run's HTTP tokens (certified run 681: 2 of 7 drones at the 25-round
bound carried 82% of 1.02M tokens). The ceiling was hard-coded; these tests pin
the new config wiring end to end:

  - loader: ``pipeline.fanout.max_iterations`` → ``cfg.fanout.max_iterations``,
    ``providers.openai.max_agent_iterations`` → ``cfg.openai.max_agent_iterations``,
    both None when unset (the safe fallback — existing configs run unchanged);
  - hive.py: ``build_provider_kwargs`` injects the global knob,
    ``fanout_provider_kwargs`` lays the stage knob over it (stage wins);
  - providers: ``_call_openai_compatible`` forwards ``max_iterations`` to
    ``run_agent_loop``; absent → the loop keeps DEFAULT_MAX_ITERATIONS;
  - shipped default profile carries the certified 12-round drone ceiling.
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import providers
from hive.config import load_config
from hive.http_tools import DEFAULT_MAX_ITERATIONS

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_cfg(raw: dict):
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "hive.config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return load_config(path=path)


def _load_hive_cli():
    """Load the top-level ``hive.py`` by file path (same pattern as
    test_acceptance_cli_wiring — hive.py's top level is import-cheap by design)."""
    path = os.path.join(_REPO, "hive.py")
    spec = importlib.util.spec_from_file_location("hive_cli_entry_0077", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLoaderKnobs(unittest.TestCase):
    def test_fanout_max_iterations_parsed(self):
        cfg = _load_cfg({"pipeline": {"fanout": {
            "provider": "openai", "model": "m", "enabled": True,
            "max_iterations": 12}}})
        self.assertEqual(cfg.fanout.max_iterations, 12)

    def test_fanout_max_iterations_absent_is_none(self):
        # The safe fallback: an existing config without the knob must leave the
        # loop at its code default — None, never an implicit lowering.
        cfg = _load_cfg({"pipeline": {"fanout": {
            "provider": "openai", "model": "m", "enabled": True}}})
        self.assertIsNone(cfg.fanout.max_iterations)

    def test_openai_max_agent_iterations_parsed(self):
        cfg = _load_cfg({"providers": {"openai": {
            "base_url": "u", "api_key_env": "E", "max_agent_iterations": 10}}})
        self.assertEqual(cfg.openai.max_agent_iterations, 10)

    def test_openai_max_agent_iterations_absent_is_none(self):
        self.assertIsNone(_load_cfg({}).openai.max_agent_iterations)


class TestHiveCliKwargs(unittest.TestCase):
    def setUp(self):
        self.cli = _load_hive_cli()

    def test_global_knob_injected_into_provider_kwargs(self):
        cfg = _load_cfg({"providers": {"openai": {
            "base_url": "u", "api_key_env": "E", "max_agent_iterations": 10}}})
        self.assertEqual(self.cli.build_provider_kwargs(cfg)["max_iterations"], 10)

    def test_no_knob_no_key(self):
        cfg = _load_cfg({})
        self.assertNotIn("max_iterations", self.cli.build_provider_kwargs(cfg))

    def test_fanout_stage_knob_overrides_global(self):
        cfg = _load_cfg({
            "providers": {"openai": {"base_url": "u", "api_key_env": "E",
                                     "max_agent_iterations": 20}},
            "pipeline": {"fanout": {"provider": "openai", "model": "m",
                                    "enabled": True, "max_iterations": 12}}})
        base = self.cli.build_provider_kwargs(cfg)
        self.assertEqual(base["max_iterations"], 20)          # global everywhere else
        fo = self.cli.fanout_provider_kwargs(cfg, base)
        self.assertEqual(fo["max_iterations"], 12)            # stage wins for drones
        self.assertEqual(base["max_iterations"], 20)          # base not mutated

    def test_fanout_unset_passes_base_through(self):
        cfg = _load_cfg({})
        base = self.cli.build_provider_kwargs(cfg)
        self.assertIs(self.cli.fanout_provider_kwargs(cfg, base), base)


class TestProviderPassthrough(unittest.TestCase):
    """_call_openai_compatible forwards max_iterations to run_agent_loop; unset
    keeps the loop's own default (existing behaviour byte-for-byte)."""

    _SENTINEL = object()

    def _run(self, **call_kwargs):
        seen = {}

        def fake_loop(client, model, messages, *, root, tool_names, temperature,
                      max_tokens, extra, max_iterations=self._SENTINEL,
                      prune_keep_rounds=self._SENTINEL):
            seen["max_iterations"] = max_iterations
            seen["prune_keep_rounds"] = prune_keep_rounds
            return "ok", 5

        import types
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda **kw: object()
        with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
             mock.patch.dict(os.environ, {"DEEPINFRA_TOKEN": "tok"}, clear=True), \
             mock.patch.object(providers.http_tools, "run_agent_loop", fake_loop):
            wr = providers.call_worker("openai", "m", "p", cwd="/x", timeout=30,
                                       **call_kwargs)
        return wr, seen

    def test_max_iterations_forwarded(self):
        wr, seen = self._run(max_iterations=12)
        self.assertEqual(wr.exit_code, 0)
        self.assertEqual(seen["max_iterations"], 12)

    def test_prune_keep_rounds_forwarded_and_absent_by_default(self):
        _, seen = self._run(prune_keep_rounds=2)
        self.assertEqual(seen["prune_keep_rounds"], 2)
        _, seen = self._run()
        self.assertIs(seen["prune_keep_rounds"], self._SENTINEL)

    def test_absent_leaves_loop_default(self):
        # No kwarg passed → the loop's own signature default applies (the fake's
        # sentinel proves the handler did NOT supply the argument).
        _, seen = self._run()
        self.assertIs(seen["max_iterations"], self._SENTINEL)
        self.assertEqual(DEFAULT_MAX_ITERATIONS, 25)  # code default unchanged

    def test_cli_provider_ignores_the_key(self):
        # A mixed-provider run shares ONE kwargs dict; copilot must swallow the
        # key via **_ignored instead of crashing.
        def fake_run(cmd, input=None, cwd=None, timeout=None, env=None):
            import subprocess
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        with mock.patch.object(providers.shutil, "which",
                               return_value="copilot.cmd"), \
             mock.patch.object(providers, "_run_capture", side_effect=fake_run):
            wr = providers.call_worker("copilot", "gpt-5-mini", "p", cwd="/x",
                                       timeout=30, max_iterations=12)
        self.assertEqual(wr.exit_code, 0)


class TestPruneKnobWiring(unittest.TestCase):
    """prune_keep_rounds rides the same config → provider_kwargs → loop path."""

    def test_fanout_prune_keep_rounds_parsed_and_absent_is_none(self):
        cfg = _load_cfg({"pipeline": {"fanout": {
            "provider": "openai", "model": "m", "enabled": True,
            "prune_keep_rounds": 2}}})
        self.assertEqual(cfg.fanout.prune_keep_rounds, 2)
        self.assertIsNone(_load_cfg({}).fanout.prune_keep_rounds)

    def test_openai_prune_keep_rounds_parsed_and_absent_is_none(self):
        cfg = _load_cfg({"providers": {"openai": {
            "base_url": "u", "api_key_env": "E", "prune_keep_rounds": 3}}})
        self.assertEqual(cfg.openai.prune_keep_rounds, 3)
        self.assertIsNone(_load_cfg({}).openai.prune_keep_rounds)

    def test_fanout_stage_prune_knob_laid_over_base(self):
        cli = _load_hive_cli()
        cfg = _load_cfg({"pipeline": {"fanout": {
            "provider": "openai", "model": "m", "enabled": True,
            "prune_keep_rounds": 2}}})
        base = cli.build_provider_kwargs(cfg)
        self.assertNotIn("prune_keep_rounds", base)
        self.assertEqual(cli.fanout_provider_kwargs(cfg, base)["prune_keep_rounds"], 2)


class TestShippedDefaultCarriesDroneCeiling(unittest.TestCase):
    def test_default_profile_fanout_loop_knobs(self):
        cfg = load_config(path=os.path.join(_REPO, "config",
                                            "hive.config.default.json"))
        self.assertEqual(cfg.fanout.max_iterations, 12)
        self.assertEqual(cfg.fanout.prune_keep_rounds, 2)


if __name__ == "__main__":
    unittest.main()
