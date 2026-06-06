"""Unit tests for hive.config — per-role provider/model configuration."""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.config import load_config, Config, RoleConfig, _DEFAULTS


class TestConfigDefaults(unittest.TestCase):
    """With no file, defaults must reproduce today's behavior."""

    def setUp(self):
        # Point load_config at a non-existent path to force defaults
        self.cfg = load_config(path="/nonexistent/hive.config.json")

    def test_queen_provider_copilot(self):
        self.assertEqual(self.cfg.queen.provider, "copilot")

    def test_queen_model_gpt5mini(self):
        self.assertEqual(self.cfg.queen.model, "gpt-5-mini")

    def test_swarm_provider_copilot(self):
        self.assertEqual(self.cfg.swarm.provider, "copilot")

    def test_swarm_model_gpt5mini(self):
        self.assertEqual(self.cfg.swarm.model, "gpt-5-mini")

    def test_assemble_provider_copilot(self):
        self.assertEqual(self.cfg.role("assemble").provider, "copilot")

    def test_assemble_model_gpt5mini(self):
        self.assertEqual(self.cfg.role("assemble").model, "gpt-5-mini")

    def test_ledger_enabled(self):
        self.assertTrue(self.cfg.ledger.enabled)

    def test_ledger_db_path(self):
        self.assertEqual(self.cfg.ledger.db_path, "hive_ledger.db")

    def test_copilot_exe_none(self):
        self.assertIsNone(self.cfg.copilot.exe)

    def test_openai_endpoint_defaults_to_deepinfra_preset(self):
        # Back-compat: a config that only names provider 'deepinfra' keeps working.
        self.assertEqual(self.cfg.openai.base_url,
                         "https://api.deepinfra.com/v1/openai")
        self.assertEqual(self.cfg.openai.api_key_env, "DEEPINFRA_TOKEN")

    def test_judge_role_model_sonnet(self):
        self.assertEqual(self.cfg.role("judge").model, "claude-sonnet-4.5")

    def test_judge_role_provider_copilot(self):
        self.assertEqual(self.cfg.role("judge").provider, "copilot")

    def test_judge_caps_defaults(self):
        self.assertEqual(self.cfg.judge.max_calls_per_axis, 2)
        # max_axes is a runaway-ceiling (judge every leaf up to this), not an
        # aggressive cap — raised from 3 after N164 (decisive axis cut at cap).
        self.assertEqual(self.cfg.judge.max_axes, 12)
        self.assertEqual(self.cfg.judge.max_parallel, 2)

    def test_safety_allow_swarm_default_true(self):
        # Absence preserves today's behavior: the swarm run path stays available.
        self.assertTrue(self.cfg.safety.allow_swarm)


class TestConfigSafety(unittest.TestCase):
    """The swarm kill-switch: safety.allow_swarm gates the run path."""

    def _load(self, raw):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w") as f:
            json.dump(raw, f)
        return load_config(path=path)

    def test_allow_swarm_false_honored(self):
        cfg = self._load({"safety": {"allow_swarm": False}})
        self.assertFalse(cfg.safety.allow_swarm)

    def test_allow_swarm_true_honored(self):
        cfg = self._load({"safety": {"allow_swarm": True}})
        self.assertTrue(cfg.safety.allow_swarm)

    def test_missing_safety_defaults_true(self):
        cfg = self._load({"judge": {"max_axes": 2}})
        self.assertTrue(cfg.safety.allow_swarm)

    def test_reinforce_independent_of_allow_swarm(self):
        # B3 reinforcement has its OWN switch: enabling it must not flip allow_swarm,
        # and its caps parse independently.
        cfg = self._load({"safety": {"allow_swarm": False},
                          "reinforce": {"enabled": True, "max_workers": 3,
                                        "max_total_calls": 9}})
        self.assertFalse(cfg.safety.allow_swarm)        # legacy swarm stays off
        self.assertTrue(cfg.reinforce.enabled)
        self.assertEqual(cfg.reinforce.max_workers, 3)
        self.assertEqual(cfg.reinforce.max_total_calls, 9)

    def test_reinforce_defaults_off(self):
        cfg = self._load({"judge": {"max_axes": 2}})
        self.assertFalse(cfg.reinforce.enabled)

    def test_reinvestigation_defaults(self):
        # Live re-run off by default; cap defaults to 2 (one re-run + one confirm).
        cfg = self._load({"judge": {"max_axes": 2}})
        self.assertFalse(cfg.reinvestigation.live)
        self.assertEqual(cfg.reinvestigation.max_rounds, 2)

    def test_reinvestigation_overrides_honored(self):
        cfg = self._load({"reinvestigation": {"live": True, "max_rounds": 3}})
        self.assertTrue(cfg.reinvestigation.live)
        self.assertEqual(cfg.reinvestigation.max_rounds, 3)

    def test_scout_defaults_to_swarm_when_unset(self):
        # The B3 reinforcement worker (scout) inherits the swarm model when not named,
        # so an existing roles.swarm config keeps working unchanged.
        cfg = self._load({"roles": {"swarm": {"provider": "openai", "model": "m120"}}})
        self.assertEqual(cfg.scout.provider, "openai")
        self.assertEqual(cfg.scout.model, "m120")

    def test_scout_override_is_independent_of_swarm(self):
        cfg = self._load({"roles": {"swarm": {"provider": "openai", "model": "m120"},
                                    "scout": {"provider": "copilot",
                                              "model": "claude-sonnet-4.5"}}})
        self.assertEqual(cfg.scout.model, "claude-sonnet-4.5")   # the reinforcement lever
        self.assertEqual(cfg.swarm.model, "m120")                # legacy swarm untouched


class TestTestRunners(unittest.TestCase):
    """The per-codebase test_runners block (runtime red→green verify)."""

    def _load(self, overrides):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w") as f:
            json.dump(overrides, f)
        return load_config(path=path)

    def test_no_runners_by_default(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        self.assertEqual(cfg.test_runners, {})
        self.assertIsNone(cfg.test_runner_for_codebase("/x/FlowGate"))

    def test_runner_parsed_and_matched_by_key(self):
        cfg = self._load({"test_runners": {"flowgate": {
            "command": ["python", "-m", "pytest", "-q"], "cwd": "server",
            "timeout_sec": 120, "env": {"PYTHONUNBUFFERED": "1"}}}})
        r = cfg.test_runner_for_codebase("C:/work/FlowGate")
        self.assertIsNotNone(r)
        self.assertEqual(r.command, ["python", "-m", "pytest", "-q"])
        self.assertEqual(r.cwd, "server")
        self.assertEqual(r.timeout_sec, 120)
        self.assertEqual(r.env, {"PYTHONUNBUFFERED": "1"})

    def test_command_string_is_split(self):
        cfg = self._load({"test_runners": {"x": {"command": "pytest -q"}}})
        self.assertEqual(cfg.test_runners["x"].command, ["pytest", "-q"])

    def test_explicit_codebase_binding_wins(self):
        cfg = self._load({"test_runners": {"runner1": {
            "command": ["pytest"], "codebase": "/abs/MyApp"}}})
        self.assertIsNotNone(cfg.test_runner_for_codebase("/abs/MyApp"))
        self.assertIsNone(cfg.test_runner_for_codebase("/abs/Other"))


class TestHttpShapeTargets(unittest.TestCase):
    """The per-codebase http_shape harness block (lever ⑦ enabler / #1 wiring)."""

    def _load(self, overrides):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w") as f:
            json.dump(overrides, f)
        return load_config(path=path)

    def test_none_by_default(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        self.assertEqual(cfg.http_shape_targets, {})
        self.assertIsNone(cfg.http_shape_for_codebase("/x/FlowGate"))

    def test_grouped_target_block_parsed_and_matched_by_key(self):
        cfg = self._load({"targets": {"FlowGate": {"http_shape": {
            "app_fixture": "client", "test_dir": "server/tests"}}}})
        hs = cfg.http_shape_for_codebase("C:/work/FlowGate")
        self.assertIsNotNone(hs)
        self.assertEqual(hs.app_fixture, "client")
        self.assertEqual(hs.test_dir, "server/tests")

    def test_resolve_setup_block_inline_wins(self):
        cfg = self._load({"http_shape_targets": {"x": {
            "setup_block": "import pytest\n", "setup_block_file": "nope.py"}}})
        hs = cfg.http_shape_targets["x"]
        self.assertEqual(hs.resolve_setup_block("/anything"), "import pytest\n")

    def test_resolve_setup_block_from_relative_file(self):
        tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmpdir, "harness"))
        with open(os.path.join(tmpdir, "harness", "h.py"), "w") as f:
            f.write("# seeded client harness\n")
        cfg = self._load({"http_shape_targets": {"x": {
            "setup_block_file": "harness/h.py"}}})
        hs = cfg.http_shape_targets["x"]
        self.assertEqual(hs.resolve_setup_block(tmpdir), "# seeded client harness\n")
        # missing file → None (synthesis falls back / stays a no-op), never raises
        self.assertIsNone(hs.resolve_setup_block("/no/such/root"))

    def test_explicit_codebase_binding_wins(self):
        cfg = self._load({"http_shape_targets": {"h1": {
            "app_fixture": "c", "codebase": "/abs/MyApp"}}})
        self.assertIsNotNone(cfg.http_shape_for_codebase("/abs/MyApp"))
        self.assertIsNone(cfg.http_shape_for_codebase("/abs/Other"))


class TestConvergeSplit(unittest.TestCase):
    """The converge.split block (M020 per-locus elimination pass)."""

    def _load(self, overrides):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w") as f:
            json.dump(overrides, f)
        return load_config(path=path)

    def test_off_by_default(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        self.assertFalse(cfg.converge_split.enabled)
        self.assertEqual(cfg.converge_split.max_loci, 4)
        self.assertEqual(cfg.converge_split.provider, "")
        self.assertEqual(cfg.converge_split.model, "")

    def test_parsed_from_converge_split_block(self):
        cfg = self._load({"converge": {"split": {
            "enabled": True, "max_loci": 3,
            "provider": "openai", "model": "openai/gpt-oss-120b"}}})
        self.assertTrue(cfg.converge_split.enabled)
        self.assertEqual(cfg.converge_split.max_loci, 3)
        self.assertEqual(cfg.converge_split.provider, "openai")
        self.assertEqual(cfg.converge_split.model, "openai/gpt-oss-120b")

    def test_partial_block_keeps_defaults(self):
        cfg = self._load({"converge": {"split": {"enabled": True}}})
        self.assertTrue(cfg.converge_split.enabled)
        self.assertEqual(cfg.converge_split.max_loci, 4)


class TestConfigPartialFile(unittest.TestCase):
    """Partial config file: only override swarm model; others stay default."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "hive.config.json")
        partial = {"roles": {"swarm": {"provider": "copilot", "model": "claude-sonnet-4.6"}}}
        with open(self.config_path, "w") as f:
            json.dump(partial, f)
        self.cfg = load_config(path=self.config_path)

    def test_swarm_model_overridden(self):
        self.assertEqual(self.cfg.swarm.model, "claude-sonnet-4.6")

    def test_queen_model_still_default(self):
        self.assertEqual(self.cfg.queen.model, "gpt-5-mini")

    def test_assemble_model_still_default(self):
        self.assertEqual(self.cfg.role("assemble").model, "gpt-5-mini")

    def test_ledger_still_enabled(self):
        self.assertTrue(self.cfg.ledger.enabled)


class TestConfigOpenAiEndpoint(unittest.TestCase):
    """A custom openai block (e.g. OpenAI proper, or a self-hosted vLLM) is honored."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "hive.config.json")
        cfg = {"openai": {"base_url": "https://api.openai.com/v1",
                          "api_key_env": "OPENAI_API_KEY"}}
        with open(self.config_path, "w") as f:
            json.dump(cfg, f)
        self.cfg = load_config(path=self.config_path)

    def test_base_url_overridden(self):
        self.assertEqual(self.cfg.openai.base_url, "https://api.openai.com/v1")

    def test_api_key_env_overridden(self):
        self.assertEqual(self.cfg.openai.api_key_env, "OPENAI_API_KEY")


class TestConfigJudgePartial(unittest.TestCase):
    """Overriding a single judge cap leaves the others at default."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "hive.config.json")
        partial = {"judge": {"max_calls_per_axis": 4}}
        with open(self.config_path, "w") as f:
            json.dump(partial, f)
        self.cfg = load_config(path=self.config_path)

    def test_overridden_cap(self):
        self.assertEqual(self.cfg.judge.max_calls_per_axis, 4)

    def test_other_caps_still_default(self):
        self.assertEqual(self.cfg.judge.max_axes, 12)
        self.assertEqual(self.cfg.judge.max_parallel, 2)

    def test_judge_role_still_default(self):
        self.assertEqual(self.cfg.role("judge").model, "claude-sonnet-4.5")


class TestConfigCliOverride(unittest.TestCase):
    """CLI --model overrides all roles."""

    def test_cli_model_overrides_all(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        cfg.apply_cli_model("claude-opus-4.7")
        self.assertEqual(cfg.queen.model, "claude-opus-4.7")
        self.assertEqual(cfg.swarm.model, "claude-opus-4.7")
        self.assertEqual(cfg.role("assemble").model, "claude-opus-4.7")
        self.assertEqual(cfg.role("judge").model, "claude-opus-4.7")

    def test_cli_model_none_no_change(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        cfg.apply_cli_model(None)
        self.assertEqual(cfg.queen.model, "gpt-5-mini")

    def test_cli_model_does_not_change_provider(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        cfg.apply_cli_model("some-model")
        self.assertEqual(cfg.queen.provider, "copilot")


class TestGroupedLayout(unittest.TestCase):
    """The new grouped layout (roles / stages / targets / ops / providers) maps onto
    the same internal Config as the legacy flat keys — so no downstream code changes."""

    def _load(self, raw):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w") as f:
            json.dump(raw, f)
        return load_config(path=path)

    def test_providers_group_maps_to_copilot_openai(self):
        cfg = self._load({"providers": {"copilot": {"timeout_sec": 222},
                                        "openai": {"base_url": "u", "api_key_env": "E"}}})
        self.assertEqual(cfg.copilot.timeout_sec, 222)
        self.assertEqual(cfg.openai.base_url, "u")
        self.assertEqual(cfg.openai.api_key_env, "E")

    def test_copilot_token_config_pins_billing_account(self):
        # A configured token must reach the copilot subprocess as COPILOT_GITHUB_TOKEN
        # (which overrides the CLI's stored login) so the run bills THAT account.
        cfg = self._load({"providers": {"copilot": {"token": "github_pat_TESTACCT"}}})
        self.assertEqual(cfg.copilot.token, "github_pat_TESTACCT")
        # token_env names an env var to read the token from instead of inlining it.
        cfg2 = self._load({"providers": {"copilot": {"token_env": "HIVE_COPILOT_TOK"}}})
        self.assertEqual(cfg2.copilot.token_env, "HIVE_COPILOT_TOK")
        # unset → both None (handler falls back to stored login + warns).
        cfg3 = self._load({"providers": {"copilot": {}}})
        self.assertIsNone(cfg3.copilot.token)
        self.assertIsNone(cfg3.copilot.token_env)

    def test_stages_judge_maps_to_judge_caps(self):
        cfg = self._load({"stages": {"judge": {"max_axes": 7, "votes_per_axis": 5}}})
        self.assertEqual(cfg.judge.max_axes, 7)
        self.assertEqual(cfg.judge.votes_per_axis, 5)

    def test_stages_converge_split(self):
        cfg = self._load({"stages": {"converge": {"split": {"enabled": True, "max_loci": 2}}}})
        self.assertTrue(cfg.converge_split.enabled)
        self.assertEqual(cfg.converge_split.max_loci, 2)
        self.assertEqual(cfg.converge_split.model, "")   # no placeholder => reuse roles.converge

    def test_stages_reinforce_and_reinvestigation(self):
        cfg = self._load({"stages": {"reinforce": {"enabled": True, "max_workers": 3},
                                     "reinvestigation": {"live": True, "max_rounds": 4}}})
        self.assertTrue(cfg.reinforce.enabled)
        self.assertEqual(cfg.reinforce.max_workers, 3)
        self.assertTrue(cfg.reinvestigation.live)
        self.assertEqual(cfg.reinvestigation.max_rounds, 4)

    def test_stages_commit_maps_to_commit_stage(self):
        cfg = self._load({"stages": {"commit": {"filename_only_threshold": 7}}})
        self.assertEqual(cfg.commit_stage.filename_only_threshold, 7)

    def test_ops_swarm_run_maps_to_safety(self):
        cfg = self._load({"ops": {"swarm_run": {"allow": False}}})
        self.assertFalse(cfg.safety.allow_swarm)

    def test_ops_apply_and_ledger(self):
        cfg = self._load({"ops": {"apply": {"backup_dir": "b", "backup_ttl_hours": 5},
                                  "ledger": {"enabled": False, "db_path": "x.db"}}})
        self.assertEqual(cfg.apply.backup_dir, "b")
        self.assertEqual(cfg.apply.backup_ttl_hours, 5)
        self.assertFalse(cfg.ledger.enabled)
        self.assertEqual(cfg.ledger.db_path, "x.db")

    def test_targets_map_to_db_and_runners(self):
        cfg = self._load({"targets": {"FlowGate": {
            "db": {"kind": "sqlite", "path": "/x/f.db"},
            "tests": {"command": ["pytest"], "cwd": "server"}}}})
        db = cfg.db_for_codebase("/work/FlowGate")
        self.assertIsNotNone(db)
        self.assertEqual(db.path, "/x/f.db")
        r = cfg.test_runner_for_codebase("/work/FlowGate")
        self.assertIsNotNone(r)
        self.assertEqual(r.command, ["pytest"])

    def test_targets_comment_key_skipped(self):
        # A targets-level "_comment" is not a codebase entry and must not break parsing.
        cfg = self._load({"targets": {"_comment": "doc", "FlowGate": {
            "db": {"kind": "sqlite", "path": "/x/f.db"}}}})
        self.assertIsNotNone(cfg.db_for_codebase("/work/FlowGate"))

    def test_comment_keys_ignored_in_role(self):
        cfg = self._load({"roles": {"judge": {"provider": "openai", "model": "m",
                                              "_comment": "ignore me"}}})
        self.assertEqual(cfg.role("judge").provider, "openai")
        self.assertEqual(cfg.role("judge").model, "m")


class TestShippedDefaultProfile(unittest.TestCase):
    """The shipped config/hive.config.default.json loads via the default profile and
    preserves today's hand-tuned values (migration is value-preserving)."""

    def setUp(self):
        self.cfg = load_config()  # profile None -> config/hive.config.default.json

    def test_queen_is_copilot(self):
        # decompose queen was swapped codex/gpt-5.4-mini -> copilot/gpt-5-mini as part of
        # the cross-process codex serialization fix (a batch of parallel codex calls starved
        # one another; moving decompose off codex removed the contention). Validated live.
        self.assertEqual(self.cfg.queen.provider, "copilot")
        self.assertEqual(self.cfg.queen.model, "gpt-5-mini")

    def test_judge_routed_to_openai_with_tuned_caps(self):
        self.assertEqual(self.cfg.role("judge").provider, "openai")
        self.assertEqual(self.cfg.judge.max_axes, 10)
        self.assertEqual(self.cfg.judge.votes_per_axis, 5)

    def test_swarm_run_disabled(self):
        self.assertFalse(self.cfg.safety.allow_swarm)

    def test_converge_split_enabled_without_placeholder_model(self):
        self.assertTrue(self.cfg.converge_split.enabled)
        self.assertEqual(self.cfg.converge_split.model, "")   # the "..." latent bug is gone

    def test_flowgate_target_present(self):
        self.assertIsNotNone(self.cfg.db_for_codebase("X:/whatever/FlowGate"))
        self.assertIsNotNone(self.cfg.test_runner_for_codebase("X:/whatever/FlowGate"))


class TestProfileResolutionAndBootstrap(unittest.TestCase):
    """Profile path resolution and the auto-generated neutral bootstrap file."""

    def test_profile_path_default_and_named(self):
        import hive.config as C
        self.assertTrue(C._profile_path(None).endswith(
            os.path.join("config", "hive.config.default.json")))
        self.assertTrue(C._profile_path("small").endswith(
            os.path.join("config", "hive.config.small.json")))

    def test_bootstrap_writes_neutral_defaults(self):
        import hive.config as C
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "config", "hive.config.default.json")
        C._write_bootstrap_default(path)
        self.assertTrue(os.path.isfile(path))
        cfg = load_config(path=path)
        # Neutral (code) defaults, NOT the hand-tuned shipped values.
        self.assertEqual(cfg.queen.provider, "copilot")
        self.assertTrue(cfg.safety.allow_swarm)
        self.assertFalse(cfg.reinforce.enabled)
        self.assertEqual(cfg.judge.max_axes, 12)


if __name__ == "__main__":
    unittest.main()
