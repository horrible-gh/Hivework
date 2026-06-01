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

    def test_judge_role_model_sonnet(self):
        self.assertEqual(self.cfg.role("judge").model, "claude-sonnet-4.5")

    def test_judge_role_provider_copilot(self):
        self.assertEqual(self.cfg.role("judge").provider, "copilot")

    def test_judge_caps_defaults(self):
        self.assertEqual(self.cfg.judge.max_calls_per_axis, 2)
        self.assertEqual(self.cfg.judge.max_axes, 3)
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
        self.assertEqual(self.cfg.judge.max_axes, 3)
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


if __name__ == "__main__":
    unittest.main()
