"""Unit tests for hive_setup — the provider-neutral interactive installer.

Locks the key behaviour the wizard rework added: the OpenAI-compatible endpoint
is CHOSEN (preset or custom), never hard-wired to DeepInfra, and the secret file
is named after whatever api_key_env that choice implies."""
import json, os, sys, tempfile, unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hive_setup


class TestSecretsSkeleton(unittest.TestCase):
    def test_skeleton_uses_chosen_env_var_name(self):
        body = hive_setup._secrets_skeleton("OPENAI_API_KEY", "sk-123")
        self.assertIn("OPENAI_API_KEY=sk-123", body)
        self.assertNotIn("DEEPINFRA_TOKEN=", body)

    def test_skeleton_blank_token(self):
        body = hive_setup._secrets_skeleton("DEEPINFRA_TOKEN")
        self.assertIn("DEEPINFRA_TOKEN=\n", body)


class TestChooseEndpoint(unittest.TestCase):
    def test_skip_returns_none(self):
        with mock.patch.object(hive_setup, "ask_choice", return_value="skip"):
            self.assertIsNone(hive_setup.choose_endpoint())

    def test_deepinfra_preset(self):
        with mock.patch.object(hive_setup, "ask_choice", return_value="deepinfra"):
            base_url, env = hive_setup.choose_endpoint()
        self.assertEqual(base_url, "https://api.deepinfra.com/v1/openai")
        self.assertEqual(env, "DEEPINFRA_TOKEN")

    def test_openai_preset(self):
        with mock.patch.object(hive_setup, "ask_choice", return_value="openai"):
            base_url, env = hive_setup.choose_endpoint()
        self.assertEqual(base_url, "https://api.openai.com/v1")
        self.assertEqual(env, "OPENAI_API_KEY")

    def test_custom_endpoint_asks_both(self):
        answers = iter(["https://my-vllm.local/v1", "MYVLLM_KEY"])
        with mock.patch.object(hive_setup, "ask_choice", return_value="custom"), \
             mock.patch.object(hive_setup, "ask_text",
                               side_effect=lambda *a, **k: next(answers)):
            base_url, env = hive_setup.choose_endpoint()
        self.assertEqual(base_url, "https://my-vllm.local/v1")
        self.assertEqual(env, "MYVLLM_KEY")

    def test_custom_blank_url_skips(self):
        with mock.patch.object(hive_setup, "ask_choice", return_value="custom"), \
             mock.patch.object(hive_setup, "ask_text", return_value=""):
            self.assertIsNone(hive_setup.choose_endpoint())


class TestGuidedPresetGenerator(unittest.TestCase):
    def test_codex_only_is_all_codex(self):
        rm = hive_setup.build_tier_preset({"codex"}, "mix")
        self.assertTrue(all(p == "codex" for p, _m in rm.values()))

    def test_all_providers_mix_across_providers(self):
        # Requirement: with several providers available, the layout must MIX across
        # them (not pile onto one): heavy authors->codex, light agentic->copilot,
        # tool-OFF rulings->HTTP.
        rm = hive_setup.build_tier_preset({"codex", "copilot", "openai"}, "mix")
        self.assertEqual(rm["queen"][0], "codex")       # heavy author
        self.assertEqual(rm["fanout"][0], "copilot")    # light agentic
        self.assertEqual(rm["judge"][0], "openai")      # tool-OFF ruling
        self.assertEqual({p for p, _m in rm.values()}, {"codex", "copilot", "openai"})

    def test_models_come_from_the_tier_table(self):
        rm = hive_setup.build_tier_preset({"codex", "copilot", "openai"}, "mix")
        self.assertEqual(rm["queen"][1], hive_setup._MODEL_TIERS["codex"]["mix"])
        self.assertEqual(rm["judge"][1], hive_setup._MODEL_TIERS["openai"]["mix"])

    def test_copilot_only_tool_on_uses_copilot(self):
        rm = hive_setup.build_tier_preset({"copilot"}, "minimum")
        self.assertEqual(rm["queen"][0], "copilot")
        self.assertEqual(rm["judge"][0], "copilot")  # no HTTP -> falls back to CLI

    def test_http_only_cannot_satisfy_tool_on(self):
        # openai (HTTP) is tool-OFF only; a tool-ON role has no provider -> None
        self.assertIsNone(hive_setup.build_tier_preset({"openai"}, "mix"))

    def test_tool_on_never_routes_to_http(self):
        rm = hive_setup.build_tier_preset({"codex", "openai"}, "mix")
        for name in hive_setup._TOOL_ON_ROLES:
            self.assertNotEqual(rm[name][0], "openai")

    def test_tier_changes_models_not_providers(self):
        lo = hive_setup.build_tier_preset({"copilot", "openai"}, "minimum")
        hi = hive_setup.build_tier_preset({"copilot", "openai"}, "maximum")
        self.assertEqual(lo["queen"][0], hi["queen"][0])        # same provider
        self.assertNotEqual(lo["queen"][1], hi["queen"][1])     # different model tier

    def test_apply_preserves_per_role_extras(self):
        data = {"roles": {"specify": {"provider": "x", "model": "y",
                                      "timeout_sec": 900}}}
        rm = hive_setup.build_tier_preset({"codex"}, "mix")
        hive_setup.apply_tier_preset(data, rm)
        self.assertEqual(data["roles"]["specify"]["timeout_sec"], 900)
        self.assertEqual(data["roles"]["specify"]["provider"], "codex")


class TestEnvUpsert(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.env_path = os.path.join(self.tmpdir, ".env")
        self._p = mock.patch.object(hive_setup, "_secrets_path",
                                    return_value=self.env_path)
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_create_then_update_preserves_other_keys(self):
        self.assertEqual(hive_setup._upsert_env("OPENAI_API_KEY", "sk-1"), "created")
        self.assertEqual(hive_setup._upsert_env("DB_PASSWORD", "pw"), "appended")
        self.assertEqual(hive_setup._upsert_env("OPENAI_API_KEY", "sk-2"), "updated")
        keys = hive_setup._read_env()
        self.assertEqual(keys["OPENAI_API_KEY"], "sk-2")
        self.assertEqual(keys["DB_PASSWORD"], "pw")

    def test_mask_hides_secret(self):
        self.assertEqual(hive_setup._mask("sk-abcdef"), "sk****ef")
        self.assertIn("empty", hive_setup._mask(""))


class TestPatchConfigEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg_path = os.path.join(self.tmpdir, "hive.config.json")
        # A config carrying the reference-only presets key the example ships.
        with open(self.cfg_path, "w", encoding="utf-8") as f:
            json.dump({"roles": {}, "_openai_presets": {"x": 1},
                       "openai": {"base_url": "old", "api_key_env": "OLD"}}, f)
        self._cfg_patch = mock.patch.object(hive_setup, "CONFIG", self.cfg_path)
        self._cfg_patch.start()

    def tearDown(self):
        self._cfg_patch.stop()

    def test_writes_chosen_endpoint_and_drops_presets(self):
        hive_setup.patch_config_endpoint("https://api.openai.com/v1", "OPENAI_API_KEY")
        with open(self.cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["openai"],
                         {"base_url": "https://api.openai.com/v1",
                          "api_key_env": "OPENAI_API_KEY"})
        self.assertNotIn("_openai_presets", data)
        self.assertIn("roles", data)  # untouched keys survive


if __name__ == "__main__":
    unittest.main()
