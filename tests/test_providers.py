"""Unit tests for hive.providers — copilot cmd construction (no real CLI runs)."""
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import providers


class _FakeProc:
    def __init__(self):
        self.returncode = 0
        self.stdout = "{}"
        self.stderr = ""


def _capture_cmd(**call_kwargs):
    """Call call_worker with subprocess.run patched; return the cmd list passed."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _FakeProc()

    with mock.patch.object(providers.shutil, "which", return_value="copilot.cmd"), \
         mock.patch.object(providers.subprocess, "run", side_effect=fake_run):
        providers.call_worker("copilot", "gpt-5-mini", "hi", cwd="/x",
                              timeout=30, **call_kwargs)
    return seen["cmd"]


class TestCopilotCmd(unittest.TestCase):
    def test_default_has_no_available_tools_flag(self):
        cmd = _capture_cmd()
        self.assertFalse(any(c.startswith("--available-tools") for c in cmd))

    def test_empty_available_tools_disables_all(self):
        cmd = _capture_cmd(available_tools=[])
        self.assertIn("--available-tools=", cmd)

    def test_named_available_tools_csv(self):
        cmd = _capture_cmd(available_tools=["read", "grep"])
        self.assertIn("--available-tools=read,grep", cmd)

    def test_model_and_allow_present(self):
        cmd = _capture_cmd()
        self.assertIn("--model", cmd)
        self.assertIn("gpt-5-mini", cmd)
        self.assertIn("--allow-all", cmd)


class _FakeUsage:
    def __init__(self, total):
        self.prompt_tokens = 84
        self.completion_tokens = total - 84
        self.total_tokens = total


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResp:
    def __init__(self, content="hi there", total_tokens=206):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(total_tokens)


class _FakeOpenAI:
    """Stand-in for openai.OpenAI; records create() kwargs, returns a canned resp."""
    last_create_kwargs: dict = {}
    last_init_kwargs: dict = {}
    raise_on_create: Exception | None = None

    def __init__(self, **kwargs):
        _FakeOpenAI.last_init_kwargs = kwargs
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        _FakeOpenAI.last_create_kwargs = kwargs
        if _FakeOpenAI.raise_on_create is not None:
            raise _FakeOpenAI.raise_on_create
        return _FakeResp()


def _run_deepinfra(env_key="set", **call_kwargs):
    """Call the deepinfra handler with openai.OpenAI patched. env_key='set'|'unset'."""
    _FakeOpenAI.raise_on_create = None
    import types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    env = {"DEEPINFRA_TOKEN": "tok"} if env_key == "set" else {}
    with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
         mock.patch.dict(os.environ, env, clear=True):
        return providers.call_worker("deepinfra", "openai/gpt-oss-120b",
                                     "explain X", cwd="/x", timeout=30, **call_kwargs)


class TestDeepInfraHandler(unittest.TestCase):
    def test_registered(self):
        self.assertIn("deepinfra", providers._REGISTRY)

    def test_success_returns_content_and_real_tokens(self):
        wr = _run_deepinfra()
        self.assertEqual(wr.exit_code, 0)
        self.assertEqual(wr.stdout, "hi there")
        self.assertEqual(wr.real_tokens, 206)

    def test_base_url_and_key_passed_to_client(self):
        _run_deepinfra()
        self.assertEqual(_FakeOpenAI.last_init_kwargs["base_url"],
                         "https://api.deepinfra.com/v1/openai")
        self.assertEqual(_FakeOpenAI.last_init_kwargs["api_key"], "tok")

    def test_missing_key_returns_exit_1(self):
        wr = _run_deepinfra(env_key="unset")
        self.assertEqual(wr.exit_code, 1)
        self.assertIn("DEEPINFRA_TOKEN", wr.stderr)
        self.assertIsNone(wr.real_tokens)

    def test_api_error_returns_exit_1(self):
        _FakeOpenAI.raise_on_create = RuntimeError("502 upstream")
        import types
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = _FakeOpenAI
        with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
             mock.patch.dict(os.environ, {"DEEPINFRA_TOKEN": "tok"}, clear=True):
            wr = providers.call_worker("deepinfra", "m", "p", cwd="/x", timeout=5)
        self.assertEqual(wr.exit_code, 1)
        self.assertIn("502", wr.stderr)

    def test_available_tools_ignored_not_sent(self):
        """available_tools (copilot-only concept) must not leak into the API call."""
        _run_deepinfra(available_tools=[])
        self.assertNotIn("available_tools", _FakeOpenAI.last_create_kwargs)
        self.assertNotIn("tools", _FakeOpenAI.last_create_kwargs)

    def test_reasoning_effort_forwarded_when_set(self):
        _run_deepinfra(reasoning_effort="low")
        self.assertEqual(_FakeOpenAI.last_create_kwargs.get("reasoning_effort"), "low")

    def test_reasoning_effort_omitted_when_none(self):
        _run_deepinfra()
        self.assertNotIn("reasoning_effort", _FakeOpenAI.last_create_kwargs)


if __name__ == "__main__":
    unittest.main()
