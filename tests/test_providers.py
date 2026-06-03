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
         mock.patch.object(providers, "_run_capture", side_effect=fake_run):
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


def _run_deepinfra(env_key="set", provider="deepinfra", env=None, **call_kwargs):
    """Call the OpenAI-compatible handler with openai.OpenAI patched.

    ``provider`` selects the registry alias ('deepinfra' or 'openai' — same backend).
    ``env`` overrides the patched environment; by default DEEPINFRA_TOKEN=tok is set
    (env_key='unset' clears it)."""
    _FakeOpenAI.raise_on_create = None
    import types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    if env is None:
        env = {"DEEPINFRA_TOKEN": "tok"} if env_key == "set" else {}
    with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
         mock.patch.dict(os.environ, env, clear=True):
        return providers.call_worker(provider, "openai/gpt-oss-120b",
                                     "explain X", cwd="/x", timeout=30, **call_kwargs)


class TestDeepInfraHandler(unittest.TestCase):
    def test_registered(self):
        self.assertIn("deepinfra", providers._REGISTRY)

    def test_openai_alias_is_same_backend(self):
        """'openai' and 'deepinfra' map to the one OpenAI-compatible handler."""
        self.assertIn("openai", providers._REGISTRY)
        self.assertIs(providers._REGISTRY["openai"],
                      providers._REGISTRY["deepinfra"])

    def test_openai_provider_name_works(self):
        wr = _run_deepinfra(provider="openai")
        self.assertEqual(wr.exit_code, 0)
        self.assertEqual(wr.stdout, "hi there")

    def test_custom_base_url_and_api_key_env_override(self):
        """A non-DeepInfra endpoint: base_url + api_key_env flow through to the client."""
        wr = _run_deepinfra(provider="openai",
                            env={"OPENAI_API_KEY": "sk-xyz"},
                            base_url="https://api.openai.com/v1",
                            api_key_env="OPENAI_API_KEY")
        self.assertEqual(wr.exit_code, 0)
        self.assertEqual(_FakeOpenAI.last_init_kwargs["base_url"],
                         "https://api.openai.com/v1")
        self.assertEqual(_FakeOpenAI.last_init_kwargs["api_key"], "sk-xyz")

    def test_custom_api_key_env_missing_reports_that_name(self):
        wr = _run_deepinfra(provider="openai", env={},
                            api_key_env="OPENAI_API_KEY")
        self.assertEqual(wr.exit_code, 1)
        self.assertIn("OPENAI_API_KEY", wr.stderr)

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


def _run_codex(out_content=None, rc=0, stdout="(event trace)", model="gpt-5-codex",
               **call_kwargs):
    """Call the codex handler with subprocess.run patched. ``out_content`` (when
    given) is written to the --output-last-message path, simulating codex's final
    message; otherwise the temp file stays empty and the handler falls back to stdout."""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        if out_content is not None and "--output-last-message" in cmd:
            p = cmd[cmd.index("--output-last-message") + 1]
            with open(p, "w", encoding="utf-8") as f:
                f.write(out_content)
        proc = _FakeProc()
        proc.returncode = rc
        proc.stdout = stdout
        return proc

    with mock.patch.object(providers.shutil, "which", return_value="codex.cmd"), \
         mock.patch.object(providers, "_run_capture", side_effect=fake_run):
        wr = providers.call_worker("codex", model, "do X", cwd="/x", timeout=30,
                                   **call_kwargs)
    return wr, seen


class TestCodexHandler(unittest.TestCase):
    def test_registered(self):
        self.assertIn("codex", providers._REGISTRY)

    def test_exec_cmd_construction(self):
        _, seen = _run_codex()
        cmd = seen["cmd"]
        self.assertEqual(cmd[1], "exec")
        self.assertIn("-", cmd)                       # stdin marker
        self.assertIn("--model", cmd)
        self.assertIn("gpt-5-codex", cmd)
        self.assertEqual(cmd[cmd.index("--cd") + 1], "/x")
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "read-only")
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertIn("--output-last-message", cmd)

    def test_prompt_piped_via_stdin(self):
        _, seen = _run_codex()
        self.assertEqual(seen["kw"].get("input"), "do X")

    def test_final_message_file_becomes_stdout(self):
        wr, _ = _run_codex(out_content='{"verdict": {"located": true}}')
        self.assertEqual(wr.stdout, '{"verdict": {"located": true}}')
        self.assertEqual(wr.exit_code, 0)

    def test_falls_back_to_stdout_when_no_final_message(self):
        wr, _ = _run_codex(out_content=None, stdout="trace only, no final")
        self.assertEqual(wr.stdout, "trace only, no final")

    def test_empty_model_omits_model_flag(self):
        _, seen = _run_codex(model="")
        self.assertNotIn("--model", seen["cmd"])

    def test_copilot_kwargs_are_ignored(self):
        # provider_kwargs carries copilot's exe/allow_flag/available_tools; codex
        # must tolerate them (resolve its own exe, not copilot's) without error.
        wr, seen = _run_codex(allow_flag="--allow-all", available_tools=[],
                              exe="copilot.cmd")
        self.assertEqual(seen["cmd"][0], "codex.cmd")   # codex exe, not copilot's
        self.assertNotIn("--allow-all", seen["cmd"])
        self.assertEqual(wr.exit_code, 0)

    def test_nonzero_exit_surfaced(self):
        wr, _ = _run_codex(out_content="{}", rc=2)
        self.assertEqual(wr.exit_code, 2)


class _FakeProcTree:
    """Popen stand-in whose first communicate() times out, exercising the abort
    path of _run_capture. Records whether the tree-kill ran."""
    def __init__(self, killed):
        self.pid = 4242
        self.returncode = None
        self._killed = killed
        self._first = True

    def communicate(self, input=None, timeout=None):
        if self._first:
            self._first = False
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return ("", "")  # drain after kill

    def poll(self):
        return None  # still "running" so _kill_tree proceeds

    def kill(self):
        self._killed["direct"] = True


class TestRunCaptureKillsTree(unittest.TestCase):
    """The leak fix: a timed-out / interrupted worker must reap its WHOLE child
    tree (copilot.cmd → node → agent), not just the direct child."""

    def test_timeout_reaps_tree_then_reraises(self):
        killed = {}
        fp = _FakeProcTree(killed)

        def fake_tree_reap(*a, **k):       # taskkill (Windows path)
            killed["tree"] = True
            return mock.Mock(returncode=0)

        with mock.patch.object(providers.subprocess, "Popen", return_value=fp), \
             mock.patch.object(providers.subprocess, "run", side_effect=fake_tree_reap), \
             mock.patch.object(providers.os, "killpg", create=True,
                               side_effect=lambda *a: killed.__setitem__("tree", True)):
            with self.assertRaises(subprocess.TimeoutExpired):
                providers._run_capture(["worker"], input="p", cwd="/x", timeout=1)

        self.assertTrue(killed.get("tree"), "whole-tree kill (taskkill/killpg) must run")
        self.assertTrue(killed.get("direct"), "direct child kill must also run")


if __name__ == "__main__":
    unittest.main()
