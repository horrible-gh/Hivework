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


if __name__ == "__main__":
    unittest.main()
