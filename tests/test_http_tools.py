"""Tests for the HTTP provider's client-side tools + agent loop (hive.http_tools).

No network: the sandboxed read/list/grep tools run against a tmp tree, and the
agent loop is driven by a scripted fake OpenAI client. The provider-level routing
(single-shot vs loop) is checked by monkeypatching ``openai.OpenAI``.
"""
import json
import os
from types import SimpleNamespace

import pytest

from hive import http_tools as T
from hive.http_tools import (
    read_file, list_dir, grep, select_tools, execute_tool, run_agent_loop,
    ALL_TOOL_NAMES,
)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("import os\nx = 1\nprint(x)\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello world\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("foo = 1\n", encoding="utf-8")
    return str(tmp_path)


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_numbers_lines(tree):
    out = read_file(tree, path="a.py")
    assert "1\timport os" in out
    assert "3\tprint(x)" in out


def test_read_file_slice(tree):
    out = read_file(tree, path="a.py", start_line=2, end_line=2)
    assert "2\tx = 1" in out
    assert "import os" not in out
    assert "print(x)" not in out


def test_read_file_missing(tree):
    assert read_file(tree, path="nope.py").startswith("[error]")


def test_read_file_binary_refused(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02ABC")
    assert "binary" in read_file(str(tmp_path), path="blob.bin")


def test_read_file_sandbox_escape_refused(tree):
    assert execute_tool("read_file", {"path": "../secret"}, tree).startswith("[error]")


def test_read_file_truncates_long(tmp_path):
    big = "\n".join(str(i) for i in range(T._MAX_READ_LINES + 50)) + "\n"
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")
    out = read_file(str(tmp_path), path="big.txt")
    assert "truncated" in out


# ── list_dir ──────────────────────────────────────────────────────────────────

def test_list_dir_marks_directories(tree):
    out = list_dir(tree, path=".")
    assert "sub/" in out
    assert "a.py" in out


def test_list_dir_not_a_dir(tree):
    assert list_dir(tree, path="a.py").startswith("[error]")


# ── grep ──────────────────────────────────────────────────────────────────────

def test_grep_finds_matches_with_location(tree):
    out = grep(tree, pattern=r"return 42")
    assert "sub/b.py:2:" in out


def test_grep_glob_filter(tree):
    # 'hello' lives only in notes.txt; restricting to *.py finds nothing.
    assert grep(tree, pattern="hello", glob="*.py").startswith("[no matches")
    assert "notes.txt" in grep(tree, pattern="hello", glob="*.txt")


def test_grep_skips_git_dir(tree):
    # ".git/config" contains 'foo = 1' but the VCS dir is skipped.
    assert grep(tree, pattern=r"foo = 1").startswith("[no matches")


def test_grep_bad_regex(tree):
    assert grep(tree, pattern="(").startswith("[error]")


def test_grep_sandbox_escape_refused(tree):
    assert execute_tool("grep", {"pattern": "x", "path": ".."}, tree).startswith("[error]")


# ── select_tools (the available_tools convention) ─────────────────────────────

def test_select_tools_none_is_all(tree):
    assert set(select_tools(None, have_cwd=True)) == set(ALL_TOOL_NAMES)


def test_select_tools_empty_is_single_shot(tree):
    assert select_tools([], have_cwd=True) == []


def test_select_tools_subset(tree):
    assert select_tools(["read_file", "bogus"], have_cwd=True) == ["read_file"]


def test_select_tools_no_cwd_disables(tree):
    assert select_tools(None, have_cwd=False) == []


# ── execute_tool error handling ───────────────────────────────────────────────

def test_execute_unknown_tool():
    assert execute_tool("nope", {}, ".").startswith("[error] unknown tool")


def test_execute_bad_arguments(tree):
    assert execute_tool("read_file", {"bogus": 1}, tree).startswith("[error]")


# ── agent loop (scripted fake client) ─────────────────────────────────────────

def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name,
                                                    arguments=json.dumps(args)))


def _resp(content=None, tool_calls=None, total_tokens=10):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(total_tokens=total_tokens))


class _FakeClient:
    """Replays a scripted list of responses and records each create() kwargs."""
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


def test_loop_executes_tool_then_returns_answer(tree):
    scripted = [
        _resp(tool_calls=[_tool_call("c1", "read_file", {"path": "a.py"})],
              total_tokens=10),
        _resp(content="a.py defines x = 1", total_tokens=7),
    ]
    client = _FakeClient(scripted)
    messages = [{"role": "user", "content": "what is in a.py?"}]
    content, tokens = run_agent_loop(
        client, "m", messages, root=tree, tool_names=list(ALL_TOOL_NAMES),
        temperature=0.2, max_tokens=1024, extra={})
    assert content == "a.py defines x = 1"
    assert tokens == 17  # summed across both round-trips
    # The transcript grew: assistant tool-call turn + tool result + (final answer not appended).
    roles = [m["role"] for m in messages]
    assert "assistant" in roles and "tool" in roles
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert "import os" in tool_msg["content"]  # real file content fed back


def test_loop_passes_tools_when_enabled(tree):
    client = _FakeClient([_resp(content="done")])
    run_agent_loop(client, "m", [{"role": "user", "content": "hi"}], root=tree,
                   tool_names=["read_file"], temperature=0.2, max_tokens=64, extra={})
    assert "tools" in client.calls[0]
    assert client.calls[0]["tools"][0]["function"]["name"] == "read_file"


def test_loop_withholds_tools_on_final_iteration(tree):
    # Model keeps asking for tools; the iteration bound must force a final answer
    # by switching tool_choice to 'none' (no tools) on the last round.
    never_stops = [_resp(tool_calls=[_tool_call(f"c{i}", "list_dir", {"path": "."})])
                   for i in range(10)]
    client = _FakeClient(never_stops)
    run_agent_loop(client, "m", [{"role": "user", "content": "loop"}], root=tree,
                   tool_names=["list_dir"], temperature=0.2, max_tokens=64,
                   extra={}, max_iterations=3)
    assert len(client.calls) == 3
    assert "tools" not in client.calls[-1]  # last round withholds tools


# ── provider routing: single-shot vs loop ─────────────────────────────────────

def test_provider_single_shot_when_tools_empty(tree, monkeypatch):
    import openai
    from hive import providers
    monkeypatch.setenv("DEEPINFRA_TOKEN", "k")
    client = _FakeClient([_resp(content="verdict", total_tokens=5)])
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    wr = providers._call_openai_compatible(
        "m", "judge this", cwd=tree, available_tools=[])
    assert wr.exit_code == 0 and wr.stdout == "verdict"
    assert wr.real_tokens == 5
    assert "tools" not in client.calls[0]  # single-shot, no tool schemas


def test_provider_runs_loop_when_tools_default(tree, monkeypatch):
    import openai
    from hive import providers
    monkeypatch.setenv("DEEPINFRA_TOKEN", "k")
    scripted = [
        _resp(tool_calls=[_tool_call("c1", "list_dir", {"path": "."})], total_tokens=4),
        _resp(content="found a.py", total_tokens=3),
    ]
    client = _FakeClient(scripted)
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    wr = providers._call_openai_compatible(
        "m", "explore", cwd=tree, available_tools=None)
    assert wr.stdout == "found a.py"
    assert wr.real_tokens == 7
    assert "tools" in client.calls[0]


def test_provider_no_cwd_forces_single_shot(tree, monkeypatch):
    import openai
    from hive import providers
    monkeypatch.setenv("DEEPINFRA_TOKEN", "k")
    client = _FakeClient([_resp(content="x", total_tokens=1)])
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    # available_tools=None but no cwd → nothing to read → single-shot.
    providers._call_openai_compatible("m", "p", cwd=None, available_tools=None)
    assert "tools" not in client.calls[0]
