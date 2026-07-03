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
    # by switching tool_choice to 'none' (no tools) on the last round, where a real
    # model — unable to call tools — returns a plain answer.
    scripted = [
        _resp(tool_calls=[_tool_call("c0", "list_dir", {"path": "."})]),
        _resp(tool_calls=[_tool_call("c1", "list_dir", {"path": "."})]),
        _resp(content="forced final answer"),  # tools withheld → plain answer
    ]
    client = _FakeClient(scripted)
    run_agent_loop(client, "m", [{"role": "user", "content": "loop"}], root=tree,
                   tool_names=["list_dir"], temperature=0.2, max_tokens=64,
                   extra={}, max_iterations=3)
    assert len(client.calls) == 3
    assert "tools" not in client.calls[-1]  # last round withholds tools


# ── reasoning-channel fallback + empty-comb retry (NR 0004.0003 §2) ───────────

def _reasoning_resp(reasoning, content=None, attr="reasoning_content", total_tokens=5):
    msg = SimpleNamespace(content=content, tool_calls=None, **{attr: reasoning})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(total_tokens=total_tokens))


def test_message_text_prefers_content():
    msg = SimpleNamespace(content="real answer", reasoning_content="thinking")
    assert T.message_text(msg) == "real answer"


def test_message_text_falls_back_to_reasoning_content():
    msg = SimpleNamespace(content="", reasoning_content="answer in reasoning")
    assert T.message_text(msg) == "answer in reasoning"


def test_message_text_falls_back_to_reasoning():
    msg = SimpleNamespace(content=None, reasoning="answer via reasoning")
    assert T.message_text(msg) == "answer via reasoning"


def test_message_text_empty_when_nothing():
    assert T.message_text(SimpleNamespace(content=None)) == ""


def test_loop_recovers_answer_from_reasoning_channel(tree):
    # Forced-answer turn: empty content but the answer is on the reasoning channel.
    client = _FakeClient([_reasoning_resp("the real comb body", content="")])
    content, _ = run_agent_loop(
        client, "m", [{"role": "user", "content": "investigate"}], root=tree,
        tool_names=["read_file"], temperature=0.2, max_tokens=64, extra={})
    assert content == "the real comb body"  # not an empty comb


def test_loop_retries_once_when_fully_empty(tree):
    # Both content and reasoning empty → ONE forced-answer retry recovers it.
    client = _FakeClient([
        _resp(content="", total_tokens=3),          # empty first answer
        _resp(content="recovered on retry", total_tokens=4),  # the retry
    ])
    content, tokens = run_agent_loop(
        client, "m", [{"role": "user", "content": "go"}], root=tree,
        tool_names=["read_file"], temperature=0.2, max_tokens=64, extra={})
    assert content == "recovered on retry"
    assert len(client.calls) == 2          # original + exactly one retry
    assert "tools" not in client.calls[-1]  # retry forces a plain answer
    assert tokens == 7                      # tokens summed across both


def test_loop_forced_answer_on_tool_arg_text_at_iteration_bound(tree):
    # RC-1 (NR 0005.0003): the model prints its next search's arguments as the
    # answer instead of concluding. On the LAST iteration the RC-C bridge cannot
    # help (no budget to investigate), so the terminal tool-arg guard forces ONE
    # conclusion turn (max_iterations=1 pins the loop to that terminal case).
    memo = json.dumps({"path": "", "pattern": "create_button", "glob": "*.vue"})
    client = _FakeClient([
        _resp(content=memo, total_tokens=3),                         # tool-arg noise
        _resp(content='{"axis_id":"A","findings":[]}', total_tokens=4),  # forced conclusion
    ])
    content, tokens = run_agent_loop(
        client, "m", [{"role": "user", "content": "go"}], root=tree,
        tool_names=["grep"], temperature=0.2, max_tokens=64, extra={},
        max_iterations=1)
    assert content == '{"axis_id":"A","findings":[]}'
    assert len(client.calls) == 2          # original + exactly one forced-answer turn
    assert "tools" not in client.calls[-1]  # the forced turn withholds tools


def test_loop_does_not_retry_a_real_comb(tree):
    # A genuine comb carries non-tool keys (axis_id/findings) → not tool-arg text,
    # so no wasteful extra turn is spent on a valid conclusion.
    comb = '{"axis_id":"A","findings":[{"claim":"x"}]}'
    client = _FakeClient([_resp(content=comb, total_tokens=5)])
    content, _ = run_agent_loop(
        client, "m", [{"role": "user", "content": "go"}], root=tree,
        tool_names=["grep"], temperature=0.2, max_tokens=64, extra={})
    assert content == comb
    assert len(client.calls) == 1          # no retry for a valid comb


def test_looks_like_tool_call_text_predicate(tree):
    names = T._tool_arg_param_names(["read_file", "grep"])
    assert {"path", "pattern", "glob", "ignore_case"} <= names
    # tool-arg objects (keys ⊆ tool params) are flagged
    assert T._looks_like_tool_call_text('{"path":"x","pattern":"y"}', names)
    assert T._looks_like_tool_call_text('{"path":"x"}', names)
    # a real comb has non-tool keys → not flagged
    assert not T._looks_like_tool_call_text('{"axis_id":"A","findings":[]}', names)
    # prose and empties → not flagged
    assert not T._looks_like_tool_call_text("I will search next.", names)
    assert not T._looks_like_tool_call_text("", names)
    # with no tools in play, nothing is a tool-call
    assert not T._looks_like_tool_call_text('{"path":"x"}', set())


def test_looks_like_tool_call_text_hallucinated_params(tree):
    # RC-2 (NR hivework.default.0007.0005): the drone hallucinates param names for
    # an imagined search API, so the keys are NOT a subset of the real schema. The
    # original exact-subset guard missed these (caught 1/5 of a live run's noise);
    # the generalised guard flags any flat, path-bearing, non-comb object.
    names = T._tool_arg_param_names(["read_file", "grep"])
    for noise in (
        '{"path":"client/src","query":"ToastContainer","max_results":20}',
        '{"path":"client/src","depth":2}',
        '{"path":"server","query":"group","max_results":20}',
        '{"path":"server/modules/flow_gate/db.py","line_start":340,"line_end":380}',
    ):
        assert T._looks_like_tool_call_text(noise, names), noise
    # Still conservative: a real comb that merely carries a path field, and a
    # substantive object with no path key, are NOT swept up.
    assert not T._looks_like_tool_call_text(
        '{"axis_id":"A","findings":[{"file":"a.py"}],"path":"a.py"}', names)
    assert not T._looks_like_tool_call_text(
        '{"summary":"two R docs created","recommendation":"add unique constraint"}', names)
    # A nested (non-flat) object is an answer shape, not a memo.
    assert not T._looks_like_tool_call_text('{"path":"a.py","note":{"k":1}}', names)


# ── RC-C text→execution bridge (NR hivework.default.0008.0009) ────────────────

def test_infer_tool_call_grep_from_pattern(tree):
    # A printed search with a pattern (or hallucinated 'query') maps to grep.
    assert T._infer_tool_call_from_text(
        '{"path":"sub","pattern":"foo","glob":"*.py"}', ["grep", "read_file"]) \
        == ("grep", {"pattern": "foo", "path": "sub", "glob": "*.py"})
    name, kw = T._infer_tool_call_from_text(
        '{"path":"client/src","query":"toast","max_results":20}', ["grep", "read_file"])
    assert name == "grep" and kw["pattern"] == "toast" and kw["path"] == "client/src"


def test_infer_tool_call_read_with_line_bounds(tree):
    # path + line bounds (or hallucinated line_start/line_end) → read_file slice.
    assert T._infer_tool_call_from_text(
        '{"path":"a.py","line_start":1,"line_end":2}', ["grep", "read_file"]) \
        == ("read_file", {"path": "a.py", "start_line": 1, "end_line": 2})


def test_infer_tool_call_bare_path(tree):
    # A file-looking path → read_file; a dir-looking path → list_dir.
    assert T._infer_tool_call_from_text('{"path":"a.py"}', list(ALL_TOOL_NAMES)) \
        == ("read_file", {"path": "a.py"})
    assert T._infer_tool_call_from_text('{"path":"sub"}', list(ALL_TOOL_NAMES)) \
        == ("list_dir", {"path": "sub"})


def test_infer_tool_call_none_when_unmappable(tree):
    assert T._infer_tool_call_from_text("prose, not json", ["grep"]) is None
    assert T._infer_tool_call_from_text('{"foo":"bar"}', ["grep"]) is None
    # grep memo but grep not offered → no inference
    assert T._infer_tool_call_from_text('{"pattern":"x"}', ["read_file"]) is None


def test_loop_bridges_printed_search_into_real_evidence(tree):
    # RC-A1/RC-C: the drone prints a search instead of calling a tool. The loop must
    # EXECUTE it, feed the real file content back, and let the drone conclude — not
    # break blind. The grep hits sub/b.py ('return 42').
    memo = json.dumps({"path": ".", "pattern": "return"})
    scripted = [
        _resp(content=memo, total_tokens=3),                              # printed search
        _resp(content='{"axis_id":"A","findings":[{"claim":"x"}]}', total_tokens=4),  # concludes
    ]
    client = _FakeClient(scripted)
    messages = [{"role": "user", "content": "investigate"}]
    content, tokens = run_agent_loop(
        client, "m", messages, root=tree, tool_names=["grep", "read_file"],
        temperature=0.2, max_tokens=64, extra={})
    assert content == '{"axis_id":"A","findings":[{"claim":"x"}]}'
    assert tokens == 7
    # The executed grep result was fed back as a user turn carrying real evidence.
    fed = [m for m in messages if m["role"] == "user" and "executed for you" in m["content"]]
    assert fed and "b.py" in fed[0]["content"]


def test_loop_bridge_is_bounded(tree):
    # A drone that ONLY ever prints searches must still terminate: after the bridge
    # cap it falls through to the forced-answer guard, not an infinite loop.
    memo = json.dumps({"path": ".", "pattern": "x"})
    # _MAX_TEXT_BRIDGES bridge round-trips, then one more memo that breaks the loop
    # (budget spent), then the forced-answer turn supplies the conclusion.
    scripted = [_resp(content=memo, total_tokens=1)
                for _ in range(T._MAX_TEXT_BRIDGES + 1)]
    scripted.append(_resp(content="forced conclusion", total_tokens=1))
    client = _FakeClient(scripted)
    content, _ = run_agent_loop(
        client, "m", [{"role": "user", "content": "go"}], root=tree,
        tool_names=["grep"], temperature=0.2, max_tokens=64, extra={},
        max_iterations=20)
    # Bounded: _MAX_TEXT_BRIDGES bridges + 1 break round + 1 forced-answer turn.
    assert len(client.calls) == T._MAX_TEXT_BRIDGES + 2
    assert content == "forced conclusion"


def test_loop_retry_tolerates_failure(tree):
    # If the retry itself raises, the loop returns the (empty) content, not an error.
    class _BoomThenNothing:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return _resp(content="", total_tokens=2)
            raise RuntimeError("retry boom")

    client = _BoomThenNothing()
    content, _ = run_agent_loop(
        client, "m", [{"role": "user", "content": "go"}], root=tree,
        tool_names=["read_file"], temperature=0.2, max_tokens=64, extra={})
    assert content == ""            # graceful, no exception
    assert len(client.calls) == 2   # it did attempt the retry


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


def test_provider_single_shot_recovers_reasoning(tree, monkeypatch):
    # Single-shot judge/pick: empty content, answer on the reasoning channel.
    import openai
    from hive import providers
    monkeypatch.setenv("DEEPINFRA_TOKEN", "k")
    client = _FakeClient([_reasoning_resp("verdict in reasoning", content="")])
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    wr = providers._call_openai_compatible(
        "m", "judge this", cwd=tree, available_tools=[])
    assert wr.exit_code == 0 and wr.stdout == "verdict in reasoning"


def test_provider_no_cwd_forces_single_shot(tree, monkeypatch):
    import openai
    from hive import providers
    monkeypatch.setenv("DEEPINFRA_TOKEN", "k")
    client = _FakeClient([_resp(content="x", total_tokens=1)])
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    # available_tools=None but no cwd → nothing to read → single-shot.
    providers._call_openai_compatible("m", "p", cwd=None, available_tools=None)
    assert "tools" not in client.calls[0]


# ── history pruning (R0001 0077 req 2) ────────────────────────────────────────

def _many_line_result_call(call_id):
    # grep over conftest-made tree isn't long enough; read a real multi-line file.
    return _tool_call(call_id, "read_file", {"path": "a.py"})


def _long_tool_round(call_id):
    return _resp(tool_calls=[_tool_call(call_id, "list_dir", {"path": "."})])


def _pruning_tree_file(tree, name="big.txt", lines=30):
    path = os.path.join(tree, name)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(lines):
            f.write(f"line-{i}\n")
    return name


def test_prune_off_keeps_old_tool_results_verbatim(tree):
    # Default (no prune_keep_rounds) is byte-for-byte today's behaviour.
    name = _pruning_tree_file(tree)
    scripted = [
        _resp(tool_calls=[_tool_call("c1", "read_file", {"path": name})]),
        _resp(tool_calls=[_tool_call("c2", "list_dir", {"path": "."})]),
        _resp(tool_calls=[_tool_call("c3", "list_dir", {"path": "."})]),
        _resp(content="done"),
    ]
    messages = [{"role": "user", "content": "go"}]
    run_agent_loop(_FakeClient(scripted), "m", messages, root=tree,
                   tool_names=list(ALL_TOOL_NAMES), temperature=0.2,
                   max_tokens=64, extra={})
    first_tool = next(m for m in messages if m["role"] == "tool")
    assert "line-29" in first_tool["content"]
    assert "pruned" not in first_tool["content"]


def test_prune_stubs_old_rounds_keeps_recent_full(tree):
    name = _pruning_tree_file(tree)
    scripted = [
        _resp(tool_calls=[_tool_call("c1", "read_file", {"path": name})]),
        _resp(tool_calls=[_tool_call("c2", "read_file", {"path": name})]),
        _resp(tool_calls=[_tool_call("c3", "read_file", {"path": name})]),
        _resp(content="done"),
    ]
    messages = [{"role": "user", "content": "go"}]
    content, _ = run_agent_loop(
        _FakeClient(scripted), "m", messages, root=tree,
        tool_names=list(ALL_TOOL_NAMES), temperature=0.2, max_tokens=64,
        extra={}, prune_keep_rounds=1)
    assert content == "done"
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    # Rounds 1–2 (older than the keep window when later rounds fired) are stubbed…
    assert "pruned" in tool_msgs[0]["content"]
    assert "call the tool again" in tool_msgs[0]["content"]
    # …with the head kept so the drone can still recognise what it was.
    assert tool_msgs[0]["content"].startswith("1\tline-0")
    assert "pruned" in tool_msgs[1]["content"]
    # The keep-window round stays verbatim.
    assert "line-29" in tool_msgs[2]["content"]
    assert "pruned" not in tool_msgs[2]["content"]
    # Skeleton intact: ids/roles survive so the OpenAI protocol stays valid.
    assert all(m.get("tool_call_id") for m in tool_msgs)
    assert [m["role"] for m in messages[:2]] == ["user", "assistant"]


def test_prune_leaves_tiny_results_alone(tree):
    # A result at/under the head+2 threshold isn't worth a stub — left as is.
    scripted = [
        _resp(tool_calls=[_tool_call("c1", "list_dir", {"path": "."})]),
        _resp(tool_calls=[_tool_call("c2", "list_dir", {"path": "."})]),
        _resp(tool_calls=[_tool_call("c3", "list_dir", {"path": "."})]),
        _resp(content="done"),
    ]
    messages = [{"role": "user", "content": "go"}]
    run_agent_loop(_FakeClient(scripted), "m", messages, root=tree,
                   tool_names=list(ALL_TOOL_NAMES), temperature=0.2,
                   max_tokens=64, extra={}, prune_keep_rounds=1)
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert all("pruned" not in m["content"] for m in tool_msgs)


def test_prune_helper_is_idempotent_and_protocol_safe(tree):
    from hive.http_tools import _prune_old_tool_results
    long = "\n".join(f"l{i}" for i in range(40))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "seed " + long},   # seed prompt NEVER pruned
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": long},
        {"role": "assistant", "content": "memo"},
        {"role": "user",
         "content": "[You printed a search instead of calling a tool…]\n" + long},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c2", "content": long},
    ]
    pruned_ids: set = set()
    _prune_old_tool_results(messages, 1, pruned_ids)
    snapshot = [dict(m) for m in messages]
    _prune_old_tool_results(messages, 1, pruned_ids)   # second pass: no-op
    assert [dict(m) for m in messages] == snapshot
    assert "pruned" in messages[3]["content"]          # old tool result stubbed
    assert "pruned" in messages[5]["content"]          # bridge result stubbed
    assert messages[5]["content"].startswith("[You printed a search")
    assert "pruned" not in messages[7]["content"]      # keep-window round full
    assert messages[1]["content"].startswith("seed ")  # seed untouched
    assert "pruned" not in messages[1]["content"]
    assert messages[4]["content"] == "memo"            # assistant turns untouched
