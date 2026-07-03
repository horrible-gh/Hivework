"""TS0006 (group 0077) certification scenarios for the fanout loop-cost work.

Deterministic, zero-billing re-certification of TR0005 (R0001 requirements 1+2):
scripted fake clients drive the REAL ``hive.http_tools.run_agent_loop`` /
``hive.providers`` / ``hive.config`` / ``hive.py`` wiring — no live API, no
tokens spent. Each scenario judges by OBSERVABLES (call counts, serialized
transcript bytes), never by symbol presence, so a decoy implementation cannot
pass.

  S1  cap truncation:   max_iterations=12 → exactly 12 in-loop calls (+1 guard);
                        unset → the code-default 25 (existing behaviour kept).
  S2  prune linearizes:  per-round transcript bytes plateau under
                        prune_keep_rounds=2 vs monotone growth without.
  S3  wiring ablation:   config A (stage knobs) / B (no knobs) / C (global knob,
                        stage override) reach the loop — ON=1 / OFF=0 triangle.
  S4  no-bite:           skeleton/seed/assistant invariance, final comb
                        passthrough, small outputs untouched, copilot ignores
                        the keys.
  S5  live evidence:     re-aggregate ledger runs 681/682/683 + honey anchors
                        (reads only — the paid runs were TR0005's).

Run:  python perf/ts0077_loop_cost.py            (exit 0 = all GO)
"""
import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from hive import providers
from hive.config import load_config
from hive.http_tools import run_agent_loop, DEFAULT_MAX_ITERATIONS, ALL_TOOL_NAMES

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    status = "GO" if cond else "NO-GO"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


# ── scripted client ────────────────────────────────────────────────────────────

def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name,
                                                    arguments=json.dumps(args)))


def _resp(content=None, tool_calls=None, total_tokens=10):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(total_tokens=total_tokens))


class HungryClient:
    """A drone that requests a (large) read_file every round, forever — the
    runaway-tail pathology. Records the serialized transcript size of every
    create() so history growth is measurable."""

    def __init__(self, path, final_comb=None, rounds_before_comb=None):
        self.calls = 0
        self.sizes = []
        self.path = path
        self.final_comb = final_comb
        self.rounds_before_comb = rounds_before_comb
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        self.sizes.append(len(json.dumps(kwargs["messages"])))
        if (self.rounds_before_comb is not None
                and self.calls > self.rounds_before_comb):
            return _resp(content=self.final_comb)
        return _resp(tool_calls=[_tool_call(f"c{self.calls}", "read_file",
                                            {"path": self.path})])


def _make_root(lines=800) -> tuple[str, str]:
    root = tempfile.mkdtemp(prefix="ts0077_")
    name = "big.txt"
    with open(os.path.join(root, name), "w", encoding="utf-8") as f:
        for i in range(lines):
            f.write(f"payload line {i} " + "x" * 40 + "\n")
    return root, name


def _run_loop(root, client, **kw):
    return run_agent_loop(client, "m", [{"role": "user", "content": "seed"}],
                          root=root, tool_names=list(ALL_TOOL_NAMES),
                          temperature=0.2, max_tokens=64, extra={}, **kw)


# ── S1 cap truncation ─────────────────────────────────────────────────────────

def s1():
    print("S1 — cap truncation")
    root, name = _make_root(lines=10)
    capped = HungryClient(name)
    _run_loop(root, capped, max_iterations=12)
    # 12 in-loop rounds + exactly one forced-answer guard call.
    check(capped.calls == 13, "capped drone bills 12 rounds + 1 guard",
          f"calls={capped.calls}")
    uncapped = HungryClient(name)
    _run_loop(root, uncapped)
    check(uncapped.calls == DEFAULT_MAX_ITERATIONS + 1,
          "unset keeps the code default (25 + guard)",
          f"calls={uncapped.calls}, DEFAULT={DEFAULT_MAX_ITERATIONS}")


# ── S2 pruning linearizes transcript growth ───────────────────────────────────

def s2():
    print("S2 — pruning linearizes per-round transcript bytes")
    root, name = _make_root(lines=800)
    off = HungryClient(name)
    _run_loop(root, off, max_iterations=12)
    on = HungryClient(name)
    _run_loop(root, on, max_iterations=12, prune_keep_rounds=2)
    grow_off = off.sizes[-1] / off.sizes[3]
    grow_on = on.sizes[-1] / on.sizes[3]
    check(all(b > a for a, b in zip(off.sizes, off.sizes[1:])),
          "prune-off transcript grows every round (the pathology)",
          f"first={off.sizes[0]}B last={off.sizes[-1]}B")
    check(grow_on < 1.5, "prune-on plateaus after the keep window",
          f"round4→last growth ×{grow_on:.2f} (off ×{grow_off:.2f})")
    check(on.sizes[-1] < off.sizes[-1] * 0.5,
          "prune-on final round ships <50% of prune-off bytes",
          f"{on.sizes[-1]}B vs {off.sizes[-1]}B "
          f"(-{100 - 100 * on.sizes[-1] / off.sizes[-1]:.0f}%)")


# ── S3 config wiring ablation triangle ────────────────────────────────────────

def _hive_cli():
    spec = importlib.util.spec_from_file_location(
        "hive_cli_ts0077", os.path.join(_REPO, "hive.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _loop_kwargs_for(raw_cfg: dict) -> dict:
    """Drive config → build_provider_kwargs → fanout_provider_kwargs →
    _call_openai_compatible → (mocked) run_agent_loop; return what the loop saw."""
    tmp = tempfile.mkdtemp(prefix="ts0077_cfg_")
    cfg_path = os.path.join(tmp, "hive.config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(raw_cfg, f)
    cfg = load_config(path=cfg_path)
    cli = _hive_cli()
    pk = cli.fanout_provider_kwargs(cfg, cli.build_provider_kwargs(cfg))
    seen = {}
    sentinel = object()

    def fake_loop(client, model, messages, *, root, tool_names, temperature,
                  max_tokens, extra, max_iterations=sentinel,
                  prune_keep_rounds=sentinel):
        seen["max_iterations"] = max_iterations
        seen["prune_keep_rounds"] = prune_keep_rounds
        return "ok", 5

    import types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kw: object()
    with mock.patch.dict("sys.modules", {"openai": fake_openai}), \
         mock.patch.dict(os.environ, {"DEEPINFRA_TOKEN": "tok"}), \
         mock.patch.object(providers.http_tools, "run_agent_loop", fake_loop):
        providers.call_worker("openai", "m", "p", cwd=_REPO, timeout=30, **pk)
    seen["sentinel"] = sentinel
    return seen


def s3():
    print("S3 — config wiring ablation (ON=1 / OFF=0 / override)")
    a = _loop_kwargs_for({"pipeline": {"fanout": {
        "provider": "openai", "model": "m", "enabled": True,
        "max_iterations": 12, "prune_keep_rounds": 2}}})
    check(a["max_iterations"] == 12 and a["prune_keep_rounds"] == 2,
          "A: stage knobs reach the loop (ON=1)",
          f"loop saw {a['max_iterations']}/{a['prune_keep_rounds']}")
    b = _loop_kwargs_for({"pipeline": {"fanout": {
        "provider": "openai", "model": "m", "enabled": True}}})
    check(b["max_iterations"] is b["sentinel"]
          and b["prune_keep_rounds"] is b["sentinel"],
          "B: no knobs → loop keeps its own defaults (OFF=0, acceptance №3)")
    c = _loop_kwargs_for({
        "providers": {"openai": {"base_url": "https://x", "api_key_env":
                                 "DEEPINFRA_TOKEN", "max_agent_iterations": 10}},
        "pipeline": {"fanout": {"provider": "openai", "model": "m",
                                "enabled": True, "max_iterations": 12}}})
    check(c["max_iterations"] == 12,
          "C: stage knob overrides the global one", f"loop saw {c['max_iterations']}")
    c2 = _loop_kwargs_for({
        "providers": {"openai": {"base_url": "https://x", "api_key_env":
                                 "DEEPINFRA_TOKEN", "max_agent_iterations": 10}},
        "pipeline": {"fanout": {"provider": "openai", "model": "m",
                                "enabled": True}}})
    check(c2["max_iterations"] == 10,
          "C2: global knob alone reaches every tool-ON stage",
          f"loop saw {c2['max_iterations']}")


# ── S4 no-bite ─────────────────────────────────────────────────────────────────

def s4():
    print("S4 — no-bite")
    root, name = _make_root(lines=800)
    comb = json.dumps({"axis_id": "X", "findings": [{"t": "f1"}]})
    client = HungryClient(name, final_comb=comb, rounds_before_comb=5)
    messages = [{"role": "user", "content": "seed"}]
    content, _ = run_agent_loop(
        client, "m", messages, root=root, tool_names=list(ALL_TOOL_NAMES),
        temperature=0.2, max_tokens=64, extra={},
        max_iterations=12, prune_keep_rounds=2)
    check(content == comb, "final comb passes through pruning untouched")
    check(messages[0]["content"] == "seed", "seed prompt never pruned")
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    check(all(m.get("tool_call_id") for m in tool_msgs),
          "tool_call_id pairing intact (protocol-safe)",
          f"{len(tool_msgs)} tool turns")
    check(any("pruned" in m["content"] for m in tool_msgs[:-2])
          and "pruned" not in tool_msgs[-1]["content"],
          "old rounds stubbed, keep-window verbatim")
    assistant_texts = [m.get("content") for m in messages
                       if m.get("role") == "assistant" and m.get("content")]
    check(all("pruned" not in (t or "") for t in assistant_texts),
          "assistant turns untouched")

    root2, name2 = _make_root(lines=4)  # tiny outputs: not worth stubbing
    small = HungryClient(name2)
    msgs2 = [{"role": "user", "content": "seed"}]
    run_agent_loop(small, "m", msgs2, root=root2,
                   tool_names=list(ALL_TOOL_NAMES), temperature=0.2,
                   max_tokens=64, extra={}, max_iterations=6,
                   prune_keep_rounds=1)
    check(all("pruned" not in m["content"] for m in msgs2
              if m.get("role") == "tool"), "small outputs left verbatim")

    import subprocess
    def fake_run(cmd, input=None, cwd=None, timeout=None, env=None):
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
    with mock.patch.object(providers.shutil, "which", return_value="copilot.cmd"), \
         mock.patch.object(providers, "_run_capture", side_effect=fake_run):
        wr = providers.call_worker("copilot", "gpt-5-mini", "p", cwd="/x",
                                   timeout=30, max_iterations=12,
                                   prune_keep_rounds=2)
    check(wr.exit_code == 0 and wr.stdout == "ok",
          "copilot handler ignores both keys (mixed-provider safe)")


# ── S5 live-evidence cross-check (reads only) ─────────────────────────────────

_EXPECTED = {681: 1024903, 682: 1076407, 683: 307409}
_ANCHORS = ("remote_routes.py", "remote_tool_service", "prompt_copy_service")


def s5():
    print("S5 — live evidence cross-check (no new billing)")
    db = os.path.join(_REPO, "hive_ledger.db")
    conn = sqlite3.connect(db)
    for run_id, expected in _EXPECTED.items():
        got = conn.execute(
            "SELECT SUM(real_tokens) FROM worker_calls "
            "WHERE run_id=? AND stage='fanout'", (run_id,)).fetchone()[0]
        check(got == expected, f"ledger run {run_id} fanout tokens match TR0005",
              f"{got} vs {expected}")
    conn.close()
    honey = os.path.join(_REPO, ".live", "0117_nr", "honey_cap12_prune2.md")
    with open(honey, encoding="utf-8") as f:
        text = f.read()
    for anchor in _ANCHORS:
        check(anchor in text, f"honey (run 683) carries answer-key anchor {anchor!r}")
    saved = _EXPECTED[681] - _EXPECTED[683]
    print(f"  [info] certified saving: {saved:,} tokens/run "
          f"(-{100 * saved / _EXPECTED[681]:.1f}%)")


def main() -> int:
    for scenario in (s1, s2, s3, s4, s5):
        scenario()
    print()
    if FAILURES:
        print(f"RESULT: NO-GO ({len(FAILURES)} failure(s)): {FAILURES}")
        return 1
    print("RESULT: GO — all scenarios passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
