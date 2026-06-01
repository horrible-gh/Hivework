"""Provider adapter — thin dispatch for worker calls.

Dispatches worker calls to the appropriate provider backend. Currently only
the 'copilot' provider is implemented. Other providers can be added by
extending the _REGISTRY dict.

Usage:
    result = call_worker("copilot", "gpt-5-mini", prompt, cwd=root, timeout=300)
    print(result.stdout, result.latency_s)
"""
import os, shutil, signal, subprocess, tempfile, time, logging
from dataclasses import dataclass

logger = logging.getLogger("hive.providers")


def _tee_call(model, latency_s, exit_code, stderr) -> None:
    """When HIVE_CALL_LOG is set, append the per-call copilot footer (carries the
    'AI Credits N.NN' line on stderr) to that file. Lets a run's true credit spend
    be summed from disk without relying on a manual account-balance read."""
    path = os.environ.get("HIVE_CALL_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n=== call model={model} latency={latency_s:.1f}s "
                    f"exit={exit_code} ===\n{stderr}\n")
    except OSError:
        pass


@dataclass
class WorkerResult:
    stdout: str
    stderr: str
    exit_code: int
    latency_s: float
    # EXACT total token count when the provider reports it (HTTP providers like
    # deepinfra via response.usage). None for copilot — the CLI exposes no token
    # counts. Feeds the ledger's real_tokens column (ledger.py docstring).
    real_tokens: int | None = None


def _kill_tree(proc) -> None:
    """Kill ``proc`` AND every descendant it spawned.

    A bare ``proc.kill()`` only reaps the direct child. The agentic CLIs launch a
    tree (copilot.cmd → node → agent; codex.cmd likewise), and the grandchildren
    keep running — and keep billing per internal turn — if the parent alone dies.
    Windows: ``taskkill /T`` walks the tree. POSIX: kill the whole process group
    (the worker is started in its own session/group, see ``_run_capture``).
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # best-effort reap; fall through to the direct kill below
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def _run_capture(cmd, *, input=None, cwd=None, timeout=None) -> subprocess.CompletedProcess:
    """``subprocess.run`` replacement that kills the WHOLE child tree on timeout
    or interrupt — not just the direct child.

    ``subprocess.run``'s own timeout kills only the immediate process, leaking the
    node/agent grandchildren that dominate cost. Here we own the ``Popen`` handle,
    start it in its own process group/session, and on ANY abort (TimeoutExpired,
    KeyboardInterrupt, …) reap the entire tree before re-raising — so a timed-out
    or Ctrl-C'd worker can't strand live drones. The exception still propagates,
    so callers (e.g. fanout's ``except subprocess.TimeoutExpired``) are unchanged.
    """
    kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, text=True, encoding="utf-8",
                  errors="replace", cwd=cwd)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except BaseException:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=10)  # drain pipes / reap after the tree dies
        except Exception:
            pass
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _call_copilot(model, prompt, cwd, timeout, exe=None, allow_flag="--allow-all",
                  available_tools=None) -> WorkerResult:
    """Call the copilot CLI. Prompt sent via stdin (never -p) to avoid cp932 truncation.

    ``available_tools``: when not None, restrict the model to exactly this tool
    list via ``--available-tools=<csv>``. An empty list disables ALL tools,
    forcing a single-shot completion (no file/shell access). The JUDGE uses this
    to rule on the supplied bundle instead of turning into an agentic explorer —
    which both defeats the retrieval redesign and blows the timeout.
    """
    if exe is None:
        exe = shutil.which("copilot.cmd") or shutil.which("copilot") or "copilot"
    cmd = [exe, allow_flag, "--model", model]
    if available_tools is not None:
        cmd.append("--available-tools=" + ",".join(available_tools))
    logger.debug("call_worker copilot: model=%s cwd=%s timeout=%d", model, cwd, timeout)
    t0 = time.monotonic()
    result = _run_capture(cmd, input=prompt, cwd=cwd, timeout=timeout)
    latency_s = time.monotonic() - t0
    if result.returncode != 0:
        logger.warning("copilot rc=%d (%.1fs) stderr: %s",
                       result.returncode, latency_s, result.stderr[:500])
    _tee_call(model, latency_s, result.returncode, result.stderr)
    return WorkerResult(stdout=result.stdout, stderr=result.stderr,
                        exit_code=result.returncode, latency_s=latency_s)


def _call_deepinfra(model, prompt, cwd=None, timeout=120, *, system=None,
                    temperature=0.2, max_tokens=1024, reasoning_effort=None,
                    api_key_env="DEEPINFRA_TOKEN",
                    base_url="https://api.deepinfra.com/v1/openai",
                    available_tools=None, **_ignored) -> WorkerResult:
    """Call a DeepInfra OpenAI-compatible chat model (e.g. ``openai/gpt-oss-120b``).

    HTTP via the openai SDK, not subprocess. Unlike copilot this returns EXACT
    token counts (``response.usage.total_tokens``), surfaced on
    ``WorkerResult.real_tokens`` for the ledger's real_tokens column.

    Tool-OFF single-shot path: the model has no access to the local filesystem,
    so ``cwd`` and ``available_tools`` are accepted only for ``call_worker``
    signature parity and ignored (a non-empty ``available_tools`` is meaningless
    here and logged at debug). Intended for the judge / review roles, where a
    ~20s round-trip is acceptable.

    Config errors (missing key, openai not installed) and API failures return a
    WorkerResult with exit_code=1 and the message on stderr — matching the
    copilot handler's contract so callers' existing error handling applies.
    """
    if available_tools:
        logger.debug("deepinfra: ignoring available_tools=%s (no local tool access)",
                     available_tools)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        msg = f"{api_key_env} not set in environment"
        logger.warning("deepinfra: %s", msg)
        return WorkerResult(stdout="", stderr=msg, exit_code=1, latency_s=0.0)
    try:
        from openai import OpenAI
    except ImportError:
        msg = "openai package not installed (pip install openai)"
        logger.warning("deepinfra: %s", msg)
        return WorkerResult(stdout="", stderr=msg, exit_code=1, latency_s=0.0)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra: dict = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    logger.debug("call_worker deepinfra: model=%s timeout=%d", model, timeout)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **extra)
    except Exception as e:
        latency_s = time.monotonic() - t0
        logger.warning("deepinfra call failed (%.1fs): %s", latency_s, e)
        return WorkerResult(stdout="", stderr=str(e), exit_code=1, latency_s=latency_s)
    latency_s = time.monotonic() - t0

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    real_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    _tee_call(model, latency_s, 0, "")
    return WorkerResult(stdout=content, stderr="", exit_code=0,
                        latency_s=latency_s, real_tokens=real_tokens)


def _call_codex(model, prompt, cwd=None, timeout=300, *, sandbox="read-only",
                codex_exe=None, **_ignored) -> WorkerResult:
    """Call the Codex CLI in non-interactive ``codex exec`` mode (tool-ON agentic).

    The OpenAI-equivalent of the copilot worker: an agentic CLI that reads/explores
    the live tree and emits a final message. Wired for the tool-ON roles
    (queen / specify author), NOT the tool-OFF single-shot judge/review (those go to
    deepinfra). On a ChatGPT subscription login Codex is flat-rate, so it sidesteps
    copilot's per-internal-turn billing for exactly the stages that dominate cost.

    Mechanics:
      - the prompt is piped on stdin (``codex exec -`` reads instructions from stdin);
      - ``--cd`` sets the working root; ``--sandbox read-only`` lets the agent
        read/run but never write (specify is propose-only, the queen only explores),
        so there are no approval prompts and no accidental edits;
      - ``--output-last-message FILE`` captures the agent's FINAL message — we return
        THAT as ``stdout``, not the event-trace stream, so ``extract_first_json`` sees
        a clean comb without scraping tool-trace lines (cf. copilot's parse.py);
      - ``--ephemeral`` / ``--skip-git-repo-check`` keep automation hygienic.

    An empty ``model`` omits ``--model`` so Codex uses its configured default (lets a
    caller route to Codex without pinning a model string). ``available_tools`` (a
    copilot concept) has no clean Codex equivalent and is ignored — route tool-OFF
    work to deepinfra. Token usage is not parsed yet (flat-rate subscription), so
    ``real_tokens`` is None. A non-zero exit is surfaced on the WorkerResult,
    matching the other handlers' contract; a timeout propagates like copilot's.
    """
    exe = codex_exe or shutil.which("codex.cmd") or shutil.which("codex") or "codex"
    fd, out_path = tempfile.mkstemp(suffix=".codexmsg.txt")
    os.close(fd)
    cmd = [exe, "exec", "-", "--cd", cwd or ".", "--sandbox", sandbox,
           "--skip-git-repo-check", "--color", "never", "--ephemeral",
           "--output-last-message", out_path]
    if model:
        cmd[2:2] = ["--model", model]  # insert after "exec" (before the "-" stdin marker)
    logger.debug("call_worker codex: model=%s cwd=%s sandbox=%s timeout=%d",
                 model, cwd, sandbox, timeout)
    t0 = time.monotonic()
    try:
        result = _run_capture(cmd, input=prompt, cwd=cwd, timeout=timeout)
    except BaseException:  # timeout / interrupt: clean up the temp file, then propagate
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
    latency_s = time.monotonic() - t0
    # Read the final agent message (the comb), then clean up the temp file.
    final = ""
    try:
        with open(out_path, "r", encoding="utf-8", errors="replace") as f:
            final = f.read()
    except OSError:
        final = ""
    try:
        os.remove(out_path)
    except OSError:
        pass
    if not final.strip():
        final = result.stdout  # codex wrote no final message → fall back to stdout
    if result.returncode != 0:
        logger.warning("codex rc=%d (%.1fs) stderr: %s",
                       result.returncode, latency_s, result.stderr[:500])
    _tee_call(model, latency_s, result.returncode, result.stderr)
    return WorkerResult(stdout=final, stderr=result.stderr,
                        exit_code=result.returncode, latency_s=latency_s)


# Extension point: add new providers here.
# Handler signature: (model, prompt, cwd, timeout, **kwargs) -> WorkerResult
_REGISTRY: dict = {
    "copilot": _call_copilot,
    "deepinfra": _call_deepinfra,
    "codex": _call_codex,
}


def call_worker(provider, model, prompt, cwd=None, timeout=300, **provider_kwargs) -> WorkerResult:
    """Dispatch a worker call to the named provider.

    Raises NotImplementedError for unknown providers.
    To add a provider: register a handler in _REGISTRY above.
    """
    handler = _REGISTRY.get(provider)
    if handler is None:
        raise NotImplementedError(
            f"Provider '{provider}' is not implemented. "
            f"Available: {list(_REGISTRY)}. "
            "To add one, register a handler in hive/providers.py _REGISTRY."
        )
    return handler(model=model, prompt=prompt, cwd=cwd, timeout=timeout, **provider_kwargs)
