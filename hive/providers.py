"""Provider adapter — thin dispatch for worker calls.

Dispatches worker calls to the appropriate provider backend. Currently only
the 'copilot' provider is implemented. Other providers can be added by
extending the _REGISTRY dict.

Usage:
    result = call_worker("copilot", "gpt-5-mini", prompt, cwd=root, timeout=300)
    print(result.stdout, result.latency_s)
"""
import os, shutil, signal, subprocess, tempfile, time, logging, contextlib
from dataclasses import dataclass

from hive import http_tools

# OS-level byte-range lock primitive — used to serialize codex across PROCESSES
# (see _codex_serial_lock). msvcrt on Windows, fcntl on POSIX; either may be
# absent on an exotic platform, in which case the lock degrades to a no-op.
try:
    import msvcrt
except ImportError:  # not Windows
    msvcrt = None
try:
    import fcntl
except ImportError:  # not POSIX
    fcntl = None

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
                  available_tools=None, **_ignored) -> WorkerResult:
    """Call the copilot CLI. Prompt sent via stdin (never -p) to avoid cp932 truncation.

    ``available_tools``: when not None, restrict the model to exactly this tool
    list via ``--available-tools=<csv>``. An empty list disables ALL tools,
    forcing a single-shot completion (no file/shell access). The JUDGE uses this
    to rule on the supplied bundle instead of turning into an agentic explorer —
    which both defeats the retrieval redesign and blows the timeout.

    ``**_ignored`` absorbs the OpenAI-compatible endpoint kwargs (``base_url`` /
    ``api_key_env``) that callers put in the single shared ``provider_kwargs`` dict
    for the HTTP provider — they are meaningless to the copilot CLI and dropped.
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


# A reasoning model (gpt-oss-120b) spends COMPLETION tokens on its hidden
# reasoning before emitting the answer, so a tight cap truncates the JSON verdict
# mid-stream → unbalanced object → unparseable, and the axis is dropped to
# located=False even after the JSON-only retry (Defect 4b / T892 axis D: a large
# bundle drove long reasoning that overran the old 1024 cap, twice). 4096 leaves
# ample room for reasoning + the small JSON object; billed only for tokens the
# model actually generates, so it is free on the common short path.
_DEEPINFRA_MAX_TOKENS = 4096


def _call_openai_compatible(model, prompt, cwd=None, timeout=120, *, system=None,
                            temperature=0.2, max_tokens=_DEEPINFRA_MAX_TOKENS,
                            reasoning_effort=None,
                            api_key_env="DEEPINFRA_TOKEN",
                            base_url="https://api.deepinfra.com/v1/openai",
                            available_tools=None, **_ignored) -> WorkerResult:
    """Call any OpenAI-compatible chat endpoint (DeepInfra, OpenAI, vLLM, …).

    This is the generic HTTP handler behind both the ``openai`` and ``deepinfra``
    provider names — they are the SAME backend; "deepinfra" is simply the preset
    whose ``base_url`` / ``api_key_env`` defaults below point at DeepInfra. Any
    other OpenAI-compatible vendor is reached by passing a different ``base_url``
    and ``api_key_env`` (wired from the config ``openai`` block), so the engine is
    not bound to one provider.

    HTTP via the openai SDK, not subprocess. Unlike copilot this returns EXACT
    token counts (``response.usage.total_tokens``), surfaced on
    ``WorkerResult.real_tokens`` for the ledger's real_tokens column.

    Two paths, selected by ``available_tools`` (same convention as copilot):

      * Single-shot (tool-OFF): ``available_tools=[]`` — one chat completion, no
        tools, no loop (~20s round-trip). The judge / converge / review roles use
        this; it can't read local files and bills only the single exchange.
      * Agentic (tool-ON): ``available_tools=None`` (all local tools) or a named
        subset, AND a ``cwd`` to root them — runs ``hive.http_tools.run_agent_loop``,
        which gives the model read_file / list_dir / grep over the codebase via the
        OpenAI function-calling protocol and executes those calls client-side. This
        is the missing client half that lets an HTTP-only operator (no copilot/
        codex) run the tool-ON roles (queen / specify / …) over plain HTTP. Cost
        note: a long agent trace bills per token across every round-trip — the
        reason tool-ON defaults to flat-rate CLIs; output caps + an iteration bound
        keep it bounded.

    Without a ``cwd`` there is nothing to read, so any tool request degrades to the
    single-shot path. ``real_tokens`` is the sum of ``usage.total_tokens`` over all
    round-trips, for the ledger.

    Config errors (missing key, openai not installed) and API failures return a
    WorkerResult with exit_code=1 and the message on stderr — matching the
    copilot handler's contract so callers' existing error handling applies.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        msg = f"{api_key_env} not set in environment"
        logger.warning("openai: %s", msg)
        return WorkerResult(stdout="", stderr=msg, exit_code=1, latency_s=0.0)
    try:
        from openai import OpenAI
    except ImportError:
        msg = "openai package not installed (pip install openai)"
        logger.warning("openai: %s", msg)
        return WorkerResult(stdout="", stderr=msg, exit_code=1, latency_s=0.0)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra: dict = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    tool_names = http_tools.select_tools(available_tools, have_cwd=bool(cwd))
    logger.debug("call_worker openai: model=%s base_url=%s timeout=%d tools=%s",
                 model, base_url, timeout, tool_names or "none")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    t0 = time.monotonic()
    try:
        if tool_names:
            content, real_tokens = http_tools.run_agent_loop(
                client, model, messages, root=cwd, tool_names=tool_names,
                temperature=temperature, max_tokens=max_tokens, extra=extra)
        else:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, **extra)
            content = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            real_tokens = (getattr(usage, "total_tokens", None)
                           if usage is not None else None)
    except Exception as e:
        latency_s = time.monotonic() - t0
        logger.warning("openai call failed (%.1fs): %s", latency_s, e)
        return WorkerResult(stdout="", stderr=str(e), exit_code=1, latency_s=latency_s)
    latency_s = time.monotonic() - t0

    _tee_call(model, latency_s, 0, "")
    return WorkerResult(stdout=content, stderr="", exit_code=0,
                        latency_s=latency_s, real_tokens=real_tokens)


# Codex (ChatGPT-subscription login) tolerates only ONE concurrent request: its
# backend caps subscription concurrency and every live `codex exec` shares the same
# ~/.codex auth/session, so firing N codex workers at once lets one win and starves
# the rest (observed in a 3-item parallel batch: N182 completed, N181 timed out at
# 300s, N183 returned empty). copilot fan-out has no such limit, which is why the
# swarm parallelizes fine — so the guard is codex-ONLY, leaving copilot untouched.
#
# The batch runner launches each hive item as its OWN process (pl_batch_runner spawns
# `python hive_runner.py` per task inside a thread pool), so an in-process semaphore
# can't coordinate them. We need an OS-level lock: a byte-range lock on a shared temp
# file, which the kernel auto-releases if the holder dies — a crashed/killed worker
# never strands the queue. Set HIVE_CODEX_NO_LOCK=1 to disable (e.g. tests).
_CODEX_LOCK_PATH = os.path.join(tempfile.gettempdir(), "hivework_codex.lock")
_CODEX_LOCK_ACQUIRE_TIMEOUT = float(os.environ.get("HIVE_CODEX_LOCK_TIMEOUT", "1800"))


@contextlib.contextmanager
def _codex_serial_lock(acquire_timeout=None, poll=1.0):
    """Serialize codex calls ACROSS processes via an OS byte-range lock.

    Held only around the codex subprocess (not the whole hive run), so queued
    workers wait their turn rather than racing. Acquisition polls a non-blocking
    lock so we can bound the wait (``acquire_timeout``); on timeout we raise
    ``TimeoutError`` rather than hang a batch overnight. If neither msvcrt nor
    fcntl is available, or HIVE_CODEX_NO_LOCK is set, this is a no-op.
    """
    if os.environ.get("HIVE_CODEX_NO_LOCK") or (msvcrt is None and fcntl is None):
        yield
        return
    if acquire_timeout is None:
        acquire_timeout = _CODEX_LOCK_ACQUIRE_TIMEOUT
    f = open(_CODEX_LOCK_PATH, "a+")
    try:
        deadline = time.monotonic() + acquire_timeout
        waited = False
        while True:
            try:
                if msvcrt is not None:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # acquired
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"codex serialization lock not acquired within "
                        f"{acquire_timeout:.0f}s ({_CODEX_LOCK_PATH}); another "
                        f"codex worker is holding it longer than expected")
                if not waited:
                    logger.info("codex busy — waiting for the serialization lock "
                                "(another codex worker is running)")
                    waited = True
                time.sleep(poll)
        try:
            yield
        finally:
            try:
                if msvcrt is not None:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        f.close()


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
    latency_s = 0.0
    try:
        # Cross-process serialization: only one codex subprocess runs at a time.
        # t0 is set INSIDE the lock so latency reflects codex time, not queue wait.
        with _codex_serial_lock():
            t0 = time.monotonic()
            result = _run_capture(cmd, input=prompt, cwd=cwd, timeout=timeout)
            latency_s = time.monotonic() - t0
    except BaseException:  # timeout / interrupt: clean up the temp file, then propagate
        try:
            os.remove(out_path)
        except OSError:
            pass
        raise
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
#
# ``openai`` and ``deepinfra`` map to the SAME OpenAI-compatible handler — the
# generic provider name plus a DeepInfra-preset alias kept for back-compat with
# existing configs/tests. Point either at another vendor via the config ``openai``
# block (base_url / api_key_env).
_REGISTRY: dict = {
    "copilot": _call_copilot,
    "openai": _call_openai_compatible,
    "deepinfra": _call_openai_compatible,
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
