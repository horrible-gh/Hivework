"""Provider adapter — thin dispatch for worker calls.

Dispatches worker calls to the appropriate provider backend. Currently only
the 'copilot' provider is implemented. Other providers can be added by
extending the _REGISTRY dict.

Usage:
    result = call_worker("copilot", "gpt-5-mini", prompt, cwd=root, timeout=300)
    print(result.stdout, result.latency_s)
"""
import os, shutil, subprocess, time, logging
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
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", cwd=cwd, timeout=timeout)
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


# Extension point: add new providers here.
# Handler signature: (model, prompt, cwd, timeout, **kwargs) -> WorkerResult
_REGISTRY: dict = {
    "copilot": _call_copilot,
    "deepinfra": _call_deepinfra,
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
