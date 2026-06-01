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


# Extension point: add new providers here.
# Handler signature: (model, prompt, cwd, timeout, **kwargs) -> WorkerResult
_REGISTRY: dict = {
    "copilot": _call_copilot,
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
