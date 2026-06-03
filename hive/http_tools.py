"""Client-side local tools + agent loop for the OpenAI-compatible HTTP provider.

The OpenAI chat API has function-calling *protocol* (the model can request
``tool_calls``), but it cannot read the user's disk — executing a tool that
touches local files is inherently client-side work. ``copilot``/``codex`` ship
that half (a loop + local tools), which is why the agentic tool-ON roles route to
them. This module supplies the SAME missing half for the HTTP provider so an
HTTP-only operator (no copilot/codex installed) can still run the tool-ON roles
(queen / specify / …) over a plain OpenAI-compatible endpoint.

Three read-only tools, all sandboxed to a single ``root`` directory (the
codebase root passed as ``cwd``):

  - ``read_file``  — read a text file (optionally a line range), with 1-based
                     line numbers, so the model can cite ``file:line``.
  - ``list_dir``   — list a directory's entries (dirs marked with a trailing /).
  - ``grep``       — regex search over the tree (skips VCS/build/binary noise).

Everything is read-only and path-confined: a tool argument that resolves outside
``root`` (via ``..`` or an absolute path) is refused, never executed. Every tool
output is capped (bytes / lines / matches) so a single call can't balloon the
per-token bill on the next round-trip — the central cost concern that pushed
tool-ON to flat-rate CLIs in the first place. The loop itself is bounded by
``max_iterations`` and accumulates ``usage.total_tokens`` across every round-trip
for the ledger.
"""
import json
import logging
import os
import re

logger = logging.getLogger("hive.http_tools")

# ── Output budgets (cap a single tool result so the next prompt stays bounded) ──
_MAX_READ_LINES = 800          # lines returned by one read_file call
_MAX_READ_BYTES = 100_000      # hard byte ceiling for a read_file result
_MAX_LIST_ENTRIES = 500        # entries returned by one list_dir call
_MAX_GREP_MATCHES = 100        # matches returned by one grep call
_MAX_FILE_SCAN_BYTES = 1_000_000  # skip files larger than this during grep
_GREP_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".apply_backups",
                   "hive_workdir", ".venv", ".mypy_cache", ".pytest_cache"}

# Default ceiling on tool-call rounds. A bound is what keeps a runaway model from
# spending unbounded tokens; the common path terminates in a handful of rounds.
DEFAULT_MAX_ITERATIONS = 25


def _resolve(root: str, path: str) -> str:
    """Resolve ``path`` against ``root`` and refuse anything that escapes it.

    Both sides are realpath'd so symlinks and ``..`` can't tunnel out of the
    sandbox. Raises ValueError on escape; the caller turns that into a tool-error
    string the model can recover from (it never crashes the loop).
    """
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, path or "."))
    if full != root_real and not full.startswith(root_real + os.sep):
        raise ValueError(f"path {path!r} escapes the sandbox root")
    return full


def _is_probably_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def read_file(root: str, *, path: str, start_line: int | None = None,
              end_line: int | None = None) -> str:
    """Read a text file under ``root``, returning 1-based numbered lines.

    Optional ``start_line``/``end_line`` (1-based, inclusive) scope a slice; the
    result is still capped at ``_MAX_READ_LINES`` / ``_MAX_READ_BYTES`` so a huge
    file can't flood the context. A trailing note flags truncation."""
    full = _resolve(root, path)
    if not os.path.isfile(full):
        return f"[error] not a file: {path}"
    with open(full, "rb") as f:
        head = f.read(4096)
    if _is_probably_binary(head):
        return f"[error] {path} looks binary; refusing to read"
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    lo = (start_line - 1) if start_line and start_line > 0 else 0
    hi = end_line if end_line and end_line > 0 else total
    lo = max(0, min(lo, total))
    hi = max(lo, min(hi, total))
    sliced = lines[lo:hi]
    truncated = False
    if len(sliced) > _MAX_READ_LINES:
        sliced = sliced[:_MAX_READ_LINES]
        hi = lo + _MAX_READ_LINES
        truncated = True
    out_lines = []
    nbytes = 0
    for i, ln in enumerate(sliced, start=lo + 1):
        rendered = f"{i}\t{ln.rstrip(chr(10))}"
        nbytes += len(rendered)
        if nbytes > _MAX_READ_BYTES:
            truncated = True
            break
        out_lines.append(rendered)
    body = "\n".join(out_lines)
    if truncated:
        body += f"\n[... truncated; file has {total} lines total]"
    return body or "[empty file]"


def list_dir(root: str, *, path: str = ".") -> str:
    """List entries of a directory under ``root`` (dirs get a trailing slash)."""
    full = _resolve(root, path)
    if not os.path.isdir(full):
        return f"[error] not a directory: {path}"
    try:
        names = sorted(os.listdir(full))
    except OSError as e:
        return f"[error] cannot list {path}: {e}"
    entries = []
    for name in names[:_MAX_LIST_ENTRIES]:
        suffix = "/" if os.path.isdir(os.path.join(full, name)) else ""
        entries.append(name + suffix)
    body = "\n".join(entries) if entries else "[empty directory]"
    if len(names) > _MAX_LIST_ENTRIES:
        body += f"\n[... {len(names) - _MAX_LIST_ENTRIES} more entries omitted]"
    return body


def grep(root: str, *, pattern: str, path: str = ".", glob: str | None = None,
         ignore_case: bool = False) -> str:
    """Regex-search text files under ``root``/``path`` for ``pattern``.

    Returns ``relpath:lineno: line`` matches (capped at ``_MAX_GREP_MATCHES``).
    Skips VCS/build dirs, oversized files, and binaries. ``glob`` (e.g. ``*.py``)
    filters by filename. A bad regex returns an error string, never raises."""
    base = _resolve(root, path)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as e:
        return f"[error] bad regex {pattern!r}: {e}"
    if os.path.isfile(base):
        roots = [(os.path.dirname(base), [os.path.basename(base)])]
    else:
        roots = None
    matches: list[str] = []
    root_real = os.path.realpath(root)
    capped = False

    def _scan(filepath: str) -> bool:
        """Append matches from one file. Returns False once the global cap is hit."""
        rel = os.path.relpath(filepath, root_real).replace(os.sep, "/")
        if glob and not _fnmatch(os.path.basename(filepath), glob):
            return True
        try:
            if os.path.getsize(filepath) > _MAX_FILE_SCAN_BYTES:
                return True
            with open(filepath, "rb") as f:
                if _is_probably_binary(f.read(4096)):
                    return True
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    if rx.search(line):
                        matches.append(f"{rel}:{n}: {line.rstrip()[:300]}")
                        if len(matches) >= _MAX_GREP_MATCHES:
                            return False
        except OSError:
            return True
        return True

    if roots is not None:  # single-file grep
        for _d, names in roots:
            for name in names:
                _scan(os.path.join(_d, name))
    else:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _GREP_SKIP_DIRS]
            for name in filenames:
                if not _scan(os.path.join(dirpath, name)):
                    capped = True
                    break
            if capped:
                break
    if not matches:
        return f"[no matches for {pattern!r}]"
    body = "\n".join(matches)
    if capped:
        body += f"\n[... capped at {_MAX_GREP_MATCHES} matches]"
    return body


def _fnmatch(name: str, glob: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(name, glob)


# ── Tool registry: name -> (callable, OpenAI schema) ──────────────────────────
_TOOLS = {
    "read_file": (
        read_file,
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a text file from the codebase and return its "
                           "content with 1-based line numbers. Use start_line/"
                           "end_line to read just a slice of a large file.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string",
                         "description": "File path relative to the codebase root."},
                "start_line": {"type": "integer",
                               "description": "1-based first line (optional)."},
                "end_line": {"type": "integer",
                             "description": "1-based last line, inclusive (optional)."},
            }, "required": ["path"]}}}),
    "list_dir": (
        list_dir,
        {"type": "function", "function": {
            "name": "list_dir",
            "description": "List the entries of a directory in the codebase. "
                           "Directories are shown with a trailing slash.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string",
                         "description": "Directory path relative to the codebase "
                                        "root (default: the root itself)."},
            }, "required": []}}}),
    "grep": (
        grep,
        {"type": "function", "function": {
            "name": "grep",
            "description": "Search the codebase for a regular expression and "
                           "return matching lines as 'path:line: text'.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string",
                            "description": "Python regular expression to search for."},
                "path": {"type": "string",
                         "description": "Subtree or file to search (default: root)."},
                "glob": {"type": "string",
                         "description": "Filename glob filter, e.g. '*.py' (optional)."},
                "ignore_case": {"type": "boolean",
                                "description": "Case-insensitive match (optional)."},
            }, "required": ["pattern"]}}}),
}

ALL_TOOL_NAMES = tuple(_TOOLS)


def select_tools(available_tools, *, have_cwd: bool) -> list[str]:
    """Map the ``available_tools`` convention to a concrete tool-name list.

    Mirrors the copilot handler's semantics exactly:
      - ``[]`` (explicit empty)  -> no tools / single-shot;
      - ``None``                 -> all local tools (the agentic default);
      - a list of names          -> just those (unknown names dropped).
    Without a ``cwd`` there is nothing to read, so the set is empty regardless.
    """
    if not have_cwd:
        return []
    if available_tools is None:
        return list(ALL_TOOL_NAMES)
    if available_tools == []:
        return []
    return [t for t in available_tools if t in _TOOLS]


def schemas_for(names) -> list[dict]:
    return [_TOOLS[n][1] for n in names if n in _TOOLS]


def execute_tool(name: str, arguments: dict, root: str) -> str:
    """Run one tool by name with kwargs, confined to ``root``. Never raises:
    any failure (bad path, bad args, unknown tool) becomes an ``[error] …``
    string so the model can see it and adjust on the next round."""
    entry = _TOOLS.get(name)
    if entry is None:
        return f"[error] unknown tool: {name}"
    fn = entry[0]
    if not isinstance(arguments, dict):
        return f"[error] tool {name} expects an object of arguments"
    try:
        return fn(root, **arguments)
    except TypeError as e:
        return f"[error] bad arguments for {name}: {e}"
    except ValueError as e:  # sandbox escape, etc.
        return f"[error] {e}"
    except Exception as e:  # defensive: a tool bug must not kill the loop
        logger.warning("tool %s raised: %s", name, e)
        return f"[error] {name} failed: {e}"


def run_agent_loop(client, model, messages, *, root, tool_names,
                   temperature, max_tokens, extra, max_iterations=DEFAULT_MAX_ITERATIONS):
    """Drive the tool-calling loop against an OpenAI-compatible ``client``.

    Repeatedly calls ``client.chat.completions.create`` with the tool schemas;
    when the model returns ``tool_calls`` they are executed locally (under
    ``root``) and fed back, until the model returns a plain answer or
    ``max_iterations`` is reached. Returns ``(content, total_tokens)`` where
    ``total_tokens`` is summed across every round-trip (None if the endpoint
    reports no usage at all).

    ``messages`` is mutated in place (the running transcript). On the final
    iteration tools are withheld (``tool_choice='none'``) so the model is forced
    to answer instead of requesting yet another call it has no budget to run.
    """
    tools = schemas_for(tool_names)
    total_tokens = 0
    saw_usage = False
    content = ""
    for i in range(max_iterations):
        last = i == max_iterations - 1
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, **extra)
        if tools and not last:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        tot = getattr(usage, "total_tokens", None) if usage is not None else None
        if tot is not None:
            total_tokens += tot
            saw_usage = True
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        content = msg.content or ""
        if not tool_calls:
            break
        # Echo the assistant's tool-call turn, then answer each call.
        messages.append({
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            } for tc in tool_calls],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                result = f"[error] could not parse arguments: {e}"
            else:
                result = execute_tool(tc.function.name, args, root)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})
        logger.debug("agent loop round %d: %d tool call(s)", i + 1, len(tool_calls))
    return content, (total_tokens if saw_usage else None)
