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

from hive.parse import extract_first_json, is_comb_dict

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

# ── History pruning (R0001 0077 requirement 2) ────────────────────────────────
# The loop resends the WHOLE transcript every round and DeepInfra has no cached-
# input discount, so a long drone's bill grows ~quadratically with rounds — old
# tool outputs (read/grep dumps it already mined) are re-billed verbatim on every
# later round. Pruning replaces tool outputs older than the last
# ``prune_keep_rounds`` tool-call rounds with a short head + a stub note, keeping
# growth near-linear. The model keeps its OWN turns (its reasoning and citations
# survive) and can always re-run a tool if it truly needs pruned content back —
# that costs one cheap call, only when needed, instead of billing every round.

# Lines of a pruned tool result kept verbatim (enough to recognise WHAT the call
# returned — the file path / first matches — without carrying the full dump).
_PRUNE_HEAD_LINES = 5


def _prune_stub(text: str) -> str:
    """Shrink one old tool result to a head + stub note. Idempotent by caller
    bookkeeping (the loop prunes each message object at most once)."""
    lines = text.splitlines()
    if len(lines) <= _PRUNE_HEAD_LINES + 2:  # already tiny — not worth a stub
        return text
    head = "\n".join(lines[:_PRUNE_HEAD_LINES])
    return (f"{head}\n[... pruned {len(lines) - _PRUNE_HEAD_LINES} older lines to "
            f"save context — call the tool again if you need this content]")


def _prune_old_tool_results(messages: list, keep_rounds: int,
                            pruned_ids: set) -> None:
    """Stub tool outputs older than the last ``keep_rounds`` tool-call rounds.

    A "round" is an assistant turn carrying ``tool_calls``; everything before the
    keep-window's first such turn is old. Only RESULT payloads are shrunk — the
    ``role=='tool'`` messages and the RC-C bridge's user-turn results (identified
    by their fixed prefix). The message SKELETON is untouched (roles, tool_call_id
    pairing, the assistant's own turns, the system/seed prompt), so the OpenAI
    protocol stays valid and the model keeps its reasoning trail. ``pruned_ids``
    (object ids, loop-local) makes repeated passes no-ops.
    """
    round_starts = [idx for idx, m in enumerate(messages)
                    if m.get("role") == "assistant" and m.get("tool_calls")]
    if len(round_starts) <= keep_rounds:
        return
    cutoff = round_starts[-keep_rounds]
    for m in messages[:cutoff]:
        if id(m) in pruned_ids:
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        is_tool_result = m.get("role") == "tool"
        is_bridge_result = (m.get("role") == "user"
                            and content.startswith("[You printed a search"))
        if is_tool_result or is_bridge_result:
            m["content"] = _prune_stub(content)
            pruned_ids.add(id(m))


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


def message_text(msg):
    """Extract the assistant's answer, falling back to the reasoning channel.

    A reasoning model (gpt-oss-120b on deepinfra) driven through the tool loop's
    forced-answer turn (tool_choice='none') routinely emits the answer on the
    ``reasoning`` / ``reasoning_content`` channel and leaves ``content`` empty
    while still reporting finish_reason=stop with tokens spent (NR
    hivework.default.0004.0003 §2: run 418 swarm returned 9/10 empty combs this
    way). Reading only ``content`` discards that answer → empty comb → parse
    failure, recorded downstream as a 0-char "successful" call. Fall back to the
    reasoning channel so the work is not silently dropped.
    """
    txt = getattr(msg, "content", None)
    if txt and txt.strip():
        return txt
    for attr in ("reasoning_content", "reasoning"):
        alt = getattr(msg, attr, None)
        if alt and str(alt).strip():
            return str(alt)
    return txt or ""


def _tool_arg_param_names(tool_names) -> set[str]:
    """Union of every parameter name across the given tools' schemas.

    These are the keys a *legitimate* tool call would carry (path, pattern,
    glob, ignore_case, start_line, end_line, …). Used to recognise when the
    model printed a tool-call's arguments as plain text instead of issuing a
    structured ``tool_calls`` request."""
    names: set[str] = set()
    for schema in schemas_for(tool_names):
        props = (schema.get("function", {})
                 .get("parameters", {}).get("properties", {}))
        names.update(props)
    return names


# File-navigation keys a search/read memo carries. The real schema params
# (path/pattern/…) PLUS the synonyms a reasoning drone routinely HALLUCINATES
# for a search API it imagines (query/max_results/depth/line_start/…). The
# anchor of the set — recognising "this object is navigating files" — is the
# path-like group; the rest just widens recall.
_PATH_KEYS = frozenset({"path", "file", "filename", "filepath", "dir",
                        "directory", "paths", "files"})
_NAV_SYNONYM_KEYS = frozenset({
    "query", "q", "search", "regex", "max_results", "limit", "depth",
    "line_start", "line_end", "context", "case_insensitive", "recursive",
    "include", "exclude", "head", "tail", "lines",
})


def _is_scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


# How many times one agent loop will INTERPRET a printed tool-arg memo as a real
# tool call and feed the result back (RC-C bridge, below). A small cap: it converts
# a "blind drone" into a few real searches without letting a model that only ever
# prints memos spin to the iteration bound. After the cap the loop falls through to
# the forced-answer guard, which demands a conclusion.
_MAX_TEXT_BRIDGES = 3

# Synonym map: a hallucinated/aliased memo key -> the real grep/read_file param it
# means. Lets the RC-C bridge execute a printed search even when the drone invented
# its own field names (query/regex for pattern, line_start for start_line, …).
_PATH_SYNONYMS = ("path", "file", "filename", "filepath", "dir", "directory")
_PATTERN_SYNONYMS = ("pattern", "regex", "query", "q", "search")


def _infer_tool_call_from_text(content: str, tool_names) -> tuple[str, dict] | None:
    """Map a printed tool-arg memo to a concrete ``(tool_name, kwargs)`` to run.

    The RC-C counterpart to :func:`_looks_like_tool_call_text` (NR
    hivework.default.0008.0009 RC-A1/RC-C): when a reasoning drone PRINTS its next
    search as text instead of issuing a structured ``tool_calls`` request, this
    decides which local tool that text was asking for so the loop can execute it and
    feed the evidence back — turning a blind, zero-tool drone into a real search.

    Inference (first match wins), normalising hallucinated key names:
      - a search ``pattern`` (or query/regex/…) + ``grep`` available  → ``grep``;
      - a ``path`` plus line bounds + ``read_file`` available          → ``read_file``;
      - a bare file-looking ``path`` (has an extension)                → ``read_file``;
      - a bare dir-looking ``path``                                    → ``list_dir``.
    Returns ``None`` when nothing sensible maps or the needed tool isn't offered —
    the caller then falls through to break/forced-answer. Never raises."""
    try:
        obj = extract_first_json(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    def _first_str(keys):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None

    path = _first_str(_PATH_SYNONYMS)
    pattern = _first_str(_PATTERN_SYNONYMS)
    start = obj.get("start_line") if isinstance(obj.get("start_line"), int) \
        else (obj.get("line_start") if isinstance(obj.get("line_start"), int) else None)
    end = obj.get("end_line") if isinstance(obj.get("end_line"), int) \
        else (obj.get("line_end") if isinstance(obj.get("line_end"), int) else None)
    glob = obj.get("glob") if isinstance(obj.get("glob"), str) \
        else (obj.get("include") if isinstance(obj.get("include"), str) else None)
    ic = obj.get("ignore_case")
    if not isinstance(ic, bool):
        ic = obj.get("case_insensitive") if isinstance(obj.get("case_insensitive"), bool) else None

    if pattern and "grep" in tool_names:
        kw: dict = {"pattern": pattern}
        if path:
            kw["path"] = path
        if glob:
            kw["glob"] = glob
        if ic is not None:
            kw["ignore_case"] = ic
        return "grep", kw
    if path and (start is not None or end is not None) and "read_file" in tool_names:
        kw = {"path": path}
        if start is not None:
            kw["start_line"] = start
        if end is not None:
            kw["end_line"] = end
        return "read_file", kw
    if path:
        has_ext = bool(os.path.splitext(path)[1])
        if has_ext and "read_file" in tool_names:
            return "read_file", {"path": path}
        if "list_dir" in tool_names:
            return "list_dir", {"path": path}
        if "read_file" in tool_names:
            return "read_file", {"path": path}
    return None


def _looks_like_tool_call_text(content: str, tool_param_names: set[str]) -> bool:
    """True iff ``content`` is a bare tool-call argument object emitted as text.

    The pathology (NR hivework.default.0005.0003 RC-1): a reasoning drone ends
    the loop by *printing* its next search step — e.g.
    ``{"path":"x","pattern":"y","glob":"*.py"}`` — as its answer instead of
    issuing a real ``tool_calls`` request or concluding. That object decodes as
    valid JSON and is non-empty, so the empty-content guard waves it through and
    it is accepted as a (0-finding) answer.

    RC-2 (NR hivework.default.0007.0005): the original guard required the keys to
    be a subset of the *exact* schema params, so it missed the dominant case —
    the drone HALLUCINATES param names for a search API it imagines
    (``{"path":..,"query":..,"max_results":..}``, ``{"path":..,"depth":..}``).
    Empirically that exact-match check caught only 1/5 of a live run's noise.
    The fix discriminates by what a memo is NOT rather than by exact key spelling:
    a CONCLUSION is comb-shaped (carries a ``findings`` list — :func:`is_comb_dict`,
    the SSOT); a search memo is a small FLAT object (all-scalar values) that is not
    comb-shaped and is navigating files (carries a path-like key, or — preserving
    the original behaviour — keys all within the schema params). Robust to new
    hallucinated key names; still conservative — a real comb (has ``findings``),
    prose (does not decode to a dict), and any structured/nested answer are left
    alone. Never raises."""
    # No tools were offered → the model had nothing to "call", so its output
    # cannot be a tool-call memo. Preserves the original single-shot contract.
    if not tool_param_names:
        return False
    try:
        obj = extract_first_json(content)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict) or not obj:
        return False
    # A conclusion is comb-shaped; never treat one as a memo.
    if is_comb_dict(obj):
        return False
    keys = set(obj.keys())
    # Original exact-schema-subset signal (kept for backward compatibility).
    if tool_param_names and keys <= tool_param_names:
        return True
    # Generalised signal: a small flat object navigating files. Flat = every
    # value is a scalar (a real answer object would nest). Navigating = carries a
    # path-like key, and every key is a recognised navigation term (real param,
    # path-like, or a known hallucinated synonym) — so a substantive object that
    # merely mentions a "path" field is not swept up.
    nav_vocab = _PATH_KEYS | _NAV_SYNONYM_KEYS | set(tool_param_names)
    if (keys & _PATH_KEYS) and keys <= nav_vocab \
            and all(_is_scalar(v) for v in obj.values()):
        return True
    return False


def run_agent_loop(client, model, messages, *, root, tool_names,
                   temperature, max_tokens, extra, max_iterations=DEFAULT_MAX_ITERATIONS,
                   prune_keep_rounds=None):
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

    ``prune_keep_rounds`` (R0001 0077 req 2): when a positive int, tool outputs
    older than the last N tool-call rounds are shrunk to a head + stub before
    every round-trip (see :func:`_prune_old_tool_results`), so the re-billed
    transcript stops growing quadratically. None/0 = no pruning — existing
    behaviour byte-for-byte.
    """
    tools = schemas_for(tool_names)
    tool_param_names = _tool_arg_param_names(tool_names)
    total_tokens = 0
    saw_usage = False
    content = ""
    bridges = 0
    pruned_ids: set = set()
    for i in range(max_iterations):
        last = i == max_iterations - 1
        if prune_keep_rounds:
            _prune_old_tool_results(messages, prune_keep_rounds, pruned_ids)
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
            # Per-round spend trace: the loop resends the whole transcript every
            # round, so per-round usage is the ground truth for measuring history
            # growth (R0001 0077). Debug-level — visible under -v only.
            logger.debug("agent loop round %d: usage %d tokens (cumulative %d)",
                         i + 1, tot, total_tokens)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        content = message_text(msg)
        if not tool_calls:
            # RC-C bridge (NR hivework.default.0008.0009 RC-A1/RC-C): a reasoning
            # drone (gpt-oss-120b) routinely PRINTS its next search as text —
            # {"path":..,"pattern":..} — instead of issuing a structured tool_calls
            # request. The loop used to break right here, so that drone executed ZERO
            # tools and went blind (the dominant low-token swarm failure). Instead:
            # if the plain answer is a tool-call memo and we still have iterations and
            # bridge budget, INTERPRET it, run the tool locally, feed the result back
            # as a user turn (a 'tool' role needs a preceding tool_calls turn, which we
            # don't have), and CONTINUE — so the printed search becomes real evidence
            # and the drone gets another turn to conclude. Bounded by _MAX_TEXT_BRIDGES
            # and max_iterations; once exhausted it falls through to the forced-answer
            # guard. Genuine answers (prose, real combs) don't match the memo predicate.
            if (not last and bridges < _MAX_TEXT_BRIDGES
                    and _looks_like_tool_call_text(content, tool_param_names)):
                inferred = _infer_tool_call_from_text(content, tool_names)
                if inferred is not None:
                    name, args = inferred
                    result = execute_tool(name, args, root)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[You printed a search instead of calling a tool, so it "
                            f"was executed for you — {name}({json.dumps(args)})]\n"
                            f"{result}\n\nNow CONCLUDE: output the single comb JSON "
                            f"with a `findings` array. Do not print another search."),
                    })
                    bridges += 1
                    logger.debug("agent loop: bridged a printed %s call (#%d)",
                                 name, bridges)
                    continue
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
    # Forced-answer guard. Two failure modes get ONE more forced-answer turn:
    #   (1) Empty output (NR hivework.default.0004.0003 §2/§5.2): the loop ends with
    #       BOTH content and reasoning empty (finish_reason=stop, tokens spent) → the
    #       caller gets a 0-char comb that parses to nothing yet looks like a clean
    #       success. (dominant run-418 mode.)
    #   (2) A tool-call emitted as plain text (NR hivework.default.0005.0003 RC-1): the
    #       model prints its next search's arguments — {"path":..,"pattern":..,"glob":..}
    #       — instead of calling the tool or concluding. That is non-empty valid JSON,
    #       so (1)'s empty trigger never fired and the noise was accepted as the answer
    #       (and downstream became fake honey evidence). (dominant run-424/449 mode.)
    # Both are the model failing to deliver a conclusion; one explicit forced-answer
    # turn is cheap insurance. Skipped for genuine answers (prose or any object that
    # carries a non-tool key, e.g. a real comb).
    is_empty = not content.strip()
    is_tool_arg_text = (not is_empty) and _looks_like_tool_call_text(
        content, tool_param_names)
    if is_empty or is_tool_arg_text:
        logger.warning("agent loop ended with %s; retrying once with a forced answer",
                       "empty output" if is_empty else "a tool-call emitted as plain text")
        messages.append({
            "role": "user",
            "content": "Output your final answer now as plain message content — "
                       "not a tool call, not a tool-argument JSON object (e.g. "
                       "{\"path\":..,\"pattern\":..}), and not hidden reasoning. "
                       "Conclude from the evidence you have already gathered. Do "
                       "not return an empty message.",
        })
        if prune_keep_rounds:
            _prune_old_tool_results(messages, prune_keep_rounds, pruned_ids)
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, **extra)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:  # a failed retry must not lose the (empty) result
            logger.warning("empty-comb retry failed: %s", e)
        else:
            usage = getattr(resp, "usage", None)
            tot = getattr(usage, "total_tokens", None) if usage is not None else None
            if tot is not None:
                total_tokens += tot
                saw_usage = True
            content = message_text(resp.choices[0].message)
    return content, (total_tokens if saw_usage else None)
