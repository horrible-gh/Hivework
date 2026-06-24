"""Comb stdout parser — extracts the first complete balanced JSON object.

Copilot worker (drone) stdout contains:
  1. Tool-trace lines starting with "●" or "✗" (and continuation lines)
  2. The comb JSON object
  3. Optionally: broken/duplicate JSON fragments at the end

This parser:
  - Skips all leading non-JSON content (tool-trace lines)
  - Extracts exactly the FIRST complete top-level balanced `{...}` object
  - Handles string literals (won't be fooled by `{` / `}` inside strings)
  - Handles escape sequences inside strings
  - Discards any trailing content after the first complete JSON object

Returns the parsed dict, or raises ValueError on failure.
"""

import json
import os
import re
from typing import Any, Iterator


def extract_first_json(raw: str) -> dict[str, Any]:
    """Extract the comb JSON object from raw worker stdout.

    Worker stdout interleaves tool-trace lines with the comb JSON, and a trace
    line may itself contain a brace fragment (e.g. an echoed shell snippet like
    ``ForEach-Object { $_.Name }``). Selecting the *first* balanced ``{...}`` is
    therefore unsafe — such a fragment would win and fail to decode. Instead we
    scan every top-level balanced ``{...}`` block, JSON-decode each, and return
    the largest block that decodes to an object. The comb is always the largest
    real JSON object; brace fragments and broken trailing JSON are skipped.

    If nothing decodes, a single best-effort repair pass is tried (see
    :func:`_repair_stray_escapes`) before giving up — workers periodically emit a
    well-formed-looking object with one over-escaped string element, and a
    paid-for decompose/comb call should not be thrown away over that.

    Args:
        raw: The raw stdout text from a copilot worker.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no balanced block decodes to a JSON object.
    """
    obj, last_error = _best_object(raw)
    if obj is not None:
        return obj

    repaired = _repair_stray_escapes(raw)
    if repaired != raw:
        obj, last_error = _best_object(repaired)
        if obj is not None:
            return obj

    if last_error is not None:
        raise ValueError(
            f"Extracted JSON block failed to parse: {last_error}"
        ) from last_error
    raise ValueError("No complete top-level JSON object found in comb output")


def is_comb_dict(obj: Any) -> bool:
    """True iff ``obj`` is a comb-shaped dict — one carrying a ``findings`` list.

    A comb is a CONCLUSION; its signature is a ``findings`` list. A drone that
    emits its NEXT search step instead — a tool-argument object like
    ``{"path":..,"pattern":..,"glob":..}`` — decodes to a dict with no
    ``findings`` key. This predicate is the SINGLE SOURCE OF TRUTH for "is this a
    comb", shared by the ledger honesty check (``fanout.is_comb_shaped``) and the
    pipeline-input gate (``hive.py`` parse stage) so the SAME noise is judged the
    same way in telemetry and in the evidence fed to assemble (NR
    hivework.default.0005.0003 RC-2). Never raises."""
    return isinstance(obj, dict) and isinstance(obj.get("findings"), list)


def _best_object(raw: str) -> tuple[dict[str, Any] | None, json.JSONDecodeError | None]:
    """Return the largest top-level block that decodes to a dict (or None)."""
    best: dict[str, Any] | None = None
    best_len = -1
    last_error: json.JSONDecodeError | None = None
    for block in _iter_top_level_objects(raw):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if isinstance(obj, dict) and len(block) > best_len:
            best, best_len = obj, len(block)
    return best, last_error


# A pretty-printed array element whose string-delimiter quotes were themselves
# backslash-escaped: ``          \"mode='next'\",`` instead of ``"mode='next'"``.
# Observed from a real decompose worker (T890) — the model over-escaped a value
# containing single quotes, which is invalid JSON and desyncs the brace scanner.
_ESCAPED_ELEMENT_RE = re.compile(r'^(\s*)\\"(.*)\\"(\s*,?\s*)$')

# The same defect, but emitted INLINE inside a single-line array, mixed with
# well-formed elements: ``["a", \"mode='info'\", \"mode='next'\", "b"]``. The
# whole-line rule above never fires here. We rewrite only ``\"value\"`` tokens
# sitting at an array-element boundary — a ``[`` or ``,`` before and a ``,`` or
# ``]`` after — which is where a delimiter quote belongs. A genuinely escaped
# quote inside a string value (e.g. ``mode=\"info\"``, preceded by ``=``) is not
# at a boundary and is therefore left intact. ``[^"\\\n]*`` keeps each match on
# one line and stops at the closing token's backslash. (T890 follow-up: queen
# emitted the keyword array on one line.)
_INLINE_ESCAPED_ELEMENT_RE = re.compile(
    r'(?<=[\[,])(\s*)\\"([^"\\\n]*)\\"(\s*)(?=\s*[,\]])'
)


def _repair_stray_escapes(raw: str) -> str:
    """Best-effort, conservative repair of recurring worker JSON defects.

    Rewrites array elements whose string-delimiter quotes were backslash-escaped
    (``\\"value\\"``) back to plain JSON strings (``"value"``), in two shapes:

    - whole-line pretty-printed elements (``          \\"value\\",``), and
    - inline elements at an array boundary on a single line.

    Both rules are deliberately narrow: a real ``"`` at the start of a line and a
    genuinely escaped quote mid-value never match, so Windows-path globs and
    legitimately escaped in-string quotes are left untouched. This runs only as a
    fallback after strict parsing fails, and the result is re-validated by
    ``json.loads`` — a bad repair still raises rather than returning junk.
    """
    raw = _INLINE_ESCAPED_ELEMENT_RE.sub(r'\1"\2"\3', raw)
    out: list[str] = []
    for line in raw.split("\n"):
        m = _ESCAPED_ELEMENT_RE.match(line)
        if m and '\\"' not in m.group(2):
            line = f'{m.group(1)}"{m.group(2)}"{m.group(3)}'
        out.append(line)
    return "\n".join(out)


def _iter_top_level_objects(text: str) -> Iterator[str]:
    """Yield every balanced top-level ``{...}`` substring, respecting strings.

    A character-by-character state machine:
    - Outside strings: counts brace depth; each time depth returns to 0 a
      complete top-level object substring is yielded.
    - Inside strings (delimited by `"`, only entered while inside an object):
      ignores braces, handles `\\` escapes.

    An unbalanced trailing ``{`` (JSON cut off mid-stream) yields nothing.
    """
    start = None
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if in_string:
            if ch == '\\':
                escape_next = True
            elif ch == '"':
                in_string = False
            continue

        # Not in string
        if ch == '"':
            if depth > 0:  # Only track strings inside an object
                in_string = True
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


_SOURCE_ABS_PATH_RE = re.compile(
    r'[A-Za-z]:\\[^\s"<>|]*(?:\\server\\|\\client\\)[^\s"<>|]*',
    re.IGNORECASE,
)


def _norm_path_prefix(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)).rstrip("\\/")


def find_out_of_root_source_paths(raw: str, codebase_root: str | None) -> list[str]:
    """Return absolute source paths in raw output that are outside codebase_root."""
    if not codebase_root:
        return []
    root = _norm_path_prefix(codebase_root)
    bad: list[str] = []
    seen: set[str] = set()
    for match in _SOURCE_ABS_PATH_RE.finditer(raw or ""):
        path = _norm_path_prefix(match.group(0))
        if path.startswith(root + os.sep) or path == root:
            continue
        if path not in seen:
            seen.add(path)
            bad.append(match.group(0))
    return bad


def partition_combs(
    comb_files: dict[str, str],
    codebase_root: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Parse each comb file and split comb-shaped conclusions from noise (G1).

    The fan-out stage writes one ``comb_<axis>.txt`` per axis; some of them are
    not combs at all but search-memos — a tool-argument object the drone printed
    instead of concluding (NR hivework.default.0005.0003 RC-2). Such an object is
    valid JSON, so ``parse_comb_file`` happily returns it; left ungated it would
    flow through conflict-scan → reconcile → assemble as fake "honey" evidence.

    This partitions the parsed files using the single comb-shape predicate
    (:func:`is_comb_dict`), so the evidence set carries ONLY genuine conclusions
    and excluded/failed axes stay visible to the caller for telemetry:

    Returns ``(combs, excluded_notes, parse_fail_notes)`` where
      - ``combs``        — comb-shaped dicts, in axis-id order (the evidence set);
      - ``excluded_notes`` — ``"axis: …"`` notes for parsed-but-non-comb axes;
      - ``parse_fail_notes`` — ``"axis: error"`` notes for files that did not parse.
    """
    combs: list[dict[str, Any]] = []
    excluded_notes: list[str] = []
    parse_fail_notes: list[str] = []
    for axis_id, comb_path in sorted(comb_files.items()):
        try:
            with open(comb_path, "r", encoding="utf-8") as f:
                raw = f.read()
            parsed = extract_first_json(raw)
        except (ValueError, FileNotFoundError) as e:
            parse_fail_notes.append(f"{axis_id}: {e}")
            continue
        if not is_comb_dict(parsed):
            excluded_notes.append(
                f"{axis_id}: non-comb output (no findings array) — excluded from evidence")
            continue
        bad_paths = find_out_of_root_source_paths(raw, codebase_root)
        if bad_paths:
            excluded_notes.append(
                f"{axis_id}: out-of-root source path {bad_paths[0]} — excluded from evidence")
            continue
        combs.append(parsed)
    return combs, excluded_notes, parse_fail_notes


def parse_comb_file(filepath: str) -> dict[str, Any]:
    """Read a comb output file and extract the JSON comb.

    Args:
        filepath: Path to comb_*.txt file.

    Returns:
        Parsed comb dict.

    Raises:
        ValueError: If no valid JSON found.
        FileNotFoundError: If file doesn't exist.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    return extract_first_json(raw)
