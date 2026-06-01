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


def _repair_stray_escapes(raw: str) -> str:
    """Best-effort, conservative repair of one recurring worker JSON defect.

    Rewrites only WHOLE-LINE array elements that begin with a backslash-escaped
    delimiter quote (``\\"value\\"``) back to a plain JSON string (``"value"``).
    Lines that start with a real ``"`` (e.g. a Windows path glob) never match, so
    legitimately escaped in-string quotes are left untouched. This runs only as a
    fallback after strict parsing fails, and the result is re-validated by
    ``json.loads`` — a bad repair still raises rather than returning junk.
    """
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
