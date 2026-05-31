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

    Args:
        raw: The raw stdout text from a copilot worker.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no balanced block decodes to a JSON object.
    """
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

    if best is not None:
        return best
    if last_error is not None:
        raise ValueError(
            f"Extracted JSON block failed to parse: {last_error}"
        ) from last_error
    raise ValueError("No complete top-level JSON object found in comb output")


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
