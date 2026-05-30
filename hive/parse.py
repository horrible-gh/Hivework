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
from typing import Any


def extract_first_json(raw: str) -> dict[str, Any]:
    """Extract the first complete balanced top-level JSON object from raw stdout.

    Args:
        raw: The raw stdout text from a copilot worker.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no complete JSON object is found.
    """
    json_str = _extract_first_balanced_braces(raw)
    if json_str is None:
        raise ValueError("No complete top-level JSON object found in comb output")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted JSON block failed to parse: {e}") from e


def _extract_first_balanced_braces(text: str) -> str | None:
    """Find the first balanced `{...}` in text, respecting strings/escapes.

    Uses a character-by-character state machine:
    - Outside strings: counts brace depth
    - Inside strings (delimited by `"`): ignores braces, handles `\\` escapes

    Returns the substring from first `{` to its matching `}`, or None.
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
            if depth > 0:  # Only enter string tracking inside JSON
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
                    return text[start:i + 1]

    return None


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
