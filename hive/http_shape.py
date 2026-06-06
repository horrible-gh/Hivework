"""Generate pytest gates for HTTP JSON response shapes.

The generated test deliberately relies on a caller-provided TestClient fixture.
That keeps application startup, dependency overrides, and seeded temporary
databases in the target project's existing pytest infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
import keyword
import re


_SUPPORTED_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
_SUPPORTED_MUST = {"exists", "non_empty"}


@dataclass(frozen=True)
class JsonPathPart:
    """One dotted JSON path component."""

    key: str
    iterates: bool = False


def parse_json_path(json_path: str) -> tuple[JsonPathPart, ...]:
    """Parse dotted object keys with optional ``[]`` array traversal markers."""
    if not isinstance(json_path, str) or not json_path.strip():
        raise ValueError("json_path must be a non-empty string")

    parts: list[JsonPathPart] = []
    for raw_part in json_path.split("."):
        if not raw_part:
            raise ValueError(f"invalid json_path {json_path!r}: empty component")
        iterates = raw_part.endswith("[]")
        key = raw_part[:-2] if iterates else raw_part
        if not key or "[" in key or "]" in key:
            raise ValueError(
                f"invalid json_path component {raw_part!r}; "
                "only a trailing [] traversal marker is supported"
            )
        parts.append(JsonPathPart(key=key, iterates=iterates))
    return tuple(parts)


def build_json_path_assertions(
    json_path: str,
    must: str,
    *,
    root_name: str = "payload",
    indent: str = "    ",
) -> str:
    """Return pytest assertion statements for a parsed JSON path.

    Intermediate arrays must contain at least one item. Without that guard,
    an assertion such as "every project has modules" would pass vacuously for
    an empty projects list.
    """
    parts = parse_json_path(json_path)
    if must not in _SUPPORTED_MUST:
        supported = ", ".join(sorted(_SUPPORTED_MUST))
        raise ValueError(f"unsupported must {must!r}; expected one of: {supported}")
    if not root_name.isidentifier() or keyword.iskeyword(root_name):
        raise ValueError("root_name must be a valid Python identifier")

    lines: list[str] = []
    current = root_name
    current_indent = indent
    traversed: list[str] = []

    for index, part in enumerate(parts):
        location = ".".join(traversed) or "<root>"
        key_literal = repr(part.key)
        value_name = f"_path_value_{index}"

        lines.append(
            f"{current_indent}assert isinstance({current}, dict), "
            f"{repr(f'json_path {json_path!r}: {location} must be an object')}"
        )
        lines.append(
            f"{current_indent}assert {key_literal} in {current}, "
            f"{repr(f'json_path {json_path!r}: missing field {part.key!r}')}"
        )
        lines.append(f"{current_indent}{value_name} = {current}[{key_literal}]")
        traversed.append(part.key + ("[]" if part.iterates else ""))
        is_final = index == len(parts) - 1

        if part.iterates:
            lines.append(
                f"{current_indent}assert isinstance({value_name}, list), "
                f"{repr(f'json_path {json_path!r}: {part.key!r} must be an array')}"
            )
            if not is_final or must == "non_empty":
                lines.append(
                    f"{current_indent}assert {value_name}, "
                    f"{repr(f'json_path {json_path!r}: {part.key!r} must not be empty')}"
                )
            if not is_final or must == "non_empty":
                item_name = f"_path_item_{index}"
                lines.append(f"{current_indent}for {item_name} in {value_name}:")
                current_indent += "    "
                current = item_name
                if is_final:
                    lines.append(
                        f"{current_indent}assert {current}, "
                        f"{repr(f'json_path {json_path!r}: each selected value must be non-empty')}"
                    )
        else:
            current = value_name
            if is_final and must == "non_empty":
                lines.append(
                    f"{current_indent}assert {current}, "
                    f"{repr(f'json_path {json_path!r}: value must be non-empty')}"
                )

    return "\n".join(lines)


def _test_name(verb: str, route: str, json_path: str, must: str) -> str:
    raw = f"test_http_shape_{verb}_{route}_{json_path}_{must}"
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    return normalized or "test_http_shape"


def build_http_shape_test(
    route: str,
    verb: str,
    response_assertion: dict,
    *,
    app_fixture: str,
) -> str:
    """Build executable pytest source for one HTTP response-shape gate."""
    if not isinstance(route, str) or not route.startswith("/"):
        raise ValueError("route must be an absolute HTTP path starting with '/'")
    if not isinstance(verb, str):
        raise ValueError("verb must be a string")
    normalized_verb = verb.strip().lower()
    if normalized_verb not in _SUPPORTED_VERBS:
        supported = ", ".join(sorted(_SUPPORTED_VERBS))
        raise ValueError(f"unsupported verb {verb!r}; expected one of: {supported}")
    if (
        not isinstance(app_fixture, str)
        or not app_fixture.isidentifier()
        or keyword.iskeyword(app_fixture)
    ):
        raise ValueError("app_fixture must be a valid Python fixture identifier")
    if not isinstance(response_assertion, dict):
        raise ValueError("response_assertion must be a dict")

    json_path = response_assertion.get("json_path")
    must = response_assertion.get("must")
    assertion_source = build_json_path_assertions(json_path, must)
    test_name = _test_name(normalized_verb, route, json_path, must)

    return (
        '"""Generated HTTP response-shape gate.\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        '"""\n\n'
        f"def {test_name}({app_fixture}):\n"
        f"    response = {app_fixture}.{normalized_verb}({route!r})\n"
        "    assert 200 <= response.status_code < 300, response.text\n"
        "    payload = response.json()\n"
        f"{assertion_source}\n"
    )
