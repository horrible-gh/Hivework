"""Write-sink behaviour oracle — lever L2 (hivework.default.0048.0003-NR).

Why this module exists
----------------------
L1 (``specify._apply_same_facet_consistency_gate``) normalises the 0082 Lv3 defect:
two write-sites sharing ONE diagnosed FK-misrouting facet (``insert_event(group_id,
...)`` routing a groups value into ``events.doc_id`` FK→documents) are forced onto the
SAME canonical callee-swap (``insert_group_event(group_id, ...)`` → ``events.group_id``
FK→groups). But L1 alone cannot drive the loop to GREEN: NR0003 root cause RC1 is that a
single author writes BOTH the fix AND the self-test that certifies it, so there is no
INDEPENDENT oracle. When that self-test pins the decoy, L1 normalises the source but then
loops back — the spec never certifies by execution.

L2 is the missing independent oracle. It generalises lever ⑦'s HTTP-shape red-test
synthesis (``hive.http_shape_synth``): instead of "an FE-bound response field comes back
empty on a GET", the symptom here is "a MUTATING endpoint whose write mis-routes an FK
raises a runtime FOREIGN KEY violation → HTTP 500". The oracle is a TestClient red test
that issues the mutating request and asserts the response does NOT 500 — RED while the
sink mis-routes (FK violation), GREEN once the callee-swap routes the value to the table
its FK references. It is synthesised DETERMINISTICALLY (never authored by the model that
wrote the fix), so it is the independent certification L1's loop-back was missing.

Division of labour (mirrors lever ⑦)
------------------------------------
* This module RECOGNISES the symptom (the spec carries a routing-call repair, and a single
  mutating route is grounded / supplied) and builds the executable test.
* The HARNESS (a ``setup_block`` recipe, like ``recipes/flowgate_http_shape_harness.py``)
  seeds the state the mutation needs and hands back a ``TestClient`` fixture. A mutating
  endpoint needs pre-seeded rows (a group to dispose), which cannot be derived from
  grounding — so, exactly as lever ⑦ requires a harness, L2 declines (fail-open) when no
  runnable harness is supplied rather than emit a test that errors.
* The TEST IS NOT A GATE. ``hive/verify.py`` certifies it by EXECUTION: a test that is
  green WITHOUT the fix is rejected as non-biting, so a mis-seeded harness can only fail to
  certify, never wave a bad fix through.

Everything here is pure-local, deterministic, zero model cost, fail-open (any missing
piece → ``None`` / spec untouched), and never raises.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

from hive.http_shape_synth import _fixture_name_in
from hive.retriever import _resolve_http_bindings

# A persistence/event ROUTING call — the exact shape of the 0082 Lv3 write-sink repair:
# ``insert_*event(arg0, ...)``. Its presence in a SOURCE edit is what tells L2 the spec is
# repairing a write-sink (so a not-500 oracle on the triggering endpoint is meaningful).
# Mirrors specify._ROUTING_CALL_RE (kept local to avoid a specify→here import cycle).
_ROUTING_CALL_RE = re.compile(r"\binsert_\w*event\s*\(")

# Verbs whose handler MUTATES — the endpoints whose write can mis-route an FK.
_MUTATING_VERBS = ("post", "put", "patch", "delete")

# An unfilled path parameter (``/groups/{group_id}/dispose``): a template we cannot seed a
# concrete value into from grounding alone, so a concrete request path must be supplied.
_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")


@dataclass(frozen=True)
class WriteSinkRequest:
    """The concrete mutating request the red test issues to trigger the write.

    ``verb``/``path`` are the request line; ``json`` is the optional request body. ``path``
    is CONCRETE (path params already substituted with values the harness seeds) — a request
    still carrying ``{...}`` is rejected, since it cannot reach the write.
    """

    verb: str
    path: str
    json: dict[str, Any] | None = None


def _has_routing_repair(spec: dict[str, Any]) -> bool:
    """True when a SOURCE edit's replacement is a routing call (``insert_*event(...)``).

    This is the write-sink repair signature: the spec is changing where an event row lands.
    Looks at non-test source edits only; deterministic, never raises.
    """
    for e in (spec.get("edits") or []):
        if not isinstance(e, dict):
            continue
        if e.get("kind", "edit") == "create_file":
            continue
        new = (e.get("replacement_new") or "") + "\n" + (e.get("anchor_old") or "")
        if _ROUTING_CALL_RE.search(new):
            return True
    return False


def _detect_mutating_route(honey_text: str, codebase_root: str) -> tuple[str, str] | None:
    """``(verb, full_path)`` of the single mutating route grounded in the honey, or ``None``.

    Resolves the honey's request bindings via the retriever (the same grounding lever ⑦
    uses) and returns the unique mutating route. Declines (``None``) on zero or several
    distinct mutating routes, or when the route still carries a path parameter we cannot
    seed. Pure-local, fail-open, never raises.
    """
    if not honey_text or not codebase_root:
        return None
    try:
        bindings = _resolve_http_bindings([{"text": honey_text}], codebase_root)
    except Exception:
        return None
    muts = [b for b in bindings if (b.get("verb") or "").lower() in _MUTATING_VERBS]
    if not muts:
        return None
    paths = {(b.get("full_path") or b.get("route") or "") for b in muts}
    paths.discard("")
    if len(paths) != 1:
        return None  # zero or several mutating routes → don't guess
    full_path = next(iter(paths))
    verbs = {(b.get("verb") or "").lower() for b in muts
             if (b.get("full_path") or b.get("route") or "") == full_path}
    verbs &= set(_MUTATING_VERBS)
    if len(verbs) != 1:
        return None
    return (next(iter(verbs)), full_path)


def _coerce_request(request: Any, honey_text: str,
                    codebase_root: str) -> WriteSinkRequest | None:
    """Resolve the concrete mutating request, or ``None`` (fail-open).

    A caller-supplied ``request`` (dict / :class:`WriteSinkRequest`) wins — its concrete
    ``path`` may already substitute a seeded id. Otherwise the verb+path are grounded from
    the honey, but only accepted when the route carries NO unfilled path parameter (we never
    invent an id to seed). Never raises.
    """
    verb = path = None
    body: dict[str, Any] | None = None
    if isinstance(request, WriteSinkRequest):
        verb, path, body = request.verb, request.path, request.json
    elif isinstance(request, dict):
        verb = request.get("verb")
        path = request.get("path")
        b = request.get("json")
        body = b if isinstance(b, dict) else None
    if not (verb and path):
        grounded = _detect_mutating_route(honey_text, codebase_root)
        if grounded is None:
            return None
        verb, path = grounded
    verb = str(verb).lower()
    path = str(path)
    if verb not in _MUTATING_VERBS:
        return None
    if not path.startswith("/") or _PATH_PARAM_RE.search(path):
        return None  # not absolute, or an unseeded template → decline
    return WriteSinkRequest(verb=verb, path=path, json=body)


def _test_fn_name(verb: str, path: str) -> str:
    """A valid, descriptive pytest function name for the not-500 oracle. Never raises."""
    raw = f"test_write_sink_{verb}_{path}_not_500"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name or "test_write_sink_not_500"


def _build_write_sink_test(req: WriteSinkRequest, fixture: str) -> tuple[str, str]:
    """``(source, test_fn_name)`` for the not-500 red test. Deterministic, never raises."""
    fn = _test_fn_name(req.verb, req.path)
    call_args = [repr(req.path)]
    if req.json is not None:
        call_args.append(f"json={req.json!r}")
    call = f"{fixture}.{req.verb}({', '.join(call_args)})"
    src = (
        "def {fn}({fixture}):\n"
        "    response = {call}\n"
        "    assert response.status_code != 500, (\n"
        "        \"{verb} {path} must not 500: a write-sink FK mis-routing surfaces as a \"\n"
        "        \"server error (the mutating handler's event write violates its FK); \"\n"
        "        \"got %s: %s\" % (response.status_code, response.text))\n"
    ).format(fn=fn, fixture=fixture, call=call,
             verb=req.verb.upper(), path=req.path)
    return src, fn


def synthesize_write_sink_red_test(
    spec: dict[str, Any],
    honey_text: str,
    codebase_root: str,
    *,
    setup_block: str | None = None,
    request: Any = None,
    test_dir: str = "tests",
    test_id: str = "WRITE_SINK_RED",
) -> dict[str, Any] | None:
    """Build the ``create_file`` not-500 red-test edit + node id, or ``None`` (fail-open).

    Returns ``{"edit", "node", "request"}`` when ALL hold: the spec carries a routing-call
    repair (a write-sink fix to certify), a runnable ``setup_block`` harness with a
    TestClient fixture is supplied, and a concrete mutating request resolves (caller-supplied
    or grounded with no unfilled path parameter). Otherwise ``None``. Zero model cost.
    """
    if not _has_routing_repair(spec):
        return None  # no write-sink repair in this spec → a not-500 oracle is not meaningful
    if not setup_block:
        return None  # a mutation needs seeded state → no harness, no runnable test
    fixture = _fixture_name_in(setup_block)
    if not fixture:
        return None  # the harness exposes no TestClient fixture to bind to
    req = _coerce_request(request, honey_text, codebase_root)
    if req is None:
        return None

    body, fn = _build_write_sink_test(req, fixture)
    content = setup_block.rstrip() + "\n\n\n" + body
    slug = re.sub(r"[^a-z0-9]+", "_", req.path.lower()).strip("_") or "endpoint"
    fname = f"test_write_sink_{slug}.py"
    rel = f"{test_dir.rstrip('/')}/{fname}"
    edit = {
        "id": test_id,
        "kind": "create_file",
        "file": rel,
        "content": content,
        "rationale": (
            f"write-sink behaviour oracle (lever L2): {req.verb.upper()} {req.path} must "
            "not 500 — the FK mis-routing the spec repairs surfaces as a server error; RED "
            "until the event write routes to the table its FK references, certified by the "
            "red→green run (independent of the author's self-test)."),
        "confidence": "high",
    }
    return {"edit": edit, "node": f"{rel}::{fn}", "request": req}
