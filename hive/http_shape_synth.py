"""Consumer for GPT's ``build_http_shape_test`` scaffold — mid-term lever ⑦.

Why this module exists
----------------------
The decoy producer never stops being produced: a function merely *named* like the
fix (``get_effective_head``, ``list_modules``) keeps outscoring the real source on
keywords, so STATIC matching is fooled round after round. The only thing never
fooled is a LIVE observation — running the target and watching the symptom.

Converge already grew a live data-read, but it fires only on its *undecidable*
branch; when converge is (wrongly) CONFIDENT the scenario is consistent it never
observes, and a confidently-wrong attribution (head · N183 · M036) sails through.
Lever ⑦ is the backstop that does NOT depend on converge's confidence: whenever the
symptom is "an FE-bound response field comes back empty, so the control that renders
it disappears from the screen", apply ALWAYS has a red test to observe red→green. It
is not a new stage — it is the *observation-isation* of the apply stage that already
exists (``hive/verify.py`` + ``apply --verify``).

Division of labour
------------------
``hive.http_shape.build_http_shape_test`` (GPT's scaffold) turns a ``(route, verb,
json_path, must, app_fixture)`` contract into executable pytest source. This module
is the layer ABOVE it: it RECOGNISES the symptom and fills that contract
DETERMINISTICALLY from the grounding the retriever already builds — never from a
model. Specifically it reuses :func:`hive.retriever._resolve_http_bindings` (which,
since the mount-prefix fold, resolves a client ``getRequest('/api/v1/projects')`` to
the REAL handler's full path) for the route, reads the FE-bound array field off the
response items (``Array.isArray(it.modules)`` → ``modules``) for the field, and reads
the response container key off the handler's own ``return`` for the json_path.

Discipline: the scaffold's TEST IS NOT A GATE. ``verify.py`` certifies it by
EXECUTION (a test green WITHOUT the fix is rejected as non-biting), so a
mis-synthesised test can only fail to certify, never wave a bad fix through. We do
not trust the scaffold's word — the red→green run is the proof. Everything here is
pure-local, deterministic, fail-open (any missing piece → ``None``, today's behaviour
is preserved), zero model cost, never raises.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

# retriever → here is the only import direction (no cycle): we consume the route
# grounding the retriever already builds. http_shape is GPT's scaffold (the contract).
from hive.http_shape import build_http_shape_test, _test_name
from hive.retriever import _resolve_http_bindings

# An array field read off a response ITEM in front-end code. ``Array.isArray(it.modules)``
# is the canonical, highly specific signal that ``modules`` is an array field the server
# serialises onto each item AND that the FE gates rendering on it — exactly the M036
# symptom (empty array → ``v-if="currentModules.length"`` removes the selector). The
# receiver token (``it`` / ``item`` / ``p`` …) is irrelevant; only the field name is.
_FE_ARRAY_FIELD_RE = re.compile(
    r"""Array\.isArray\(\s*[A-Za-z_$][\w$]*\.([A-Za-z_]\w*)\s*\)""")

# The response container key: the handler's own ``return {"projects": …}`` / ``return
# JSONResponse({"projects": …})``. The key whose value is the item list is what makes
# the json_path (``projects[].modules``) match the REAL serialized shape. Read from the
# resolved handler body, never guessed.
_RETURN_DICT_KEY_RE = re.compile(r"""\{\s*['"](\w+)['"]\s*:""")

# A pytest fixture in the target that hands back a TestClient — what GPT's generated
# ``def test_…(app_fixture):`` depends on. We discover the NAME so the synthesised
# test binds to the target's OWN harness (app startup, dependency overrides, seeded
# temp DB) instead of fabricating one.
_FIXTURE_DEF_RE = re.compile(
    r"@pytest\.fixture[^\n]*\n(?:\s*(?:async\s+)?def\s+(\w+)\s*\()", re.MULTILINE)
_TESTCLIENT_RE = re.compile(r"\bTestClient\s*\(")


@dataclass(frozen=True)
class HttpShapeSymptom:
    """A recognised "FE-bound field grounded by an HTTP route" symptom.

    ``full_path`` is the mount-folded route the TestClient must call
    (``/api/v1/projects``); ``field`` is the FE-bound array field that comes back empty
    (``modules``); ``container`` is the response key wrapping the item list
    (``projects``) so the json_path matches the real shape; ``handler_file`` /
    ``handler_line`` locate the RESOLVED handler for provenance — NOT the fix site (the
    fix is wherever the data is produced, which is exactly what the red→green run finds).
    """

    verb: str
    full_path: str
    field: str
    container: str
    fe_url: str
    handler_file: str
    handler_line: str

    @property
    def json_path(self) -> str:
        return f"{self.container}[].{self.field}"


def _fe_array_fields(text: str) -> list[str]:
    """Array fields read off response items in ``text`` (most-mentioned first).

    Conservative: only the ``Array.isArray(x.<field>)`` shape votes, so a plain
    ``.length`` on a FE local or a scalar read never registers. Distinct names ordered
    by frequency then name (stable, deterministic)."""
    freq: dict[str, int] = {}
    for m in _FE_ARRAY_FIELD_RE.finditer(text or ""):
        freq[m.group(1)] = freq.get(m.group(1), 0) + 1
    return sorted(freq, key=lambda f: (-freq[f], f))


def _response_container_key(handler_text: str, field: str) -> str | None:
    """The response key wrapping the item list, read off the handler's ``return``.

    Scans the resolved handler body for ``return {... "<key>": ...}`` and returns the
    FIRST dict key that is NOT the field itself (the wrapper, e.g. ``projects``, not the
    leaf ``modules``). Returns ``None`` when the handler returns a bare list/array or no
    literal-keyed dict is visible — the caller then declines to synthesise (we never
    guess a container that would make the json_path miss the real shape)."""
    for m in _RETURN_DICT_KEY_RE.finditer(handler_text or ""):
        key = m.group(1)
        if key != field:
            return key
    return None


def detect_http_shape_symptom(honey_text: str,
                              codebase_root: str) -> HttpShapeSymptom | None:
    """Recognise the HTTP-shape symptom from grounding, or ``None`` (fail-open).

    Fires ONLY when every piece resolves UNAMBIGUOUSLY from deterministic grounding:
      * exactly one GET fetch URL in the honey resolves (unambiguously) to one backend
        handler via :func:`_resolve_http_bindings` (the mount-prefix-folded full path),
      * exactly one FE-bound array field is read off response items in the honey, and
      * the handler's own ``return`` names the container key wrapping the item list.
    Anything less returns ``None`` so the caller keeps today's behaviour — we never
    guess a route, a field, or a shape. Pure-local, zero model cost, never raises.
    """
    if not honey_text or not codebase_root:
        return None
    try:
        bindings = _resolve_http_bindings([{"text": honey_text}], codebase_root)
    except Exception:
        return None
    gets = [b for b in bindings
            if (b.get("verb") or "").lower() in ("get", "route")]
    if not gets:
        return None  # no read URL in the honey → not this symptom

    # SEVERAL handlers may bind the SAME full path (FlowGate M036: both
    # ``project_settings.list_projects_endpoint`` and the shadowed
    # ``legacy_misc_routes.api_projects`` declare ``GET /api/v1/projects``). That
    # ambiguity is precisely WHY a live test is needed — static analysis cannot tell
    # which handler FastAPI actually dispatches to, but a TestClient hitting the URL
    # exercises the real one. So we key on the PATH, not the handler: require the GETs
    # to agree on ONE full path (they all test the same URL), and never guess across
    # two DIFFERENT paths.
    full_paths = {(b.get("full_path") or b.get("route") or "") for b in gets}
    full_paths.discard("")
    if len(full_paths) != 1:
        return None  # zero or several distinct read URLs → don't guess
    full_path = next(iter(full_paths))
    if not full_path.startswith("/"):
        return None

    fields = _fe_array_fields(honey_text)
    if len(fields) != 1:
        return None  # zero (not this symptom) or several (ambiguous) → don't guess
    field = fields[0]

    # Container key from the handler(s): the candidates almost always agree
    # (``return {"projects": …}``). Take the first non-field key any candidate names;
    # decline only if NONE names one (bare-array / unreadable shape → json_path would miss).
    container = None
    handler_file = handler_line = ""
    for b in gets:
        c = _response_container_key(b.get("text") or "", field)
        if c:
            container = c
            handler_file = b.get("file") or ""
            handler_line = str(b.get("lines") or "")
            break
    if not container:
        return None

    return HttpShapeSymptom(
        verb="get",
        full_path=full_path,
        field=field,
        container=container,
        fe_url=gets[0].get("url") or full_path,
        handler_file=handler_file,
        handler_line=handler_line,
    )


def _fixture_name_in(text: str) -> str | None:
    """Name of the first ``@pytest.fixture`` whose body constructs a ``TestClient``.

    Best-effort and deterministic: matches a fixture def then checks ``TestClient(``
    appears in the following ~60 lines (the fixture body). Returns the fixture name or
    ``None``. Never raises."""
    for m in _FIXTURE_DEF_RE.finditer(text or ""):
        tail = text[m.end(): m.end() + 4000]
        # stop the body scan at the next top-level def/class so we don't borrow a
        # TestClient from an unrelated later fixture.
        nxt = re.search(r"\n(?:@pytest\.fixture|def |class )", tail)
        body = tail[: nxt.start()] if nxt else tail
        if _TESTCLIENT_RE.search(m.group(0) + body):
            return m.group(1)
    return None


def _all_fixture_names_in(text: str) -> list[str]:
    """Every ``@pytest.fixture`` in ``text`` whose body constructs a ``TestClient``."""
    out: list[str] = []
    for m in _FIXTURE_DEF_RE.finditer(text or ""):
        tail = text[m.end(): m.end() + 4000]
        nxt = re.search(r"\n(?:@pytest\.fixture|def |class )", tail)
        body = tail[: nxt.start()] if nxt else tail
        if _TESTCLIENT_RE.search(m.group(0) + body):
            out.append(m.group(1))
    return out


def discover_app_fixture(codebase_root: str, test_dir: str = "tests") -> str | None:
    """The target's TestClient fixture name — ONLY when it is UNAMBIGUOUS.

    Scans ``conftest.py`` + the test files for ``@pytest.fixture``s that build a
    ``TestClient``. Returns the name ONLY when the whole test tree defines EXACTLY ONE
    such fixture; with zero (no harness) or several (FlowGate has many: an inbox client,
    a settings client, …) it returns ``None`` and synthesis is declined.

    The uniqueness bar is the fail-open discipline: auto-binding to an ARBITRARY client
    fixture is unsafe — a fixture whose app does not mount the symptom's route (or does
    not seed it) makes the red test fail regardless of the fix, i.e. a FALSE still_red
    that OVER-BLOCKS a correct fix (a new gate, exactly what lever ⑦ must not become).
    When the harness cannot be chosen unambiguously we observe nothing and keep today's
    behaviour; a faithful harness must then be supplied explicitly (``setup_block``).
    Deterministic, never raises.
    """
    seen: set[str] = set()
    names: list[str] = []
    roots = [os.path.join(codebase_root, test_dir), codebase_root]
    paths: list[str] = []
    for r in roots:
        cf = os.path.join(r, "conftest.py")
        if os.path.isfile(cf):
            paths.append(cf)
    tdir = os.path.join(codebase_root, test_dir)
    if os.path.isdir(tdir):
        try:
            for fn in sorted(os.listdir(tdir)):
                if fn.startswith("test_") and fn.endswith(".py"):
                    paths.append(os.path.join(tdir, fn))
        except OSError:
            pass
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
        except OSError:
            continue
        for name in _all_fixture_names_in(txt):
            if name not in seen:
                seen.add(name)
                names.append(name)
        if len(names) > 1:
            return None  # ambiguous harness → decline (never auto-bind a wrong fixture)
    return names[0] if len(names) == 1 else None


def synthesize_http_shape_red_test(
    honey_text: str,
    codebase_root: str,
    *,
    setup_block: str | None = None,
    app_fixture: str | None = None,
    test_dir: str = "tests",
    test_id: str = "HTTP_SHAPE_RED",
) -> dict[str, Any] | None:
    """Build the ``create_file`` red-test edit + node id for the symptom, or ``None``.

    Returns ``{"edit", "node", "symptom"}`` when the HTTP-shape symptom is recognised
    AND a TestClient harness is available; otherwise ``None`` (fail-open). The harness
    is supplied one of two ways, never invented:

      * ``app_fixture`` — the name of an EXISTING TestClient fixture in the target
        (discovered automatically when not passed). GPT's ``def test_…(app_fixture):``
        then binds to the target's own conftest harness.
      * ``setup_block`` — explicit pytest source (imports + a ``@pytest.fixture`` that
        builds a seeded TestClient) prepended to the generated test. Used when the
        target has no reusable client fixture; the fixture name is read from it.

    The route/verb/json_path are interpolated from :func:`detect_http_shape_symptom`'s
    deterministic grounding — this function never decides them. Zero model cost.
    """
    symptom = detect_http_shape_symptom(honey_text, codebase_root)
    if symptom is None:
        return None

    fixture = app_fixture
    if fixture is None and setup_block:
        fixture = _fixture_name_in(setup_block)
    if fixture is None and not setup_block:
        fixture = discover_app_fixture(codebase_root, test_dir)
    if not fixture:
        return None  # no runnable harness → decline rather than emit a test that errors

    response_assertion = {"json_path": symptom.json_path, "must": "non_empty"}
    try:
        gen = build_http_shape_test(symptom.full_path, symptom.verb,
                                    response_assertion, app_fixture=fixture)
    except Exception:
        return None  # scaffold rejected the contract (bad verb/path) → fail-open

    content = (setup_block.rstrip() + "\n\n\n" + gen) if setup_block else gen
    fn = _test_name(symptom.verb, symptom.full_path,
                    symptom.json_path, "non_empty")
    fname = f"test_http_shape_{re.sub(r'[^a-z0-9]+', '_', symptom.field.lower()).strip('_') or 'field'}.py"
    rel = f"{test_dir.rstrip('/')}/{fname}"
    edit = {
        "id": test_id,
        "kind": "create_file",
        "file": rel,
        "content": content,
        "rationale": (
            f"HTTP-shape red test (lever ⑦): {symptom.verb.upper()} "
            f"{symptom.full_path} must return a non-empty {symptom.field!r}; RED until "
            f"the producer of {symptom.field!r} is fixed — certified by the red→green run."),
        "confidence": "high",
    }
    return {"edit": edit, "node": f"{rel}::{fn}", "symptom": symptom}
