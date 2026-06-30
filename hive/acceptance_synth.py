"""box-0: synthesise an acceptance RED test from a design doc's ``## 수용기준``.

Why this module exists
----------------------
Levers ⑦ (``http_shape_synth``) and L2/L3 grow a red→green oracle from the *bug
symptom* (an empty FE-bound field, a 500 on a write, a last-write-wins race). box-0
is the same machine pointed at a DIFFERENT seed: a DESIGN document's acceptance
criteria. When a feature card ships, "did the author actually build what the design
promised?" has had no executable answer — the design's ``## 수용기준`` was prose a
human read. box-0 reads that prose, grounds it against the live code, and (when it
resolves UNAMBIGUOUSLY) synthesises a TestClient/unit RED test wired into
``spec.verify`` so ``apply --verify`` certifies the feature by EXECUTION — exactly
the posture ⑦ takes, but the oracle's seed is the design instead of the honey.

Division of labour
------------------
This module RECOGNISES an acceptance criterion and fills the
``(verb, route, json_path, must)`` / ``(target, expected)`` contract
DETERMINISTICALLY — either lifted verbatim from an explicit ``oracle:`` block the
design author wrote (lowest risk, NR0003 recommendation A), or derived from prose
under the §2.3 rules. It NEVER guesses a route, a field, or a value: the single
hard cell of box-0 (prose → contract, NR0003 §3.2-(2)) is gated so anything less
than a unique reading declines (``None``) — a decline is a no-go candidate (the
human/Opus boundary, M0002 §3), never a guessed test.

Discipline (inherited from lever ⑦): the synthesised test IS NOT A GATE. ``verify.py``
certifies it by EXECUTION — a test green WITHOUT the source edit is rejected as
non-biting — so a mis-synthesised acceptance test can only fail to certify, never
wave a bad feature through. Everything here is pure-local, deterministic, fail-open
(any missing/ambiguous piece → ``None``, today's behaviour preserved), zero model
cost, never raises.

Logic SSOT: hivework.default.0064.0006-L (data contract: 0005-P).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# Reuse GPT's scaffold (the contract) and the route grounding the retriever already
# builds. Reuse ⑦'s harness discovery / container reader verbatim — box-0 differs only
# in its SEED (design criteria, not honey), never in how it grounds or runs.
from hive.http_shape import (
    _SUPPORTED_VERBS,
    _test_name,
    build_http_shape_test,
    build_json_path_assertions,
    parse_json_path,
)
from hive.http_shape_synth import (
    _response_container_key,
    discover_app_fixture,
    _fixture_name_in,
)
from hive.retriever import _resolve_http_bindings

# PyYAML parses the ``## 수용기준`` list. Guarded so a stripped environment fails open
# (no parser → no criteria → today's behaviour), keeping box-0 zero-HARD-dependency.
try:  # pragma: no cover - import guard
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None

# ── L0006 §1 parameters ──────────────────────────────────────────────────────
ACCEPTANCE_MARKER = "## 수용기준"          # P0005 DD-1 fixed marker (oracle source)
SUPPORTED_KIND = {"http_read", "unit_value"}
SUPPORTED_MUST_HTTP_NOW = {"exists", "non_empty"}   # build_json_path_assertions today
SUPPORTED_MUST_BOX0_EXT = {"equals"}                # box-0 value-compare extension
EDIT_CONFIDENCE = "high"
KILL_SWITCH_ENV = "HIVE_NO_ACCEPTANCE"
TEST_ID = "ACCEPTANCE_RED"

# A response field name a prose criterion names, written as a backticked identifier
# (``the `modules` field``). Backticks are the deterministic, low-false-positive signal
# — bare words in prose are NOT fields. Exactly one distinct backticked id may survive
# the §2.3 gate; zero or several → decline.
_BACKTICK_ID_RE = re.compile(r"`([A-Za-z_][\w]*)`")

# A ``file.py::symbol`` target token (unit_value). The only shape that promotes a
# criterion to the unit kind from prose; anything fuzzier declines.
_SYMBOL_TARGET_RE = re.compile(r"`?([\w./-]+\.py)::([A-Za-z_]\w*)`?")

# An absolute HTTP path token in prose (``/api/v1/projects``). Presence (or a verb
# keyword) routes the criterion to the http_read kind.
_HTTP_PATH_RE = re.compile(r"(?<![\w])/(?:[\w\-]+/)*[\w\-]+")
_VERB_KEYWORD_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", re.IGNORECASE)

# A literal expected value in prose for ``equals`` (``정확히 3``, ``== 'done'``,
# ``= "x"``). Conservative: only an explicit number or quoted string is lifted.
_EXPECTED_LITERAL_RE = re.compile(
    r"""(?:정확히|==|=)\s*(?P<v>"[^"]*"|'[^']*'|-?\d+(?:\.\d+)?)""")


@dataclass(frozen=True)
class AcceptanceSymptom:
    """A grounded, runnable acceptance contract — the oracle seed for one criterion.

    ``kind`` selects the assertion shape: ``http_read`` (a TestClient call asserts a
    json_path) or ``unit_value`` (a framework-neutral value equality). ``full_path`` /
    ``verb`` / ``json_path`` / ``must`` carry the HTTP contract; ``target`` / ``expected``
    carry the unit contract. ``source_ac_id`` traces back to the design criterion so the
    synthesised note/node is auditable.
    """

    kind: str
    source_ac_id: str
    must: str = ""
    # http_read
    verb: str = ""
    full_path: str = ""
    json_path: str = ""
    field: str = ""
    container: str = ""
    # unit_value / equals
    target: str = ""
    expected: Any = None


# ── §2.1 read_acceptance_criteria ────────────────────────────────────────────
def extract_marker_section(design_text: str, marker: str = ACCEPTANCE_MARKER) -> str:
    """The body under ``marker`` up to the next ``## `` heading (or EOF), or ``""``.

    Deterministic, never raises. A missing marker yields an empty string → the caller
    treats box-0's input as empty (L scenario 3), preserving today's behaviour."""
    if not design_text or marker not in design_text:
        return ""
    idx = design_text.index(marker)
    after = design_text[idx + len(marker):]
    # stop at the next top-level (## or #) heading so we don't swallow later sections.
    nxt = re.search(r"\n#{1,2}\s", after)
    return after[: nxt.start()] if nxt else after


def read_acceptance_criteria(design_text: str) -> list[dict[str, Any]]:
    """Parse the ``## 수용기준`` marker section into criterion dicts (id + prose required).

    Each item carries ``id`` / ``prose`` and optionally an ``oracle`` block and a
    ``harness`` block (P0005 DD-1). Items without both ``id`` and ``prose`` are dropped
    (L §2.1). Fail-open: no marker / no yaml / malformed → ``[]``."""
    section = extract_marker_section(design_text)
    if not section.strip() or _yaml is None:
        return []
    try:
        data = _yaml.safe_load(section)
    except Exception:
        return []
    items: list[Any]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("acceptance"), list):
        items = data["acceptance"]
    else:
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict) and it.get("id") and it.get("prose"):
            out.append(it)
    return out


# ── §2.3 prose helpers ───────────────────────────────────────────────────────
def infer_must(prose: str) -> str:
    """L §4.2 decision tree: prose modality → assertion. Default ``non_empty``.

    Every branch terminates with a default so an unrecognised modality is conservatively
    treated as "the field must come back non-empty" rather than declined."""
    p = prose or ""
    if re.search(r"비어\s*있지\s*않|non-?empty|적어도\s*하나|목록을\s*반환|at least one", p, re.I):
        return "non_empty"
    if _EXPECTED_LITERAL_RE.search(p) or re.search(r"정확히", p):
        return "equals"
    if re.search(r"존재|있어야|포함|present|contains?", p, re.I):
        return "exists"
    return "non_empty"


def _extract_response_field_names(prose: str) -> list[str]:
    """Distinct backticked identifiers a prose criterion names (the response field)."""
    seen: list[str] = []
    for m in _BACKTICK_ID_RE.finditer(prose or ""):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def _extract_literal_expected(prose: str) -> Any:
    """The single literal expected value in prose for ``equals``, or ``None``.

    ``"x"`` / ``'x'`` → str; an integer/float token → the parsed number. Returns ``None``
    when no explicit literal is present (so ``equals`` declines rather than compares to a
    guess)."""
    m = _EXPECTED_LITERAL_RE.search(prose or "")
    if not m:
        return None
    raw = m.group("v")
    if raw[:1] in "\"'":
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return None


def _distinct_path_tokens(text: str) -> list[str]:
    """Absolute HTTP path tokens (``/api/v1/projects``) named in ``text``, in order."""
    out: list[str] = []
    for m in _HTTP_PATH_RE.finditer(text or ""):
        tok = m.group(0)
        if tok.count("/") >= 1 and tok not in out:
            out.append(tok)
    return out


def _resolve_gets(text: str, codebase_root: str) -> list[dict[str, Any]]:
    """Backend bindings the retriever resolves for the path tokens named in ``text``.

    The retriever grounds CLIENT FETCH literals, but a design criterion names a route as
    PLAIN PROSE (``GET /api/v1/projects``) or a bare ``route:`` string. So we lift the path
    tokens and feed the resolver a synthetic ``getRequest("<path>")`` probe per token —
    reusing the exact mount-prefix-folded route grounding lever ⑦ relies on. The binding's
    verb comes from the backend decorator (so a POST route resolves with verb=post)."""
    probe = "\n".join(f'getRequest("{p}")' for p in _distinct_path_tokens(text))
    if not probe:
        return []
    try:
        bindings = _resolve_http_bindings([{"text": probe}], codebase_root)
    except Exception:
        return []
    return [b for b in bindings
            if (b.get("verb") or "").lower() in _SUPPORTED_VERBS
            or (b.get("verb") or "").lower() == "route"]


def _distinct_full_paths(gets: list[dict[str, Any]]) -> set[str]:
    paths = {(b.get("full_path") or b.get("route") or "") for b in gets}
    paths.discard("")
    return paths


def _dominant_verb(gets: list[dict[str, Any]]) -> str | None:
    """The single supported verb the GETs agree on, or ``None`` (disagreement → decline)."""
    verbs = {(b.get("verb") or "").lower() for b in gets}
    verbs = {v for v in verbs if v in _SUPPORTED_VERBS}
    if "get" in verbs and len(verbs) == 1:
        return "get"
    if len(verbs) == 1:
        return next(iter(verbs))
    # mixed/unknown verbs but a read URL resolved → default to GET (the read contract).
    return "get" if gets else None


def _container_for(gets: list[dict[str, Any]], field: str) -> str | None:
    for b in gets:
        c = _response_container_key(b.get("text") or "", field)
        if c:
            return c
    return None


def derive_contract_from_prose(prose: str, codebase_root: str,
                               ac_id: str) -> AcceptanceSymptom | None:
    """L §2.3 — derive a runnable contract from free prose, or ``None`` (decline).

    The single hard cell of box-0. Resolves a UNIQUE route (1) and UNIQUE field (1)
    from grounding before it will synthesise; any ambiguity (0 or ≥2) declines. Guessing
    is forbidden — a decline is a no-go candidate, not a fabricated test."""
    if not prose:
        return None
    has_path = bool(_HTTP_PATH_RE.search(prose)) or bool(_VERB_KEYWORD_RE.search(prose))
    sym_m = _SYMBOL_TARGET_RE.search(prose)

    if has_path:
        gets = _resolve_gets(prose, codebase_root)
        paths = _distinct_full_paths(gets)
        if len(paths) != 1:          # 0 or ≥2 distinct read URLs → ambiguous, decline
            return None
        full_path = next(iter(paths))
        if not full_path.startswith("/"):
            return None
        verb = _dominant_verb(gets)
        if not verb:
            return None
        fields = _extract_response_field_names(prose)
        if len(fields) != 1:         # 0 or ≥2 candidate fields → ambiguous, decline
            return None
        field = fields[0]
        must = infer_must(prose)
        container = _container_for(gets, field)
        if must == "equals":
            expected = _extract_literal_expected(prose)
            if expected is None:     # equals with no literal → can't compare, decline
                return None
            json_path = f"{container}.{field}" if container else field
            return AcceptanceSymptom(
                kind="http_read", source_ac_id=ac_id, verb=verb,
                full_path=full_path, json_path=json_path, field=field,
                container=container or "", must="equals", expected=expected)
        json_path = f"{container}[].{field}" if container else field
        if must not in SUPPORTED_MUST_HTTP_NOW:
            return None
        return AcceptanceSymptom(
            kind="http_read", source_ac_id=ac_id, verb=verb, full_path=full_path,
            json_path=json_path, field=field, container=container or "", must=must)

    if sym_m:
        target = f"{sym_m.group(1)}::{sym_m.group(2)}"
        if not _resolve_symbol(target, codebase_root):
            return None
        expected = _extract_literal_expected(prose)
        if expected is None:
            return None
        return AcceptanceSymptom(kind="unit_value", source_ac_id=ac_id,
                                 target=target, must="equals", expected=expected)

    return None  # kind unknown → decline (L scenario 4)


# ── §2.2 explicit-oracle validation ──────────────────────────────────────────
def _resolve_symbol(target: str, codebase_root: str) -> bool:
    """True when ``file.py::symbol`` exists: the file is under root and defines symbol."""
    if "::" not in (target or ""):
        return False
    rel, sym = target.split("::", 1)
    path = os.path.join(codebase_root, rel)
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
    except OSError:
        return False
    return bool(re.search(rf"(?m)^\s*(?:async\s+def|def|class)\s+{re.escape(sym)}\b", txt)
                or re.search(rf"(?m)^\s*{re.escape(sym)}\s*[:=]", txt))


def validate_explicit_oracle(o: dict[str, Any], codebase_root: str,
                             ac_id: str) -> AcceptanceSymptom | None:
    """L §2.2 — confirm an author-written ``oracle:`` block GROUNDS, lift it verbatim.

    The author supplies verb/route/json_path/must (http_read) or target/must/expected
    (unit_value); box-0 confirms existence against the live code and NEVER overrides the
    author's values. Anything unsupported or ungrounded declines."""
    if not isinstance(o, dict):
        return None
    kind = o.get("kind")
    if kind not in SUPPORTED_KIND:
        return None
    if kind == "http_read":
        verb = str(o.get("verb", "")).lower()
        route = o.get("route", "")
        must = o.get("must", "")
        json_path = o.get("json_path", "")
        if verb not in _SUPPORTED_VERBS:
            return None
        if not isinstance(route, str) or not route.startswith("/"):
            return None
        if must not in (SUPPORTED_MUST_HTTP_NOW | SUPPORTED_MUST_BOX0_EXT):
            return None
        if must == "equals" and "expected" not in o:
            return None
        if not json_path:
            return None
        gets = _resolve_gets(route, codebase_root)
        if len(_distinct_full_paths(gets)) != 1:
            return None
        full_path = next(iter(_distinct_full_paths(gets)))
        return AcceptanceSymptom(
            kind="http_read", source_ac_id=ac_id, verb=verb, full_path=full_path,
            json_path=json_path, must=must, expected=o.get("expected"),
            field=str(o.get("field", "")), container=str(o.get("container", "")))
    # unit_value
    target = o.get("target", "")
    if not _resolve_symbol(target, codebase_root):
        return None
    if o.get("must") != "equals" or "expected" not in o:
        return None
    return AcceptanceSymptom(kind="unit_value", source_ac_id=ac_id, target=target,
                             must="equals", expected=o.get("expected"))


def detect_acceptance(ac: dict[str, Any],
                      codebase_root: str) -> AcceptanceSymptom | None:
    """L §2.2 — route a criterion to the explicit-oracle or prose-derivation path."""
    if not codebase_root or not isinstance(ac, dict):
        return None
    ac_id = str(ac.get("id", ""))
    if isinstance(ac.get("oracle"), dict):
        return validate_explicit_oracle(ac["oracle"], codebase_root, ac_id)
    return derive_contract_from_prose(str(ac.get("prose", "")), codebase_root, ac_id)


# ── §2.4 synthesis ───────────────────────────────────────────────────────────
def _path_expr(json_path: str, root_name: str = "payload") -> str:
    """``a.b.c`` → ``payload['a']['b']['c']`` (non-iterating paths only)."""
    parts = parse_json_path(json_path)
    expr = root_name
    for part in parts:
        expr += f"[{part.key!r}]"
    return expr


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "field"


def _build_http_equals_test(route: str, verb: str, json_path: str,
                            expected: Any, app_fixture: str) -> str:
    """A TestClient test asserting ``json_path == expected`` (box-0 ``equals`` extension).

    Reuses ⑦'s ``exists`` chain for the existence/KeyError-safe walk, then appends one
    value-equality assert. Declines (raises → caller catches) on an iterating path; an
    array equality is not a value compare."""
    exists = build_json_path_assertions(json_path, "exists")
    expr = _path_expr(json_path)            # raises on '[]' → caller fails open
    fn = _test_name(verb, route, json_path, "equals")
    return (
        '"""Generated acceptance value gate (box-0).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"    response = {app_fixture}.{verb}({route!r})\n"
        "    assert 200 <= response.status_code < 300, response.text\n"
        "    payload = response.json()\n"
        f"{exists}\n"
        f"    assert {expr} == {expected!r}, 'acceptance value mismatch'\n"
    )


def _build_unit_value_test(target: str, expected: Any) -> str:
    """A framework-neutral pytest unit test asserting ``symbol == expected``.

    Loads ``file.py::symbol`` from the file path (relative to the run cwd = codebase
    root) and compares the symbol's value. Framework-specific skeletons (vitest) are a
    follow-up T (L DEFERRED)."""
    rel, sym = target.split("::", 1)
    fn = f"test_acceptance_unit_{_slug(sym)}"
    return (
        '"""Generated acceptance unit-value gate (box-0)."""\n'
        "import importlib.util as _u\n\n"
        f"def {fn}():\n"
        f"    _s = _u.spec_from_file_location('_box0_mod', {rel!r})\n"
        "    assert _s and _s.loader, 'target module not importable'\n"
        "    _m = _u.module_from_spec(_s)\n"
        "    _s.loader.exec_module(_m)\n"
        f"    assert getattr(_m, {sym!r}) == {expected!r}, 'acceptance value mismatch'\n"
    )


def _test_fn_name(symptom: AcceptanceSymptom) -> str:
    if symptom.kind == "unit_value":
        return f"test_acceptance_unit_{_slug(symptom.target.split('::')[-1])}"
    return _test_name(symptom.verb, symptom.full_path, symptom.json_path, symptom.must)


def synthesize_acceptance_red_test(
    symptom: AcceptanceSymptom | None,
    codebase_root: str,
    *,
    setup_block: str | None = None,
    app_fixture: str | None = None,
    test_dir: str = "tests",
    test_id: str = TEST_ID,
) -> dict[str, Any] | None:
    """Build the ``create_file`` red-test edit + node id for ``symptom``, or ``None``.

    Mirrors :func:`hive.http_shape_synth.synthesize_http_shape_red_test`. http_read needs
    a runnable TestClient harness (explicit ``app_fixture`` / ``setup_block`` / unique
    discovery); without one it declines rather than emit a test that errors. unit_value
    needs no harness. Zero model cost, never raises beyond construction errors the caller
    catches."""
    if symptom is None:
        return None
    try:
        if symptom.kind == "unit_value":
            content = _build_unit_value_test(symptom.target, symptom.expected)
            slug = _slug(symptom.target.split("::")[-1])
        else:
            fixture = app_fixture
            if fixture is None and setup_block:
                fixture = _fixture_name_in(setup_block)
            if fixture is None and not setup_block:
                fixture = discover_app_fixture(codebase_root, test_dir)
            if not fixture:
                return None  # no runnable harness → decline (L scenario 5)
            if symptom.must == "equals":
                body = _build_http_equals_test(
                    symptom.full_path, symptom.verb, symptom.json_path,
                    symptom.expected, fixture)
            else:
                body = build_http_shape_test(
                    symptom.full_path, symptom.verb,
                    {"json_path": symptom.json_path, "must": symptom.must},
                    app_fixture=fixture)
            content = (setup_block.rstrip() + "\n\n\n" + body) if setup_block else body
            slug = _slug(symptom.field or symptom.full_path)
    except Exception:
        return None  # scaffold/path rejected the contract → fail-open

    fn = _test_fn_name(symptom)
    rel = f"{test_dir.rstrip('/')}/test_acceptance_{slug}.py"
    if symptom.kind == "unit_value":
        rationale = (f"acceptance unit test (box-0, AC {symptom.source_ac_id}): "
                     f"{symptom.target} must equal {symptom.expected!r}; RED until the "
                     "feature is built — certified by the red→green run.")
    else:
        rationale = (f"acceptance red test (box-0, AC {symptom.source_ac_id}): "
                     f"{symptom.verb.upper()} {symptom.full_path} json_path "
                     f"{symptom.json_path!r} must {symptom.must}"
                     + (f" == {symptom.expected!r}" if symptom.must == "equals" else "")
                     + "; RED until the feature is built — certified by the red→green run.")
    edit = {
        "id": test_id,
        "kind": "create_file",
        "file": rel,
        "content": content,
        "rationale": rationale,
        "confidence": EDIT_CONFIDENCE,
    }
    return {"edit": edit, "node": f"{rel}::{fn}", "symptom": symptom}
