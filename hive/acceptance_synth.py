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
# box-5 (group 0070, level-6, GAP-5): the RELATIONSHIP axis. Unlike ``equals`` (which
# compares an observed json_path to a LITERAL constant), a relational must compares one
# observed value to ANOTHER observed value on the same payload — the cross-field/cross-entity
# invariant NR0003 named (``total == len(items)``). ``equals_len``: ``payload[json_path] ==
# len(payload[other_path])`` (a count field agrees with a list length). ``equals_path``:
# ``payload[json_path] == payload[other_path]`` (two scalar paths are equal). Both carry an
# ``other_path`` companion instead of an ``expected`` literal. Explicit-oracle ONLY — prose
# can NOT resolve WHICH two paths relate without guessing, and box never guesses.
#
# box-5b (group 0072, level-6b, GAP-5b): the CROSS-ROUTE relation — box-5 (relation) composed
# with box-4 (multi-route). The SAME ``equals_len``/``equals_path`` musts, but the RHS observed
# value lives on ANOTHER read's payload (``GET /dashboard/summary``'s ``project_count`` ==
# ``len(GET /projects``'s ``items``)``). A relational assert inside a ``reads`` oracle carries an
# ``other_read`` INDEX locator selecting which read supplies ``other_path``; the synthesised test
# fetches every read first, then asserts across payloads. No new must — a scope extension, not a
# new axis. Explicit-oracle ONLY (which read's which field relates can NOT come from prose).
SUPPORTED_MUST_BOX5_REL = {"equals_len", "equals_path"}
# box-6 (group 0071, level-7, GAP-6): the DELTA axis. Every level so far certifies an
# observation at a SINGLE point in time — an absolute value, or a relation between two
# values on one payload. The delta axis certifies the *change* a mutation causes: read a
# scalar BEFORE, apply a state-changing call, read the SAME scalar AFTER, assert the
# difference. This is orthogonal to box-3 (the time/step axis) — box-3 asserts the absolute
# POST-state (``status == 'done'``), whereas box-6 asserts the MAGNITUDE OF CHANGE
# (``count went up by exactly one``), the single most common acceptance criterion that had
# no executable form. ``increases_by``: ``after == before + by``. ``decreases_by``:
# ``after == before - by``. ``delta_equals``: ``after - before == by`` (signed). Each carries
# a numeric ``by`` companion. Explicit-oracle ONLY — which scalar a mutation should move, and
# by how much, can NOT be lifted from prose without guessing, and box never guesses.
SUPPORTED_MUST_BOX6_DELTA = {"increases_by", "decreases_by", "delta_equals"}
# Verbs that CHANGE state (the mutation half of a delta). A delta's ``observe`` half must be
# a read verb (``_READ_VERBS``); its ``mutate`` half must be one of these — a read can not be
# the thing that moves the counter, and a mutation can not be the thing that observes it.
_MUTATING_VERBS = {"post", "put", "patch", "delete"}
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
    # box-2 (group 0067, level-3, GAP-2): field cross-validation. When a SINGLE
    # criterion names ≥2 response fields on ONE route, ``asserts`` carries the per-field
    # assertions (each ``{json_path, must, expected}``) so one fetch certifies them all
    # in ONE test = ONE node (verify.py is untouched — this is NOT a multi-gate). Empty
    # for a single-field criterion, whose scalar ``json_path``/``must``/``expected``
    # slots stay authoritative so the level-1/2 output is byte-identical.
    asserts: tuple = ()
    # box-3 (group 0068, level-4, GAP-3): multi-step POST→GET. When a criterion's
    # satisfaction requires STATE CHANGE then OBSERVATION (create a resource, then read it
    # back), ``steps`` carries the ordered call sequence — each mutation step
    # ``{verb, full_path, body, expect_status}`` followed by a terminal read step
    # ``{verb, full_path, asserts}`` whose json_path assertions certify the effect. The whole
    # sequence runs inside ONE synthesised test = ONE node (verify.py is STILL untouched — a
    # multi-step test is one function, not a multi-gate). Empty for a single-shot criterion,
    # whose scalar/``asserts`` slots stay authoritative so the level-1/2/3 output is
    # byte-identical. Only an explicit ``steps:`` oracle populates this: a request body can
    # NOT be derived from prose without guessing, and box-0 never guesses.
    steps: tuple = ()
    # box-4 (group 0069, level-5, GAP-4): multiple INDEPENDENT routes. When a criterion's
    # satisfaction is defined as the conjunction of observations across ≥2 UNORDERED read
    # routes (``GET /projects`` shows the item AND ``GET /dashboard/summary`` reflects the
    # count), ``reads`` carries each independent read ``{verb, full_path, asserts}``. Unlike
    # ``steps`` there is NO causal order and EVERY read carries its own assertions — the whole
    # conjunction runs inside ONE synthesised test = ONE node (verify.py is STILL untouched: N
    # independent reads bundled by one criterion are one function, not a multi-gate). Empty for
    # a single-route criterion, whose scalar/``asserts`` slots stay authoritative so the
    # level-1/2/3/4 output is byte-identical. Only an explicit ``reads:`` oracle populates this
    # (which N independent routes a prose criterion means can NOT be resolved without guessing).
    reads: tuple = ()
    # box-6 (group 0071, level-7, GAP-6): the DELTA axis. When a criterion is satisfied only
    # if a mutation moves an observable by a specific amount (creating a project raises the
    # dashboard count by exactly one), ``delta`` carries a SINGLE spec dict
    # ``{observe:{verb, full_path, json_path}, mutate:{verb, full_path, body?, expect_status?},
    # must, by}``. The synthesised test reads the scalar BEFORE, mutates, reads AFTER, and
    # asserts the difference — all inside ONE function = ONE node (verify.py is STILL untouched:
    # a before/mutate/after bracket is one test, not a multi-gate). Empty for a non-delta
    # criterion, whose scalar/``asserts``/``steps``/``reads`` slots stay authoritative so the
    # level-1..6 output is byte-identical. Only an explicit ``delta:`` oracle populates this —
    # which scalar moves, and by how much, can NOT be derived from prose without guessing.
    delta: tuple = ()


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
        if not fields:               # 0 candidate fields → nothing to assert, decline
            return None
        must = infer_must(prose)
        if len(fields) == 1:
            field = fields[0]
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

        # box-2 (group 0067, level-3, GAP-2): ≥2 fields on ONE route = field
        # cross-validation. Prose can only attribute a UNIFORM, LITERAL-FREE must to every
        # field (each named field simply must be present / non-empty). A literal (equals)
        # can NOT be attributed to one of N fields unambiguously — deriving "which field ==
        # 3?" is a guess, and box-0's whole trust basis is that it never guesses — so a
        # multi-field criterion carrying a literal declines (§3 of NR0003). Per-field
        # must/expected is available only through an explicit ``asserts`` oracle.
        if must not in SUPPORTED_MUST_HTTP_NOW:          # exists / non_empty only
            return None
        if _extract_literal_expected(prose) is not None:  # literal + N fields → ambiguous
            return None
        # The container heuristic (``_response_container_key``) returns the FIRST return-dict
        # key that is not the field — built for a single leaf nested in one wrapper
        # (``modules`` inside ``{"projects": [...]}``). With ≥2 TOP-LEVEL sibling fields
        # (``{"total": .., "items": ..}``) it would wrongly latch onto a SIBLING as the
        # container (``total[].items``). Reject any container that is itself one of the named
        # fields → that field is top-level (json_path = the field). A genuine shared wrapper
        # (``projects`` for leaves ``modules``/``tags``) is not among the fields, so it stays.
        field_set = set(fields)
        asserts: list[dict[str, Any]] = []
        for f in fields:
            c = _container_for(gets, f)
            if c in field_set:
                c = None
            jp = f"{c}[].{f}" if c else f
            asserts.append({"json_path": jp, "must": must, "expected": None})
        first_c = _container_for(gets, fields[0])
        if first_c in field_set:
            first_c = None
        return AcceptanceSymptom(
            kind="http_read", source_ac_id=ac_id, verb=verb, full_path=full_path,
            json_path=asserts[0]["json_path"], field=fields[0],
            container=first_c or "", must=must, asserts=tuple(asserts))

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
        # box-3 (group 0068, level-4, GAP-3): an explicit multi-step ``steps:`` oracle
        # (POST→GET) is validated separately — its route/verb live PER STEP, not at the
        # top level, so this must win before the single-route shape below.
        if isinstance(o.get("steps"), list):
            return _validate_steps_oracle(o["steps"], codebase_root, ac_id)
        # box-6 (group 0071, level-7, GAP-6): an explicit ``delta:`` oracle brackets a
        # mutation with a before/after read of the SAME scalar. Its verbs/routes live INSIDE
        # the observe/mutate halves, not at the top level, so this wins before the single-route
        # shape below. Mutually exclusive with ``steps``/``reads`` (an author writing more than
        # one block gets ``steps`` first, then ``delta``, then ``reads`` — deterministic).
        if isinstance(o.get("delta"), dict):
            return _validate_delta_oracle(o["delta"], codebase_root, ac_id)
        # box-4 (group 0069, level-5, GAP-4): an explicit ``reads:`` oracle bundles ≥2
        # INDEPENDENT read routes. Its route/verb/asserts live PER READ, not at the top
        # level, so this must win before the single-route shape below (and after ``steps``
        # — the two blocks are mutually exclusive; ``steps`` wins if an author writes both).
        if isinstance(o.get("reads"), list):
            return _validate_reads_oracle(o["reads"], codebase_root, ac_id)
        verb = str(o.get("verb", "")).lower()
        route = o.get("route", "")
        if verb not in _SUPPORTED_VERBS:
            return None
        if not isinstance(route, str) or not route.startswith("/"):
            return None
        raw_asserts = o.get("asserts")
        if isinstance(raw_asserts, list):
            # box-2 (group 0067, level-3, GAP-2): explicit multi-field / field
            # cross-validation. Each entry is an independent ``{json_path, must, expected?}``
            # the author wrote; box-0 confirms the route grounds ONCE and lifts every assert
            # verbatim (never overrides an author's value). A single-entry list reduces to
            # the scalar shape so the output stays byte-identical to a plain json_path oracle.
            gets = _resolve_gets(route, codebase_root)
            if len(_distinct_full_paths(gets)) != 1:
                return None
            full_path = next(iter(_distinct_full_paths(gets)))
            norm: list[dict[str, Any]] = []
            for a in raw_asserts:
                no = _normalize_one_assert(a)
                if no is None:
                    return None
                norm.append(no)
            if not norm:
                return None
            # A single NON-relational assert reduces to the scalar shape so the output stays
            # byte-identical to a plain json_path oracle (levels 1-5). A single RELATIONAL
            # assert (box-5) can NOT reduce — the scalar slots have no ``other_path`` cell — so
            # it stays in ``asserts`` and routes through the multi-assert builder.
            if len(norm) == 1 and norm[0]["must"] not in SUPPORTED_MUST_BOX5_REL:
                a0 = norm[0]
                return AcceptanceSymptom(
                    kind="http_read", source_ac_id=ac_id, verb=verb, full_path=full_path,
                    json_path=a0["json_path"], must=a0["must"], expected=a0["expected"])
            return AcceptanceSymptom(
                kind="http_read", source_ac_id=ac_id, verb=verb, full_path=full_path,
                json_path=norm[0]["json_path"], must=norm[0]["must"],
                expected=norm[0].get("expected"), asserts=tuple(norm))
        must = o.get("must", "")
        json_path = o.get("json_path", "")
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


def _normalize_one_assert(a: dict[str, Any], *,
                          allow_cross_read: bool = False) -> dict[str, Any] | None:
    """Validate + lift ONE assertion entry into ``{json_path, must, expected|other_path}``.

    The single per-assert gate reused by every multi-assert shape (box-2 ``asserts:``, box-3
    terminal read, box-4 each read) so a new assertion primitive lands in all of them at once.
    A ``must`` must be supported; an ``equals`` must ship an explicit ``expected``; a box-5
    RELATIONAL must (``equals_len``/``equals_path``) must ship a NON-empty ``other_path``, must
    NOT also carry an ``expected`` literal (a value + a relation is ambiguous), and neither side
    may be an iterating ``[]`` path (a relation is over whole values / list lengths, never
    per-item). Anything malformed → ``None`` (decline).

    box-5b (group 0072, level-6b): when ``allow_cross_read`` (a multiroute ``reads`` context), a
    relational assert may carry an ``other_read`` INDEX naming which OTHER read's payload supplies
    ``other_path`` — a cross-route invariant. Cross-read RELAXES the self-reference guard (the same
    field name on a DIFFERENT route is legitimate); the index's range/self validation needs the
    read count and is deferred to :func:`_validate_reads_oracle`. An ``other_read`` outside a reads
    context (box-2/3, one payload) declines — cross-payload has no meaning there."""
    if not isinstance(a, dict):
        return None
    jp = a.get("json_path", "")
    m = a.get("must", "")
    supported = SUPPORTED_MUST_HTTP_NOW | SUPPORTED_MUST_BOX0_EXT | SUPPORTED_MUST_BOX5_REL
    if not jp or m not in supported:
        return None
    if m in SUPPORTED_MUST_BOX5_REL:
        other = a.get("other_path", "")
        if not isinstance(other, str) or not other:
            return None
        if "[]" in jp or "[]" in other:   # relation is over whole values, not per-item
            return None
        if "expected" in a:               # literal + relation → ambiguous, decline
            return None
        cross = a.get("other_read")
        if cross is not None:
            # box-5b: the RHS lives on another read's payload — valid ONLY inside a reads oracle.
            if not allow_cross_read:
                return None
            # a bool is an int subclass — reject it so ``other_read: true`` never means read 1.
            if isinstance(cross, bool) or not isinstance(cross, int) or cross < 0:
                return None
            return {"json_path": jp, "must": m, "other_path": other,
                    "other_read": cross, "expected": None}
        if other == jp:                   # within-payload relation → the two paths must differ
            return None
        return {"json_path": jp, "must": m, "other_path": other, "expected": None}
    if m == "equals" and "expected" not in a:
        return None
    return {"json_path": jp, "must": m, "expected": a.get("expected")}


def _normalize_read_asserts(step: dict[str, Any], *,
                            allow_cross_read: bool = False) -> list[dict[str, Any]] | None:
    """Lift a read step's assertions into ``[{json_path, must, expected}]`` or ``None``.

    A terminal read step carries either an ``asserts:`` list (box-2 multi-field shape) or a
    scalar ``json_path`` + ``must`` pair. Each ``must`` must be supported and an ``equals``
    must ship an explicit ``expected`` (never compares to a guess). Malformed/empty → None.
    ``allow_cross_read`` is forwarded to the per-assert gate so a box-5b cross-route relation
    (``other_read``) is accepted only in a multiroute ``reads`` context, never a single read."""
    raw = step.get("asserts")
    if isinstance(raw, list):
        norm: list[dict[str, Any]] = []
        for a in raw:
            no = _normalize_one_assert(a, allow_cross_read=allow_cross_read)
            if no is None:
                return None
            norm.append(no)
        return norm or None
    no = _normalize_one_assert(step, allow_cross_read=allow_cross_read)
    return [no] if no else None


def _validate_steps_oracle(raw_steps: list[Any], codebase_root: str,
                           ac_id: str) -> AcceptanceSymptom | None:
    """L §5 (group 0068) — validate an explicit multi-step POST→GET oracle, or decline.

    ``steps`` is an ORDERED list: every entry but the last is a MUTATION step
    ``{verb, route, body?, expect_status?}`` (a POST/PUT/PATCH/DELETE that changes state),
    and the LAST is a READ step ``{verb, route, asserts?|json_path+must}`` whose json_path
    assertions certify the effect. box-0 confirms EVERY step's route grounds to exactly one
    full_path (reusing the retriever grounding the single-shot path uses) and lifts the
    author's verbs/bodies/asserts verbatim — it never overrides or guesses a value. Requires
    ≥2 steps (a single step is the level-1/2/3 shape, handled elsewhere) and a terminal step
    carrying ≥1 assertion; anything unsupported or ungrounded declines."""
    if len(raw_steps) < 2:
        return None
    steps: list[dict[str, Any]] = []
    last = len(raw_steps) - 1
    for i, st in enumerate(raw_steps):
        if not isinstance(st, dict):
            return None
        verb = str(st.get("verb", "")).lower()
        route = st.get("route", "")
        if verb not in _SUPPORTED_VERBS:
            return None
        if not isinstance(route, str) or not route.startswith("/"):
            return None
        gets = _resolve_gets(route, codebase_root)
        paths = _distinct_full_paths(gets)
        if len(paths) != 1:
            return None
        full_path = next(iter(paths))
        if i == last:
            norm = _normalize_read_asserts(st)
            if not norm:
                return None
            steps.append({"verb": verb, "full_path": full_path, "asserts": tuple(norm)})
        else:
            body = st.get("body")
            if body is not None and not isinstance(body, (dict, list)):
                return None  # a request body must be JSON-shaped (never a guess/scalar)
            expect = st.get("expect_status")
            if expect is not None and not isinstance(expect, int):
                return None
            steps.append({"verb": verb, "full_path": full_path,
                          "body": body, "expect_status": expect})
    terminal = steps[-1]
    a0 = terminal["asserts"][0]
    # Mirror the terminal read's first assert into the scalar slots for the fn name /
    # rationale; ``asserts`` stays empty so ``steps`` is the sole multi-step signal.
    return AcceptanceSymptom(
        kind="http_read", source_ac_id=ac_id, verb=terminal["verb"],
        full_path=terminal["full_path"], json_path=a0["json_path"], must=a0["must"],
        expected=a0["expected"], steps=tuple(steps))


# Read-only verbs a level-5 independent read may use. A mutating verb inside ``reads``
# would change state — that belongs in a level-4 ordered ``steps`` sequence, not an
# unordered observation set — so it declines here, keeping the route axis a pure read.
_READ_VERBS = {"get", "head"}


def _validate_reads_oracle(raw_reads: list[Any], codebase_root: str,
                           ac_id: str) -> AcceptanceSymptom | None:
    """L §3 (group 0069) — validate an explicit multi-route ``reads:`` oracle, or decline.

    ``reads`` is an UNORDERED list of ≥2 INDEPENDENT read routes; each entry is a read
    ``{verb, route, asserts?|json_path+must}`` whose json_path assertions certify one facet
    of the criterion. box-0 confirms EVERY read's route grounds to exactly one full_path
    (reusing the retriever grounding the single-shot path uses), rejects any mutating verb
    (state change belongs in ``steps``), and lifts the author's verbs/asserts verbatim — it
    never overrides or guesses a value. Requires ≥2 reads (a single read is the level-3
    shape, handled elsewhere) and each read carrying ≥1 assertion; anything unsupported or
    ungrounded declines."""
    if len(raw_reads) < 2:
        return None
    reads: list[dict[str, Any]] = []
    for rd in raw_reads:
        if not isinstance(rd, dict):
            return None
        verb = str(rd.get("verb", "")).lower()
        route = rd.get("route", "")
        if verb not in _READ_VERBS:
            return None
        if not isinstance(route, str) or not route.startswith("/"):
            return None
        gets = _resolve_gets(route, codebase_root)
        paths = _distinct_full_paths(gets)
        if len(paths) != 1:
            return None
        full_path = next(iter(paths))
        norm = _normalize_read_asserts(rd, allow_cross_read=True)
        if not norm:
            return None
        reads.append({"verb": verb, "full_path": full_path, "asserts": tuple(norm)})
    # box-5b (group 0072, level-6b): a cross-route relation's ``other_read`` index must select a
    # DIFFERENT, in-range read. An out-of-range index or a self-index (a relation of a read with
    # ITSELF is a within-payload box-5 relation, which omits ``other_read``) declines — the whole
    # criterion is a no-go candidate rather than a test that references an undefined payload.
    n = len(reads)
    for idx, rd_norm in enumerate(reads):
        for a in rd_norm["asserts"]:
            if "other_read" in a:
                oi = a["other_read"]
                if oi >= n or oi == idx:
                    return None
    first = reads[0]
    a0 = first["asserts"][0]
    # Mirror the first read's first assert into the scalar slots for the fn name /
    # rationale; ``reads`` stays the sole multi-route signal.
    return AcceptanceSymptom(
        kind="http_read", source_ac_id=ac_id, verb=first["verb"],
        full_path=first["full_path"], json_path=a0["json_path"], must=a0["must"],
        expected=a0["expected"], reads=tuple(reads))


def _validate_delta_oracle(o: dict[str, Any], codebase_root: str,
                           ac_id: str) -> AcceptanceSymptom | None:
    """L (group 0071, level-7) — validate an explicit before/after ``delta:`` oracle, or decline.

    ``o`` = ``{observe, mutate, must, by}``. ``observe`` is a READ ``{verb, route, json_path}``
    naming the SCALAR to bracket; ``mutate`` is a state change ``{verb, route, body?,
    expect_status?}``. box-0 grounds BOTH routes to exactly one full_path (reusing the retriever
    the single-shot path uses), requires ``observe`` to use a read verb and ``mutate`` a mutating
    verb, a supported ``must`` and a numeric ``by``, and a NON-iterating ``json_path`` (a delta is
    over a scalar count, never per item). It lifts the author's verbs/body/``by`` verbatim — never
    overrides or guesses. Anything unsupported or ungrounded declines (``None``)."""
    if not isinstance(o, dict):
        return None
    observe = o.get("observe")
    mutate = o.get("mutate")
    if not isinstance(observe, dict) or not isinstance(mutate, dict):
        return None
    must = o.get("must", "")
    if must not in SUPPORTED_MUST_BOX6_DELTA:
        return None
    by = o.get("by")
    # a bool is an int subclass — reject it so ``by: true`` never silently means +1.
    if isinstance(by, bool) or not isinstance(by, (int, float)):
        return None
    # observe half — a read of the scalar to bracket (must be a read verb, grounds once).
    o_verb = str(observe.get("verb", "")).lower()
    o_route = observe.get("route", "")
    o_jp = observe.get("json_path", "")
    if o_verb not in _READ_VERBS:
        return None
    if not isinstance(o_route, str) or not o_route.startswith("/"):
        return None
    if not o_jp or "[]" in o_jp:          # a delta is over a scalar, not an iterating path
        return None
    o_paths = _distinct_full_paths(_resolve_gets(o_route, codebase_root))
    if len(o_paths) != 1:
        return None
    o_full = next(iter(o_paths))
    # mutate half — a state change (must be a mutating verb, grounds once).
    m_verb = str(mutate.get("verb", "")).lower()
    m_route = mutate.get("route", "")
    if m_verb not in _MUTATING_VERBS:
        return None
    if not isinstance(m_route, str) or not m_route.startswith("/"):
        return None
    body = mutate.get("body")
    if body is not None and not isinstance(body, (dict, list)):
        return None  # a request body must be JSON-shaped (never a guess/scalar)
    expect = mutate.get("expect_status")
    if expect is not None and (isinstance(expect, bool) or not isinstance(expect, int)):
        return None
    m_paths = _distinct_full_paths(_resolve_gets(m_route, codebase_root))
    if len(m_paths) != 1:
        return None
    m_full = next(iter(m_paths))
    spec = {
        "observe": {"verb": o_verb, "full_path": o_full, "json_path": o_jp},
        "mutate": {"verb": m_verb, "full_path": m_full, "body": body,
                   "expect_status": expect},
        "must": must, "by": by,
    }
    # Mirror the observe read into the scalar slots for the fn name / rationale; ``delta`` stays
    # the sole delta signal so the non-delta output paths are byte-identical.
    return AcceptanceSymptom(
        kind="http_read", source_ac_id=ac_id, verb=o_verb, full_path=o_full,
        json_path=o_jp, must=must, delta=(spec,))


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


def _emit_assert_lines(a: dict[str, Any], root_name: str = "payload",
                       other_root_name: str | None = None) -> str:
    """Assertion source for ONE assert entry against ``root_name`` (box-2/3/4/5/5b shared).

    The single place a ``must`` becomes pytest lines, so every multi-assert builder (multi-field,
    multi-step terminal read, multi-route each read) grows the box-5 RELATION axis at once and
    stays byte-identical for the existing musts. exists / non_empty reuse ⑦'s
    ``build_json_path_assertions``; ``equals`` appends one value-vs-literal assert; a box-5
    ``equals_len`` guards the companion path is a list then asserts ``left == len(other)``;
    ``equals_path`` asserts ``left == other``. Raises on an iterating ``[]`` scalar path (via
    ``_path_expr``) so the caller fails open — a relation is over whole values, not per item.

    box-5b (group 0072, level-6b): ``other_root_name`` names the payload the relation's RHS
    (``other_path``) is read from. It defaults to ``root_name`` (within-payload box-5, byte-identical
    to before); a cross-route relation passes the OTHER read's payload variable so the RHS resolves
    against the right response."""
    jp = a["json_path"]
    must = a["must"]
    orn = other_root_name or root_name
    if must in SUPPORTED_MUST_BOX5_REL:
        other = a["other_path"]
        left_exists = build_json_path_assertions(jp, "exists", root_name=root_name)
        other_exists = build_json_path_assertions(other, "exists", root_name=orn)
        left_expr = _path_expr(jp, root_name)          # raises on '[]' → caller fails open
        other_expr = _path_expr(other, orn)            # raises on '[]' → caller fails open
        lines = [left_exists, other_exists]
        if must == "equals_len":
            lines.append(
                f"    assert isinstance({other_expr}, list), "
                f"{repr(f'acceptance relation: {other!r} must be an array')}")
            lines.append(
                f"    assert {left_expr} == len({other_expr}), "
                "'acceptance relation mismatch (count != length)'")
        else:  # equals_path
            lines.append(
                f"    assert {left_expr} == {other_expr}, 'acceptance relation mismatch'")
        return "\n".join(lines)
    block = build_json_path_assertions(
        jp, "exists" if must == "equals" else must, root_name=root_name)
    if must == "equals":
        expr = _path_expr(jp, root_name)               # raises on '[]' → caller fails open
        block = f"{block}\n    assert {expr} == {a['expected']!r}, 'acceptance value mismatch'"
    return block


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


def _multifield_test_name(verb: str, route: str, asserts: tuple) -> str:
    """A deterministic test-fn name for a multi-field gate (route + every json_path)."""
    sig = "_".join(a.get("json_path", "") for a in asserts)
    raw = f"test_acceptance_multifield_{verb}_{route}_{sig}"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name or "test_acceptance_multifield"


def _build_http_multifield_test(route: str, verb: str, asserts: tuple,
                                app_fixture: str) -> str:
    """A TestClient test asserting SEVERAL json_paths on ONE response (box-2, level-3).

    Field cross-validation from a single acceptance criterion: one fetch, then each
    field's assertion block accumulated so the criterion certifies as a unit. exists /
    non_empty reuse ⑦'s ``build_json_path_assertions``; ``equals`` appends one value
    assert (non-iterating paths only, like :func:`_build_http_equals_test`). Declines
    (raises → caller fails open) on an iterating ``equals`` path — an array is not a
    value compare. This stays ONE test = ONE node; verify.py is untouched."""
    blocks: list[str] = [_emit_assert_lines(a) for a in asserts]
    fn = _multifield_test_name(verb, route, asserts)
    body = "\n".join(blocks)
    return (
        '"""Generated acceptance multi-field gate (box-2, level-3).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        f"Certifies {len(asserts)} field assertions from ONE criterion on a single fetch.\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"    response = {app_fixture}.{verb}({route!r})\n"
        "    assert 200 <= response.status_code < 300, response.text\n"
        "    payload = response.json()\n"
        f"{body}\n"
    )


def _multistep_test_name(steps: tuple) -> str:
    """A deterministic test-fn name for a multi-step gate (every step's verb + route)."""
    sig = "_".join(f"{s['verb']}_{s['full_path']}" for s in steps)
    raw = f"test_acceptance_multistep_{sig}"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name or "test_acceptance_multistep"


def _build_http_multistep_test(steps: tuple, app_fixture: str) -> str:
    """A TestClient test running an ORDERED POST→GET sequence (box-3, level-4).

    Multi-step acceptance from a single criterion: each mutation step issues its request
    (with an optional JSON body) and asserts its status, then the terminal read step fetches
    and asserts every json_path — so create→persist→read-back certifies as a unit. exists /
    non_empty reuse ⑦'s ``build_json_path_assertions``; ``equals`` appends one value assert
    (non-iterating paths only, like :func:`_build_http_equals_test`). Declines (raises →
    caller fails open) on an iterating ``equals`` path. The whole sequence lives in ONE test
    function = ONE node; verify.py is untouched — this is not a multi-gate."""
    lines: list[str] = []
    n = len(steps)
    for i, s in enumerate(steps):
        var = f"r{i + 1}"
        verb = s["verb"]
        route = s["full_path"]
        if i == n - 1:                       # terminal read step
            lines.append(f"    {var} = {app_fixture}.{verb}({route!r})")
            lines.append(f"    assert 200 <= {var}.status_code < 300, {var}.text")
            lines.append(f"    payload = {var}.json()")
            for a in s["asserts"]:
                lines.append(_emit_assert_lines(a))
        else:                                # mutation step (POST/PUT/PATCH/DELETE)
            body = s.get("body")
            call = (f"{app_fixture}.{verb}({route!r}, json={body!r})"
                    if body is not None else f"{app_fixture}.{verb}({route!r})")
            lines.append(f"    {var} = {call}")
            exp = s.get("expect_status")
            if exp is not None:
                lines.append(f"    assert {var}.status_code == {exp}, {var}.text")
            else:
                lines.append(f"    assert 200 <= {var}.status_code < 300, {var}.text")
    fn = _multistep_test_name(steps)
    body = "\n".join(lines)
    return (
        '"""Generated acceptance multi-step gate (box-3, level-4).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        f"Certifies a {n}-step POST→GET sequence from ONE criterion.\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"{body}\n"
    )


def _reads_have_cross(reads: tuple) -> bool:
    """True when any read carries a box-5b cross-route relational assert (``other_read``)."""
    return any("other_read" in a for r in reads for a in r["asserts"])


def _multiroute_test_name(reads: tuple) -> str:
    """A deterministic test-fn name for a multi-route gate (every read's verb + route).

    A cross-route relation (box-5b) uses a distinct ``multiroute_relation`` stem so it triangulates
    against a plain independent-route conjunction (box-4)."""
    sig = "_".join(f"{r['verb']}_{r['full_path']}" for r in reads)
    stem = "multiroute_relation" if _reads_have_cross(reads) else "multiroute"
    raw = f"test_acceptance_{stem}_{sig}"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name or f"test_acceptance_{stem}"


def _build_http_multiroute_test(reads: tuple, app_fixture: str) -> str:
    """A TestClient test issuing ≥2 INDEPENDENT reads on one response set (box-4, level-5).

    Route cross-validation from a single acceptance criterion: each independent read fetches
    its own route into its own ``payload`` and asserts every json_path — so a feature whose
    contract spans endpoints (the created item appears in the list AND the summary count)
    certifies as a unit. There is NO ordering between reads (unlike level-4 ``steps``).
    exists / non_empty reuse ⑦'s ``build_json_path_assertions``; ``equals`` appends one value
    assert (non-iterating paths only, like :func:`_build_http_equals_test`). Declines (raises
    → caller fails open) on an iterating ``equals`` path. The whole set lives in ONE test
    function = ONE node; verify.py is untouched — this is not a multi-gate."""
    lines: list[str] = []
    for i, r in enumerate(reads):
        var = f"r{i + 1}"
        payload = f"payload{i + 1}"
        verb = r["verb"]
        route = r["full_path"]
        lines.append(f"    {var} = {app_fixture}.{verb}({route!r})")
        lines.append(f"    assert 200 <= {var}.status_code < 300, {var}.text")
        lines.append(f"    {payload} = {var}.json()")
        for a in r["asserts"]:
            lines.append(_emit_assert_lines(a, payload))
    fn = _multiroute_test_name(reads)
    body = "\n".join(lines)
    n = len(reads)
    return (
        '"""Generated acceptance multi-route gate (box-4, level-5).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        f"Certifies {n} INDEPENDENT read routes from ONE criterion.\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"{body}\n"
    )


def _build_http_multiroute_relation_test(reads: tuple, app_fixture: str) -> str:
    """A TestClient test asserting a relation ACROSS ≥2 independent reads (box-5b, level-6b).

    box-5 (relation) composed with box-4 (multi-route): a criterion satisfied only when an observed
    value on one route agrees with an observed value on ANOTHER route (a summary count equals the
    length of the list route it summarises). Because a relation's RHS lives on a DIFFERENT payload,
    this builder runs in TWO passes: pass one fetches EVERY read into its own ``payload{i}`` (so all
    payloads are defined), pass two emits the assertions — a within-payload assert against its own
    payload, a cross-route relation against its own payload for the LHS and read ``other_read``'s
    payload for the RHS. exists / non_empty / equals reuse the shared emitter; the whole set lives
    in ONE test function = ONE node (verify.py is untouched — this is not a multi-gate). This is why
    box-4's builder is left alone: its fetch-then-assert-inline ordering can NOT reference a payload
    that has not been fetched yet, so cross-route gets its own two-pass builder."""
    lines: list[str] = []
    for i, r in enumerate(reads):                      # pass 1: fetch every payload
        var = f"r{i + 1}"
        payload = f"payload{i + 1}"
        lines.append(f"    {var} = {app_fixture}.{r['verb']}({r['full_path']!r})")
        lines.append(f"    assert 200 <= {var}.status_code < 300, {var}.text")
        lines.append(f"    {payload} = {var}.json()")
    for i, r in enumerate(reads):                      # pass 2: assert across payloads
        payload = f"payload{i + 1}"
        for a in r["asserts"]:
            if "other_read" in a:
                other_payload = f"payload{a['other_read'] + 1}"
                lines.append(_emit_assert_lines(a, payload, other_payload))
            else:
                lines.append(_emit_assert_lines(a, payload))
    fn = _multiroute_test_name(reads)
    body = "\n".join(lines)
    n = len(reads)
    return (
        '"""Generated acceptance cross-route relation gate (box-5b, level-6b).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        f"Certifies a relational invariant spanning {n} INDEPENDENT read routes from ONE criterion.\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"{body}\n"
    )


def _delta_test_name(spec: dict[str, Any]) -> str:
    """A deterministic test-fn name for a delta gate (mutate route + observe route + must)."""
    ov, mu = spec["observe"], spec["mutate"]
    raw = (f"test_acceptance_delta_{mu['verb']}_{mu['full_path']}"
           f"_{ov['verb']}_{ov['full_path']}_{spec['must']}")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    return name or "test_acceptance_delta"


def _build_http_delta_test(spec: dict[str, Any], app_fixture: str) -> str:
    """A TestClient test asserting a mutation moves a scalar by ``by`` (box-6, level-7).

    Reads the scalar BEFORE, applies the mutation (asserting its status), reads the SAME scalar
    AFTER, and asserts the SIGNED difference — so "creating a project raises the count by exactly
    one" certifies by execution. exists guards on both reads reuse ⑦'s
    ``build_json_path_assertions``; ``_path_expr`` builds the KeyError-safe before/after accessors
    (raises on an iterating ``[]`` path → caller fails open). The whole bracket lives in ONE test
    function = ONE node; verify.py is untouched — this is not a multi-gate."""
    observe, mutate = spec["observe"], spec["mutate"]
    must, by = spec["must"], spec["by"]
    jp = observe["json_path"]
    o_route, o_verb = observe["full_path"], observe["verb"]
    m_route, m_verb = mutate["full_path"], mutate["verb"]
    before_exists = build_json_path_assertions(jp, "exists", root_name="before")
    after_exists = build_json_path_assertions(jp, "exists", root_name="after")
    before_expr = _path_expr(jp, "before")     # raises on '[]' → caller fails open
    after_expr = _path_expr(jp, "after")
    body = mutate.get("body")
    call = (f"{app_fixture}.{m_verb}({m_route!r}, json={body!r})"
            if body is not None else f"{app_fixture}.{m_verb}({m_route!r})")
    exp = mutate.get("expect_status")
    status_line = (f"    assert r1.status_code == {exp}, r1.text" if exp is not None
                   else "    assert 200 <= r1.status_code < 300, r1.text")
    if must == "increases_by":
        rel = (f"    assert {after_expr} == {before_expr} + {by!r}, "
               f"'acceptance delta mismatch (expected +{by})'")
    elif must == "decreases_by":
        rel = (f"    assert {after_expr} == {before_expr} - {by!r}, "
               f"'acceptance delta mismatch (expected -{by})'")
    else:  # delta_equals (signed)
        rel = (f"    assert {after_expr} - {before_expr} == {by!r}, "
               f"'acceptance delta mismatch (expected delta {by})'")
    fn = _delta_test_name(spec)
    lines = [
        f"    r0 = {app_fixture}.{o_verb}({o_route!r})",
        "    assert 200 <= r0.status_code < 300, r0.text",
        "    before = r0.json()",
        before_exists,
        f"    r1 = {call}",
        status_line,
        f"    r2 = {app_fixture}.{o_verb}({o_route!r})",
        "    assert 200 <= r2.status_code < 300, r2.text",
        "    after = r2.json()",
        after_exists,
        rel,
    ]
    body_src = "\n".join(lines)
    return (
        '"""Generated acceptance delta gate (box-6, level-7).\n\n'
        f"Requires the existing pytest TestClient fixture {app_fixture!r}.\n"
        f"Certifies {m_verb.upper()} {m_route} moves {jp!r} on {o_route} ({must} {by}).\n"
        '"""\n\n'
        f"def {fn}({app_fixture}):\n"
        f"{body_src}\n"
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
    if symptom.delta:                  # box-6 level-7 delta gate
        return _delta_test_name(symptom.delta[0])
    if symptom.reads:                  # box-4 level-5 multi-route gate
        return _multiroute_test_name(symptom.reads)
    if symptom.steps:                  # box-3 level-4 multi-step gate
        return _multistep_test_name(symptom.steps)
    if symptom.asserts:                # box-2 level-3 multi-field / box-5 level-6 relation gate
        return _multifield_test_name(symptom.verb, symptom.full_path, symptom.asserts)
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
            if symptom.delta:                  # box-6 level-7: before/mutate/after delta
                body = _build_http_delta_test(symptom.delta[0], fixture)
                slug = _slug((symptom.full_path or symptom.field) + "_delta")
            elif symptom.reads:                # box-4 level-5 / box-5b level-6b multi-route
                if _reads_have_cross(symptom.reads):   # cross-route relation → two-pass builder
                    body = _build_http_multiroute_relation_test(symptom.reads, fixture)
                    slug = _slug((symptom.full_path or symptom.field) + "_multiroute_relation")
                else:
                    body = _build_http_multiroute_test(symptom.reads, fixture)
                    slug = _slug((symptom.full_path or symptom.field) + "_multiroute")
            elif symptom.steps:                # box-3 level-4: multi-step POST→GET
                body = _build_http_multistep_test(symptom.steps, fixture)
                slug = _slug((symptom.full_path or symptom.field) + "_multistep")
            elif symptom.asserts:              # box-2 level-3 field / box-5 level-6 relation
                body = _build_http_multifield_test(
                    symptom.full_path, symptom.verb, symptom.asserts, fixture)
                is_rel = any(a.get("must") in SUPPORTED_MUST_BOX5_REL
                             for a in symptom.asserts)
                slug = _slug((symptom.full_path or symptom.field)
                             + ("_relation" if is_rel else "_multifield"))
            elif symptom.must == "equals":
                body = _build_http_equals_test(
                    symptom.full_path, symptom.verb, symptom.json_path,
                    symptom.expected, fixture)
                slug = _slug(symptom.field or symptom.full_path)
            else:
                body = build_http_shape_test(
                    symptom.full_path, symptom.verb,
                    {"json_path": symptom.json_path, "must": symptom.must},
                    app_fixture=fixture)
                slug = _slug(symptom.field or symptom.full_path)
            content = (setup_block.rstrip() + "\n\n\n" + body) if setup_block else body
    except Exception:
        return None  # scaffold/path rejected the contract → fail-open

    fn = _test_fn_name(symptom)
    rel = f"{test_dir.rstrip('/')}/test_acceptance_{slug}.py"
    if symptom.kind == "unit_value":
        rationale = (f"acceptance unit test (box-0, AC {symptom.source_ac_id}): "
                     f"{symptom.target} must equal {symptom.expected!r}; RED until the "
                     "feature is built — certified by the red→green run.")
    elif symptom.delta:
        spec = symptom.delta[0]
        ov, mu = spec["observe"], spec["mutate"]
        rationale = (f"acceptance delta red test (box-6 level-7, AC {symptom.source_ac_id}): "
                     f"{mu['verb'].upper()} {mu['full_path']} moves {ov['json_path']!r} on "
                     f"{ov['full_path']} — {spec['must']} {spec['by']}; RED until the feature "
                     "is built — certified by one red→green run.")
    elif symptom.reads:
        routes = ", ".join(f"{r['verb'].upper()} {r['full_path']}" for r in symptom.reads)

        def _read_facet(a: dict[str, Any]) -> str:
            if a["must"] in SUPPORTED_MUST_BOX5_REL:
                rel = "len" if a["must"] == "equals_len" else "value"
                tgt = f"{rel}({a['other_path']!r}"
                tgt += f" on read #{a['other_read'] + 1})" if "other_read" in a else ")"
                return f"{a['json_path']!r} must equal {tgt}"
            return (f"{a['json_path']!r} must {a['must']}"
                    + (f" == {a['expected']!r}" if a['must'] == "equals" else ""))
        facets = "; ".join(", ".join(_read_facet(a) for a in r["asserts"])
                           for r in symptom.reads)
        cross = _reads_have_cross(symptom.reads)
        label = "box-5b level-6b cross-route relation" if cross else "box-4 level-5"
        rationale = (f"acceptance multi-route red test ({label}, AC "
                     f"{symptom.source_ac_id}): {routes} — {facets}; RED until the feature "
                     "is built — certified by one red→green run.")
    elif symptom.steps:
        seq = " → ".join(f"{s['verb'].upper()} {s['full_path']}" for s in symptom.steps)
        term = ", ".join(f"{a['json_path']!r} must {a['must']}"
                         + (f" == {a['expected']!r}" if a['must'] == "equals" else "")
                         for a in symptom.steps[-1]["asserts"])
        rationale = (f"acceptance multi-step red test (box-3 level-4, AC "
                     f"{symptom.source_ac_id}): {seq} — then {term}; RED until the feature "
                     "is built — certified by one red→green run.")
    elif symptom.asserts:
        def _facet(a: dict[str, Any]) -> str:
            m = a["must"]
            if m == "equals":
                return f"{a['json_path']!r} must equals == {a['expected']!r}"
            if m in SUPPORTED_MUST_BOX5_REL:
                rel = "len" if m == "equals_len" else "value"
                return f"{a['json_path']!r} must equal {rel}({a['other_path']!r})"
            return f"{a['json_path']!r} must {m}"
        paths = ", ".join(_facet(a) for a in symptom.asserts)
        is_rel = any(a.get("must") in SUPPORTED_MUST_BOX5_REL for a in symptom.asserts)
        label = ("box-5 level-6 relation" if is_rel else "box-2 level-3")
        rationale = (f"acceptance {'relational' if is_rel else 'multi-field'} red test ({label}, AC "
                     f"{symptom.source_ac_id}): {symptom.verb.upper()} {symptom.full_path} — "
                     f"{paths}; RED until the feature is built — certified by one red→green run.")
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
