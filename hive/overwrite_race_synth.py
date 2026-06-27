"""Overwrite-race guarded-write lowering + behaviour oracle — lever L3 (hivework.default.0057.0004-T).

Why this module exists
----------------------
NR0003 (hivework.default.0057) confirmed the 0062 class — a client-side write-write race
on ONE reactive ref (``mentionCopy``) written by a LIVE event handler (``_onMentionCopied``)
AND an ASYNC refetch (``fetchMentionCopy``) — is DETECTED soundly (fanout.py contract item 5
+ assemble.py time-signature ranking, FULL PASS on 0056), but the pipeline could never
AUTO-FIX it (apply RED→GREEN) because the fix-synthesis backend has three holes:

  ① specify carries NO race-class → guarded-write lowering template. Its only red-test /
     lowering assets (``http_shape_synth`` lever ⑦, ``write_sink_synth`` lever L2) are
     FK-misrouting / HTTP-shape specific (call-site / callee swap). The 0062 fix is a
     GUARD INSERTION: a generation token threaded through both writers so a late stale
     async write is discarded.
  ② there is NO behaviour oracle for the race class. ``write_sink_synth`` asserts "must not
     500", ``http_shape_synth`` asserts "field not empty"; neither can express "after two
     writers race, the final state == the latest write". Without an INDEPENDENT oracle
     ``verify.py`` non-bitingly rejects (no certified RED→GREEN).
  ③ the only sibling-consistency gate (``specify._apply_same_facet_consistency_gate``, L1)
     bundles FK callee-swap siblings; it has no equivalent for "the guard's three symbols
     (token decl + live-writer advance + async-writer stale-reject) must land TOGETHER, or
     the guard is inert".

This module closes ① and ②; the matching ③ gate lives in ``specify`` as
``_apply_guard_consistency_gate`` (mirroring L1's home).

The guarded-write shape (mirrors the SAME-FILE ``docFetchGeneration`` guard 0062 already
uses for the document detail, just never applied to the ``mentionCopy`` state):

    let <ref>Generation = 0
    function _shouldReplace<Ref>(generation: number): boolean {
      return generation === <ref>Generation
    }
    // live writer (event handler / optimistic update): advance so any in-flight async is stale
    function _onMentionCopied(...) { <ref>Generation += 1; <ref>.value = <live value> }
    // async writer (refetch / SSE / poll): capture at start, reject if superseded after await
    async function fetchMentionCopy(...) {
      const fetchGeneration = ++<ref>Generation
      ... await ...
      if (!_shouldReplace<Ref>(fetchGeneration)) return
      <ref>.value = <value derived from the awaited response>
    }

Division of labour (mirrors levers ⑦ / L2)
------------------------------------------
* This module RECOGNISES the symptom from the LIVE file (a ref written by exactly one async
  clobber-writer whose assignment derives from an awaited response, plus ≥1 live writer) and
  builds BOTH the deterministic guarded-write edits (①) and the executable last-write-wins
  red test (②).
* The HARNESS (a vitest ``setup_block`` that constructs the two writers over the ref) seeds
  what the oracle drives — exactly as lever ⑦ requires an HTTP harness, L3 declines fail-open
  for the ORACLE when no runnable harness is supplied rather than emit a test that errors.
  The LOWERING (①) needs no harness — it is a pure source transform of the live anchors.
* THE TEST IS NOT A GATE. ``verify.py`` certifies it by EXECUTION: a test green WITHOUT the
  guard is rejected as non-biting, so a mis-seeded harness can only fail to certify, never
  wave a bad fix through.

Everything here is pure-local, deterministic, zero model cost, fail-open (any missing piece
→ ``None`` / no edits), and never raises.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hive.overwrite_race_synth")

# --- recognition regexes (Vue <script setup> / TS) --------------------------------------

# A reactive ref declaration: ``const mentionCopy = ref<...>(...)`` (ref / shallowRef).
# The captured name is the state both writers race on.
_REF_DECL_RE = re.compile(
    r"""^[ \t]*const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:shallowRef|ref)\b""", re.MULTILINE)

# An assignment to that ref's ``.value`` (the write-site marker). Built per-ref.
def _assign_re(ref: str) -> re.Pattern[str]:
    return re.compile(r"^([ \t]*)" + re.escape(ref) + r"\.value\s*=\s*(.+)$", re.MULTILINE)

# A function header (named ``function`` or arrow ``const x = (...) =>``), used to find the
# enclosing writer of an assignment and whether it is ``async``.
_FN_HEADER_RE = re.compile(
    r"""^([ \t]*)(?:(async)\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(""", re.MULTILINE)
_ARROW_HEADER_RE = re.compile(
    r"""^([ \t]*)const\s+([A-Za-z_$][\w$]*)\s*=\s*(async\s+)?\([^)]*\)\s*(?::[^=]+)?=>""",
    re.MULTILINE)

@dataclass(frozen=True)
class WriterSite:
    """One function that writes ``<ref>.value``.

    ``name`` is the function name; ``is_async`` whether it is declared ``async``; ``header``
    is the exact header line text (the anchor for inserting a generation capture); ``assign``
    is the exact ``<ref>.value = …`` line text (the anchor for inserting the live-writer
    advance or the async stale-reject); ``indent`` is the assignment's leading whitespace;
    ``derives_from_await`` is True when the assignment RHS references a variable bound from an
    ``await`` in the same function (the CLOBBER signature — a stale response overwriting live
    state), False for a constant/reset write."""

    name: str
    is_async: bool
    header: str
    assign: str
    indent: str
    derives_from_await: bool


@dataclass(frozen=True)
class RaceSymptom:
    """A recognised client write-write overwrite-race on one reactive ref.

    ``ref`` is the state both writers race on; ``decl`` is the exact ref-declaration line
    (anchor for the generation-token + helper insert); ``async_writer`` is the unique async
    clobber-writer (its stale write must be guarded); ``live_writers`` are the live/optimistic
    writers (each must ADVANCE the generation so an in-flight async write becomes stale)."""

    ref: str
    decl: str
    async_writer: WriterSite
    live_writers: tuple[WriterSite, ...] = field(default_factory=tuple)
    source_file: str = ""  # codebase-relative path of the file the writers live in

    @property
    def gen_token(self) -> str:
        return f"{self.ref}Generation"

    @property
    def guard_fn(self) -> str:
        return "_shouldReplace" + self.ref[:1].upper() + self.ref[1:]


def _enclosing_writer(text: str, assign_start: int) -> tuple[str, bool, str] | None:
    """``(fn_name, is_async, header_line)`` of the function enclosing the assignment at
    ``assign_start``, or ``None`` when no header precedes it. The nearest header above the
    assignment wins (function bodies do not overlap in this scan). Never raises."""
    best: tuple[int, str, bool, str] | None = None
    for rx in (_FN_HEADER_RE, _ARROW_HEADER_RE):
        for m in rx.finditer(text):
            if m.start() >= assign_start:
                break
            if rx is _FN_HEADER_RE:
                is_async = bool(m.group(2))
                name = m.group(3)
            else:
                is_async = bool(m.group(3))
                name = m.group(2)
            header = text[m.start(): text.find("\n", m.start())]
            if best is None or m.start() > best[0]:
                best = (m.start(), name, is_async, header)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _next_header_offset(text: str, after: int) -> int:
    """Offset of the next top-level function header after ``after`` (end of the current
    function body), or ``len(text)``. Never raises."""
    nxt = len(text)
    for rx in (_FN_HEADER_RE, _ARROW_HEADER_RE):
        for m in rx.finditer(text, after + 1):
            nxt = min(nxt, m.start())
            break
    return nxt


def _collect_writer_sites(text: str, ref: str) -> list[WriterSite]:
    """Every function that assigns ``<ref>.value`` in ``text``, classified. Never raises.

    ``derives_from_await`` (the CLOBBER signature) is decided by POSITION: an assignment that
    sits AFTER the first ``await`` in its enclosing function commits a value the await
    produced (or could be raced by it) — a stale response overwriting live state. An
    assignment BEFORE the await is the intentional "blank then load" reset, not a clobber.
    This cleanly separates ``fetchMentionCopy`` (assigns after its GET) from ``fetchDoc``
    (``= null`` reset before its GET) on the real 0062 file."""
    sites: list[WriterSite] = []
    arx = _assign_re(ref)
    # Pre-compute function header offsets so we can bound each body.
    headers = sorted(
        [(m.start(), m) for m in _FN_HEADER_RE.finditer(text)]
        + [(m.start(), m) for m in _ARROW_HEADER_RE.finditer(text)])
    header_offsets = [h[0] for h in headers]
    for m in arx.finditer(text):
        enc = _enclosing_writer(text, m.start())
        if enc is None:
            continue
        name, is_async, header = enc
        indent = m.group(1)
        nl = text.find("\n", m.start())
        assign_line = text[m.start(): nl] if nl != -1 else text[m.start():]
        derives = False
        if is_async:
            # body span = [enclosing header, next header)
            h_off = max([o for o in header_offsets if o < m.start()], default=0)
            b_end = _next_header_offset(text, h_off)
            body_before = text[h_off: m.start()]
            first_await = body_before.find("await ")
            derives = first_await != -1 and (h_off + first_await) < m.start() <= b_end
        sites.append(WriterSite(name=name, is_async=is_async, header=header,
                                assign=assign_line, indent=indent, derives_from_await=derives))
    return sites


def detect_overwrite_race_symptom(honey_text: str, codebase_root: str,
                                  *, source_file: str | None = None) -> RaceSymptom | None:
    """Recognise the 0062 overwrite-race symptom from the LIVE file, or ``None`` (fail-open).

    Fires ONLY when every piece resolves UNAMBIGUOUSLY:
      * the honey names a client component file (``.vue``/``.ts``) that exists under the
        codebase root (or ``source_file`` is supplied),
      * that file declares exactly one reactive ref written by MORE THAN ONE function,
      * exactly ONE of those writers is an async CLOBBER writer (assigns the ref from an
        awaited response) — the stale write that must be guarded, and
      * at least one LIVE writer (a non-async / non-await-derived writer) exists to advance
        the generation.
    Several refs, several async clobber-writers, or an already-guarded async writer (it
    already compares the generation token) → ``None`` so the caller keeps today's behaviour.
    Pure-local, zero model cost, never raises.

    Source-file resolution (T0009 — autonomous, no config crutch):
      1. an explicit ``source_file`` (config binding OR an already-resolved path) wins;
      2. else the honey names exactly ONE existing client file → use it (fast path);
      3. else (the MULTI-file 0062 honey, where the single-file resolve declines because
         several client files are named) DRIVE the per-file recogniser across every named
         candidate and accept the UNIQUE file that actually exhibits the race. Zero matches →
         honest "nothing to fix" signal; several matches → "ambiguous" signal. Both decline.
    Step 3 is what lets the hive derive ``source_file`` ITSELF on a multi-file honey instead of
    being hand-fed the answer in the preset — the matcher (per-file recognition) already
    existed; only the multi-candidate driver was missing.
    """
    if not honey_text and not source_file:
        return None
    # (1) Explicit binding / already-resolved path.
    if source_file:
        path = source_file
        # A caller may pass a repo-RELATIVE source_file (run_specify does); resolve it against
        # the codebase root when it is not already an existing absolute path.
        if path and not os.path.isfile(path) and codebase_root:
            joined = os.path.join(
                codebase_root, *str(path).replace("\\", "/").lstrip("/").split("/"))
            if os.path.isfile(joined):
                path = joined
        if not path or not os.path.isfile(path):
            return None
        return _recognize_symptom_in_file(path, honey_text, codebase_root)

    # (2) Single named client file → today's behaviour.
    path = _resolve_component_file(honey_text, codebase_root)
    if path:
        return _recognize_symptom_in_file(path, honey_text, codebase_root)

    # (3) Multi-file honey: scan every named candidate, accept the unique racing file.
    cands = _candidate_component_files(honey_text, codebase_root)
    matches: list[RaceSymptom] = []
    for p in cands:
        sym = _recognize_symptom_in_file(p, honey_text, codebase_root)
        if sym is not None:
            matches.append(sym)
    if len(matches) == 1:
        logger.info("overwrite-race: auto-resolved source_file from %d honey-named "
                    "candidate(s) → %s (ref=%s)",
                    len(cands), matches[0].source_file, matches[0].ref)
        return matches[0]
    if not matches:
        # Smoking-gun (failure mode #1): never a silent no-op. Either no file was named, or
        # none of the named files exhibits the race → there is honestly nothing to lower.
        if cands:
            logger.info("overwrite-race: no race candidate among %d honey-named file(s) "
                        "(smoking-gun: nothing to auto-fix)", len(cands))
        return None
    # Smoking-gun (failure mode #2 guard): several files race → we never guess which.
    logger.warning("overwrite-race: %d candidate files exhibit a race (%s) — ambiguous, "
                   "declining (smoking-gun: cannot auto-pick a source_file)",
                   len(matches), ", ".join(m.source_file for m in matches))
    return None


def _recognize_symptom_in_file(path: str, honey_text: str,
                               codebase_root: str) -> RaceSymptom | None:
    """Recognise the overwrite-race symptom in ONE already-resolved file, or ``None``.

    The per-file matcher (factored out of :func:`detect_overwrite_race_symptom` so the
    multi-candidate driver can run it against each honey-named file). Reads ``path``, grounds
    the racing ref from the honey, and returns the unique :class:`RaceSymptom` or ``None`` when
    the file does not exhibit an unambiguous race. Pure-local, zero model cost, never raises.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None

    # Ground the racing state from the honey: a real pipeline run's seed/converge NAMES the
    # ref whose value appears-then-vanishes. When the honey names ≥1 ref, only those are
    # considered (the other levers ground route/field/sink from the honey the same way); a
    # file routinely has several async-written refs and only the named one is the symptom.
    # With no honey naming a ref, structural uniqueness is required (conservative → decline
    # when several refs qualify) so we never guess which state the report meant.
    named = {m.group(1) for m in re.finditer(r"\b([A-Za-z_$][\w$]*)\b", honey_text or "")}

    best: RaceSymptom | None = None
    for dm in _REF_DECL_RE.finditer(text):
        ref = dm.group(1)
        if named and ref not in named:
            continue
        decl = text[dm.start(): text.find("\n", dm.start())]
        sites = _collect_writer_sites(text, ref)
        if len({s.name for s in sites}) < 2:
            continue  # a ref written by a single function cannot race itself
        async_clobbers = [s for s in sites if s.is_async and s.derives_from_await]
        if len(async_clobbers) != 1:
            continue  # zero (not this symptom) or several (ambiguous) → don't guess
        async_writer = async_clobbers[0]
        # Already guarded? An async writer that already compares a generation token is fixed.
        if re.search(r"\bGeneration\b", async_writer.assign) or _already_guarded(
                text, async_writer):
            continue
        # LIVE writers = the synchronous event-handler / optimistic-update writers (NOT async).
        # Each must ADVANCE the generation so an in-flight async write is marked stale. An
        # async sibling reset (e.g. fetchDoc's pre-await ``= null``) is deliberately left
        # untouched: its own trailing async refetch re-stamps the generation, so the minimal
        # canonical fix is exactly "two write-sites (one live, one async) + the three guard
        # symbols" (NR0003), not every writer of the ref.
        live = tuple(s for s in sites
                     if s.name != async_writer.name and not s.is_async)
        if not live:
            continue  # no live writer to advance the generation → guard would be inert
        rel = path
        if codebase_root:
            try:
                rel = os.path.relpath(path, codebase_root)
            except ValueError:
                rel = path
        rel = rel.replace("\\", "/")
        cand = RaceSymptom(ref=ref, decl=decl, async_writer=async_writer,
                           live_writers=live, source_file=rel)
        if best is not None:
            return None  # more than one racing ref in the file → ambiguous, decline
        best = cand
    return best


def _already_guarded(text: str, writer: WriterSite) -> bool:
    """True when ``writer``'s body already captures+compares a per-ref generation token —
    i.e. the stale-reject guard is present. Conservative; never raises."""
    start = text.find(writer.header)
    if start == -1:
        return False
    body = text[start: start + 4000]
    nxt = re.search(r"\n(?:async\s+)?function |\nconst \w+\s*=\s*(?:async\s+)?\(", body[1:])
    body = body[: nxt.start()] if nxt else body
    return bool(re.search(r"Generation\b", body) and re.search(r"return\b", body))


def _candidate_component_files(honey_text: str, codebase_root: str) -> list[str]:
    """Every existing ``.vue``/``.ts`` file the honey names, resolved against the codebase root.

    The candidate set the multi-file driver scans (and the single-file resolve's source of
    truth). Order-preserving, de-duplicated. Empty when the honey names no existing client
    file. Never raises."""
    if not honey_text or not codebase_root:
        return []
    cands: list[str] = []
    for m in re.finditer(r"[\w./\\-]+\.(?:vue|ts)\b", honey_text):
        rel = m.group(0).replace("\\", "/").lstrip("/")
        p = os.path.join(codebase_root, *rel.split("/"))
        if os.path.isfile(p) and p not in cands:
            cands.append(p)
    return cands


def _resolve_component_file(honey_text: str, codebase_root: str) -> str | None:
    """The single client component file the honey points at, or ``None``.

    Returns the UNIQUE existing ``.vue``/``.ts`` file the honey names. Zero or several distinct
    existing files → ``None`` (the multi-candidate driver in
    :func:`detect_overwrite_race_symptom` then disambiguates the several-files case by which
    one actually races). Never raises."""
    cands = _candidate_component_files(honey_text, codebase_root)
    return cands[0] if len(cands) == 1 else None


# --- ① lowering: race-class → guarded-write edits ---------------------------------------

def synthesize_guarded_write_edits(symptom: RaceSymptom,
                                   *, id_prefix: str = "RACE_GUARD") -> list[dict[str, Any]]:
    """Deterministic guarded-write edits that close the overwrite race, or ``[]`` (fail-open).

    Builds, from the live anchors the symptom carries:
      * the generation token + ``_shouldReplace<Ref>`` helper, appended after the ref decl,
      * a ``<ref>Generation += 1`` advance prepended to EACH live writer's assignment,
      * a ``const fetchGeneration = ++<ref>Generation`` capture after the async writer's
        header + a ``if (!_shouldReplace<Ref>(fetchGeneration)) return`` stale-reject
        prepended to the async writer's assignment.

    The token is threaded so the LAST writer to START is authoritative and a late async
    response that a live event superseded is discarded — exactly the SAME-FILE
    ``docFetchGeneration`` discipline 0062 already trusts for the document detail. Pure
    string transform of the anchors; never raises. Each edit carries ``guard_role`` so the
    consistency gate (③) can verify the three guard symbols all landed.
    """
    if symptom is None or not symptom.decl or not symptom.async_writer:
        return []
    edits: list[dict[str, Any]] = []
    gen = symptom.gen_token
    guard = symptom.guard_fn

    # Edit A — generation token + helper, right after the ref declaration.
    helper = (f"{symptom.decl}\n"
              f"// Overwrite-race guard (lever L3): only the newest write generation may "
              f"commit {symptom.ref}.\n"
              f"let {gen} = 0\n"
              f"function {guard}(generation: number): boolean {{\n"
              f"  return generation === {gen}\n"
              f"}}")
    edits.append({
        "id": f"{id_prefix}_TOKEN",
        "file": "",  # filled by the wrapper from the symptom's source file
        "anchor_old": symptom.decl,
        "replacement_new": helper,
        "guard_role": "token_decl",
        "confidence": "high",
        "rationale": (f"overwrite-race guard (lever L3): declare the {gen} token + "
                      f"{guard} stale-filter that thread last-write-wins through both writers "
                      f"of {symptom.ref}."),
    })

    # Edit(s) B — each live writer advances the generation so an in-flight async write is stale.
    seen_live: set[str] = set()
    for i, lw in enumerate(symptom.live_writers):
        if lw.assign in seen_live:
            continue
        seen_live.add(lw.assign)
        new = f"{lw.indent}{gen} += 1\n{lw.assign}"
        edits.append({
            "id": f"{id_prefix}_LIVE_{i}",
            "file": "",
            "anchor_old": lw.assign,
            "replacement_new": new,
            "guard_role": "live_advance",
            "confidence": "high",
            "rationale": (f"overwrite-race guard (lever L3): live writer {lw.name!r} advances "
                          f"{gen} so a stale in-flight async write to {symptom.ref} is rejected."),
        })

    # Edit C — async writer captures its generation at the top of its body (body indent =
    # the header's own indent + one level, NOT the assignment's deeper in-block indent).
    aw = symptom.async_writer
    body_indent = (aw.header[: len(aw.header) - len(aw.header.lstrip())]) + "  "
    edits.append({
        "id": f"{id_prefix}_ASYNC_CAPTURE",
        "file": "",
        "anchor_old": aw.header,
        "replacement_new": f"{aw.header}\n{body_indent}const fetchGeneration = ++{gen}",
        "guard_role": "async_capture",
        "confidence": "high",
        "rationale": (f"overwrite-race guard (lever L3): async writer {aw.name!r} captures its "
                      f"{gen} at start so a later live/async write can mark it stale."),
    })

    # Edit D — async writer rejects its own stale response before committing.
    new_assign = (f"{aw.indent}if (!{guard}(fetchGeneration)) return\n{aw.assign}")
    edits.append({
        "id": f"{id_prefix}_ASYNC_REJECT",
        "file": "",
        "anchor_old": aw.assign,
        "replacement_new": new_assign,
        "guard_role": "async_reject",
        "confidence": "high",
        "rationale": (f"overwrite-race guard (lever L3): async writer {aw.name!r} discards its "
                      f"response when a newer generation superseded it — the clobber that made "
                      f"{symptom.ref} appear-then-vanish."),
    })
    return edits


# --- ② behaviour oracle: last-write-wins red test ---------------------------------------

def _test_fn_name(ref: str) -> str:
    raw = f"test_{ref}_last_write_wins_over_stale_async"
    name = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
    return re.sub(r"_+", "_", name) or "test_last_write_wins"


def _build_race_test(symptom: RaceSymptom, fixture_call: str) -> tuple[str, str]:
    """``(source, test_fn_name)`` for the last-write-wins vitest red test.

    Drives the documented race ORDER against the harness's two writers: the async writer
    starts (its response will resolve LATE and stale), the live writer commits the fresh
    value, then the stale async response resolves — and asserts the ref still holds the live
    value. RED while the async write is unguarded (it clobbers), GREEN once the generation
    guard discards the stale response. Deterministic, never raises."""
    fn = _test_fn_name(symptom.ref)
    src = (
        "import {{ test, expect }} from 'vitest'\n\n"
        "test('{fn}', async () => {{\n"
        "  const ctx = {call}\n"
        "  // 1. async clobber-writer starts; its response is rigged to resolve LATE + stale\n"
        "  const stale = ctx.startStaleAsyncWrite()\n"
        "  // 2. live writer commits the fresh value (advancing the generation if guarded)\n"
        "  ctx.liveWrite()\n"
        "  const live = ctx.read()\n"
        "  // 3. the stale async response resolves AFTER the live write\n"
        "  await stale\n"
        "  // last write wins: the live value must survive the stale async response\n"
        "  expect(ctx.read()).toEqual(live)\n"
        "}})\n"
    ).format(fn=fn, call=fixture_call)
    return src, fn


def synthesize_overwrite_race_red_test(
    spec: dict[str, Any],
    honey_text: str,
    codebase_root: str,
    *,
    symptom: RaceSymptom | None = None,
    setup_block: str | None = None,
    fixture_call: str = "makeRaceHarness()",
    test_dir: str = "client/src/test",
    test_id: str = "RACE_ORACLE_RED",
) -> dict[str, Any] | None:
    """Build the last-write-wins red-test edit + node id, or ``None`` (fail-open).

    Returns ``{"edit", "node", "symptom"}`` when ALL hold: the symptom resolves (passed in
    or recognised from the live file), the spec carries an overwrite-race guard repair (a
    ``guard_role`` edit — so a last-write-wins oracle is meaningful), and a runnable vitest
    ``setup_block`` harness exposing ``makeRaceHarness()`` (``startStaleAsyncWrite`` /
    ``liveWrite`` / ``read``) is supplied. Otherwise ``None``. Zero model cost.

    The harness cannot be derived from grounding (it must construct the two writers over the
    component's ref), so — exactly as lever ⑦ requires an HTTP harness — L3 declines fail-open
    when none is supplied rather than emit a test that errors. ``verify.py`` certifies the
    test by execution; a mis-seeded harness only fails to certify, never ships a bad fix.
    """
    if not _has_guard_repair(spec):
        return None  # no guard insertion in this spec → a race oracle is not meaningful
    if not setup_block:
        return None  # the race needs a constructed harness → no harness, no runnable test
    sym = symptom or detect_overwrite_race_symptom(honey_text, codebase_root)
    if sym is None:
        return None

    body, fn = _build_race_test(sym, fixture_call)
    content = setup_block.rstrip() + "\n\n\n" + body
    slug = re.sub(r"[^a-z0-9]+", "_", sym.ref.lower()).strip("_") or "ref"
    rel = f"{test_dir.rstrip('/')}/{slug}_race.test.ts"
    edit = {
        "id": test_id,
        "kind": "create_file",
        "file": rel,
        "content": content,
        "rationale": (
            f"overwrite-race behaviour oracle (lever L3): {sym.ref} must equal the LIVE write "
            "after a stale async response resolves last — RED while the async write is "
            "unguarded (it clobbers the live value, the appear-then-vanish symptom), GREEN once "
            "the generation guard discards the stale response; certified by the red→green run."),
        "confidence": "high",
    }
    return {"edit": edit, "node": f"{rel}::{fn}", "symptom": sym}


def _has_guard_repair(spec: dict[str, Any]) -> bool:
    """True when a SOURCE edit carries a ``guard_role`` (the overwrite-race guard repair
    signature) OR introduces a generation-token assign+compare by hand. Never raises."""
    for e in (spec.get("edits") or []):
        if not isinstance(e, dict):
            continue
        if e.get("guard_role"):
            return True
        new = (e.get("replacement_new") or "")
        if re.search(r"Generation\b", new) and re.search(r"\+\+|\+= *1|=== ", new):
            return True
    return False
