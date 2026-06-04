"""Apply stage — renders an edit-spec into a human-facing proposal (propose only).

Pipeline position (fix-extension):

  investigate (fan-out)  →  merge (honey)  →  specify (edit-spec)  →  APPLY (this)

apply is the third and final box of the fix loop. Unlike specify it calls NO
worker: it is a pure, deterministic local pass over the edit-spec JSON (the SSOT
authored by specify) and the LIVE codebase. It re-opens each target file, checks
that the spec's ``anchor_old`` still exists there exactly once, renders the
``anchor_old → replacement_new`` change as a unified diff, and emits a proposal.

Key invariants (mirrored from recipes/edit_spec_contract_v1.md):
  - Default is PROPOSE ONLY. apply does not touch the target codebase unless the
    caller explicitly opts in with ``write=True`` (CLI ``--write``). ``gate.apply``
    in the spec is model-authored and is NEVER trusted to drive a write — the
    write switch is the human-held CLI flag.
  - The JSON edit-spec is the SSOT; the unified diff is a DERIVED view rendered
    here from ``anchor_old``/``replacement_new`` — apply does not invent edits.
  - Anchors are re-verified against LIVE code at apply time. Even an edit specify
    marked ``verified`` can have drifted (live code changed since specify ran), so
    apply re-locates the anchor and refuses to present a drifted edit as ready.
  - A write only happens when the proposal is READY (every edit applicable +
    termination ready_to_apply). Before writing, originals are snapshotted into a
    scratch backup bundle (see hive.backup) and the edits are applied
    all-or-nothing: any anchor that is no longer unique at the moment of writing
    aborts and rolls every touched file back.

An edit is "applicable" only when its ``anchor_old`` is found EXACTLY ONCE in the
current file. Zero matches (missing / already-applied) or multiple matches
(ambiguous) make the edit non-applicable and drop the overall verdict to
not-ready — the same conservative posture specify takes with stale anchors.
"""

import difflib
import json
import logging
import os
import py_compile
import re
import tempfile
from typing import Any

from hive import backup as backup_store
# Share the ONE canonical termination vocabulary with specify so apply never rejects a
# value specify legitimately emits (N174: 'needs_runtime' was missing from apply's copy).
# Reuse specify's deterministic import↔usage wiring check so the import-pairing rule below
# applies the SAME logic to the would-be-written subset (no second, drifting copy).
from hive.specify import VALID_TERMINATION as _VALID_TERMINATION
from hive.specify import _incomplete_wiring_ids

logger = logging.getLogger("hive.apply")

# Per-edit applicability statuses (only "applicable" can contribute to a ready verdict).
APPLICABLE = "applicable"
ANCHOR_MISSING = "anchor_missing"
ANCHOR_AMBIGUOUS = "anchor_ambiguous"
ALREADY_APPLIED = "already_applied"
FILE_MISSING = "file_missing"
NO_CHANGE = "no_change"
# create_file branch (additive — apply.py also creates new files, not only edits)
CREATE_OK = "create_ok"          # target absent + content non-empty → applicable
FILE_EXISTS = "file_exists"      # create_file target already on disk → not applicable
EMPTY_CONTENT = "empty_content"  # content empty/whitespace-only → not applicable
POST_APPLY_BROKEN = "post_apply_broken"  # anchor unique but the applied result is broken


# A test-expectation edit must never be written ahead of the source edit it asserts
# (Defect 3 / partial atomicity). We recognise test files by path convention so that an
# unwritable source edit can hold its test siblings out of a partial write.
_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__)/"             # inside a tests/ test/ __tests__/ dir
    r"|(?:^|/)test_[^/]+$"                       # file named test_*.*
    r"|(?:^|/)[^/]+_test\.[^./]+$"               # file named *_test.ext
    r"|(?:^|/)[^/]+\.(?:test|spec)\.[^./]+$",    # file named *.test.ext / *.spec.ext
    re.IGNORECASE,
)


def _is_test_file(rel_path: str) -> bool:
    """True when a path is a test file/dir by convention (tests/ dir, test_*, *_test, *.spec.*)."""
    if not rel_path:
        return False
    return bool(_TEST_PATH_RE.search(rel_path.replace("\\", "/")))


def load_spec(spec_path: str) -> dict[str, Any]:
    """Load the edit-spec JSON (the SSOT produced by specify)."""
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


_UTF8_BOM = b"\xef\xbb\xbf"


def _norm_nl(s: str) -> str:
    """Normalize any EOL flavour in ``s`` to ``\\n`` (universal-newline form)."""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _decode_preserving(raw: bytes) -> tuple[str, str, bool]:
    """Decode file bytes to ``\\n``-normalized text, remembering its EOL + BOM.

    Returns ``(text, eol, had_bom)`` where ``text`` is normalized to ``\\n`` so the
    LF-based anchors specify captures match regardless of the file's on-disk EOL,
    ``eol`` is the file's dominant newline (``\\r\\n`` / ``\\n`` / ``\\r``) and
    ``had_bom`` flags a UTF-8 BOM. Pairing this with :func:`_encode_preserving`
    lets apply re-write a file in its ORIGINAL EOL + encoding rather than the
    host's ``os.linesep`` — Hive must never reformat a file it only edited a few
    lines of (the EOL-drift FlowGate hit).
    """
    had_bom = raw.startswith(_UTF8_BOM)
    if had_bom:
        raw = raw[len(_UTF8_BOM):]
    s = raw.decode("utf-8")
    crlf = s.count("\r\n")
    cr = s.count("\r") - crlf
    lf = s.count("\n") - crlf
    if crlf and crlf >= lf and crlf >= cr:
        eol = "\r\n"
    elif cr and cr > lf:
        eol = "\r"
    else:
        eol = "\n"
    return _norm_nl(s), eol, had_bom


def _encode_preserving(text: str, eol: str, had_bom: bool) -> bytes:
    """Inverse of :func:`_decode_preserving`: re-apply EOL + BOM, encode UTF-8."""
    body = text if eol == "\n" else text.replace("\n", eol)
    data = body.encode("utf-8")
    return _UTF8_BOM + data if had_bom else data


def _read_text_preserving(abs_path: str) -> tuple[str, str, bool]:
    """Read a file as ``\\n``-normalized text plus its (eol, had_bom) fingerprint."""
    with open(abs_path, "rb") as f:
        return _decode_preserving(f.read())


def render_unified_diff(rel_path: str, original: str, modified: str) -> str:
    """Render a git-style unified diff from ``original`` to ``modified``.

    Labels use ``a/<rel_path>`` and ``b/<rel_path>`` so the output reads like a
    real patch. This is the DERIVED human view; the spec's text fields are SSOT.
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=3,
    )
    return "".join(diff)


def render_creation_diff(rel_path: str, content: str) -> str:
    """Render a unified diff for a brand-new file (``/dev/null`` → content).

    The DERIVED human view of a ``create_file`` edit: a full-file addition,
    labelled the git way so it reads like a real patch.
    """
    diff = difflib.unified_diff(
        [],
        content.splitlines(keepends=True),
        fromfile="/dev/null",
        tofile=f"b/{rel_path}",
        n=3,
    )
    return "".join(diff)


def _template_dotvalue_count(text: str) -> int:
    """Count ``<ident>.value`` occurrences inside the ``<template>`` of a Vue SFC.

    In Vue 3 ``<script setup>`` refs/computed are auto-unwrapped in the template, so
    ``foo.value`` written there reads ``.value`` off the already-unwrapped value
    (``undefined``). A template should therefore never contain ``<ref>.value``.
    """
    m = re.search(r"<template[^>]*>(.*)</template>", text, re.DOTALL | re.IGNORECASE)
    region = m.group(1) if m else ""
    return len(re.findall(r"\b[A-Za-z_$][\w$]*\.value\b", region))


def _distinctive(line: str) -> bool:
    """A line carrying enough signal to flag as a duplicate (excludes trivia).

    Trivial structural lines (``)``, ``},``, ``else:``, ``pass``) recur all over a
    file and would false-match the overlap check; a distinctive line is ≥6 chars
    of stripped content and contains an identifier-ish token.
    """
    s = line.strip()
    return len(s) >= 6 and re.search(r"[A-Za-z_]\w\w", s) is not None


def _post_apply_defects(rel_path: str, file_text: str, anchor_old: str,
                        replacement_new: str, modified: str) -> list[str]:
    """Deterministic, env-free checks that the dry-applied result isn't broken.

    A unique anchor only proves *where* the edit lands, not that the result is
    sound — the gap that let Hive call a crashing edit "READY, applicable" (T889):
    an anchor that ended mid-block while the replacement re-stated the lines that
    FOLLOW it (→ duplicated block + a None-branch attribute access that still
    *parses*), and a Vue ``<template>`` edit that introduced ref ``.value``. These
    checks run no model and need no project environment; they raise the floor
    (no broken edit ships as READY) without judging whether the fix is correct.

    Returns a list of defect messages — empty means clean.
    """
    defects: list[str] = []

    # 1. Anchor-boundary overlap → duplication. A sound replacement rewrites the
    #    anchored span; it must not re-emit the lines that already FOLLOW the
    #    anchor (those stay in the file → duplicated). The signature is precise:
    #    replacement_new reproduces the CONTIGUOUS run of code lines immediately
    #    after the anchor. We compare only *distinctive* lines (≥6 chars with an
    #    identifier) so trivial lines like ``)`` / ``else:`` can't false-match, and
    #    require a prefix run ≥2 so a lone coincidental line doesn't trip it.
    pos = file_text.find(anchor_old)
    if anchor_old and pos != -1:
        after = file_text[pos + len(anchor_old):]
        after_distinct = [ln for ln in after.splitlines() if _distinctive(ln)][:8]
        repl_lines = {ln for ln in replacement_new.splitlines() if _distinctive(ln)}
        run = 0
        for ln in after_distinct:
            if ln in repl_lines:
                run += 1
            else:
                break
        if run >= 2:
            defects.append(
                f"anchor likely too short: replacement re-emits {run} line(s) that "
                f"already follow the anchor (e.g. {after_distinct[0].strip()[:60]!r})"
                " — applying would duplicate them")

    # 2. Vue SFC: flag ``.value`` this edit INTRODUCES into the <template> region.
    if rel_path.endswith(".vue"):
        before_n = _template_dotvalue_count(file_text)
        after_n = _template_dotvalue_count(modified)
        if after_n > before_n:
            defects.append(
                f"edit introduces `.value` inside <template> ({before_n}→{after_n}); "
                "Vue auto-unwraps refs there, so `.value` evaluates to undefined")

    # 3. Python: the applied file must still compile.
    if rel_path.endswith(".py"):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "edited.py")
            with open(src, "w", encoding="utf-8") as f:
                f.write(modified)
            try:
                py_compile.compile(src, cfile=os.path.join(d, "out.pyc"),
                                   doraise=True)
            except py_compile.PyCompileError as e:
                defects.append(f"applied file fails to compile: "
                               f"{str(e.msg).strip()[:160]}")

    return defects


def evaluate_edit(edit: dict[str, Any], codebase_root: str) -> dict[str, Any]:
    """Re-verify one edit against LIVE code and render its diff.

    Returns a result dict describing applicability. The edit is ``applicable``
    only when ``anchor_old`` occurs exactly once in the current file AND the
    dry-applied result passes :func:`_post_apply_defects`; the diff is rendered
    for that single, unambiguous replacement.
    """
    edit_id = edit.get("id", "?")
    rel_path = edit.get("file", "")
    kind = edit.get("kind", "edit")
    # Normalize anchor/replacement EOL to '\n' so matching is EOL-agnostic against
    # the '\n'-normalized file text (the file's real EOL is re-applied at write time).
    anchor_old = _norm_nl(edit.get("anchor_old", ""))
    replacement_new = _norm_nl(edit.get("replacement_new", ""))

    result: dict[str, Any] = {
        "id": edit_id,
        "file": rel_path,
        "confidence": edit.get("confidence", "?"),
        "anchor_status_claimed": edit.get("anchor_status", "?"),
        "status": ANCHOR_MISSING,
        "applicable": False,
        "diff": "",
        "messages": [],
    }

    if not rel_path:
        result["status"] = FILE_MISSING
        result["messages"].append("edit has no 'file' field")
        return result

    abs_path = os.path.join(codebase_root, rel_path)

    # create_file branch — MUST precede the anchor logic. An empty anchor would
    # mis-evaluate on the anchor path (``"".count("")`` quirk). Applicability is
    # INVERTED: a create_file edit is applicable only when the target is ABSENT.
    # os.path.exists (not isfile) so a directory collision is caught here too,
    # matching the write-time re-check in write_edits.
    if kind == "create_file":
        content = edit.get("content", "")
        if os.path.exists(abs_path):
            result["status"] = FILE_EXISTS
            result["messages"].append(
                f"create_file target already exists: {rel_path} — refusing to "
                "clobber without an anchor review")
            return result
        if not content or not content.strip():
            result["status"] = EMPTY_CONTENT
            result["messages"].append(
                "create_file content is empty or whitespace-only — inert edit")
            return result
        result["status"] = CREATE_OK
        result["applicable"] = True
        result["diff"] = render_creation_diff(rel_path, content)
        return result

    if not os.path.isfile(abs_path):
        result["status"] = FILE_MISSING
        result["messages"].append(f"file not found under codebase root: {rel_path}")
        return result

    file_text, _eol, _bom = _read_text_preserving(abs_path)

    if anchor_old == replacement_new:
        result["status"] = NO_CHANGE
        result["messages"].append("anchor_old equals replacement_new — no-op edit")
        return result

    occurrences = file_text.count(anchor_old) if anchor_old else 0

    if occurrences == 1:
        modified = file_text.replace(anchor_old, replacement_new, 1)
        result["diff"] = render_unified_diff(rel_path, file_text, modified)

        # Anchor is unique — but is the APPLIED result sound? Re-verify deterministically.
        defects = _post_apply_defects(
            rel_path, file_text, anchor_old, replacement_new, modified)
        if defects:
            result["status"] = POST_APPLY_BROKEN
            result["applicable"] = False
            result["messages"].extend(defects)
            return result

        result["status"] = APPLICABLE
        result["applicable"] = True
        if str(edit.get("anchor_status", "")).lower() != "verified":
            # Live matches, but specify did not mark it verified — surface, don't block.
            result["messages"].append(
                f"anchor located in live code but spec anchor_status="
                f"{edit.get('anchor_status')!r}")
        return result

    if occurrences > 1:
        result["status"] = ANCHOR_AMBIGUOUS
        result["messages"].append(
            f"anchor_old occurs {occurrences}x in {rel_path} — not unique, "
            "cannot target a single edit")
        return result

    # Zero occurrences. Distinguish "already applied" from "drifted/missing".
    if replacement_new and file_text.count(replacement_new) >= 1:
        result["status"] = ALREADY_APPLIED
        result["messages"].append(
            "anchor_old absent but replacement_new already present — edit looks "
            "already applied")
    else:
        result["status"] = ANCHOR_MISSING
        msg = f"anchor_old not found in live {rel_path}"
        if str(edit.get("anchor_status", "")).lower() == "verified":
            msg += " — DRIFT: spec marked it 'verified' but live code has changed"
        result["messages"].append(msg)
    return result


def build_proposal(spec: dict[str, Any], codebase_root: str) -> dict[str, Any]:
    """Evaluate every edit against live code and decide the overall verdict.

    The proposal is ``ready`` only when ALL of these hold:
      - termination == "ready_to_apply"
      - there is at least one edit
      - every edit is APPLICABLE (anchor unique in live code right now)

    Anything else (deferred-only specs, drifted anchors, ambiguous anchors,
    non-ready termination) yields ``ready = False`` with explicit reasons.
    """
    termination = spec.get("termination")
    edits = spec.get("edits") if isinstance(spec.get("edits"), list) else []
    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []
    gate = spec.get("gate") if isinstance(spec.get("gate"), dict) else {}
    eff = spec.get("effectiveness") if isinstance(spec.get("effectiveness"), dict) else {}
    ineffective = {str(x) for x in (eff.get("ineffective_ids") or [])}

    edit_results = [evaluate_edit(e, codebase_root) for e in edits if isinstance(e, dict)]
    n_applicable = sum(1 for r in edit_results if r["applicable"])

    # Per-edit writability (Defect 3): an edit is individually safe to write when
    # its anchor is unique in live code (applicable) AND the effectiveness review
    # did not flag IT as ineffective. This is decoupled from the GLOBAL termination
    # so a verified, effective edit is not held hostage by a deferred sibling — a
    # lone "anchor_not_grounded" defer on an unrelated item used to flip termination
    # to needs_reinvestigation and block an already-ready root-cause fix (T892 E1).
    for r in edit_results:
        r["writable"] = bool(r["applicable"]) and str(r["id"]) not in ineffective

    # Import↔usage atomicity (N178): a partial write can land an import edit whose paired
    # USAGE edit is held (its anchor drifted), leaving a dangling import. We assemble ONLY
    # the would-be-written edits and reuse specify's wiring check: an edit whose added
    # import binding is unused once we keep just the writable subset is held too, so the
    # import and its use ship all-or-nothing. (When every edit is writable the binding is
    # used, so nothing is held — this only bites a genuine partial split.) Runs BEFORE the
    # test-hold below so holding an import cascades correctly into source_unwritable.
    edits_by_id = {str(e.get("id", "?")): e for e in edits if isinstance(e, dict)}
    writable_now = [edits_by_id[str(r["id"])] for r in edit_results
                    if r["writable"] and str(r["id"]) in edits_by_id]
    if len(writable_now) < len(edit_results):  # only meaningful on a partial split
        dangling = _incomplete_wiring_ids({"edits": writable_now}, codebase_root)
        for r in edit_results:
            if str(r["id"]) in dangling and r["writable"]:
                r["writable"] = False
                r["held_reason"] = (
                    "import edit held: its added binding is unused without a held sibling "
                    f"edit — {dangling[str(r['id'])]} (writing it alone leaves a dangling "
                    "import)")

    # Partial atomicity (Defect 3): a test-expectation edit must not be written ahead
    # of the source edit it asserts. --partial decides writability per edit, so a clean
    # test-edit could land while its paired code-edit failed (non-unique / drifted
    # anchor) — leaving the suite asserting behavior the code does not yet have, which is
    # worse than writing nothing. We cannot cheaply prove the exact code↔test pairing, so
    # we hold ALL test edits whenever ANY source (non-test) edit is unwritable. Over-
    # holding is safe (the held edit just waits for the next round); writing a test ahead
    # of its code is not.
    test_ids = {str(r["id"]) for r in edit_results if _is_test_file(r["file"])}
    source_unwritable = any(
        not r["writable"] for r in edit_results if str(r["id"]) not in test_ids)
    if source_unwritable and test_ids:
        for r in edit_results:
            if str(r["id"]) in test_ids and r["writable"]:
                r["writable"] = False
                r["held_reason"] = (
                    "test-expectation edit held: a source edit in this spec is not "
                    "applicable — writing it alone would assert unshipped behavior")
    writable_ids = [r["id"] for r in edit_results if r["writable"]]

    reasons: list[str] = []
    if termination != "ready_to_apply":
        reasons.append(f"termination is {termination!r}, not 'ready_to_apply'")
    if not edit_results:
        reasons.append("spec contains no edits (nothing to apply)")
    for r in edit_results:
        if not r["applicable"]:
            reasons.append(f"{r['id']} ({r['file']}): {r['status']}")

    # Stage-1 safety net: gate.apply must never drive an actual write here. apply
    # only ever proposes, but we still flag a spec that tried to flip the switch.
    if gate.get("apply") is True:
        reasons.append("gate.apply was True — ignored (Stage-1 is propose-only)")

    ready = (
        termination == "ready_to_apply"
        and bool(edit_results)
        and all(r["applicable"] for r in edit_results)
    )
    # Partial-ready: not fully ready, but ≥1 edit is individually writable — the
    # operator can apply just those with --partial without waiting on the deferred
    # items. We never down-rank a writable edit for a sibling's unresolved state.
    partial_ready = (not ready) and bool(writable_ids)

    return {
        "ready": ready,
        "partial_ready": partial_ready,
        "writable_ids": writable_ids,
        "not_ready_reasons": reasons,
        "codebase_root": os.path.abspath(codebase_root),
        "source_honey": spec.get("source_honey", ""),
        "termination": termination,
        "edits": edit_results,
        "deferred": deferred,
        "gate_commands": list(gate.get("commands") or []),
        "n_edits": len(edit_results),
        "n_applicable": n_applicable,
    }


def write_edits(
    spec: dict[str, Any],
    codebase_root: str,
    backup_root: str,
    ttl_hours: int,
    only_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Apply a READY spec's edits to disk, with a scratch backup + rollback.

    Precondition: the caller has already confirmed the edits to write are each
    applicable (anchor unique in live code a moment ago). This function still
    re-verifies uniqueness at the instant of each write (a file may have changed
    in between, or one edit may collide with another that targets the same file).

    When ``only_ids`` is given, only those edit ids are written (Defect 3 partial
    apply: the individually-writable subset, when the spec is not globally ready);
    the backup bundle snapshots only the touched files. The sequence is:

      1. purge expired backup bundles (time-boxed undo window upkeep),
      2. snapshot every target file's original bytes into a fresh bundle,
      3. apply edits one at a time, re-reading each file so multiple edits to the
         same file compose correctly,
      4. if any edit's anchor is no longer unique, roll ALL touched files back from
         the in-memory snapshot and report failure (all-or-nothing).

    Returns a result dict: ``ok``, ``attempted``, ``written`` (rel paths actually
    changed), ``bundle`` (backup dir, the undo handle), and ``reason`` on failure.
    """
    result: dict[str, Any] = {
        "ok": False, "attempted": True, "written": [], "bundle": None,
        "reason": "", "rolled_back": False,
    }

    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    if only_ids is not None:
        edits = [e for e in edits if str(e.get("id", "?")) in only_ids]

    # Modified paths (anchor edits) are snapshotted; created paths are recorded
    # in the manifest so a restore deletes them (they have no original bytes).
    rel_paths: list[str] = []
    created_paths: list[str] = []
    for e in edits:
        rel = e.get("file", "")
        if not rel:
            continue
        if e.get("kind", "edit") == "create_file":
            if rel not in created_paths:
                created_paths.append(rel)
        elif rel not in rel_paths:
            rel_paths.append(rel)
    if not edits or (not rel_paths and not created_paths):
        result["reason"] = "spec has no writable edits"
        return result

    os.makedirs(backup_root, exist_ok=True)
    backup_store.purge_expired(backup_root, fallback_ttl_hours=ttl_hours)

    try:
        bundle = backup_store.create_bundle(
            backup_root, spec.get("_spec_path", "spec"), codebase_root,
            rel_paths, ttl_hours, created_paths=created_paths)
    except OSError as e:
        result["reason"] = f"could not snapshot originals for backup: {e}"
        return result
    result["bundle"] = bundle["dir"]
    originals: dict[str, bytes] = bundle["originals"]
    created_written: list[str] = []  # create_file paths written so far (for rollback)

    def _rollback() -> None:
        # Restore the exact pre-write bytes (binary) — a text-mode rewrite here would
        # re-normalize EOL and defeat the whole point of the snapshot.
        for rel, data in originals.items():
            try:
                with open(os.path.join(codebase_root, rel), "wb") as f:
                    f.write(data)
            except OSError as e:
                logger.error("rollback failed for %s: %s", rel, e)
        for rel in created_written:
            try:
                os.remove(os.path.join(codebase_root, rel))
            except OSError as e:
                logger.error("rollback: could not delete created file %s: %s", rel, e)
        result["rolled_back"] = True

    written: list[str] = []
    for edit in edits:
        rel = edit.get("file", "")
        abs_path = os.path.join(codebase_root, rel)

        if edit.get("kind", "edit") == "create_file":
            # Re-verify the target is still absent at write time.
            if os.path.exists(abs_path):
                result["reason"] = (
                    f"{edit.get('id', '?')} ({rel}): create_file target already "
                    "exists at write time — rolled back")
                _rollback()
                return result
            content = edit.get("content", "")
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            # Binary write: emit the spec's content bytes verbatim (no os.linesep
            # translation), so a new file isn't silently reformatted to host EOL.
            with open(abs_path, "wb") as f:
                f.write(content.encode("utf-8"))
            created_written.append(rel)
            if rel not in written:
                written.append(rel)
            continue

        anchor_old = _norm_nl(edit.get("anchor_old", ""))
        replacement_new = _norm_nl(edit.get("replacement_new", ""))
        # Read '\n'-normalized text for matching, but remember the file's real EOL
        # and BOM so the rewrite preserves them byte-for-byte.
        text, eol, had_bom = _read_text_preserving(abs_path)
        occurrences = text.count(anchor_old) if anchor_old else 0
        if occurrences != 1:
            result["reason"] = (
                f"{edit.get('id', '?')} ({rel}): anchor no longer unique at "
                f"write time ({occurrences} matches) — rolled back")
            _rollback()
            return result

        modified = text.replace(anchor_old, replacement_new, 1)
        with open(abs_path, "wb") as f:
            f.write(_encode_preserving(modified, eol, had_bom))
        if rel not in written:
            written.append(rel)

    result["ok"] = True
    result["written"] = written
    logger.info("Applied %d edit(s) across %d file(s); backup at %s",
                len(edits), len(written), bundle["dir"])
    return result


def render_proposal_markdown(proposal: dict[str, Any]) -> str:
    """Render the human-facing apply proposal (diffs + gate + verdict)."""
    lines: list[str] = []
    write = proposal.get("write")
    if write and write.get("ok"):
        lines.append("# Apply result — WRITTEN to live codebase (backup taken)")
    elif write and write.get("attempted"):
        lines.append("# Apply result — write attempted but NOT applied")
    else:
        lines.append("# Apply proposal — propose only (nothing was written)")
    lines.append("")
    lines.append(f"- codebase_root: `{proposal['codebase_root']}`")
    if proposal.get("source_honey"):
        lines.append(f"- source_honey: `{proposal['source_honey']}`")
    lines.append(f"- spec termination: `{proposal['termination']}`")
    lines.append(f"- edits: {proposal['n_edits']} "
                 f"({proposal['n_applicable']} applicable)")
    lines.append(f"- deferred: {len(proposal['deferred'])}")
    lines.append("")

    if proposal["ready"]:
        lines.append("## ✅ VERDICT: READY TO APPLY")
        lines.append("")
        lines.append("Every edit's anchor was re-verified unique in live code. "
                     "Review the diffs below, then apply by hand and run the gate.")
    else:
        lines.append("## ⛔ VERDICT: NOT READY")
        lines.append("")
        lines.append("Do not apply. Reasons:")
        for reason in proposal["not_ready_reasons"]:
            lines.append(f"- {reason}")
        if proposal.get("partial_ready"):
            lines.append("")
            lines.append(f"> **Partial apply available:** {len(proposal['writable_ids'])} "
                         f"edit(s) {proposal['writable_ids']} are individually "
                         "applicable and passed the effectiveness review — they are NOT "
                         "blocked by the unresolved items above. Re-run `apply --write "
                         "--partial` to write just those.")
    lines.append("")

    rv = proposal.get("runtime_verify")
    if rv:
        ok = rv.get("transition") in ("red_to_green",)
        lines.append("## Runtime verify — red→green " + ("✅ CONFIRMED" if ok else "⛔ NOT confirmed"))
        lines.append("")
        lines.append(f"- node: `{rv.get('node', '')}`")
        lines.append(f"- transition: `{rv.get('transition')}` — {rv.get('reason', '')}")
        for phase in ("red", "green"):
            run = rv.get(phase)
            if isinstance(run, dict):
                lines.append(f"- {phase}: `{run.get('status')}` "
                             f"(exit {run.get('returncode')})")
        lines.append("")

    if write and write.get("attempted"):
        if write.get("ok"):
            lines.append("## Write — APPLIED")
            lines.append("")
            lines.append("These files were modified in the live codebase:")
            for rel in write.get("written", []):
                lines.append(f"- `{rel}`")
            lines.append("")
            lines.append(f"Backup bundle (undo window): `{write.get('bundle')}`")
            lines.append("")
            lines.append("To undo before the bundle expires:")
            lines.append("")
            lines.append("```")
            lines.append(f"python hive.py restore --bundle {write.get('bundle')}")
            lines.append("```")
        else:
            lines.append("## Write — NOT APPLIED")
            lines.append("")
            lines.append(f"- {write.get('reason', 'write did not complete')}")
            if write.get("rolled_back"):
                lines.append("- all touched files were rolled back to their "
                             "pre-write state")
            if write.get("bundle"):
                lines.append(f"- backup bundle: `{write.get('bundle')}`")
        lines.append("")

    if proposal["gate_commands"]:
        lines.append("## Gate — run these AFTER you apply the edits")
        lines.append("")
        for cmd in proposal["gate_commands"]:
            lines.append(f"- `{cmd}`")
        lines.append("")

    lines.append("## Edits")
    lines.append("")
    if not proposal["edits"]:
        lines.append("_(none)_")
        lines.append("")
    for r in proposal["edits"]:
        flag = "✔" if r["applicable"] else "✗"
        lines.append(f"### {flag} {r['id']} — `{r['file']}`")
        lines.append(f"- status: `{r['status']}` | confidence: "
                     f"`{r['confidence']}` | anchor_status (spec): "
                     f"`{r['anchor_status_claimed']}`")
        for msg in r["messages"]:
            lines.append(f"- ⚠ {msg}")
        lines.append("")
        if r["diff"]:
            lines.append("```diff")
            lines.append(r["diff"].rstrip("\n"))
            lines.append("```")
            lines.append("")

    if proposal["deferred"]:
        lines.append("## Deferred (stay as investigation/surface — not edits)")
        lines.append("")
        for d in proposal["deferred"]:
            if isinstance(d, dict):
                lines.append(f"- **{d.get('reason', '?')}**: "
                             f"{d.get('issue', '')} "
                             f"(stays_as: {d.get('stays_as', '?')})")
            else:
                lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)


def _resolve_effective_root(spec: dict[str, Any], candidate_roots: list[str]) -> str:
    """Choose the single base root the spec's target files actually live under.

    Code and design docs often live in separate trees (e.g. a ``FlowGate`` source
    checkout vs. a ``Documents/.../FlowGate`` design-doc tree). investigate is told
    about both (``--codebase`` + ``--docs``) and resolves a doc anchor under the
    docs tree, so specify records *that* tree as the spec's ``codebase_root``. apply
    must not lose this: a caller that passes only ``--codebase <source>`` would
    otherwise resolve a doc edit's path against the source tree and report
    ``file_missing`` for every anchor.

    We probe each anchor (non-create) edit's file against the candidate roots in
    order and return the root under which the most files resolve. ``create_file``
    edits are absent by design and don't vote. When nothing resolves we return the
    first candidate, preserving today's ``file_missing`` behaviour rather than
    guessing. A single resolved root is returned so the write/backup path stays
    single-root (a spec whose files genuinely straddle two trees keeps the first
    candidate; the unresolved files surface as ``file_missing``, never a wrong write).
    """
    roots: list[str] = []
    for r in candidate_roots:
        if r and r not in roots:
            roots.append(r)
    if not roots:
        return ""
    anchor_files = [
        e.get("file", "") for e in (spec.get("edits") or [])
        if isinstance(e, dict) and e.get("kind", "edit") != "create_file" and e.get("file")
    ]
    if not anchor_files or len(roots) == 1:
        return roots[0]
    best_root, best_hits = roots[0], -1
    for r in roots:
        hits = sum(1 for rel in anchor_files if os.path.isfile(os.path.join(r, rel)))
        if hits > best_hits:
            best_root, best_hits = r, hits
    return best_root


def run_apply(
    spec_path: str,
    codebase_root: str | None = None,
    docs_root: str | None = None,
    output_path: str | None = None,
    write: bool = False,
    backup_root: str | None = None,
    ttl_hours: int = 168,
    partial: bool = False,
    verify: bool = False,
    runner: Any = None,
) -> dict[str, Any]:
    """Run the apply stage: edit-spec JSON + live code → proposal (and optional write).

    Loads the SSOT edit-spec, re-verifies each anchor against the live codebase,
    renders unified diffs, and writes a human-facing proposal markdown. Returns
    the proposal dict.

    With ``write=False`` (default) nothing is written to the target codebase. With
    ``write=True`` AND a READY proposal, the edits are applied to disk after the
    originals are snapshotted into a scratch backup bundle (all-or-nothing, with
    rollback). A non-ready proposal is never written.

    Args:
        spec_path: Path to the edit-spec JSON produced by specify.
        codebase_root: Live codebase root. Defaults to the spec's ``codebase_root``.
        docs_root: Optional separate design-doc tree, used as an additional base to
            resolve an edit's file path when code and docs live in different trees.
        output_path: Where to write the proposal markdown. If None, nothing is
            written to disk and the caller uses the returned dict.
        write: When True, apply a READY proposal's edits to the live codebase.
        backup_root: Directory for scratch backup bundles. Required when
            ``write`` is True.
        ttl_hours: Backup retention window; expired bundles are purged on write.

    Raises:
        FileNotFoundError: if the spec file is missing.
        ValueError: if codebase_root cannot be resolved (not given and absent
            from the spec), or if ``write`` is True without a ``backup_root``.
    """
    spec = load_spec(spec_path)
    spec["_spec_path"] = spec_path  # so a backup bundle can name itself after the spec

    # Resolve where the spec's files actually live. Candidates, in priority order:
    # an explicit --codebase, an explicit --docs (separate design-doc tree), and the
    # tree specify recorded in the spec. Whichever holds the anchor files wins, so
    # apply works whether the caller points --codebase at the code or the docs tree.
    root = _resolve_effective_root(
        spec, [codebase_root or "", docs_root or "", spec.get("codebase_root") or ""]
    )
    if not root:
        raise ValueError(
            "codebase_root not provided and not present in the edit-spec; "
            "cannot re-verify anchors against live code")
    if not os.path.isdir(root):
        logger.warning("apply: codebase_root %r is not a directory — anchor "
                       "re-verification will fail for every edit", root)

    term = spec.get("termination")
    if term is not None and term not in _VALID_TERMINATION:
        logger.warning("apply: spec termination is invalid: %r", term)

    proposal = build_proposal(spec, root)

    # Runtime red→green gate (the closed loop). build_proposal proves the anchor lands
    # and the effectiveness review reasons ABOUT the edit; this OBSERVES the target's own
    # test go red→green. It is opt-in (--verify) AND needs a configured runner + a
    # spec.verify.red_test_node — otherwise it is skipped, leaving today's behaviour. A
    # spec that ships a verify target but does NOT transition red→green is NOT ready: a
    # fix unconfirmed by execution must not be presented as applicable.
    if verify and runner is not None and isinstance(spec.get("verify"), dict) \
            and spec.get("verify", {}).get("red_test_node"):
        if not backup_root:
            raise ValueError("verify=True requires a backup_root (the dry-run snapshot)")
        from hive import verify as verify_mod  # lazy: verify imports apply
        rv = verify_mod.verify_red_green(spec, root, runner, backup_root, ttl_hours)
        proposal["runtime_verify"] = rv
        if rv["transition"] not in verify_mod.VERIFIED_TRANSITIONS:
            proposal["ready"] = False
            proposal["not_ready_reasons"].append(
                f"runtime verify did not confirm the fix: {rv['transition']} "
                f"({rv.get('reason', '')})")
            logger.warning("apply: runtime verify blocked READY — %s (%s)",
                           rv["transition"], rv.get("reason", ""))
        else:
            logger.info("apply: runtime verify CONFIRMED red→green for node %s",
                        rv.get("node"))

    if write:
        if not backup_root:
            raise ValueError("write=True requires a backup_root")
        if proposal["ready"]:
            proposal["write"] = write_edits(spec, root, backup_root, ttl_hours)
        elif partial and proposal["writable_ids"]:
            # Partial apply (Defect 3): the spec is not globally ready, but some
            # edits are individually applicable + effective. Write JUST those so a
            # verified root-cause fix ships instead of being blocked by a deferred
            # sibling. The deferred/unresolved items are reported, not applied.
            only = set(proposal["writable_ids"])
            logger.warning("apply: proposal NOT fully ready, but --partial set — "
                           "writing %d individually-ready edit(s) %s; %d item(s) "
                           "remain unresolved", len(only), sorted(only),
                           len(proposal["not_ready_reasons"]))
            w = write_edits(spec, root, backup_root, ttl_hours, only_ids=only)
            w["partial"] = True
            w["applied_ids"] = sorted(only)
            proposal["write"] = w
            proposal["applied_partial"] = bool(w.get("ok"))
        else:
            why = ("no individually-writable edit (every edit is non-applicable "
                   "or flagged ineffective)" if not proposal["writable_ids"]
                   else "proposal not ready — rerun with --partial to apply the "
                        f"{len(proposal['writable_ids'])} individually-ready edit(s)")
            logger.warning("apply: --write requested but proposal is NOT READY — "
                           "nothing written (%s)", why)
            proposal["write"] = {
                "ok": False, "attempted": False, "written": [], "bundle": None,
                "reason": why, "rolled_back": False,
            }
    else:
        proposal["write"] = None

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(render_proposal_markdown(proposal))
        logger.info("Apply proposal written to %s", output_path)

    verdict = "READY" if proposal["ready"] else "NOT READY"
    logger.info("Apply: %s — %d/%d edits applicable, termination=%s",
                verdict, proposal["n_applicable"], proposal["n_edits"],
                proposal["termination"])
    if not proposal["ready"]:
        for reason in proposal["not_ready_reasons"]:
            logger.info("  not-ready: %s", reason)

    w = proposal.get("write")
    if w and w.get("ok"):
        logger.info("Write: APPLIED %d file(s); backup at %s",
                    len(w.get("written", [])), w.get("bundle"))
    elif w and w.get("attempted"):
        logger.warning("Write: NOT applied — %s", w.get("reason"))
    return proposal
