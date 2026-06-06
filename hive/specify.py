"""Specify stage — lowers a honey's prose fix directions into a precise edit-spec.

Pipeline position (fix-extension):

  investigate (fan-out)  →  merge (honey)  →  SPECIFY (this)  →  apply (propose-only)

Unlike investigate, specify is NOT a swarm. A single consistent author takes the
assembled honey (whose fix directions are written as prose) plus the LIVE target
codebase and lowers each direction into a concrete ``anchor_old → replacement_new``
edit. Code edits must be internally coherent, so this stage is never fan-out.

Key invariants (mirrored from recipes/edit_spec_contract_v1.md):
  - Anchors come from LIVE code, never from the honey — the honey may be stale.
    The author records anchor_status (verified | stale | not_found) as feedback.
  - The spec defines its own boundary: a direction that cannot be expressed as an
    edit goes to ``deferred[]``, it is not forced into an edit.
  - Stage-1 safety: gate.apply is ALWAYS false here. specify proposes; the PM (or
    a later promotion) applies. specify never writes to the target codebase.
  - The JSON edit-spec is the SSOT; the human-facing unified diff is a DERIVED view
    rendered later by hive/apply.py — specify does not author the diff.
  - Effectiveness gate: an edit whose anchor is valid but whose change does not
    alter the behavior the honey identified — a no-op assignment, a guard whose
    condition can never be true, a whitespace-only diff — must NOT be presented as
    ready. After authoring, specify re-reads the edits (a deterministic no-op check
    plus an independent model review) and downgrades a ready_to_apply spec that does
    not actually change the reported behavior. A ready claim that cannot be verified
    loops back to re-investigate (needs_reinvestigation) rather than being trusted —
    there is no "hand it to a human" terminal state (the tool fixes autonomously; an
    unverifiable claim is re-worked, not punted).

The author's role prompt is the contract file itself, loaded at runtime so the
contract stays the single source of authoring rules (no duplicated prompt here).
"""

import difflib
import glob
import json
import logging
import os
import re
from typing import Any

from hive import dbread
from hive.http_shape_synth import synthesize_http_shape_red_test
from hive.investigate import SEED_TARGET_SECTION, CONVERGE_TARGET_SECTION
from hive.parse import extract_first_json
from hive.providers import call_worker

logger = logging.getLogger("hive.specify")

# The authoring contract doubles as the specify author's role/system prompt.
_DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "recipes", "edit_spec_contract_v1.md"
)

# The single specify author can be a SLOW agentic CLI (codex), unlike the tool-OFF
# API reviewer. This is its per-attempt wall-clock cap; it is overridable per-role
# via hive.config.json (``specify.timeout_sec``) so the operator can right-size it.
_AUTHOR_TIMEOUT_DEFAULT = 600

# Providers whose author worker has live file-system tools. An author on any other
# provider (e.g. deepinfra) is a tool-OFF single-shot call and must lift anchors
# from the grounding pre-flight's "Anchor ground truth" block, not by reading files.
_TOOL_PROVIDERS = frozenset({"copilot"})

# Structural expectations for the emitted edit-spec JSON.
_REQUIRED_KEYS = ("edits", "deferred", "gate", "termination")
# CANONICAL termination vocabulary — the single source of truth shared with apply
# (imported there as VALID_TERMINATION) so the two can never drift. ``needs_runtime`` is
# first-class: investigate.py instructs the author to emit it when a fix's correctness
# depends on a runtime fact that cannot be confirmed statically (N174 — apply used to
# reject it as "invalid" because its copy of this set was stale). There is NO ``needs_pm``:
# the tool's whole purpose is to fix autonomously, so there is no "hand this to a human"
# terminal — a claim it cannot stand behind loops back to re-investigate, not punt.
VALID_TERMINATION = {"ready_to_apply", "needs_reinvestigation", "needs_runtime"}
_VALID_TERMINATION = VALID_TERMINATION  # backward-compatible local alias
_STALE_STATUSES = {"stale", "not_found"}

# Effectiveness-gate outcomes. An ineffective edit means the fix does not change
# behavior; an inconclusive review (the check could not be obtained) likewise cannot
# vouch for a ready claim. BOTH loop back to re-investigate — the tool re-works the
# fix autonomously rather than punting an unverifiable claim to a human.
_INEFFECTIVE_TERMINATION = "needs_reinvestigation"
_INCONCLUSIVE_TERMINATION = "needs_reinvestigation"

# Decisiveness gate: a conservatively-authored needs_reinvestigation spec whose edits are
# all verified+effective+confident is promoted to ready_to_apply (never a blanket drop).
_DECISIVE_CONFIDENCE = {"high", "medium"}
# Deferred reasons that are genuinely "optional/surface" — a side note that does not
# contradict the edits, so its presence must not block applying an independently-verified
# edit. ONLY policy_direction qualifies: a policy/UX suggestion the fix does not depend on.
#
# ``not_expressible_as_edit`` and ``multi_file_design`` were here too — that was the N176
# hole. They do not mean "optional"; they mean "the real fix is BIGGER than what I
# authored" (it spans multiple files, or could not be reduced to a single anchored edit).
# Treating them as harmless let specify ship the easy half of a fix as ready_to_apply while
# filing the hard root cause as a footnote (N176: a front-end one-liner shipped "done"
# while the back-end cause that actually clears the symptom sat in deferred). They now live
# in _SUBSTANTIVE_DEFERRED_REASONS and BLOCK a ready claim instead of being waved through.
_OPTIONAL_DEFERRED_REASONS = {"policy_direction"}
# Deferred reasons that say a SUBSTANTIVE fix was punted — very often the actual root
# cause, filed away while the symptom-level edits ship. A ready_to_apply spec carrying one
# of these is downgraded to needs_reinvestigation (the deferred-substance gate) so the loop
# re-works the full fix rather than vouching for the partial one.
_SUBSTANTIVE_DEFERRED_REASONS = {"not_expressible_as_edit", "multi_file_design"}

# A deferral whose OWN grounding says the claimed defect was REFUTED by live code is a
# disproven hypothesis, not a punted root cause (T907: the honey alleged a
# RequirementCreateView shape-mismatch; the author opened the live file, found it maps the
# API rows into {id,label} objects, and recorded that refutation in the deferral's
# evidence). Counting such a deferral as a substantive punt re-opens a settled question and
# downgrades a genuinely complete fix, so the deferred-substance gate exempts it.
_LIVE_REFUTED_RE = re.compile(
    r"not\s+observed\s+in\s+(?:the\s+)?live(?:\s+code)?|"
    r"not\s+present\s+in\s+(?:the\s+)?live|"
    r"\brefuted\b|\bdisproven\b|\bdisproved\b|contradicted\s+by\s+(?:the\s+)?live|"
    r"does\s+not\s+(?:match|appear|exist)\s+in\s+(?:the\s+)?live|"
    r"live\s+code\s+(?:shows|proves)[^.]*\bnot\b|"
    r"라이브(?:\s*코드)?(?:에서)?[^.]*(?:반박|반증)|반박됨|반증됨|관측되지\s*않",
    re.IGNORECASE)

# ── Structured reinvestigation reason codes (Step A) ───────────────────────────
# Every site that lands a spec in needs_reinvestigation stamps a MACHINE-READABLE
# reason code into ``spec["reinvestigation"]`` so the reactive re-investigation can
# route by CAUSE (which hole) instead of re-parsing the prose ``notes``. The LAST
# gate to fire wins — the structured field is overwritten, so it is the single
# source of truth for which gate had final say (the prose notes still accumulate
# the full trail). Each code maps to the targeted action the bridge should take:
#   stale_anchor          → re-anchor against live source (NOT a re-investigation)
#   anchor_not_grounded   → targeted re-retrieve of the missing grounding
#   ineffective           → exclude the ruled-out node, re-converge
#   inconclusive          → re-review / re-converge (effectiveness unconfirmed)
#   seed_target_uncovered → re-author the edit for the dropped seed target
#   converge_locus_uncovered → re-author the dropped INDEPENDENT defect locus (multi-locus)
#   deferred_root_cause   → re-retrieve the punted substantive fix's axis (if thin)
#   legacy_coerce         → a retired needs_pm coerced here (no real gap)
#   author_declared       → the author itself emitted NR (read its own detail)
RI_STALE_ANCHOR = "stale_anchor"
RI_ANCHOR_NOT_GROUNDED = "anchor_not_grounded"
RI_INEFFECTIVE = "ineffective"
RI_INCONCLUSIVE = "inconclusive"
RI_SEED_TARGET_UNCOVERED = "seed_target_uncovered"
RI_CONVERGE_LOCUS_UNCOVERED = "converge_locus_uncovered"
RI_DEFERRED_ROOT_CAUSE = "deferred_root_cause"
RI_DATASOURCE_REGRESSION = "datasource_regression"
RI_LEGACY_COERCE = "legacy_coerce"
RI_AUTHOR_DECLARED = "author_declared"


def _append_note(spec: dict[str, Any], note: str) -> None:
    """Append ``note`` to ``spec['notes']`` (space-joined), preserving prior trail."""
    prev = str(spec.get("notes", "")).strip()
    spec["notes"] = f"{prev} {note}".strip() if prev else note


def _set_reinvestigation(spec: dict[str, Any], *, reason_code: str, gate: str,
                         note: str, detail: str | None = None) -> dict[str, Any]:
    """Land ``spec`` in needs_reinvestigation with a structured, routable reason.

    Sets the canonical termination, stamps ``spec['reinvestigation']`` (last writer
    wins → the gate with final say) and appends ``note`` to the prose trail. This is
    the ONE place termination becomes needs_reinvestigation inside the gates, so the
    reactive bridge always finds a machine-readable cause, never just prose.
    """
    spec["termination"] = "needs_reinvestigation"
    spec["reinvestigation"] = {
        "reason_code": reason_code,
        "gate": gate,
        "detail": (detail if detail is not None else note),
    }
    _append_note(spec, note)
    return spec


def _ensure_reinvestigation_reason(spec: dict[str, Any]) -> dict[str, Any]:
    """Stamp an author-declared reason when NR was emitted by the author, not a gate.

    Runs last in ``run_specify``. A gate that downgrades to needs_reinvestigation
    always stamps ``spec['reinvestigation']``; if the spec is in that terminal state
    WITHOUT the structured field, the author worker emitted it directly. We record
    ``author_declared`` (carrying the author's own narrative) so the reactive bridge
    never meets a needs_reinvestigation with no routable cause. A non-NR spec keeps
    no stray reason field.
    """
    if spec.get("termination") != "needs_reinvestigation":
        spec.pop("reinvestigation", None)
        return spec
    if not isinstance(spec.get("reinvestigation"), dict):
        spec["reinvestigation"] = {
            "reason_code": RI_AUTHOR_DECLARED,
            "gate": "author",
            "detail": str(spec.get("notes", "")).strip()[:300] or "author emitted "
            "needs_reinvestigation",
        }
    return spec


def load_contract(contract_path: str | None = None) -> str:
    """Load the edit-spec authoring contract (the specify author's role prompt)."""
    path = contract_path or _DEFAULT_CONTRACT_PATH
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Anchor-grounding pre-flight (NR164/NR165/TR891 systematic gap) ──────────────
# The honey carries the LOCATION of the bug (judge file:line) but not the VALUE at
# it: e.g. CSS_RULES grounded ``DocWorkflow.vue:89-101`` yet never quoted the
# actual ``color`` of ``.wf-step.wf-undecided``. The author then re-anchors a
# location whose current value it never saw, and the effectiveness reviewer cannot
# judge whether the edit turns grey→blue — so a correct fix dies as
# needs_reinvestigation. The fix (the constructive form of the deferred grounding
# gate): before authoring, lift the CURRENT live text at each cited file:line into
# the honey so the value is on the table for BOTH the author and the reviewer.
# Pure-local, free, deterministic, never raises — an unresolvable or already-quoted
# citation is simply skipped (it never fabricates).

# A path-like token ending in an extension, then ``:line`` or ``:lo-hi``. The
# required extension is what stops a log timestamp ("21:33:01") or a bare
# "name:42" from matching — only real file citations resolve.
_CITATION_RE = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+):(\d+)(?:-(\d+))?")

_GROUND_MAX_ANCHORS = 12       # cap lifts so a citation-heavy honey can't balloon
_GROUND_MAX_LINES = 40         # per-anchor line cap (a huge range is clamped)
_GROUND_MIN_LINE_CHARS = 8     # lines shorter than this don't vote in the dedup guard
# Seed-named targets get a GENEROUS forward window (Defect 2 / T892): the seed
# cites an APPROXIMATE range ("Around lines 299-322"), but the exact assertion
# block to rewrite often sits past it in a large test file. A 40-line slice
# truncated before the real expectations, so specify had no byte-for-byte anchor
# and deferred the file a third time. Lifting a wider block from the cited start
# puts the full enclosing case set on the table for the tool-OFF author.
_GROUND_WIDE_LINES = 120
# Total line budget across ALL lifts — a hard ceiling so even a citation-storm of
# distinct regions can't balloon the author prompt past a workable size (T892
# timeout: 8 overlapping wide windows blew the codex author past its 600s cap).
# Sized to hold ~5 distinct wide windows; merge removes the redundancy first so the
# budget only bites a genuinely pathological number of DISTINCT regions.
_GROUND_MAX_TOTAL_LINES = 600
# Ceiling on a single MERGED window — the union of two overlapping wide windows can
# exceed one window's cap; this bounds it so a merge can't itself balloon.
_GROUND_MERGED_MAX_LINES = 200


def _merge_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or directly-adjacent ``[lo, hi]`` line ranges (1-based).

    The grounding loop previously deduped on the EXACT ``(file, lo, hi)`` key, so
    three near-duplicate windows of one region (queries.json 124-243 / 127-246 /
    129-248 — all the same ~19-line tail of a 142-line file) each lifted in full and
    tripled the author prompt (T892). Collapsing overlapping ranges into one window
    per region kills that redundancy. Ranges touching at the boundary (``hi+1 ==
    next.lo``) are merged too, since a one-line gap is not worth a second fenced block.
    """
    if not ivs:
        return []
    ordered = sorted(ivs)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + 1:          # overlap or directly adjacent
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _read_lines(path: str, lo: int, hi: int) -> str:
    """Read 1-based line range [lo, hi] from a file. '' on any error/empty."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return ""
    lo = max(1, lo)
    hi = min(len(all_lines), hi)
    if hi < lo:
        return ""
    return "".join(all_lines[lo - 1:hi])


def _already_grounded(text: str, honey_text: str) -> bool:
    """True when the honey already quotes this snippet's substance (dedup guard).

    Only substantial lines vote, so a snippet whose every meaningful line already
    appears in the honey is not re-lifted (e.g. an assemble-authored honey that
    quoted the code). A snippet with no substantial line is treated as not-present.
    """
    sig = [ln.strip() for ln in text.splitlines()
           if len(ln.strip()) >= _GROUND_MIN_LINE_CHARS]
    return bool(sig) and all(ln in honey_text for ln in sig)


def _in_wide(rel: str, wide: set[str]) -> bool:
    """True when ``rel`` is one of the seed-named wide-grounding files (path-aligned)."""
    r = rel.lower()
    return any(r == w or r.endswith("/" + w) or w.endswith("/" + r)
               or os.path.basename(w) == os.path.basename(r) for w in wide)


def ground_anchors(honey_text: str, codebase_root: str,
                   docs_root: str | None = None,
                   *, max_anchors: int = _GROUND_MAX_ANCHORS,
                   max_lines: int = _GROUND_MAX_LINES,
                   wide_files: set[str] | None = None,
                   wide_lines: int = _GROUND_WIDE_LINES) -> tuple[str, dict[str, Any]]:
    """Lift the CURRENT live text at each ``file:line`` the honey cites into the honey.

    Returns ``(enriched_honey, diag)`` where ``diag`` records lifted / skipped /
    unresolved citations for logging. Citations are resolved against the code tree
    first, then the docs tree. Already-quoted or unresolvable citations are skipped.
    Pure-local and free — no model call — and never raises.

    ``wide_files`` (seed-named edit targets) get a GENEROUS forward window of
    ``wide_lines`` from the cited start, so an approximate seed range still pulls
    the full enclosing block — clamped to the file length (Defect 2 / T892).
    """
    lifted: list[dict[str, Any]] = []
    skipped_present: list[str] = []
    unresolved: list[str] = []
    roots = [("code", codebase_root)] + ([("docs", docs_root)] if docs_root else [])
    wide = {f.replace("\\", "/").lstrip("/").lower() for f in (wide_files or set())}

    # Pass 1 — collect the RAW honey-cited ranges per file, preserving the order in
    # which each file first appears (so lifts read in citation order downstream).
    raw: dict[str, list[tuple[int, int]]] = {}
    order: dict[str, int] = {}
    for idx, m in enumerate(_CITATION_RE.finditer(honey_text)):
        rel = m.group(1).replace("\\", "/").lstrip("/")
        lo = int(m.group(2))
        hi = int(m.group(3)) if m.group(3) else lo
        if hi < lo:
            lo, hi = hi, lo
        order.setdefault(rel, idx)
        raw.setdefault(rel, []).append((lo, hi))

    # Pass 2 — expand each citation to its window (forward-widening seed targets),
    # THEN merge overlapping windows per file. Expanding BEFORE merging is what
    # collapses near-duplicate windows of one region: three 120-line windows that
    # only overlap once widened (queries.json 124-243 / 127-246 / 129-248 of a
    # 142-line file) merge to a single lift instead of three (the T892 balloon).
    windows: list[tuple[int, str, int, int, bool]] = []  # (order, rel, lo, hi, truncated)
    for rel, ivs in raw.items():
        is_wide = bool(wide) and _in_wide(rel, wide)
        cap = wide_lines if is_wide else max_lines
        expanded: list[tuple[int, int]] = []
        trunc_hi: dict[int, int] = {}   # lo -> requested hi before the per-window cap
        for lo, hi in ivs:
            if is_wide:                          # seed target: forward window from start
                hi = max(hi, lo + cap - 1)
            if hi - lo + 1 > cap:                # clamp a single window to its cap
                trunc_hi[lo] = hi
                hi = lo + cap - 1
            expanded.append((lo, hi))
        for lo, hi in _merge_intervals(expanded):
            # A merged span may exceed a single window's cap (two overlapping wide
            # windows union wider); clamp only at the merged ceiling so the union of
            # two near-duplicate regions is kept whole rather than re-split.
            requested_hi = max([hi] + [v for k, v in trunc_hi.items() if lo <= k <= hi])
            if requested_hi - lo + 1 > _GROUND_MERGED_MAX_LINES:
                hi = lo + _GROUND_MERGED_MAX_LINES - 1
            else:
                hi = min(hi, requested_hi)
            truncated = hi < requested_hi
            windows.append((order[rel], rel, lo, hi, truncated))
    windows.sort(key=lambda w: (w[0], w[2]))

    total_lines = 0
    for _ord, rel, lo, hi, truncated in windows:
        text, tree = "", ""
        for label, root in roots:
            if root and os.path.isfile(os.path.join(root, rel)):
                text = _read_lines(os.path.join(root, rel), lo, hi)
                tree = label
                break
        if not text.strip():
            unresolved.append(f"{rel}:{lo}-{hi}")
            continue
        if _already_grounded(text, honey_text):
            skipped_present.append(f"{rel}:{lo}-{hi}")
            continue
        n_lines = text.count("\n") + 1
        # Total-line budget: stop lifting once the cumulative block size would
        # exceed the ceiling, so a citation-storm can't balloon the prompt even
        # when every window is a distinct region (count cap alone is not enough).
        if lifted and total_lines + n_lines > _GROUND_MAX_TOTAL_LINES:
            break
        total_lines += n_lines
        lifted.append({"cite": f"{rel}:{lo}-{hi}", "tree": tree,
                       "text": text, "truncated": truncated})
        if len(lifted) >= max_anchors:
            break

    diag = {"lifted": [x["cite"] for x in lifted],
            "skipped_present": skipped_present, "unresolved": unresolved}
    if not lifted:
        return honey_text, diag

    parts = [
        honey_text, "",
        "## Anchor ground truth (CURRENT live text at each cited location)", "",
        "The fix directions above give LOCATIONS but not always the VALUES at them. "
        "Below is the text lifted live from those lines. Author the edit so it changes "
        "these VALUES (do not merely re-anchor at the same location), and judge "
        "effectiveness against them. Re-confirm byte-for-byte before anchoring.", "",
    ]
    for x in lifted:
        trunc = " (truncated)" if x["truncated"] else ""
        parts += [f"--- {x['cite']} (under {x['tree']} root){trunc}",
                  "```", x["text"].rstrip("\n"), "```", ""]
    return "\n".join(parts), diag


def _docs_root_block(docs_root: str | None, grounded_only: bool = False) -> str:
    """Prompt fragment telling the author about a separate design-doc tree.

    Design docs (PM-facing D/P/L specs) commonly live in a tree separate from the
    code. The author worker runs with cwd=codebase_root and would otherwise never
    see them — so a "edit the D031 design doc" direction gets mis-lowered onto the
    nearest-looking source file. This block makes the docs tree visible and tells
    the author to PREFER it when the honey's directive is about a document, and to
    emit ``file`` relative to whichever tree actually holds the target.

    Under ``grounded_only`` (a tool-OFF author) the doc anchor, like a code anchor,
    is lifted from the injected ground-truth block rather than by opening the file.
    """
    if not docs_root:
        return ""
    lift = (
        "the doc's CURRENT text is in the \"Anchor ground truth\" block below (grounding "
        "resolves doc citations against this tree too) — lift `anchor_old` from there"
        if grounded_only else
        "open it there (docs root + the path the honey cites) and lift `anchor_old` from "
        "the CURRENT text byte-for-byte")
    return f"""
[Design-docs root — a SEPARATE tree from the code]
{docs_root}
When the honey's fix direction targets a design document (e.g. a Markdown D/P/L
spec), the file lives under this docs root, NOT the codebase root: {lift}.
Prefer the design document over any source file when the directive is about the
document. Emit `file` as the path relative to the tree that holds it.
"""


def _stamp_root(spec: dict[str, Any], codebase_root: str, docs_root: str | None) -> str:
    """Pick which tree to record as the spec's ``codebase_root``.

    Returns ``docs_root`` when the spec's anchor edits resolve under the docs tree
    but not the code tree (a design-doc edit); otherwise ``codebase_root``. Probing
    by file existence keeps this deterministic and avoids trusting the author's
    own (possibly wrong) sense of which tree it edited. ``create_file`` edits are
    absent by design and don't vote.
    """
    if not docs_root:
        return codebase_root
    anchor_files = [
        e.get("file", "") for e in (spec.get("edits") or [])
        if isinstance(e, dict) and e.get("kind", "edit") != "create_file" and e.get("file")
    ]
    if not anchor_files:
        return codebase_root
    in_docs = sum(1 for f in anchor_files if os.path.isfile(os.path.join(docs_root, f)))
    in_code = sum(1 for f in anchor_files if os.path.isfile(os.path.join(codebase_root, f)))
    return docs_root if in_docs > in_code else codebase_root


def build_specify_prompt(honey_text: str, contract_text: str, codebase_root: str,
                         docs_root: str | None = None,
                         grounded_only: bool = False) -> str:
    """Build the full prompt for the single specify author.

    The contract is the role/system prompt; the honey is the input to lower; the
    codebase_root tells the author where the LIVE code is. When ``docs_root`` is
    given, a separate design-doc tree is also made visible so a document-update
    direction is not mis-lowered onto the nearest source file.

    Two authoring modes:

    - tool-ON (``grounded_only=False``, the copilot author): the author re-opens
      each live file and lifts ``anchor_old`` byte-for-byte, trusting nothing the
      honey quotes.
    - tool-OFF (``grounded_only=True``, a single-shot deepinfra author): the author
      has NO file tools, so it lifts ``anchor_old`` from the deterministically
      live-lifted "Anchor ground truth" block the grounding pre-flight injected, and
      defers any direction whose value is not grounded rather than guessing. This is
      what lets the author move off the per-internal-turn-billed copilot (the
      retrieve→judge cost pattern applied to ground→author).
    """
    if grounded_only:
        lift_block = (
            "[Input honey — lower each fix direction into the edit-spec contracted above]\n"
            "You have NO file-system tools and CANNOT open files. The "
            "\"Anchor ground truth\" section in the honey below carries the CURRENT live "
            "text at each cited location, lifted deterministically from disk — treat it "
            "as authoritative and lift `anchor_old` from THERE byte-for-byte (set "
            "anchor_status=verified only for an anchor copied from that section). If a "
            "value you must anchor is NOT present in the ground truth, put that direction "
            "in `deferred[]` (reason: \"anchor_not_grounded\") rather than guessing — never "
            "fabricate an anchor from the honey's prose. "
            "IMPORTANT: an anchor_not_grounded deferred item and an edit for the SAME FILE "
            "are a CONTRADICTION — a file cannot be both ungroundable and successfully anchored. "
            "A direction belongs in one place only: edits[] (grounded) OR deferred[] "
            "(ungrounded). Never emit both for the same file.")
    else:
        lift_block = (
            "[Input honey — lower each fix direction into the edit-spec contracted above]\n"
            "Re-open every file you touch (under the codebase root, or the docs root when "
            "the direction targets a design document) and lift `anchor_old` from the "
            "CURRENT text byte-for-byte. Do NOT trust code quoted in the honey below.")
    return f"""{contract_text}

[Codebase root — read LIVE files from here]
{codebase_root}
{_docs_root_block(docs_root, grounded_only=grounded_only)}
{lift_block}
When the honey calls for a brand-new file that does not yet exist in the codebase,
emit a `create_file` edit (kind + content, no anchor) per the contract rather than
forcing an anchor edit against an existing file.

{honey_text}
"""


def _unique_line_window(text: str, start: int, end: int,
                        max_grow: int = 12) -> tuple[str, int] | None:
    """Grow ``[start, end)`` outward to whole-line boundaries until that slice is unique.

    Returns ``(window_text, window_start_offset)`` once ``text.count(window_text) == 1``,
    or ``None`` if it cannot be made unique within ``max_grow`` lines of context on each
    side. Used to disambiguate a literal that repeats across branches. Deterministic,
    never raises.
    """
    ls = text.rfind("\n", 0, start) + 1              # start of the line holding `start`
    le = text.find("\n", end)
    le = len(text) if le == -1 else le + 1           # just past the line holding `end-1`
    for grow in range(0, max_grow + 1):
        ws = ls
        for _ in range(grow):
            if ws == 0:
                break
            prev = text.rfind("\n", 0, ws - 1)
            ws = 0 if prev == -1 else prev + 1
        we = le
        for _ in range(grow):
            if we >= len(text):
                break
            nxt = text.find("\n", we)
            we = len(text) if nxt == -1 else nxt + 1
        window = text[ws:we]
        if text.count(window) == 1:
            return window, ws
    return None


def _disambiguate_anchors(spec: dict[str, Any], codebase_root: str,
                          docs_root: str | None = None) -> dict[str, Any]:
    """Widen a non-unique anchor into a uniquely-targeting one (free, deterministic).

    A literal can appear byte-identical in several branches (e.g. the same
    ``NON_HEAD_TYPES = {...}`` guard in two code paths). When the author writes one edit
    per occurrence but anchors them all with that identical snippet, apply sees
    ``occurrences > 1`` and refuses every one as ``anchor_ambiguous`` — the whole fix
    dead-ends in re-investigation over a purely mechanical collision (the M-head case).

    This pass closes that the safe way: for a group of edits that share the SAME
    ``(file, anchor_old)`` where the anchor occurs in live source exactly as many times
    as there are edits in the group AND all of them carry the SAME ``replacement_new``
    (so which occurrence maps to which edit is immaterial), each edit's ``anchor_old`` is
    grown with adjacent live lines until it targets one occurrence uniquely, and its
    ``replacement_new`` is grown the identical way so the inner change is preserved
    byte-for-byte. Anything that cannot be made provably unique is left untouched for
    ``_verify_anchors_live`` to downgrade as before. Runs BEFORE the live-verify pass so a
    successfully widened anchor then passes the uniqueness bar and stays ``verified``.
    Never raises; skips ``create_file`` edits and files it cannot read.
    """
    roots = [r for r in (codebase_root, docs_root) if r]
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in spec.get("edits") or []:
        if not isinstance(e, dict) or e.get("kind", "edit") == "create_file":
            continue
        anchor = e.get("anchor_old") or ""
        rel = e.get("file") or ""
        if anchor and rel:
            groups.setdefault((rel, anchor), []).append(e)

    for (rel, anchor), group in groups.items():
        if len(group) < 2:
            continue  # a lone edit on a non-unique anchor is genuinely ambiguous — leave it
        replacements = {e.get("replacement_new") for e in group}
        if len(replacements) != 1:
            continue  # differing replacements → occurrence↔edit mapping is not safe to guess
        text: str | None = None
        for root in roots:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    text = None
                break
        if text is None:
            continue
        starts: list[int] = []
        i = text.find(anchor)
        while i != -1:
            starts.append(i)
            i = text.find(anchor, i + 1)
        if len(starts) != len(group):
            continue  # occurrence count must match edit count to assign one-to-one safely
        repl = next(iter(replacements)) or ""
        windows: list[tuple[int, tuple[str, int]]] = []
        for occ in starts:
            win = _unique_line_window(text, occ, occ + len(anchor))
            if win is None:
                break
            windows.append((occ, win))
        if len(windows) != len(starts):
            continue  # at least one occurrence could not be made unique — leave the group
        for e, (occ, (w_text, w_start)) in zip(group, windows):
            a = occ - w_start
            b = a + len(anchor)
            new_repl = w_text[:a] + repl + w_text[b:]
            if text.count(w_text) != 1 or w_text == new_repl:
                continue  # belt-and-braces: only rewrite to a unique, still-changing anchor
            e["anchor_old"] = w_text
            e["replacement_new"] = new_repl
            e["anchor_disambiguated"] = (
                f"anchor_old occurred {len(starts)}x in live {rel}; widened with adjacent "
                "lines to target one occurrence uniquely")
        logger.info("specify: disambiguated %d non-unique anchors in %s (widened to "
                    "unique windows)", len(group), rel)
    return spec


def _verify_anchors_live(spec: dict[str, Any], codebase_root: str,
                         docs_root: str | None = None) -> dict[str, Any]:
    """Re-read each anchor from LIVE disk and downgrade a false ``verified`` status.

    N175 E7: specify stamped ``anchor_status=verified`` on an edit whose ``anchor_old``
    was NOT actually present in the live file, and apply only discovered the DRIFT at
    apply time. The authoring contract already PLEADS "re-confirm byte-for-byte before
    anchoring" — but a prose plea to a single-shot author is exactly the soft layer that
    keeps failing. This is the deterministic backstop: for every anchor edit we open the
    live file and count ``anchor_old``; ``verified`` survives ONLY when the anchor occurs
    EXACTLY ONCE (the same bar apply applies). 0 occurrences → ``not_found`` (drift);
    >1 → ``stale`` (ambiguous, cannot target one edit). Both land in ``_STALE_STATUSES``,
    so ``_normalize_spec`` then downgrades a ready_to_apply spec to needs_reinvestigation.

    Only ever DOWNGRADES — a live read cannot vouch that a non-verified anchor is correct,
    so a status the author left as stale/not_found/anchor-less is untouched, as is any file
    we cannot read (apply re-verifies once more against live code regardless). Free, local,
    deterministic, never raises. ``create_file`` edits have no anchor and are skipped.
    """
    roots = [r for r in (codebase_root, docs_root) if r]
    drifted: list[str] = []
    for e in spec.get("edits") or []:
        if not isinstance(e, dict) or e.get("kind", "edit") == "create_file":
            continue
        if str(e.get("anchor_status", "")).lower() != "verified":
            continue  # only a 'verified' claim needs the live cross-check
        anchor = e.get("anchor_old") or ""
        rel = e.get("file") or ""
        if not anchor or not rel:
            continue
        text: str | None = None
        for root in roots:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    text = None
                break
        if text is None:
            continue  # cannot read the live file → leave it; apply re-checks at write
        count = text.count(anchor)
        if count == 1:
            continue  # genuinely verified — anchor is uniquely present in live code
        new_status = "not_found" if count == 0 else "stale"
        e["anchor_status"] = new_status
        e["anchor_drift"] = (
            f"specify marked verified but anchor_old occurs {count}x in live {rel} "
            f"(expected exactly 1) — downgraded to {new_status}")
        drifted.append(str(e.get("id", "?")))
    if drifted:
        logger.warning("specify: anchor drift — edits %s claimed 'verified' but their "
                       "anchor is absent/non-unique in live code; downgraded so the "
                       "spec is not presented as ready", drifted)
    return spec


def _norm_anchor_line(line: str) -> str:
    """Collapse a line to its whitespace-insignificant form for anchor matching."""
    return re.sub(r"\s+", " ", line.strip())


def _reanchor_drifted(spec: dict[str, Any], codebase_root: str,
                      docs_root: str | None = None) -> dict[str, Any]:
    """Recover an anchor that drifted by WHITESPACE only — re-lift the exact live bytes.

    N178: a single high-confidence edit whose ``anchor_old`` is a few bytes off against
    live (indentation widened, a tab vs spaces, a trailing space, a CRLF) dies as
    ``not_found``, and the loop has NO deterministic recovery — it routes to a
    needs_reinvestigation that, for ``stale_anchor``, the reactive bridge can only
    terminate ("re-anchor is specify-local, not routable"). The design always meant
    ``stale_anchor → re-anchor against live source`` (the comment at the reason-code
    table) but that pass was never built. This is it.

    For each edit ``_verify_anchors_live`` just downgraded to ``not_found`` (count 0 —
    NOT ``stale``/count>1, which is a genuine ambiguity ``_disambiguate_anchors`` owns),
    find the UNIQUE contiguous live region whose lines equal ``anchor_old`` line-for-line
    ONCE insignificant whitespace is normalized. Re-lift those EXACT live bytes as the new
    ``anchor_old`` (so apply finds it), and rebuild ``replacement_new`` so its UNCHANGED
    context lines come from live verbatim while the author's CHANGED lines (the real edit)
    are kept as authored — the change is preserved, only the surrounding bytes are
    re-synced to live. On success the edit is re-marked ``verified``.

    Strictly fail-closed: a drift that is NOT a provable 1:1 whitespace re-sync (no match,
    more than one normalized match, a mismatched line count, or a rebuild that collapses to
    a no-op or to a non-unique anchor) is left as ``not_found`` so ``_normalize_spec`` still
    downgrades it — an honest NR beats a guessed location. Free, local, deterministic,
    never raises, never fabricates a location.
    """
    roots = [r for r in (codebase_root, docs_root) if r]
    recovered: list[str] = []
    for e in spec.get("edits") or []:
        if not isinstance(e, dict) or e.get("kind", "edit") == "create_file":
            continue
        if str(e.get("anchor_status", "")).lower() != "not_found":
            continue  # only a count==0 drift is a re-anchor candidate (stale=ambiguous)
        anchor = e.get("anchor_old") or ""
        repl = e.get("replacement_new") or ""
        rel = e.get("file") or ""
        if not anchor or not rel:
            continue
        text: str | None = None
        for root in roots:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    text = None
                break
        if text is None:
            continue

        a_keep = anchor.splitlines(keepends=True)
        if not a_keep:
            continue
        norm_a = [_norm_anchor_line(ln) for ln in a_keep]
        live_keep = text.splitlines(keepends=True)
        norm_live = [_norm_anchor_line(ln) for ln in live_keep]
        n = len(a_keep)
        # contiguous windows of live whose normalized lines equal anchor's, exactly once
        hits = [i for i in range(0, len(live_keep) - n + 1)
                if norm_live[i:i + n] == norm_a]
        if len(hits) != 1:
            continue  # 0 = genuinely gone; >1 = ambiguous — both stay not_found (honest)
        i = hits[0]
        live_window = live_keep[i:i + n]
        new_anchor = "".join(live_window)
        if text.count(new_anchor) != 1:
            continue  # the re-lifted bytes must themselves be unique for apply to target

        # Rebuild replacement: context (lines equal between author's anchor & replacement)
        # comes from LIVE verbatim; changed lines come from the author's replacement.
        r_keep = repl.splitlines(keepends=True)
        norm_r = [_norm_anchor_line(ln) for ln in r_keep]
        rebuilt: list[str] = []
        for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(
                a=norm_a, b=norm_r, autojunk=False).get_opcodes():
            if tag == "equal":
                rebuilt.extend(live_window[a1:a2])  # unchanged context → live bytes
            else:
                rebuilt.extend(r_keep[b1:b2])        # author's change → as authored
        new_repl = "".join(rebuilt)
        if _normalize_ws(new_repl) == _normalize_ws(new_anchor):
            continue  # rebuilt to a no-op — refuse to vouch (leave as not_found)

        e["anchor_old"] = new_anchor
        e["replacement_new"] = new_repl
        e["anchor_status"] = "verified"
        e.pop("anchor_drift", None)
        e["reanchored"] = (
            f"whitespace-drift re-anchor in {rel}: anchor_old was byte-off against live; "
            "re-lifted the exact live text (context from live, change preserved)")
        recovered.append(str(e.get("id", "?")))
    if recovered:
        logger.info("specify: re-anchored %s — anchor drifted by whitespace only, "
                    "re-lifted exact live text (no re-investigation needed)", recovered)
    return spec


# Edit-object fields the author may flatten onto the spec ROOT when it emits a single
# edit WITHOUT the edits[] envelope. Moved back into the edit on rewrap; everything not
# listed (gate, effectiveness, deferred, termination, notes, source_honey, codebase_root)
# is envelope-level and stays at the root.
_EDIT_LEVEL_KEYS = (
    "id", "file", "anchor_old", "replacement_new", "rationale", "evidence",
    "confidence", "anchor_status", "anchor_drift", "kind", "content", "occurrence",
)


def _rewrap_flattened_edit(spec: dict[str, Any]) -> dict[str, Any]:
    """Salvage a single edit the author flattened into the spec root.

    A well-formed spec carries its edits inside ``edits[]``; the envelope itself never
    holds ``anchor_old``/``replacement_new``. When an author emits ONE edit with those
    fields hoisted to the root and no ``edits`` list, every downstream pass that iterates
    ``spec["edits"]`` no-ops and the (often perfectly good) fix is silently dropped as
    "0 edits, not ready" (N175: a verified, high-confidence showToast/i18n fix was lost
    to exactly this shape). Detect that and wrap the root's edit-level fields back into
    ``edits=[{...}]`` so the normal verify/effectiveness/decisiveness pipeline can rule on
    it — the decisiveness gate, not this salvage, decides ready_to_apply.

    Conservative by construction: fires ONLY when ``edits`` is missing/empty AND the root
    looks like a concrete edit (an anchor pair, or a create_file with content) — a shape a
    real envelope never has, so false positives are not possible. Never raises.
    """
    edits = spec.get("edits")
    if isinstance(edits, list) and edits:
        return spec  # already a proper envelope
    is_anchor_edit = bool(spec.get("anchor_old")) and bool(spec.get("replacement_new"))
    is_create = (spec.get("kind") == "create_file"
                 and bool(str(spec.get("content") or "").strip()))
    if not (is_anchor_edit or is_create):
        return spec  # not a flattened edit — leave untouched
    edit = {k: spec.pop(k) for k in _EDIT_LEVEL_KEYS if k in spec}
    edit.setdefault("id", "E1")
    spec["edits"] = [edit]
    spec.setdefault("deferred", [])
    # No termination rode at the root; default to the conservative loop-back so the
    # decisiveness gate (after anchor-verification + the effectiveness review) is what
    # promotes to ready_to_apply, never this salvage on its own.
    spec.setdefault("termination", "needs_reinvestigation")
    logger.warning("specify: author flattened a single edit onto the spec root (no "
                   "edits[]) — rewrapped as edits[1] so the fix is not dropped")
    return spec


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Enforce Stage-1 invariants and reconcile internal inconsistencies.

    - gate.apply is ALWAYS forced false here (specify proposes only).
    - A spec containing a stale/not_found edit is not ready: if the author still
      claimed ready_to_apply, override it to needs_reinvestigation rather than
      presenting an unverified anchor as applicable.
    - A legacy/stray ``needs_pm`` (the retired "hand it to a human" terminal) is
      coerced to ``needs_reinvestigation``: the tool re-works the fix, it never punts.
    """
    # needs_pm is retired — coerce any author/legacy emission to the autonomous loop-back
    # so the decisiveness gate and apply see only the canonical vocabulary.
    if spec.get("termination") == "needs_pm":
        logger.info("specify: coercing retired termination needs_pm -> needs_reinvestigation")
        spec["termination"] = "needs_reinvestigation"
        spec["reinvestigation"] = {
            "reason_code": RI_LEGACY_COERCE, "gate": "normalize",
            "detail": "retired needs_pm coerced to needs_reinvestigation",
        }

    gate = spec.get("gate")
    if not isinstance(gate, dict):
        gate = {}
        spec["gate"] = gate
    if gate.get("apply") is not False:
        logger.warning("specify: gate.apply was %r — forcing false (Stage-1 safety)",
                       gate.get("apply"))
        gate["apply"] = False

    edits = spec.get("edits")
    edits = edits if isinstance(edits, list) else []
    unverified = [e.get("id", "?") for e in edits
                  if isinstance(e, dict)
                  and str(e.get("anchor_status", "")).lower() in _STALE_STATUSES]
    if unverified and spec.get("termination") == "ready_to_apply":
        logger.warning("specify: edits %s are stale/not_found but termination=ready_to_apply"
                       " — overriding to needs_reinvestigation", unverified)
        spec["termination"] = "needs_reinvestigation"
        spec["reinvestigation"] = {
            "reason_code": RI_STALE_ANCHOR, "gate": "normalize",
            "detail": f"stale/not_found anchor(s): {unverified}",
        }
    return spec


def _validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return a list of structural problems (empty = ok). Non-fatal; caller decides."""
    problems: list[str] = []
    for key in _REQUIRED_KEYS:
        if key not in spec:
            problems.append(f"missing required key: {key}")
    term = spec.get("termination")
    if term is not None and term not in _VALID_TERMINATION:
        problems.append(f"invalid termination: {term!r}")
    if "edits" in spec and not isinstance(spec["edits"], list):
        problems.append("edits is not a list")
    if "deferred" in spec and not isinstance(spec["deferred"], list):
        problems.append("deferred is not a list")
    return problems


def _normalize_ws(text: str) -> str:
    """Collapse only insignificant whitespace (line endings, trailing/edge blanks).

    Indentation is preserved on purpose — in languages like Python it is
    significant — so this never mislabels a real change as a no-op; it only
    catches diffs that are whitespace noise once line endings and trailing space
    are normalized.
    """
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _deterministic_noop_ids(spec: dict[str, Any]) -> list[str]:
    """Edit ids that make no textual difference — kind-aware.

    The cheap, certain half of the effectiveness gate; the finding downgrades the
    whole spec's termination rather than only flagging the single edit at apply.

    - Anchor edits (no ``kind`` / ``kind="edit"``): flagged when whitespace-
      normalized ``anchor_old == replacement_new`` (apply also rejects the exact
      equality; here we additionally catch whitespace-only "changes").
    - ``create_file`` edits: the anchor==replacement test does NOT apply (both
      fields are absent, so it would always compare ``"" == ""``). A create_file
      edit is inert only when its ``content`` is empty/whitespace-only — mirroring
      apply.py ``evaluate_edit``'s EMPTY_CONTENT rejection.
    """
    noop: list[str] = []
    for e in spec.get("edits") or []:
        if not isinstance(e, dict):
            continue
        if e.get("kind", "edit") == "create_file":
            content = e.get("content", "")
            if not content or not content.strip():
                noop.append(str(e.get("id", "?")))
        elif _normalize_ws(e.get("anchor_old", "")) == _normalize_ws(e.get("replacement_new", "")):
            noop.append(str(e.get("id", "?")))
    return noop


# An import statement (JS/TS/Vue ES or Python) — used to detect a binding that an edit
# ADDS so we can verify it is actually USED in the post-edit file (N175: an edit added
# `import { useToast }` but never wrote the `const { showToast } = useToast()` binding,
# so the imported name was dead and the call site stayed unwired, yet it was applied).
_RE_ES_NAMED = re.compile(r"""\bimport\b[^'"]*\{([^}]*)\}\s*from\s*['"]""")
_RE_ES_DEFAULT = re.compile(r"""\bimport\s+([A-Za-z_$][\w$]*)\s*(?:,\s*\{[^}]*\})?\s*from\s*['"]""")
_RE_ES_NS = re.compile(r"""\bimport\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s*['"]""")
_RE_PY_FROM = re.compile(r"^\s*from\s+[\w.]+\s+import\s+(.+)$")
_RE_PY_IMPORT = re.compile(r"^\s*import\s+([\w.]+)(?:\s+as\s+([\w$]+))?\s*$")


def _imported_names(line: str) -> list[str]:
    """Return the binding name(s) an import LINE introduces, or [] if it is not an import.

    Conservative: only the well-formed import shapes are parsed; anything ambiguous yields
    [] so the completeness check never fires on a line it did not fully understand.

    ``from __future__ import ...`` is excluded: a future-statement is a compiler directive
    whose names (``annotations`` etc.) are NEVER referenced as bindings, so treating them as
    "imported but unused" is a guaranteed false positive — it wrongly flagged an otherwise-
    valid create_file red test (which opens with the idiomatic future import) as incomplete
    wiring and downgraded a good spec to needs_reinvestigation.
    """
    if re.match(r"\s*from\s+__future__\s+import\b", line):
        return []
    names: list[str] = []
    def _split_clause(clause: str) -> list[str]:
        out = []
        for part in clause.split(","):
            p = part.strip()
            if not p:
                continue
            # "a as b" / "a AS b" → the local binding is b
            m = re.match(r"^([\w$]+)\s+as\s+([\w$]+)$", p)
            out.append(m.group(2) if m else p.split()[0] if p.split() else p)
        return [n for n in out if re.fullmatch(r"[A-Za-z_$][\w$]*", n)]
    m = _RE_ES_NAMED.search(line)
    if m:
        names += _split_clause(m.group(1))
    m = _RE_ES_DEFAULT.search(line)
    if m:
        names.append(m.group(1))
    m = _RE_ES_NS.search(line)
    if m:
        names.append(m.group(1))
    m = _RE_PY_FROM.match(line)
    if m:
        clause = m.group(1).split("#", 1)[0].strip().strip("()")
        names += _split_clause(clause)
    else:
        m = _RE_PY_IMPORT.match(line.split("#", 1)[0])
        if m:
            names.append(m.group(2) or m.group(1).split(".")[0])
    # de-dup, drop empties
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n); out.append(n)
    return out


def _post_edit_files(spec: dict[str, Any], codebase_root: str) -> dict[str, str]:
    """Best-effort reconstruction of each touched file's POST-edit text.

    create_file → its content; anchor edits → the live file with each anchor_old replaced
    by replacement_new (applied in spec order). A file we cannot read, or an anchor not
    found in it, is skipped for that edit — the completeness check then simply does not run
    for it (never a false positive on a file we could not assemble).
    """
    by_file: dict[str, str] = {}
    for e in spec.get("edits") or []:
        if not isinstance(e, dict):
            continue
        f = e.get("file") or ""
        if not f:
            continue
        if e.get("kind", "edit") == "create_file":
            by_file[f] = e.get("content") or ""
            continue
        if f not in by_file:
            path = os.path.join(codebase_root, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    by_file[f] = fh.read()
            except OSError:
                by_file[f] = None  # unreadable → mark, skip below
        if by_file.get(f) is None:
            continue
        old, new = e.get("anchor_old") or "", e.get("replacement_new") or ""
        if old and old in by_file[f]:
            by_file[f] = by_file[f].replace(old, new, 1)
    return {f: t for f, t in by_file.items() if t is not None}


def _incomplete_wiring_ids(spec: dict[str, Any], codebase_root: str) -> dict[str, str]:
    """Edit ids whose ADDED import binding is never used in the post-edit file → id→reason.

    The deterministic completeness half of the gate (N175): adding an import is only half a
    wiring; if the imported name is dead in the assembled file, the call it was meant to
    enable is unbound and the edit does not work. We flag per file using the FULL post-edit
    text, so an import added in one edit but used by a sibling edit's code is NOT flagged.
    Anything we cannot assemble or parse is left alone — completeness only fires on a
    provably-unused added import.
    """
    if not codebase_root:
        return {}
    files = _post_edit_files(spec, codebase_root)
    # Test edits are exempt: a red test is certified by its red→green run, not by import
    # usage, so an unused import in a test must not flag the spec as incomplete wiring.
    test_edit_ids = {str(i) for i in (spec.get("verify") or {}).get("test_edit_ids") or []}
    incomplete: dict[str, str] = {}
    for e in spec.get("edits") or []:
        if not isinstance(e, dict):
            continue
        f = e.get("file") or ""
        if f not in files:
            continue
        if str(e.get("id", "")) in test_edit_ids or _is_test_path(f):
            continue
        added_text = (e.get("content") if e.get("kind", "edit") == "create_file"
                      else e.get("replacement_new")) or ""
        old_text = e.get("anchor_old") or ""
        # names this edit introduces via a NEW import line (absent from its anchor_old)
        added_names: list[str] = []
        for line in added_text.splitlines():
            if line in old_text:
                continue
            added_names += _imported_names(line)
        if not added_names:
            continue
        # strip every import line from the post-edit file, then look for real usage
        body = "\n".join(ln for ln in files[f].splitlines() if not _imported_names(ln))
        for name in added_names:
            if not re.search(rf"\b{re.escape(name)}\b", body):
                incomplete[str(e.get("id", "?"))] = (
                    f"incomplete wiring — imported {name!r} is never used "
                    f"(binding/call site missing)")
                break
    return incomplete


# Test-file path convention (mirrors apply._TEST_PATH_RE; duplicated here to avoid a
# circular import — apply imports _incomplete_wiring_ids from this module). The wiring
# gate exempts test edits: a red test's job is to BITE (verified by the red→green run),
# not to wire a production call, so a stray unused import in a test (``import os``, the
# idiomatic ``from __future__``) is cosmetic — it must never downgrade the fix.
_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__)/"
    r"|(?:^|/)test_[^/]+$"
    r"|(?:^|/)[^/]+_test\.[^./]+$"
    r"|\.spec\.[^./]+$"
)


def _is_test_path(rel_path: str) -> bool:
    """True when a path is a test file/dir by convention. Never raises."""
    return bool(rel_path) and bool(_TEST_PATH_RE.search(rel_path.replace("\\", "/")))


# The PRIMARY read source of a SQL statement embedded in code — the first ``FROM <table>``.
# An alias (``FROM project_modules pm``) is captured as the table only (\w stops at the
# space). Case-insensitive; DML-read keyword only (we ground the source a SELECT returns).
_RE_SQL_FROM = re.compile(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _primary_from_table(text: str) -> str | None:
    """The first ``FROM <table>`` in a code blob, or None — the read's primary source."""
    m = _RE_SQL_FROM.search(text or "")
    return m.group(1) if m else None


def _datasource_regression_ids(spec: dict[str, Any], db_conn: Any) -> dict[str, str]:
    """Edit ids that switch a SQL read's PRIMARY source table to an empty/sparser one → id→reason.

    N176: an edit that points a SQL read at a table that does not exist, or that is
    literally empty in the live DB, is a data-location claim ("the data lives in table X")
    emitted as ready without being grounded — the data-location analogue of the
    callee-contract gap (N175). When a live DB connection is configured for the codebase,
    we check it deterministically: an anchor edit whose replacement changes the FIRST
    ``FROM`` table of a read is flagged when the new table is MISSING or EMPTY (0 rows).

    M036/회귀2 correction: we do NOT flag a "strictly fewer rows" swap. Row count is not
    coverage — the authoritative SSOT can hold fewer rows than the denormalized source it
    replaces (here ``groups`` 2 dup rows → ``project_modules`` 1 clean row, which is the
    CORRECT fix). The earlier fewer-rows branch mis-fired on exactly the right swap and
    downgraded it; only an absent or 0-row table is an unambiguous regression.

    Fail-closed: no db_conn, or any introspection/count failure, simply skips (never a
    false positive on a DB we cannot read). Downgrade-only — a flagged spec loops back to
    re-investigate (a cheap re-run beats a wrong apply).
    """
    if db_conn is None:
        return {}
    try:
        schema = dbread.list_schema(db_conn)
    except dbread.DbReadError:
        return {}
    if not schema:
        return {}
    # case-insensitive table-name resolution against the live catalog
    by_lower = {t.lower(): t for t in schema}

    counts: dict[str, int] = {}
    def _count(real_table: str) -> int | None:
        if real_table not in counts:
            try:
                counts[real_table] = dbread.count_rows(db_conn, real_table)
            except dbread.DbReadError:
                return None
        return counts[real_table]

    findings: dict[str, str] = {}
    for e in spec.get("edits") or []:
        if not isinstance(e, dict) or e.get("kind", "edit") == "create_file":
            continue
        old_primary = _primary_from_table(e.get("anchor_old") or "")
        new_primary = _primary_from_table(e.get("replacement_new") or "")
        if not new_primary or not old_primary:
            continue
        if new_primary.lower() == old_primary.lower():
            continue  # primary read source unchanged → nothing to ground
        eid = str(e.get("id", "?"))
        new_real = by_lower.get(new_primary.lower())
        if new_real is None:
            findings[eid] = (
                f"data-source not grounded — edit switches the primary read to table "
                f"{new_primary!r}, which does not exist in the live DB")
            continue
        n_new = _count(new_real)
        if n_new is None:
            continue  # cannot count → skip (fail-closed)
        if n_new == 0:
            findings[eid] = (
                f"data-source regression — edit switches the primary read from "
                f"{old_primary!r} to {new_primary!r}, which is EMPTY (0 rows) in the live DB")
            continue
        # NOTE (M036/회귀2): we deliberately do NOT flag "strictly fewer rows". Row count
        # is not coverage: the correct SSOT can legitimately hold fewer rows than the wrong
        # source (here ``groups`` has 2 rows of denormalized dupes, ``project_modules`` —
        # the authoritative table — has 1 clean row). The fewer-rows branch mis-fired on
        # exactly the RIGHT swap (groups → project_modules) and downgraded it. Only an
        # absent table or a literally empty (0-row) one is an unambiguous regression.
    return findings


# Tokens that can sit where a table alias would in ``FROM t <next>`` / ``JOIN t <next>`` —
# we must not read any of these as the table's alias when mapping aliases to real tables.
_SQL_ALIAS_STOP = frozenset({
    "on", "where", "inner", "left", "right", "outer", "full", "cross", "natural",
    "join", "using", "and", "or", "group", "order", "having", "limit", "offset",
    "set", "values", "select", "from", "as", "union", "intersect", "except",
})
_SQL_FROM_JOIN_RE = re.compile(
    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\bas\b\s+)?([A-Za-z_][A-Za-z0-9_]*)?",
    re.IGNORECASE)
_SQL_QUALIFIED_COL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")
_SQL_INSERT_COLS_RE = re.compile(
    r"\binsert\s+into\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.IGNORECASE)
_SQL_PLAIN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sql_alias_map(sql: str) -> dict[str, str]:
    """Map each table alias (and bare table name) in ``sql`` to its real table name.

    Parses ``FROM t [AS] a`` / ``JOIN t [AS] a`` clauses. Every table maps to itself; an
    alias maps to its table only when the following token is not a SQL keyword (so
    ``FROM projects WHERE`` does not read ``WHERE`` as the alias). Keys are lower-cased.
    """
    out: dict[str, str] = {}
    for m in _SQL_FROM_JOIN_RE.finditer(sql):
        table, alias = m.group(1), m.group(2)
        out[table.lower()] = table
        if alias and alias.lower() not in _SQL_ALIAS_STOP:
            out[alias.lower()] = table
    return out


def _undefined_column_ids(spec: dict[str, Any], db_conn: Any) -> dict[str, str]:
    """Edit ids whose SQL names a column ABSENT from the live DB schema → id→reason.

    The table-existence gate (``_datasource_regression_ids``) catches a read pointed at a
    missing/empty table; this is its column-level sibling. An edit that selects, filters or
    joins on ``alias.column`` — or a test fixture that does ``INSERT INTO table (columns…)``
    — naming a column the real schema does not have is an UNRUNNABLE claim shipped as ready
    (T906: ``pm.module`` / T907: ``pm.is_active`` and the fixture INSERT of ``is_active``;
    ``project_modules`` has neither per migration 028). We ground it deterministically
    against the live schema (PRAGMA/information_schema via ``dbread.list_schema``):

      * qualified ``alias.col`` references whose alias resolves to a REAL table, and
      * ``INSERT INTO real_table (cols…)`` column lists (covers test fixtures too)

    are checked; a column missing from that table's real column set is flagged. Fail-open:
    no db_conn, unreadable schema, an unresolved qualifier (CTE / json / subquery alias) or
    an unknown table is simply skipped — we only flag a column we can prove does not exist,
    never a guess. Downgrade-only, mirroring the table gate.
    """
    if db_conn is None:
        return {}
    try:
        schema = dbread.list_schema(db_conn)
    except dbread.DbReadError:
        return {}
    if not schema:
        return {}
    cols_by_table = {t.lower(): {c.lower() for c in cols} for t, cols in schema.items()}

    findings: dict[str, str] = {}
    for e in spec.get("edits") or []:
        if not isinstance(e, dict):
            continue
        # Validate the SQL the edit INTRODUCES: a replacement for an anchor edit, or the
        # body of a created file (so a test fixture's INSERT is grounded the same way).
        sql = (e.get("content") if e.get("kind") == "create_file"
               else e.get("replacement_new")) or ""
        if not sql.strip():
            continue
        amap = _sql_alias_map(sql)
        bad: list[str] = []
        seen: set[tuple[str, str]] = set()

        def _flag(table: str, col: str) -> None:
            key = (table.lower(), col.lower())
            if col.lower() not in cols_by_table[table.lower()] and key not in seen:
                seen.add(key)
                bad.append(f"{table}.{col}")

        for m in _SQL_QUALIFIED_COL_RE.finditer(sql):
            qualifier, col = m.group(1), m.group(2)
            real = amap.get(qualifier.lower())
            if real is None or real.lower() not in cols_by_table:
                continue  # unresolved alias or unknown table → cannot prove absence
            _flag(real, col)

        for m in _SQL_INSERT_COLS_RE.finditer(sql):
            table = m.group(1)
            if table.lower() not in cols_by_table:
                continue
            for raw in m.group(2).split(","):
                col = raw.strip().strip('"`[]')
                if _SQL_PLAIN_IDENT_RE.match(col):
                    _flag(table, col)

        if bad:
            findings[str(e.get("id", "?"))] = (
                "undefined SQL column(s) — " + ", ".join(bad) + " not present in the live "
                "DB schema; the edit's SQL/fixture cannot run (verify column names against "
                "the migration/PRAGMA schema)")
    return findings


# ── Callee-contract grounding (N175 round-2) ───────────────────────────────────
# _incomplete_wiring_ids above proves an added import is USED; it does NOT prove the
# call is invoked CORRECTLY. N175 round-2: an edit added a real, used call —
# ``showToast(t('...'), 'danger')`` — but the toast util's actual signature takes the
# severity differently (an options object, not the 2nd positional), so the literal
# 'danger' rendered in the toast BODY. The author GUESSED the signature because the
# callee's definition was never on the table, and the tool-OFF reviewer could not catch
# it because the review prompt carried only the edits, not the callee's real contract.
#
# The constructive fix (mirroring ground_anchors): lift each CALLED symbol's real
# definition out of the module the edit imports it from (or the edited file itself) and
# put that signature on the table — for the author (so it writes the call right) AND the
# reviewer (so it can flag a mis-invoked call). Pure-local, free, deterministic, never
# raises; a symbol whose def we cannot resolve is simply skipped (never fabricated), so
# it can never manufacture a phantom "signature mismatch" (the converge.py:194 trap).

# An identifier (optionally ``obj.method``) immediately followed by "(" — a call. The
# negative look-behind stops us splitting ``a.b(`` into a spurious ``b`` match's prefix.
_CALL_EXPR_RE = re.compile(r"(?<![\w.$])([A-Za-z_$][\w$]*)\s*\(")
# The module specifier of a JS/TS import or require, and a Python ``from x.y import``.
_RE_MODULE_FROM = re.compile(r"""\bfrom\s*['"]([^'"]+)['"]""")
_RE_MODULE_REQUIRE = re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)""")
_RE_MODULE_PY = re.compile(r"^\s*from\s+([\w.]+)\s+import\b")
# Calls never worth grounding (keywords / framework-ubiquitous / globals). The
# def-existence filter drops most external calls already; this kills obvious noise
# cheaply so we don't waste a resolve probe on ``if(`` / ``t(`` / ``JSON(``.
_CALL_GROUND_SKIP = frozenset({
    "if", "for", "while", "switch", "return", "catch", "function", "await", "typeof",
    "new", "delete", "void", "do", "else", "t", "$t", "tc", "$tc", "n", "$n",
    "ref", "computed", "watch", "reactive", "emit", "defineProps", "defineEmits",
    "require", "import", "Boolean", "Number", "String", "Array", "Object", "JSON",
    "Math", "Promise", "parseInt", "parseFloat", "isNaN", "console", "setTimeout",
    "setInterval", "print", "len", "str", "int", "float", "list", "dict", "set",
    "tuple", "range", "super", "self", "this",
})
# Front-end source roots an ``@/x`` / ``~/x`` alias resolves to when no explicit alias
# map is configured (Vite/Vue/webpack convention). Tried in order, under each tree root.
_ALIAS_SUBROOTS = ("", "src", "client/src", "app/src", "frontend/src", "ui/src", "web/src")
_MODULE_EXTS = ("", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".vue", ".py")
_MODULE_INDEX = ("index.ts", "index.tsx", "index.js", "index.jsx", "index.mjs", "index.vue")
_CALLEE_GROUND_MAX = 6        # cap on distinct callee contracts lifted (prompt budget)
_CALLEE_SIG_MAX_LINES = 24    # per-symbol lift cap (full signature + a little body shape)


def _called_symbols(text: str) -> list[str]:
    """Distinct symbols invoked as ``name(`` in ``text`` (skiplist removed), in order."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _CALL_EXPR_RE.finditer(text or ""):
        name = m.group(1)
        if name in _CALL_GROUND_SKIP or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _module_specifiers(text: str) -> list[str]:
    """Module paths imported/required in ``text`` (JS ``from '…'`` / ``require('…')`` /
    Python ``from x.y import``), de-duplicated in first-seen order."""
    specs: list[str] = []
    seen: set[str] = set()
    for rx in (_RE_MODULE_FROM, _RE_MODULE_REQUIRE):
        for m in rx.finditer(text or ""):
            s = m.group(1)
            if s and s not in seen:
                seen.add(s); specs.append(s)
    for line in (text or "").splitlines():
        m = _RE_MODULE_PY.match(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1)); specs.append(m.group(1))
    return specs


def _resolve_module_file(spec: str, from_file: str | None, roots: list[str]) -> str | None:
    """Resolve an import specifier to an on-disk file under one of ``roots``.

    Handles relative (``./x``, ``../x``), alias (``@/x``, ``~/x``), bare-as-path, and
    Python dotted (``a.b.c``) specifiers, probing the usual extensions and ``index.*``.
    Returns the first existing file, or None (an external/lib specifier resolves to
    nothing and is simply not grounded). Never raises.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    bases: list[tuple[str, str]] = []  # (root, relpath-without-ext)
    if spec.startswith("."):
        if from_file is None:
            return None
        rel = os.path.normpath(os.path.join(os.path.dirname(from_file), spec))
        for root in roots:
            bases.append((root, rel))
    elif spec[:2] in ("@/", "~/"):
        sub = spec[2:]
        for sr in _ALIAS_SUBROOTS:
            for root in roots:
                bases.append((root, os.path.join(sr, sub)))
    elif "/" not in spec and "." in spec and not spec.startswith("@"):
        # Python dotted module path (a.b.c) → a/b/c. (A JS bare lib like "lodash.merge"
        # also lands here, but it won't exist under a root, so it resolves to None.)
        rel = spec.replace(".", "/")
        for sr in _ALIAS_SUBROOTS:
            for root in roots:
                bases.append((root, os.path.join(sr, rel)))
    else:
        # bare specifier: external lib OR a project alias-less path; try as a path.
        for sr in _ALIAS_SUBROOTS:
            for root in roots:
                bases.append((root, os.path.join(sr, spec)))
    for root, rel in bases:
        for ext in _MODULE_EXTS:
            p = os.path.join(root, rel + ext)
            if os.path.isfile(p):
                return p
        for idx in _MODULE_INDEX:
            p = os.path.join(root, rel, idx)
            if os.path.isfile(p):
                return p
    return None


def _find_symbol_def(text: str, symbol: str) -> tuple[int, list[str]] | None:
    """Locate where ``symbol`` is DEFINED in ``text``. Returns ``(line_idx, lines)`` or None.

    Recognises the common JS/TS/Vue and Python definition shapes (function decl, const/
    let arrow or function expression, object/class method, ``def``). Conservative — only
    a real definition line matches, not a call — so an unresolved symbol yields None and
    is never grounded with a wrong region.
    """
    s = re.escape(symbol)
    pats = (
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*{s}\s*[(<]",
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:const|let|var)\s+{s}\s*[:=]",
        rf"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|static\s+|async\s+)*"
        rf"{s}\s*\([^)]*\)\s*(?::[^={{]+)?\{{",          # class/object method
        rf"^\s*{s}\s*:\s*(?:async\s+)?function\b",        # obj prop: function
        rf"^\s*{s}\s*:\s*(?:async\s+)?\([^)]*\)\s*=>",    # obj prop: arrow
        rf"^\s*def\s+{s}\s*\(",                           # python
    )
    lines = (text or "").split("\n")
    for i, line in enumerate(lines):
        for p in pats:
            if re.search(p, line):
                return i, lines
    return None


def _lift_signature(lines: list[str], idx: int, max_lines: int) -> str:
    """Lift the def at ``lines[idx]`` forward: the whole (possibly multi-line) parameter
    list plus a few body lines, capped at ``max_lines``. Enough to reveal arg names,
    order and the options-object shape without dragging the whole function in."""
    out: list[str] = []
    depth = 0
    seen_paren = False
    body_grace = 5
    for j in range(idx, min(len(lines), idx + max_lines)):
        line = lines[j]
        out.append(line)
        depth += line.count("(") - line.count(")")
        if "(" in line:
            seen_paren = True
        if seen_paren and depth <= 0:
            body_grace -= 1
            if body_grace <= 0:
                break
    return "\n".join(out).rstrip()


def _relpath_under(path: str, roots: list[str]) -> str:
    """Path relative to the first root that contains it (for a readable citation)."""
    ap = os.path.abspath(path)
    for root in roots:
        ar = os.path.abspath(root)
        if ap.startswith(ar + os.sep):
            return os.path.relpath(ap, ar).replace("\\", "/")
    return os.path.basename(path)


def _collect_callee_contracts(
    items: list[tuple[str, str, str | None]], roots: list[str],
    max_symbols: int = _CALLEE_GROUND_MAX,
) -> list[tuple[str, dict[str, str]]]:
    """Resolve the real definition of each project symbol CALLED across ``items``.

    Each item is ``(call_text, import_text, edit_file)``: ``call_text`` is scanned for
    ``name(`` calls, ``import_text`` for the module(s) those names come from, and
    ``edit_file`` lets a locally-defined callee resolve too. Returns an ordered list of
    ``(symbol, {"file": rel, "text": signature})`` for symbols whose def we could read
    from one of those module files. Bounded by ``max_symbols``; never raises.
    """
    contracts: dict[str, dict[str, str]] = {}
    order: list[str] = []
    file_cache: dict[str, str] = {}

    def _read(p: str) -> str:
        if p not in file_cache:
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    file_cache[p] = fh.read()
            except OSError:
                file_cache[p] = ""
        return file_cache[p]

    for call_text, import_text, edit_file in items:
        symbols = _called_symbols(call_text)
        if not symbols:
            continue
        search_files: list[str] = []
        for spec in _module_specifiers(import_text):
            f = _resolve_module_file(spec, edit_file, roots)
            if f and f not in search_files:
                search_files.append(f)
        if edit_file:
            for root in roots:
                p = os.path.join(root, edit_file)
                if os.path.isfile(p) and p not in search_files:
                    search_files.append(p)
                    break
        if not search_files:
            continue
        for sym in symbols:
            if sym in contracts:
                continue
            for f in search_files:
                found = _find_symbol_def(_read(f), sym)
                if not found:
                    continue
                i, flines = found
                contracts[sym] = {"file": _relpath_under(f, roots),
                                  "text": _lift_signature(flines, i, _CALLEE_SIG_MAX_LINES)}
                order.append(sym)
                break
            if len(contracts) >= max_symbols:
                break
        if len(contracts) >= max_symbols:
            break
    return [(s, contracts[s]) for s in order[:max_symbols]]


def _render_callee_block(contracts: list[tuple[str, dict[str, str]]]) -> str:
    """Render resolved callee contracts as a prompt section. '' when there are none."""
    if not contracts:
        return ""
    parts = [
        "## Callee contracts (real signatures of the functions these edits CALL)", "",
        "Each block below is the ACTUAL definition lifted live from the codebase of a "
        "function the edits invoke. Write/keep every call so its arguments match THIS "
        "signature exactly — correct argument ORDER and shape (e.g. an options object "
        "vs a positional value). A call that does not match the signature shown here is "
        "wired but MIS-INVOKED and will not behave as intended.", "",
    ]
    for sym, c in contracts:
        parts += [f"--- {sym}  (defined in {c['file']})", "```", c["text"], "```", ""]
    return "\n".join(parts)


# ── Test-fixture grounding (red-test isolation) ────────────────────────────────
# A specify-authored red test must run against the target's ISOLATED test harness,
# never the live production store/DB. Two live failures motivate this:
#   (1) a data-dependent fix (a DB read returning ``'' AS module``) had its red test
#       OMITTED — the author took the contract's "needs DB/app state → omit" escape
#       when a narrow test that SEEDS the table and calls the function directly was
#       feasible; and
#   (2) when a red test WAS authored it called the production ``get_store()`` and
#       wrote into the live sqlite file — unsafe to run under ``apply --verify``.
# The fix mirrors callee-contract grounding: surface the target's existing pytest
# fixtures (name + one-line intent, lifted from conftest.py) so the author builds the
# red test on the real isolation fixture instead of guessing or punting. Pure-local,
# free, deterministic, never raises; no conftest → empty block (graceful no-op).
_FIXTURE_GROUND_MAX = 14
_RE_PYTEST_FIXTURE = re.compile(r"^\s*@pytest\.fixture\b")
_RE_FIXTURE_DEF = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")
# A patch() whose TARGET is a get_store binding given as a string path, e.g.
# patch("modules.flow_gate.db.connection.get_store", ...). This is the robust wiring
# the author must copy; a file that only mentions get_store and patch() separately
# (monkeypatch.setattr on a guessed module alias) teaches the error-prone style.
_RE_PATCH_GET_STORE = re.compile(r"""patch\(\s*['"][^'"]*get_store""")


def _collect_test_fixtures(codebase_root: str) -> list[tuple[str, str]]:
    """``(fixture_name, one-line summary)`` for pytest fixtures defined in the target's
    ``conftest.py`` files — the isolated harness a red test should build on.

    Best-effort and bounded: scans each conftest for ``@pytest.fixture`` decorators and
    lifts the decorated def's name plus its docstring's first line. Never raises; an
    unreadable conftest is skipped. Returns [] when the target has no conftest at all.
    """
    if not codebase_root or not os.path.isdir(codebase_root):
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        conftests = sorted(glob.glob(
            os.path.join(codebase_root, "**", "conftest.py"), recursive=True))
    except OSError:
        return []
    for cf in conftests:
        try:
            with open(cf, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if not _RE_PYTEST_FIXTURE.match(ln):
                continue
            # the decorated def may sit a few lines below (stacked decorators / multi-
            # line fixture args), so look ahead a small window for the first def.
            for j in range(i + 1, min(i + 8, len(lines))):
                m = _RE_FIXTURE_DEF.match(lines[j])
                if not m:
                    continue
                name = m.group(1)
                if name in seen:
                    break
                seen.add(name)
                summary = ""
                for k in range(j + 1, min(j + 5, len(lines))):
                    s = lines[k].strip()
                    if s.startswith(('"""', "'''")):
                        summary = s.strip("\"' ")
                        break
                    if s and not s.startswith(("#", ")")):
                        break
                out.append((name, summary))
                break
            if len(out) >= _FIXTURE_GROUND_MAX:
                return out
    return out


def _render_fixture_block(fixtures: list[tuple[str, str]]) -> str:
    """Render the target's pytest fixtures as a prompt section. '' when there are none."""
    if not fixtures:
        return ""
    parts = [
        "## Test fixtures (the target's ISOLATED test harness — build any red test on these)", "",
        "These pytest fixtures already exist in the target's conftest. A red test you author "
        "MUST run against this isolated harness and seed the rows the symptom needs. NEVER let "
        "a red test read/write the production store/database: that mutates live data and is "
        "unsafe to run. If the symptom is data-dependent (a DB read returning the wrong/empty "
        "value), that is STILL a narrow, biting test — do not omit the verify block as 'needs "
        "app state'. CRUCIAL: production code under test usually reads through a global "
        "`get_store()`; merely requesting a raw-connection fixture and seeding it is NOT enough "
        "— the function will still hit the REAL database unless `get_store` is pointed at the "
        "test DB. Use the STRING-TARGET form `patch(\"<module.path>.get_store\", "
        "return_value=store)` (a context manager / decorator), exactly as the example below "
        "shows. Do NOT use `monkeypatch.setattr(<some_module>, \"get_store\", ...)` on a module "
        "alias you guessed at — that is brittle and is the usual cause of a red test that never "
        "actually bites. When the symptom surfaces through a HIGH-LEVEL function, patch "
        "`get_store` in the module where THAT function reads it (follow the example's import "
        "root and patch target).", "",
    ]
    for name, summary in fixtures:
        parts.append(f"- {name}" + (f" — {summary}" if summary else ""))
    return "\n".join(parts)


# A short excerpt of an existing target test that exercises store-backed code against an
# isolated DB — the missing half of fixture grounding. The conftest fixture names alone do
# not tell the author (1) the correct IMPORT ROOT (tests run from a cwd where the package is
# ``modules.flow_gate``, not ``server.modules.flow_gate``) nor (2) that the production
# ``get_store()`` must be PATCHED to the test DB or the code reads the real database anyway.
# Both are demonstrated by a real example, so we lift one verbatim and tell the author to
# copy the pattern. Best-effort, bounded, deterministic, never raises.
_DBTEST_EXAMPLE_MAX_LINES = 55


def _lift_db_test_example(codebase_root: str) -> str:
    """An excerpt of the target test that best demonstrates wiring an isolated DB into
    store-backed code (correct import root + patching ``get_store``). '' when none found."""
    if not codebase_root or not os.path.isdir(codebase_root):
        return ""
    try:
        candidates = glob.glob(os.path.join(codebase_root, "**", "test_*.py"), recursive=True)
    except OSError:
        return ""
    best: tuple[int, str, list[str]] | None = None
    for tf in sorted(candidates):
        try:
            with open(tf, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        text = "\n".join(lines)
        # Only files that DEMONSTRATE the robust wiring win — a string-target patch of
        # get_store. A file that merely mentions get_store and patch() separately teaches
        # the error-prone monkeypatch.setattr(<guessed alias>, ...) style the author then
        # botches, so it is excluded outright (count == 0 -> skip), never just out-scored
        # by a big integration file's def-test count.
        targeted = len(_RE_PATCH_GET_STORE.findall(text))
        if targeted == 0:
            continue
        # Rank by how densely the file shows the pattern; def-test count is only a faint
        # tiebreak so it can never flip the winner to a less-demonstrating file.
        score = targeted * 100 + text.count("def test")
        if best is None or score > best[0]:
            best = (score, os.path.relpath(tf, codebase_root), lines)
    if best is None:
        return ""
    rel, lines = best[1], best[2]
    # Import header: lines up to the first def/class/@fixture (the import root lives here).
    header: list[str] = []
    for ln in lines[:40]:
        if re.match(r"^\s*(def |class |@pytest)", ln):
            break
        if ln.strip():
            header.append(ln)
    # Window around the first ``get_store`` patch, backed up to its enclosing fixture/def.
    anchor = next((i for i, ln in enumerate(lines) if "get_store" in ln and "patch" in ln), None)
    if anchor is None:
        anchor = next((i for i, ln in enumerate(lines) if "get_store" in ln), 0)
    start = anchor
    for i in range(anchor, max(anchor - 25, -1), -1):
        if re.match(r"^\s*(@pytest\.fixture|def )", lines[i]):
            start = i - 1 if i > 0 and lines[i - 1].lstrip().startswith("@") else i
            break
    window = lines[start:start + _DBTEST_EXAMPLE_MAX_LINES]
    excerpt = "\n".join(header[:20] + ["..."] + window)
    return (f"## DB-test wiring example (from {rel}) — copy this import root + get_store patch\n"
            f"```python\n{excerpt}\n```")


def _edit_call_items(spec: dict[str, Any],
                     codebase_root: str | None = None) -> list[tuple[str, str, str | None]]:
    """Build ``_collect_callee_contracts`` items, grouped PER FILE across all edits.

    A real fix wires a call ACROSS sibling edits: one edit adds ``import {useToast}``
    (E2), another adds ``const {showToast} = useToast()`` (E3), a third writes the
    ``showToast(...)`` call (E4). Grouping every edit on a file into one item is what
    lets the import in E2 resolve the module for the call in E4 — a per-edit view would
    see the call with no import and never ground the signature (the actual N175 spec
    shape). ``call_text`` is the union of the edits' ADDED text (so only newly-written
    calls are grounded, not every call already in the file); ``import_text`` is the
    POST-edit file when readable (so a call to a PRE-EXISTING import resolves too),
    falling back to the added union.
    """
    by_file: dict[str, list[str]] = {}
    order: list[str] = []
    for e in spec.get("edits") or []:
        if not isinstance(e, dict):
            continue
        f = e.get("file")
        body = (e.get("content") if e.get("kind", "edit") == "create_file"
                else e.get("replacement_new")) or ""
        if not body:
            continue
        if f not in by_file:
            by_file[f] = []
            order.append(f)
        by_file[f].append(body)
    post = _post_edit_files(spec, codebase_root) if codebase_root else {}
    items: list[tuple[str, str, str | None]] = []
    for f in order:
        added = "\n".join(by_file[f])
        items.append((added, post.get(f) or added, f))
    return items


# How many lines of a create_file's content to surface to the effectiveness
# reviewer — enough to judge "non-empty and on-target" without ballooning the prompt.
_REVIEW_CONTENT_MAX_LINES = 40

# One terse JSON-only retry for the effectiveness review (mirrors judge's lever):
# now that the reviewer runs on a tool-OFF API provider (deepinfra), a stray prose
# wrapper or fence would otherwise degrade a ready spec straight to needs_reinvestigation.
# The retry is a transport reparse — recorded to the ledger (a real paid call) but it
# does not multiply the review (still one logical effectiveness pass).
_REVIEW_JSON_REMINDER = (
    "\n\n[Retry] Your previous response could not be parsed as the required JSON. "
    "Output ONLY the single JSON object with a \"reviews\" array as specified above — "
    "no prose, no explanation, no markdown code fences, nothing before or after it.")


def build_review_prompt(honey_text: str, spec: dict[str, Any], codebase_root: str,
                        docs_root: str | None = None) -> str:
    """Build the effectiveness-review prompt for a second, independent look.

    The reviewer gets the honey (the reported symptom + the behavior the fix must
    change) and the edits specify just authored, and judges — per edit, re-reading
    live code as needed — whether each edit ACTUALLY changes the behavior the honey
    identified (effective) and is consistent with the honey's conclusion (coherent).

    Anchor edits: an edit that is anchored correctly but functionally inert (a no-op
    assignment, a guard whose condition can never be true, a value set to what it
    already is) is effective=false — exactly the failure this review exists to catch.

    create_file edits: the block carries ``content`` (truncated to
    _REVIEW_CONTENT_MAX_LINES lines) instead of the absent anchor fields, so the
    reviewer can judge whether the new file is genuinely non-empty and on-target.

    The review also judges ``in_scope`` — whether the edit changes ONLY what the
    Requested change targets, or ALSO affects elements/behaviors the seed said to
    leave unchanged. This is the complement of effectiveness: an edit can be fully
    effective (it does turn the target blue) yet OVER-APPLY by recoloring a SHARED
    rule the seed explicitly told it not to touch, regressing every other element
    carrying that class (T891 v2). Effective+coherent both passed there; only an
    explicit over-application check catches it.
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    blocks = []
    for e in edits:
        if e.get("kind", "edit") == "create_file":
            content = e.get("content") or ""
            lines = content.splitlines()
            if len(lines) > _REVIEW_CONTENT_MAX_LINES:
                content_display = "\n".join(lines[:_REVIEW_CONTENT_MAX_LINES]) + "\n(truncated)"
            else:
                content_display = content
            block = {
                "id": e.get("id"),
                "kind": "create_file",
                "file": e.get("file"),
                "content": content_display,
                "rationale": e.get("rationale"),
            }
        else:
            block = {
                "id": e.get("id"),
                "file": e.get("file"),
                "anchor_old": e.get("anchor_old"),
                "replacement_new": e.get("replacement_new"),
                "rationale": e.get("rationale"),
            }
        blocks.append(json.dumps(block, ensure_ascii=False, indent=2))
    edits_json = "\n".join(blocks) if blocks else "(no edits)"
    # Callee-contract grounding (N175 round-2): lift the REAL signature of each function
    # the edits call so the tool-OFF reviewer can catch a wired-but-mis-invoked call (e.g.
    # wrong argument order) — it cannot open files, so without this it has no signature to
    # check against. Empty when nothing resolves (never fabricated → no phantom mismatch).
    roots = [r for r in (codebase_root, docs_root) if r]
    callee_block = _render_callee_block(
        _collect_callee_contracts(_edit_call_items(spec, codebase_root), roots))
    callee_section = ("\n" + callee_block + "\n") if callee_block else ""
    return f"""[Role] You are an INDEPENDENT effectiveness reviewer for Hivework's specify stage. \
You did not author these edits. Your only job is to catch edits that are anchored \
correctly but do not actually fix anything. Do not rewrite the edits; only judge them.

[Codebase root — re-read LIVE files from here]
{codebase_root}
{_docs_root_block(docs_root)}
[The honey — the reported symptom and the behavior the fix must change]
{honey_text}

[The proposed edits to judge]
{edits_json}
{callee_section}
[Judge each edit]
For every edit decide three booleans, applying the criterion that matches the edit's kind:
- effective:
  - Anchor edit (kind "edit" or absent): would applying this edit actually change the \
behavior the honey identified as wrong? An edit that is functionally inert — a no-op \
assignment, a guard whose condition can never be true, a value set to what it already is, \
a change with no runtime effect — is effective=false EVEN THOUGH its anchor is valid. \
MIS-INVOKED CALL: when a "Callee contracts" section is shown above, an edit that calls one \
of those functions with arguments that DO NOT MATCH the signature shown — wrong argument \
ORDER, a positional value where an options object is expected, too few/many args — is \
effective=false (the call is wired but mis-invoked, so it does not produce the intended \
behavior). Judge this ONLY against a signature actually shown above; if a callee is not \
shown, do NOT guess a mismatch. \
Re-open the live files to judge reachability and effect; do not assume. \
CRUCIAL — judge each edit AS PART OF THE WHOLE EDIT SET, not in isolation. A correct fix \
is often WIRED ACROSS FILES (e.g. a back-end guard in one file PLUS the front-end toast \
that surfaces it in another, or an import in one edit and the binding/call it enables in \
a sibling edit). An edit that is a NECESSARY part of such a multi-edit wiring is \
effective=true when the SET together produces the corrected behavior — do NOT mark it \
effective=false merely because that one edit ALONE does not fully reproduce the symptom \
fix. Reserve effective=false for an edit that contributes NOTHING to the behavior even \
when taken together with the others (truly inert / unreachable / redundant).
  - create_file edit (kind "create_file"): effective=true when a non-empty file is created \
in direct response to the honey's directions; effective=false if the content is empty or \
whitespace-only, or the honey did not ask for a new file at this path.
- coherent: is the edit consistent with the honey's conclusion (it does not contradict \
what the investigation concluded)?
- in_scope: does the edit change ONLY what the Requested change targets, WITHOUT also \
affecting elements or behaviors the seed said to leave unchanged? An edit that is \
effective but OVER-APPLIES — it edits a SHARED rule / common helper / broad selector so \
that elements beyond the single named target also change, especially when the Requested \
change has an explicit "DO NOT touch / DO NOT modify / must stay …" boundary or names a \
SINGLE target — is in_scope=false. Honor those boundaries literally: editing the exact \
thing the seed forbade (e.g. recolouring the shared rule it said to keep neutral) is \
in_scope=false EVEN THOUGH the named target does end up changed. A correctly narrow edit \
that touches only the named target is in_scope=true.

[Output contract] Output ONLY this JSON object. No prose, no text outside the JSON.
{{
  "reviews": [
    {{ "id": "<edit id>", "effective": true, "coherent": true, "in_scope": true, \
"reason": "<one line>" }}
  ]
}}
"""


def review_effectiveness(
    honey_text: str,
    spec: dict[str, Any],
    codebase_root: str,
    model: str,
    provider: str,
    ledger=None,
    provider_kwargs: dict | None = None,
    docs_root: str | None = None,
) -> tuple[dict[str, dict], bool]:
    """Second pass: ask a worker to judge each edit's effectiveness/coherence.

    Returns ``(judgments_by_id, inconclusive)``. ``inconclusive`` is True when the
    review could not be obtained or parsed — the caller then defers the ready
    decision to a human rather than silently trusting the original claim. This
    function never raises: a flaky review must not crash specify or lose the honey.

    On an UNUSABLE response (no parseable JSON, or valid JSON lacking a ``reviews``
    list) the review is retried ONCE with a terse JSON-only reminder (see
    ``_REVIEW_JSON_REMINDER``) — both attempts are recorded to the ledger but it is
    still one logical review pass. A worker-level failure (timeout / provider error)
    is NOT retried. The retry matters most on tool-OFF API providers (deepinfra),
    where a stray prose wrapper would otherwise needlessly downgrade a ready spec.
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    if not edits:
        return {}, False  # nothing to review

    base_prompt = build_review_prompt(honey_text, spec, codebase_root, docs_root)
    logger.info("Running specify effectiveness review (%d edits, independent pass)...",
                len(edits))
    attempt_prompt = base_prompt
    for attempt in range(2):
        call_id = ledger.begin_call("specify", "specify_review", provider, model,
                                    attempt_prompt) if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=codebase_root,
                             timeout=600,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **(provider_kwargs or {}))
        except Exception as e:  # subprocess timeout, provider error, etc.
            logger.warning("specify: effectiveness review worker failed: %s", e)
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            return {}, True

        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout,
                               latency_s=wr.latency_s, ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)

        try:
            parsed = extract_first_json(wr.stdout)
        except ValueError:
            parsed = None
        reviews = parsed.get("reviews") if isinstance(parsed, dict) else None
        if isinstance(reviews, list):
            judgments: dict[str, dict] = {}
            for r in reviews:
                if isinstance(r, dict) and r.get("id") is not None:
                    judgments[str(r["id"])] = r
            return judgments, False

        if attempt == 0:
            logger.warning("specify: effectiveness review returned no usable 'reviews' JSON "
                           "— retrying once with a JSON-only reminder")
            attempt_prompt = base_prompt + _REVIEW_JSON_REMINDER
        else:
            logger.warning("specify: effectiveness review produced no usable JSON after retry "
                           "— inconclusive")
    return {}, True


def _apply_effectiveness_gate(
    spec: dict[str, Any],
    noop_ids: list[str],
    judgments: dict[str, dict],
    inconclusive: bool,
    incomplete_wiring: dict[str, str] | None = None,
    datasource_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Downgrade a ready spec that contains ineffective edits or could not be verified.

    - An edit flagged a deterministic no-op, or judged ``effective=false`` /
      ``coherent=false`` by the review, is ineffective → a ready_to_apply spec is
      downgraded to needs_reinvestigation (the fix does not work; loop back).
    - If no edit is flagged but the review was inconclusive (worker failed /
      unparseable), a ready_to_apply spec is downgraded to needs_reinvestigation:
      effectiveness could not be confirmed, so the loop re-works it rather than the
      tool vouching for an unverified claim.
    - The spec is never upgraded; only a ready claim is guarded.
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    noop_set = {str(x) for x in noop_ids}
    incomplete = {str(k): v for k, v in (incomplete_wiring or {}).items()}
    datasource = {str(k): v for k, v in (datasource_ids or {}).items()}
    ineffective: dict[str, str] = {}
    certain_ids: set[str] = set()      # deterministic findings (no-op / incomplete wiring)
    review_flagged: set[str] = set()   # findings from the LLM review only
    for e in edits:
        eid = str(e.get("id", "?"))
        if eid in noop_set:
            if e.get("kind", "edit") == "create_file":
                ineffective[eid] = "no-op (create_file content is empty or whitespace-only)"
            else:
                ineffective[eid] = "no-op (whitespace-normalized anchor == replacement)"
            certain_ids.add(eid)
            continue
        # Deterministic completeness (N175): an added import nobody uses means the
        # wiring is half-done — flag it the same way as a no-op (the fix won't work).
        if eid in incomplete:
            ineffective[eid] = incomplete[eid]
            certain_ids.add(eid)
            continue
        # Deterministic data-source grounding (N176): a SQL read switched to an
        # empty/sparser/missing table (checked against the live DB) won't return the
        # rows the symptom needs — a certain finding, same as a no-op.
        if eid in datasource:
            ineffective[eid] = datasource[eid]
            certain_ids.add(eid)
            continue
        j = judgments.get(eid)
        if isinstance(j, dict):
            if j.get("effective") is False:
                ineffective[eid] = "review: ineffective — " + str(j.get("reason", "")).strip()
                review_flagged.add(eid)
            elif j.get("coherent") is False:
                ineffective[eid] = "review: incoherent — " + str(j.get("reason", "")).strip()
                review_flagged.add(eid)
            elif j.get("in_scope") is False:
                # Effective+coherent but over-applies (edits a shared rule / broad
                # selector beyond the seed's named target) — a regression, so loop
                # back to re-author a narrower edit rather than ship it (T891 v2).
                ineffective[eid] = "review: over-scope — " + str(j.get("reason", "")).strip()

    spec["effectiveness"] = {
        "inconclusive": inconclusive,
        "ineffective_ids": sorted(ineffective),
    }
    # Annotate the flagged edits so the proposal / audit trail shows why.
    for e in edits:
        eid = str(e.get("id", "?"))
        if eid in ineffective:
            e["effectiveness"] = {"ok": False, "reason": ineffective[eid]}

    if spec.get("termination") != "ready_to_apply":
        return spec  # never upgrade — only a ready claim needs guarding

    if ineffective:
        # Routing (N174 #3): deterministic findings (no-op / incomplete wiring) and
        # over-scope regressions are CERTAIN → loop back (needs_reinvestigation). But a
        # review-ONLY "ineffective"/"incoherent" verdict on edits that are all VERIFIED and
        # part of a CROSS-FILE wiring is unreliable — the reviewer may have judged an edit
        # in isolation and missed that a back-end guard pairs with a front-end toast in
        # another file. Do NOT trust that lone verdict to declare a verified fix wrong; hold
        # it as inconclusive (which loops back to re-investigate) rather than asserting the
        # fix is broken. Only applies when there is NO certain finding and every flagged id
        # came from the review.
        distinct_files = {e.get("file") for e in edits if e.get("file")}
        only_review = bool(review_flagged) and not certain_ids and \
            set(ineffective) == review_flagged
        flagged_all_verified = all(
            str(e.get("anchor_status", "")).lower() == "verified"
            for e in edits if str(e.get("id", "?")) in review_flagged)
        if only_review and len(distinct_files) > 1 and flagged_all_verified:
            logger.warning("specify: review flagged %s as ineffective, but these are "
                           "VERIFIED edits in a cross-file wiring — holding as "
                           "inconclusive -> %s (not asserting the fix is wrong)",
                           sorted(review_flagged), _INCONCLUSIVE_TERMINATION)
            note = ("effectiveness gate: review judged " + ", ".join(sorted(review_flagged))
                    + " ineffective in isolation, but they are verified edits in a "
                    "cross-file wiring — re-investigating rather than asserting the fix "
                    "is broken")
            reason_code = RI_INCONCLUSIVE
        else:
            logger.warning("specify: edits %s do not change the reported behavior but "
                           "termination=ready_to_apply — overriding to %s",
                           sorted(ineffective), _INEFFECTIVE_TERMINATION)
            note = "effectiveness gate: " + "; ".join(
                f"{k} {v}" for k, v in sorted(ineffective.items()))
            # A data-source regression is an evidence gap (we read the wrong table),
            # not a wrong causal stitch — route it to re_retrieve to find the real
            # source rather than re_converge.
            reason_code = (RI_DATASOURCE_REGRESSION
                           if set(ineffective) & set(datasource) else RI_INEFFECTIVE)
    elif inconclusive:
        logger.warning("specify: effectiveness review inconclusive — downgrading "
                       "ready_to_apply to %s (re-investigate)", _INCONCLUSIVE_TERMINATION)
        note = ("effectiveness gate: review inconclusive — re-investigating to confirm "
                "the edits change the reported behavior before presenting as ready")
        reason_code = RI_INCONCLUSIVE
    else:
        return spec

    return _set_reinvestigation(spec, reason_code=reason_code, gate="effectiveness",
                                note=note)


def _apply_anchor_not_grounded_gate(spec: dict[str, Any]) -> dict[str, Any]:
    """Remove edits that contradict an anchor_not_grounded deferred item for the same file.

    When the author puts a direction in deferred[] with reason=anchor_not_grounded it means
    the grounding evidence is missing for that target. Emitting an edit for the same file is a
    logical contradiction — a file cannot be both ungroundable and successfully anchored.
    Remove the contradictory edit from edits[], keep only the deferred record, log the
    contradiction, and downgrade a ready_to_apply claim to needs_reinvestigation.
    """
    deferred = [d for d in (spec.get("deferred") or []) if isinstance(d, dict)]

    ungrounded_files: set[str] = set()
    for d in deferred:
        if str(d.get("reason", "")).lower() != "anchor_not_grounded":
            continue
        for field in [str(d.get("issue", "") or ""),
                      *[str(e) for e in (d.get("evidence") or [])]]:
            for tok in re.findall(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+", field):
                ungrounded_files.add(os.path.basename(tok).lower())

    if not ungrounded_files:
        return spec

    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    contradictory: list[str] = []
    kept: list[dict] = []
    for e in edits:
        ef = str(e.get("file", "") or "").replace("\\", "/")
        if os.path.basename(ef).lower() in ungrounded_files:
            contradictory.append(str(e.get("id", "?")))
        else:
            kept.append(e)

    if not contradictory:
        return spec

    logger.warning(
        "specify: edits %s target file(s) also deferred as anchor_not_grounded — "
        "removing contradictory edits (cannot both ground and defer-as-ungrounded "
        "the same anchor)", contradictory)
    spec["edits"] = kept
    if spec.get("termination") == "ready_to_apply":
        note = ("anchor-not-grounded gate: edits %s removed — file also in deferred "
                "as anchor_not_grounded (contradictory emit)" % contradictory)
        _set_reinvestigation(spec, reason_code=RI_ANCHOR_NOT_GROUNDED,
                             gate="anchor_not_grounded", note=note)
    return spec


def _apply_decisiveness_gate(spec: dict[str, Any]) -> dict[str, Any]:
    """Promote a conservatively-authored needs_reinvestigation spec to ready_to_apply.

    ``termination`` must reflect whether the edits in ``edits[]`` are safe to APPLY,
    not whether the whole investigation is closed. A specify author often hedges to
    needs_reinvestigation because the honey surfaced optional/policy directions (which
    land in ``deferred[]``) even though the concrete edits are anchor-verified and passed
    the effectiveness review. The effectiveness gate only ever downgrades, so without this
    there is no path to ready_to_apply and apply refuses an otherwise-safe fix.

    This never lowers a bar on its own. It promotes needs_reinvestigation -> ready_to_apply
    ONLY when every edit is verified/effective/confident AND every deferred item is optional
    (not a contradiction of the edits). The guards below make this safe: any spec the
    effectiveness gate genuinely downgraded carries either ``inconclusive=True`` or a non-
    empty ``ineffective_ids`` (or stale anchors), each of which trips an early return — so
    only an over-conservative hedge on otherwise-clean edits is ever promoted. An existing
    ``ready_to_apply`` is left untouched.
    """
    if spec.get("termination") != "needs_reinvestigation":
        return spec

    # The promotion stands on the effectiveness review having actually run and been
    # conclusive — that is the evidence. No conclusive review => no promotion.
    eff = spec.get("effectiveness")
    if not isinstance(eff, dict) or eff.get("inconclusive"):
        return spec
    ineffective = {str(x) for x in (eff.get("ineffective_ids") or [])}

    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    if not edits:
        return spec
    for e in edits:
        if str(e.get("id", "?")) in ineffective:
            return spec
        if str(e.get("confidence", "")).lower() not in _DECISIVE_CONFIDENCE:
            return spec
        if e.get("kind", "edit") == "create_file":
            content = e.get("content", "")
            if not content or not content.strip():
                return spec
        elif str(e.get("anchor_status", "")).lower() != "verified":
            return spec

    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []
    for d in deferred:
        if not isinstance(d, dict):
            return spec
        if str(d.get("reason", "")).lower() not in _OPTIONAL_DEFERRED_REASONS:
            return spec

    spec["termination"] = "ready_to_apply"
    # No longer reinvestigating — drop any structured reason a downstream gate stamped.
    spec.pop("reinvestigation", None)
    note = ("decisiveness gate: promoted needs_reinvestigation -> ready_to_apply — every "
            "edit is verified/effective and confident; deferred items remain surfaced as "
            "optional directions")
    _append_note(spec, note)
    logger.info("specify: decisiveness gate promoted needs_reinvestigation -> ready_to_apply "
                "(%d verified/effective edit(s), %d optional deferred)",
                len(edits), len(deferred))
    return spec


def _deferred_is_live_refuted(d: dict[str, Any]) -> bool:
    """True when a deferred item's OWN text shows the claim was refuted by live code.

    Scans the deferral's ``issue`` / ``reason`` / ``evidence`` for a refutation marker
    (e.g. "not observed in live code", "refuted", Korean 라이브 코드에서 … 반박). Such a
    deferral is a disproven hypothesis, not a substantive punted fix — see
    ``_LIVE_REFUTED_RE``. Pure-local, never raises.
    """
    parts = [str(d.get("issue", "")), str(d.get("reason", ""))]
    ev = d.get("evidence")
    if isinstance(ev, list):
        parts.extend(str(x) for x in ev)
    elif ev:
        parts.append(str(ev))
    return bool(_LIVE_REFUTED_RE.search("\n".join(parts)))


def _apply_deferred_substance_gate(spec: dict[str, Any],
                                   honey_text: str = "") -> dict[str, Any]:
    """Downgrade a ready_to_apply spec that punted a SUBSTANTIVE fix to deferred[].

    A deferred item whose reason says the real fix is bigger than what was authored —
    it spans multiple files (``multi_file_design``) or could not be reduced to a single
    anchored edit (``not_expressible_as_edit``) — is not a harmless footnote. It is very
    often the actual root cause, filed away while the easy, symptom-level edits ship as
    "done". This is the N176 cheap-path: a front-end one-liner marked ready_to_apply while
    the back-end cause that actually clears the symptom sat in deferred. When such a
    deferral rides alongside a ready_to_apply spec we cannot vouch that the fix works, so
    we downgrade to needs_reinvestigation and let the loop re-work the FULL fix.

    Converge-certified escape (N182): the blanket downgrade above assumes the punted
    direction MIGHT be the real cause. But when converge already CERTIFIED a single causal
    locus — its cause→symptom check ruled ``consistent`` and survived every converge-side
    guard (N170/N180/M017), which is exactly when the honey renders the "Primary edit
    target" block — and an authored edit lands ON that certified locus, then the certified
    locus IS the verified cause and the substantive deferred is a genuinely secondary peer,
    not the punted root cause this gate guards against. In that case we ship the certified
    edit(s) ready and leave the peer surfaced in deferred[] (the operator no longer needs a
    manual --partial for the common case). Fail-closed and narrow: only for a SINGLE-locus
    convergence (multi-locus keeps its own stricter coverage gate), only when converge's
    attribution is grounded, and only when an edit actually covers it — absent any of these
    the gate fires exactly as before, so the N176 protection is untouched.

    Scope/limits (the deliberate side-effect): ``policy_direction`` (a genuine side note)
    is exempt; ``anchor_not_grounded`` has its own earlier gate. A spec that legitimately
    fixes the request AND merely notes a broader future refactor as multi_file_design will
    also be sent back — that over-rejection is the accepted cost: an extra cheap re-run is
    far better than shipping a non-fix (the failure this gate exists to stop). Never
    upgrades; only guards a ready claim.
    """
    if spec.get("termination") != "ready_to_apply":
        return spec
    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []
    punted: list[str] = []
    for d in deferred:
        if not isinstance(d, dict):
            continue
        if str(d.get("reason", "")).lower() not in _SUBSTANTIVE_DEFERRED_REASONS:
            continue
        # A deferral the author already refuted against live code is a disproven
        # hypothesis, not a punted root cause — exempt it so a complete fix is not
        # re-opened over a claim its own evidence retracted (T907).
        if _deferred_is_live_refuted(d):
            continue
        punted.append(str(d.get("issue", "") or d.get("reason", ""))[:160])
    if not punted:
        return spec

    # Converge-certified escape: single-locus convergence + an edit on the certified locus.
    # Set HIVE_NO_DEFERRED_ESCAPE=1 to disable (A/B isolation / kill-switch) — the gate
    # then fires the blanket downgrade exactly as it did before the N182 escape.
    if (not os.environ.get("HIVE_NO_DEFERRED_ESCAPE")
            and not _converge_target_loci(honey_text)):  # single-locus only (multi has its own gate)
        certified = _converge_certified_locus(honey_text)
        if certified:
            edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
            cb = os.path.basename(certified)
            covered = any(
                (f := str(e.get("file", "")).replace("\\", "/").lstrip("/"))
                and (f == certified or f.endswith("/" + certified)
                     or certified.endswith("/" + f) or os.path.basename(f) == cb)
                for e in edits)
            if covered:
                spec["deferred_substance_escape"] = {
                    "certified_locus": certified, "punted": punted}
                msg = ("deferred-substance gate: converge CERTIFIED single locus "
                       f"{certified} and an authored edit covers it — shipping the "
                       "certified fix ready; substantive deferred peer(s) remain surfaced "
                       "in deferred[] (" + "; ".join(punted) + ")")
                spec["notes"] = ((spec.get("notes", "") or "") + ("\n" if spec.get(
                    "notes") else "") + msg).strip()
                logger.info("specify: deferred-substance gate ESCAPE — converge-certified "
                            "locus %s covered by an edit; ready_to_apply preserved "
                            "(deferred peer(s): %s)", certified, "; ".join(punted))
                return spec

    note = ("deferred-substance gate: ready_to_apply downgraded — a substantive fix was "
            "punted to deferred[] (" + "; ".join(punted) + "); the authored edits address "
            "only the surface, so the fix is not vouched until the deferred root cause is "
            "re-worked")
    _set_reinvestigation(spec, reason_code=RI_DEFERRED_ROOT_CAUSE,
                         gate="deferred_substance", note=note)
    logger.warning("specify: deferred-substance gate downgraded ready_to_apply -> "
                   "needs_reinvestigation — substantive fix punted to deferred[] (%s)",
                   "; ".join(punted))
    return spec


def _seed_target_files(honey_text: str) -> list[str]:
    """Parse the honey's "Seed-specified edit targets" section into file paths.

    The investigate stage lists the files the user explicitly named as edit targets
    (``hive.investigate.SEED_TARGET_SECTION``) as ``- path:line`` bullets. We read
    them back so the coverage gate can enforce that each becomes an actual edit.
    Returns repo-relative paths (the ``:line`` suffix stripped), order-preserving.
    """
    files: list[str] = []
    in_section = False
    for line in honey_text.splitlines():
        if line.startswith(SEED_TARGET_SECTION):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            s = line.strip()
            if s.startswith("- "):
                tok = s[2:].strip().strip("`")
                m = re.match(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)", tok)
                if m:
                    files.append(m.group(1).replace("\\", "/").lstrip("/"))
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _apply_seed_coverage_gate(spec: dict[str, Any], honey_text: str) -> dict[str, Any]:
    """A seed-named edit target must become an edit, or the spec is NOT ready.

    The honey marks the files the user explicitly named as AUTHOR targets and
    grounds their live text (``investigate.seed_edit_targets``). If the author still
    leaves one out of ``edits[]`` — silently or as an optional defer — that is a
    user-provided instruction dropped, exactly the Defect 2 failure (T892 dropped
    queries.json + the spec test as ``not_expressible_as_edit`` yet shipped green).
    We refuse to present such a spec as ready_to_apply, and record which target was
    missed and the author's stated reason (so the gap is reported, not hidden).

    Diagnostics are recorded unconditionally; only a ``ready_to_apply`` claim is
    downgraded (to needs_reinvestigation) — a spec already at needs_reinvestigation is
    not vouching for completeness, so it is left as-is.
    """
    targets = _seed_target_files(honey_text)
    if not targets:
        return spec
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    edited = [str(e.get("file", "")).replace("\\", "/").lstrip("/")
              for e in edits if e.get("file")]

    def _covered(t: str) -> bool:
        tb = os.path.basename(t)
        return any(e == t or e.endswith("/" + t) or t.endswith("/" + e)
                   or os.path.basename(e) == tb for e in edited)

    missing = [t for t in targets if not _covered(t)]
    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []
    reasons: dict[str, str] = {}
    for t in missing:
        tb = os.path.basename(t)
        why = "absent (not authored and not deferred)"
        for d in deferred:
            if isinstance(d, dict) and tb in str(d.get("issue", "")):
                why = (str(d.get("reason", "deferred")) + ": "
                       + str(d.get("issue", ""))[:160])
                break
        reasons[t] = why
    spec["seed_coverage"] = {"targets": targets, "missing": missing, "reasons": reasons}

    if missing and spec.get("termination") == "ready_to_apply":
        logger.warning("specify: seed-named edit target(s) not authored %s — "
                       "downgrading ready_to_apply to needs_reinvestigation (a user-"
                       "specified edit must not be silently dropped)", missing)
        note = ("seed-coverage gate: seed-specified target(s) not authored — "
                + "; ".join(f"{t} [{reasons[t]}]" for t in missing))
        _set_reinvestigation(spec, reason_code=RI_SEED_TARGET_UNCOVERED,
                             gate="seed_coverage", note=note,
                             detail="; ".join(missing))
    return spec


def _converge_target_loci(honey_text: str) -> list[str]:
    """Parse the honey's "Converge-attributed edit targets" section into file paths.

    converge emits this section ONLY when a scenario has MULTIPLE INDEPENDENT defects
    (N179) — the primary attributed defect PLUS each additional_defect, one ``- path:line``
    bullet per locus. A single-defect convergence produces no such section, so this returns
    ``[]`` and the gate is a no-op there. One entry per declared locus (NOT deduped by file
    — two independent defects in the same file are two loci), order-preserving.
    """
    files: list[str] = []
    in_section = False
    for line in honey_text.splitlines():
        if line.startswith(CONVERGE_TARGET_SECTION):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            s = line.strip()
            if s.startswith("- "):
                tok = s[2:].strip().strip("`")
                m = re.match(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)", tok)
                if m:
                    files.append(m.group(1).replace("\\", "/").lstrip("/"))
    return files


# Header the honey renders for converge's single attributed defect. It is emitted ONLY in
# the converged+attributed branch (hive.investigate._render_converge_section), which is
# reached AFTER converge's own causal guards (N170 cause→symptom, N180 dropped-peer,
# M017 data-stamp) have all passed — a contradicted/undecidable/dropped-peer convergence
# renders a DIFFERENT (re-examine) section and never this one. So the presence of this
# block in the honey IS converge's certification that this locus is the verified cause.
_CONVERGE_PRIMARY_HEADER = "### Primary edit target"


def _converge_certified_locus(honey_text: str) -> str | None:
    """Return the file converge CERTIFIED as the single attributed defect, or None.

    Reads the honey's "### Primary edit target — attributed defect" block and returns the
    repo-relative file from its ``- location: <file>:<lines>`` line. Returns None when:
      * the block is absent (no converged+consistent attribution), or
      * converge flagged the attributed file ungrounded (⚠ not in evidence) — a self-
        warned attribution is not a solid certification, so we do NOT relax on it.
    This is the SAME structured signal the converge-coverage gate trusts (converge's own
    declaration, NOT a natural-language parse of the seed — the N177 rabbit hole).
    """
    in_block = False
    for line in honey_text.splitlines():
        if line.startswith(_CONVERGE_PRIMARY_HEADER):
            in_block = True
            continue
        if in_block:
            s = line.strip()
            if s.startswith("## ") or s.startswith("### "):
                break  # left the block without a location line
            if s.startswith("- location:"):
                if "⚠" in s:  # ⚠ ungrounded — converge itself is unsure
                    return None
                tok = s[len("- location:"):].strip().strip("`")
                m = re.match(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z0-9]+)", tok)
                if m:
                    return m.group(1).replace("\\", "/").lstrip("/")
                return None
    return None


def _apply_converge_coverage_gate(spec: dict[str, Any], honey_text: str) -> dict[str, Any]:
    """A genuine MULTI-locus convergence must not ship a ready spec covering only some.

    N179: the seed enumerated THREE separate broken outputs; converge collapsed them to one
    selector edit and the spec still terminated ready_to_apply — the user saw two of the
    three symptoms unchanged. When converge declares the scenario has MULTIPLE INDEPENDENT
    defects (≥2 loci listed in ``CONVERGE_TARGET_SECTION``), each locus must become an edit
    or an explicit defer; otherwise a ready_to_apply claim is downgraded.

    Deliberately keyed off converge's OWN structured multi-locus declaration, NOT off any
    natural-language parse of the seed (the N177 rabbit hole) — and it fires ONLY in the
    multi-locus case, so a single-node convergence (where specify legitimately re-grounds the
    one attributed locus onto a different-but-correct file) is NEVER fought. ``addressed``
    tolerates a legitimate re-ground/consolidation via a COUNT escape: ≥ as many distinct
    edits as declared loci passes. The gate bites only the clear under-coverage case (fewer
    edits than independent loci, with an uncovered locus that was not deferred). Downgrade-
    only; records diagnostics unconditionally and never promotes a non-ready spec.
    """
    loci = _converge_target_loci(honey_text)
    if len(loci) < 2:
        return spec  # not a declared multi-locus convergence → nothing to enforce
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    edited = [str(e.get("file", "")).replace("\\", "/").lstrip("/")
              for e in edits if e.get("file")]
    deferred = spec.get("deferred") if isinstance(spec.get("deferred"), list) else []

    def _aligned(t: str) -> bool:
        tb = os.path.basename(t)
        return any(e == t or e.endswith("/" + t) or t.endswith("/" + e)
                   or os.path.basename(e) == tb for e in edited)

    def _deferred(t: str) -> bool:
        tb = os.path.basename(t)
        return any(isinstance(d, dict) and tb in str(d.get("issue", "")) for d in deferred)

    uncovered = [t for t in loci if not _aligned(t) and not _deferred(t)]
    # Count escape: as many distinct edits as declared loci is enough — a re-ground may
    # land each locus on a different (correct) file than converge named, and the gate must
    # not fight that. It bites only when there are FEWER edits than independent loci.
    addressed_enough = len(edits) >= len(loci)
    spec["converge_coverage"] = {"loci": loci, "uncovered": uncovered}

    if uncovered and not addressed_enough and spec.get("termination") == "ready_to_apply":
        logger.warning("specify: converge declared %d INDEPENDENT defect loci but only %d "
                       "edit(s) authored — downgrading ready_to_apply (uncovered: %s)",
                       len(loci), len(edits), uncovered)
        note = (f"converge-coverage gate: converge attributed {len(loci)} INDEPENDENT "
                f"defect loci but only {len(edits)} edit(s) authored — uncovered: "
                + "; ".join(uncovered))
        _set_reinvestigation(spec, reason_code=RI_CONVERGE_LOCUS_UNCOVERED,
                             gate="converge_coverage", note=note,
                             detail="; ".join(uncovered))
    return spec


def _review_and_gate(
    spec: dict[str, Any],
    honey_text: str,
    codebase_root: str,
    model: str,
    provider: str,
    ledger=None,
    provider_kwargs: dict | None = None,
    docs_root: str | None = None,
    db_conn: Any = None,
) -> dict[str, Any]:
    """Run both halves of the effectiveness gate and adjust the spec's termination."""
    noop_ids = _deterministic_noop_ids(spec)
    incomplete_wiring = _incomplete_wiring_ids(spec, codebase_root)
    # Both schema-grounding checks route through datasource_ids (re_retrieve): a missing/
    # empty table (column-blind) and an absent column (column-level) are the same class of
    # gap — SQL that cannot return the rows the symptom needs. A table-level finding wins
    # over a column-level one for the same edit id.
    datasource_ids = _undefined_column_ids(spec, db_conn)
    datasource_ids.update(_datasource_regression_ids(spec, db_conn))
    judgments, inconclusive = review_effectiveness(
        honey_text, spec, codebase_root, model, provider, ledger, provider_kwargs,
        docs_root=docs_root)
    return _apply_effectiveness_gate(spec, noop_ids, judgments, inconclusive,
                                     incomplete_wiring, datasource_ids)


# Kill-switch for the HTTP-shape red-test synthesis (A/B isolation + safety valve),
# mirroring HIVE_NO_HTTP_BRIDGE / HIVE_NO_DEFERRED_ESCAPE. When set, the pass is a
# no-op and specify keeps its prior behaviour.
_HTTP_SHAPE_ENV_OFF = "HIVE_NO_HTTP_SHAPE"


def _synthesize_http_shape_red_test(spec: dict[str, Any], honey_text: str,
                                    codebase_root: str,
                                    setup_block: str | None = None,
                                    app_fixture: str | None = None,
                                    test_dir: str = "tests") -> dict[str, Any]:
    """Attach an HTTP-shape red test so apply ALWAYS observes red→green (lever ⑦).

    The backstop the apply stage was missing: when the symptom is an FE-bound array
    field served by an HTTP route (``Array.isArray(it.modules)`` off
    ``GET /api/v1/projects``), synthesise a TestClient red test from the SAME grounding
    the retriever already built (route via ``_resolve_http_bindings`` — mount-prefix
    folded — and the field/container read deterministically), register it in
    ``spec.verify`` so ``apply --verify`` runs it, and let ``verify.py`` certify the fix
    by EXECUTION. This does NOT depend on converge's confidence: even when converge was
    (wrongly) sure the scenario was consistent, apply now has a red test to observe.

    Fail-open and conservative by construction — it leaves the spec untouched unless ALL
    hold: the kill-switch is off, the spec carries no red-test node yet (never clobber an
    author-written one), there is at least one SOURCE edit to certify, and the symptom +
    a runnable harness both resolve. Adds a ``create_file`` test edit + the verify wiring;
    never a new gate (the existing apply red→green path is what acts on it). Never raises.
    """
    if os.environ.get(_HTTP_SHAPE_ENV_OFF):
        return spec
    try:
        verify = spec.get("verify") if isinstance(spec.get("verify"), dict) else {}
        if verify.get("red_test_node"):
            return spec  # an author-written red test already drives the loop — don't clobber
        edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
        source_edits = [e for e in edits
                        if e.get("kind", "edit") != "create_file"
                        or not _is_test_path(e.get("file", ""))]
        if not source_edits:
            return spec  # nothing to certify → no point synthesising a red test
        result = synthesize_http_shape_red_test(
            honey_text, codebase_root, setup_block=setup_block,
            app_fixture=app_fixture, test_dir=test_dir or "tests")
        if not result:
            return spec  # symptom not recognised, or no runnable harness → fail-open
        # Avoid an id/path collision with an existing edit (very unlikely; be safe).
        existing_ids = {str(e.get("id")) for e in edits}
        existing_files = {e.get("file") for e in edits}
        if (result["edit"]["id"] in existing_ids
                or result["edit"]["file"] in existing_files):
            return spec
        spec.setdefault("edits", []).append(result["edit"])
        vblock = spec.setdefault("verify", {})
        if not isinstance(vblock, dict):
            vblock = {}
            spec["verify"] = vblock
        vblock["red_test_node"] = result["node"]
        ids = list(vblock.get("test_edit_ids") or [])
        if result["edit"]["id"] not in ids:
            ids.append(result["edit"]["id"])
        vblock["test_edit_ids"] = ids
        sym = result["symptom"]
        _append_note(spec, f"http-shape red test synthesised (lever ⑦): {sym.verb.upper()} "
                     f"{sym.full_path} must return non-empty {sym.field!r}.")
        logger.info("specify: synthesised HTTP-shape red test for %s %s (field=%s) → "
                    "node %s", sym.verb.upper(), sym.full_path, sym.field, result["node"])
    except Exception as e:  # observation must never break authoring
        logger.warning("specify: HTTP-shape red-test synthesis skipped (%s)", e)
    return spec


def run_specify(
    honey_path: str,
    codebase_root: str,
    output_path: str,
    contract_path: str | None = None,
    model: str = "gpt-5-mini",
    provider: str = "copilot",
    ledger=None,
    provider_kwargs: dict | None = None,
    review: bool = True,
    docs_root: str | None = None,
    ground: bool = True,
    review_model: str | None = None,
    review_provider: str | None = None,
    author_timeout: int = _AUTHOR_TIMEOUT_DEFAULT,
    author_retries: int = 0,
    db_conn: Any = None,
    http_shape_setup_block: str | None = None,
    http_shape_app_fixture: str | None = None,
    http_shape_test_dir: str = "tests",
) -> dict[str, Any]:
    """Run the specify stage: honey + live code → edit-spec JSON.

    Calls a single author worker, extracts the first complete JSON object from its
    stdout, enforces Stage-1 invariants, runs the effectiveness gate, writes the
    spec to ``output_path`` as the SSOT JSON, and returns the parsed dict.

    The effectiveness gate (``review=True``, default) is a second, independent pass
    that downgrades a ready_to_apply spec whose edits do not actually change the
    reported behavior (see ``_review_and_gate``). It costs one extra worker call;
    pass ``review=False`` to skip it. By default the reviewer runs on the SAME
    provider/model as the author; pass ``review_provider``/``review_model`` to route
    it elsewhere (e.g. deepinfra) — the reviewer is a tool-OFF single-shot judgement
    (the edit diff and grounded anchor values are already in the prompt), so moving
    it off the per-internal-turn-billed copilot is the chief specify cost lever.

    Raises:
        ValueError: if the author produced no parseable JSON object.
        FileNotFoundError: if the honey or contract file is missing.
    """
    with open(honey_path, "r", encoding="utf-8") as f:
        honey_text = f.read()

    # A tool-OFF author (e.g. deepinfra) has no live files — it MUST rely on the
    # grounding pre-flight, so force grounding on and switch the prompt to the
    # grounded-only contract (lift anchors from the injected ground-truth block,
    # defer what is not grounded). This is what moves the author off the
    # per-internal-turn-billed copilot (ground→author mirrors judge's retrieve→judge).
    grounded_only = provider not in _TOOL_PROVIDERS
    if grounded_only and not ground:
        logger.warning("specify: tool-OFF author on %s but ground=False — the author "
                       "has no way to lift anchors; forcing grounding on", provider)
        ground = True

    # Anchor-grounding pre-flight: lift the CURRENT value at each cited file:line
    # into the honey so the author writes a real change and the effectiveness
    # reviewer can judge the behavioral delta (NR164/NR165/TR891 fix). Local/free.
    if ground:
        # Seed-named edit targets get a generous grounding window so an approximate
        # seed line range still pulls the full block the author must rewrite (T892).
        wide_files = set(_seed_target_files(honey_text))
        honey_text, gdiag = ground_anchors(honey_text, codebase_root, docs_root,
                                           wide_files=wide_files)
        if gdiag["lifted"]:
            logger.info("specify: anchor-grounding lifted %d cited location(s): %s",
                        len(gdiag["lifted"]), gdiag["lifted"])
        if gdiag["unresolved"]:
            logger.debug("specify: anchor-grounding could not resolve %s",
                         gdiag["unresolved"])

        # Callee-contract grounding (N175 round-2): when the honey itself quotes a call
        # and the import it comes from, lift that function's REAL signature so the author
        # writes the call with the right argument order/shape instead of guessing. The
        # reliable detection backstop is the same lift inside build_review_prompt (keyed
        # off the actual edits); this prose-keyed pass is best-effort prevention. Free.
        roots = [r for r in (codebase_root, docs_root) if r]
        cblock = _render_callee_block(
            _collect_callee_contracts([(honey_text, honey_text, None)], roots))
        if cblock:
            honey_text = honey_text + "\n\n" + cblock
            logger.info("specify: callee-grounding lifted signature(s) for the author prompt")

        # Test-fixture grounding: surface the target's isolated pytest fixtures so a
        # data-dependent red test is built on the real harness (seed + call directly),
        # never the production store — and is therefore safe to run under apply --verify.
        fixtures = _collect_test_fixtures(codebase_root)
        fblock = _render_fixture_block(fixtures)
        example = _lift_db_test_example(codebase_root)
        if example:
            fblock = (fblock + "\n\n" + example) if fblock else example
        if fblock:
            honey_text = honey_text + "\n\n" + fblock
            logger.info("specify: test-fixture grounding lifted %d fixture(s)%s for the "
                        "author prompt", len(fixtures),
                        " + a DB-test wiring example" if example else "")

    contract_text = load_contract(contract_path)
    prompt = build_specify_prompt(honey_text, contract_text, codebase_root, docs_root,
                                  grounded_only=grounded_only)

    logger.info("Running specify author (single, not fan-out)...")
    logger.debug("Prompt length: %d chars", len(prompt))

    # The author is a single blocking call with a hard wall-clock cap. A slow agentic
    # CLI (codex) can hit it; ``author_retries`` extra attempts cover a transient
    # timeout / provider hiccup so one slow call does not discard a completed
    # investigate stage (T892). Each attempt is recorded to the ledger (real spend).
    wr = None
    last_exc: Exception | None = None
    for attempt in range(author_retries + 1):
        call_id = ledger.begin_call("specify", "specify", provider, model, prompt) \
            if ledger is not None else None
        try:
            wr = call_worker(provider, model, prompt, cwd=codebase_root,
                             timeout=author_timeout,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **(provider_kwargs or {}))
        except Exception as e:  # subprocess timeout, provider error, etc.
            last_exc = e
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            if attempt < author_retries:
                logger.warning("specify: author call failed (%s) — retrying "
                               "(attempt %d/%d)", e, attempt + 2, author_retries + 1)
                continue
            raise
        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout, latency_s=wr.latency_s,
                               ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)
        # A non-zero exit or an empty comb is a soft failure: retry if budget remains
        # rather than crashing in extract_first_json on an empty string.
        if (wr.exit_code != 0 or not wr.stdout.strip()) and attempt < author_retries:
            logger.warning("specify: author returned exit=%d / %d-char output — "
                           "retrying (attempt %d/%d)", wr.exit_code, len(wr.stdout),
                           attempt + 2, author_retries + 1)
            continue
        break

    spec = extract_first_json(wr.stdout)  # raises ValueError if no JSON found
    # Rewrap BEFORE any edits[]-iterating pass: if the author flattened a single edit onto
    # the spec root (no edits[] envelope), wrap it so the fix is not silently dropped as
    # "0 edits, not ready" (N175). No-op for a well-formed spec.
    spec = _rewrap_flattened_edit(spec)
    # Deterministic anchor disambiguation FIRST: when sibling edits share an identical,
    # non-unique anchor (same literal in two branches), widen each anchor with adjacent
    # live lines so it targets one occurrence uniquely — rescuing a fix that would
    # otherwise dead-end in apply's anchor_ambiguous over a mechanical collision.
    spec = _disambiguate_anchors(spec, codebase_root, docs_root)
    # Then the drift check: re-read each 'verified' anchor from live disk and downgrade any
    # that is absent/non-unique, so _normalize_spec's stale-anchor rule then refuses to
    # present it as ready (N175 E7 — verified-without-live-recheck).
    spec = _verify_anchors_live(spec, codebase_root, docs_root)
    # Whitespace-drift re-anchor (N178): an anchor that drifted by insignificant whitespace
    # only is re-synced to the exact live bytes here — re-lifting the live text so apply
    # finds it, with the author's change preserved. This is the specify-local re-anchor the
    # reason-code table always promised; recovering it deterministically keeps a sound,
    # high-confidence fix out of a needs_reinvestigation that the bridge can only terminate.
    spec = _reanchor_drifted(spec, codebase_root, docs_root)
    spec = _normalize_spec(spec)
    spec = _apply_anchor_not_grounded_gate(spec)

    # Effectiveness gate: a second, independent pass that refuses to present edits
    # which are anchored but do not change the reported behavior as ready. The
    # reviewer may run on a different (cheaper, tool-OFF) provider than the author.
    if review:
        spec = _review_and_gate(spec, honey_text, codebase_root,
                                review_model or model, review_provider or provider,
                                ledger, provider_kwargs, docs_root=docs_root,
                                db_conn=db_conn)

    # Decisiveness gate: a verified+effective edit must be applyable even when the
    # honey also surfaced optional/policy directions (which sit in deferred[]).
    spec = _apply_decisiveness_gate(spec)

    # Deferred-substance gate (N176): a ready_to_apply spec that punted a substantive fix
    # (multi_file_design / not_expressible_as_edit) to deferred[] is shipping only the
    # surface — downgrade so the loop re-works the deferred root cause instead of vouching
    # for the partial fix. Runs after decisiveness (which no longer promotes when such a
    # deferral is present) so it also guards an author-emitted ready.
    spec = _apply_deferred_substance_gate(spec, honey_text)

    # Seed-coverage gate: a file the user named as an explicit edit target must become an
    # edit, or a ready_to_apply spec is downgraded with the dropped target(s) reported
    # (Defect 2 / T892).
    spec = _apply_seed_coverage_gate(spec, honey_text)

    # Converge-coverage gate (runs LAST so it has final say): when converge declared the
    # scenario has MULTIPLE INDEPENDENT defects (N179), a ready_to_apply spec that authored
    # an edit for only SOME of the loci is downgraded — shipping the easy locus while the
    # others stay broken on screen is exactly the false-ready this catches. No-op unless
    # converge itself declared ≥2 independent loci, so a single-defect converge is untouched.
    spec = _apply_converge_coverage_gate(spec, honey_text)

    # HTTP-shape red-test synthesis (lever ⑦): when the symptom is an FE-bound field
    # served by an HTTP route, attach a TestClient red test so apply --verify ALWAYS
    # observes the fix go red→green — the backstop that does not depend on converge's
    # confidence. Runs after the gates (it observes the authored SOURCE edits) and is
    # fully fail-open: a no-op unless the symptom + a runnable harness both resolve.
    spec = _synthesize_http_shape_red_test(
        spec, honey_text, codebase_root,
        setup_block=http_shape_setup_block,
        app_fixture=http_shape_app_fixture,
        test_dir=http_shape_test_dir)

    # Step A finalizer: if the spec lands in needs_reinvestigation but NO gate stamped a
    # structured reason, the AUTHOR itself emitted it — record that so the reactive bridge
    # always finds a routable cause (it reads the author's own narrative from notes/detail).
    spec = _ensure_reinvestigation_reason(spec)

    problems = _validate_spec(spec)
    if problems:
        logger.warning("specify: edit-spec has structural problems: %s", "; ".join(problems))

    # Record provenance so a derived diff / reconcile loop can trace this back.
    # FORCE codebase_root to the tree that actually holds the edited files (probed
    # deterministically): for a design-doc edit the anchors live under docs_root,
    # not the code tree. We overwrite rather than setdefault because the author
    # worker routinely echoes the prompt's code root into the spec — an untrusted
    # value (worker output is a draft). Stamping the real root keeps the spec
    # self-consistent so apply resolves the paths even when the caller never passes
    # --docs to apply (apply joins file paths against this codebase_root).
    spec.setdefault("source_honey", honey_path)
    spec["codebase_root"] = os.path.abspath(
        _stamp_root(spec, codebase_root, docs_root))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    n_edits = len(spec.get("edits") or [])
    n_deferred = len(spec.get("deferred") or [])
    logger.info("Edit-spec written to %s (%d edits, %d deferred, termination=%s)",
                output_path, n_edits, n_deferred, spec.get("termination", "?"))
    return spec
