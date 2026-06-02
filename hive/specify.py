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
    is deferred to a human (needs_pm) rather than trusted.

The author's role prompt is the contract file itself, loaded at runtime so the
contract stays the single source of authoring rules (no duplicated prompt here).
"""

import json
import logging
import os
import re
from typing import Any

from hive.investigate import SEED_TARGET_SECTION
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
_VALID_TERMINATION = {"ready_to_apply", "needs_reinvestigation", "needs_pm", "needs_runtime"}
_STALE_STATUSES = {"stale", "not_found"}

# Effectiveness-gate outcomes. An ineffective edit means the fix does not change
# behavior, so the loop must re-investigate; an inconclusive review (the check
# could not be obtained) instead defers the ready decision to a human.
_INEFFECTIVE_TERMINATION = "needs_reinvestigation"
_INCONCLUSIVE_TERMINATION = "needs_pm"

# Decisiveness gate: a conservatively-authored needs_pm spec is promoted to
# ready_to_apply only when every edit clears these bars (never a blanket drop).
_DECISIVE_CONFIDENCE = {"high", "medium"}
# Deferred reasons that are "optional/surface" — they do not contradict the edits,
# so their presence must not block applying an independently-verified edit.
_OPTIONAL_DEFERRED_REASONS = {"policy_direction", "not_expressible_as_edit", "multi_file_design"}


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


def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Enforce Stage-1 invariants and reconcile internal inconsistencies.

    - gate.apply is ALWAYS forced false here (specify proposes only).
    - A spec containing a stale/not_found edit is not ready: if the author still
      claimed ready_to_apply, override it to needs_reinvestigation rather than
      presenting an unverified anchor as applicable.
    """
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


# How many lines of a create_file's content to surface to the effectiveness
# reviewer — enough to judge "non-empty and on-target" without ballooning the prompt.
_REVIEW_CONTENT_MAX_LINES = 40

# One terse JSON-only retry for the effectiveness review (mirrors judge's lever):
# now that the reviewer runs on a tool-OFF API provider (deepinfra), a stray prose
# wrapper or fence would otherwise degrade a ready spec straight to needs_pm. The
# retry is a transport reparse — recorded to the ledger (a real paid call) but it
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

[Judge each edit]
For every edit decide three booleans, applying the criterion that matches the edit's kind:
- effective:
  - Anchor edit (kind "edit" or absent): would applying this edit actually change the \
behavior the honey identified as wrong? An edit that is functionally inert — a no-op \
assignment, a guard whose condition can never be true, a value set to what it already is, \
a change with no runtime effect — is effective=false EVEN THOUGH its anchor is valid. \
Re-open the live files to judge reachability and effect; do not assume.
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
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=codebase_root,
                             timeout=600, **(provider_kwargs or {}))
        except Exception as e:  # subprocess timeout, provider error, etc.
            logger.warning("specify: effectiveness review worker failed: %s", e)
            return {}, True

        if ledger is not None:
            ledger.record_call("specify", "specify_review", provider, model,
                               prompt=attempt_prompt, output=wr.stdout,
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
) -> dict[str, Any]:
    """Downgrade a ready spec that contains ineffective edits or could not be verified.

    - An edit flagged a deterministic no-op, or judged ``effective=false`` /
      ``coherent=false`` by the review, is ineffective → a ready_to_apply spec is
      downgraded to needs_reinvestigation (the fix does not work; loop back).
    - If no edit is flagged but the review was inconclusive (worker failed /
      unparseable), a ready_to_apply spec is downgraded to needs_pm: effectiveness
      could not be confirmed, so a human decides rather than the tool vouching.
    - The spec is never upgraded; only a ready claim is guarded.
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    noop_set = {str(x) for x in noop_ids}
    ineffective: dict[str, str] = {}
    for e in edits:
        eid = str(e.get("id", "?"))
        if eid in noop_set:
            if e.get("kind", "edit") == "create_file":
                ineffective[eid] = "no-op (create_file content is empty or whitespace-only)"
            else:
                ineffective[eid] = "no-op (whitespace-normalized anchor == replacement)"
            continue
        j = judgments.get(eid)
        if isinstance(j, dict):
            if j.get("effective") is False:
                ineffective[eid] = "review: ineffective — " + str(j.get("reason", "")).strip()
            elif j.get("coherent") is False:
                ineffective[eid] = "review: incoherent — " + str(j.get("reason", "")).strip()
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
        logger.warning("specify: edits %s do not change the reported behavior but "
                       "termination=ready_to_apply — overriding to %s",
                       sorted(ineffective), _INEFFECTIVE_TERMINATION)
        spec["termination"] = _INEFFECTIVE_TERMINATION
        note = "effectiveness gate: " + "; ".join(
            f"{k} {v}" for k, v in sorted(ineffective.items()))
    elif inconclusive:
        logger.warning("specify: effectiveness review inconclusive — downgrading "
                       "ready_to_apply to %s (human must confirm)", _INCONCLUSIVE_TERMINATION)
        spec["termination"] = _INCONCLUSIVE_TERMINATION
        note = ("effectiveness gate: review inconclusive — human must confirm the "
                "edits change the reported behavior before applying")
    else:
        return spec

    prev = str(spec.get("notes", "")).strip()
    spec["notes"] = f"{prev} {note}".strip() if prev else note
    return spec


def _apply_anchor_not_grounded_gate(spec: dict[str, Any]) -> dict[str, Any]:
    """Remove edits that contradict an anchor_not_grounded deferred item for the same file.

    When the author puts a direction in deferred[] with reason=anchor_not_grounded it means
    the grounding evidence is missing for that target. Emitting an edit for the same file is a
    logical contradiction — a file cannot be both ungroundable and successfully anchored.
    Remove the contradictory edit from edits[], keep only the deferred record, log the
    contradiction, and downgrade a ready_to_apply claim to needs_pm.
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
        spec["termination"] = "needs_pm"
        note = ("anchor-not-grounded gate: edits %s removed — file also in deferred "
                "as anchor_not_grounded (contradictory emit)" % contradictory)
        prev = str(spec.get("notes", "")).strip()
        spec["notes"] = f"{prev} {note}".strip() if prev else note
    return spec


def _apply_decisiveness_gate(spec: dict[str, Any]) -> dict[str, Any]:
    """Promote a conservatively-authored needs_pm spec to ready_to_apply.

    ``termination`` must reflect whether the edits in ``edits[]`` are safe to APPLY,
    not whether the whole investigation is closed. A specify author often sets
    needs_pm because the honey surfaced optional/policy directions (which land in
    ``deferred[]``) even though the concrete edits are anchor-verified and passed the
    effectiveness review. The effectiveness gate only ever downgrades, so without this
    there is no path to ready_to_apply and apply refuses an otherwise-safe fix.

    This never lowers a bar on its own. It promotes needs_pm -> ready_to_apply ONLY
    when every edit is verified/effective/confident AND every deferred item is optional
    (not a contradiction of the edits). ``needs_reinvestigation`` is never promoted
    (that means the fix does not work); an existing ``ready_to_apply`` is left untouched.
    """
    if spec.get("termination") != "needs_pm":
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
    note = ("decisiveness gate: promoted needs_pm -> ready_to_apply — every edit is "
            "verified/effective and confident; deferred items remain surfaced as "
            "optional for the PM")
    prev = str(spec.get("notes", "")).strip()
    spec["notes"] = f"{prev} {note}".strip() if prev else note
    logger.info("specify: decisiveness gate promoted needs_pm -> ready_to_apply "
                "(%d verified/effective edit(s), %d optional deferred)",
                len(edits), len(deferred))
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
    downgraded (to needs_pm) — a spec already at needs_pm/needs_reinvestigation is
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
                       "downgrading ready_to_apply to needs_pm (a user-specified edit "
                       "must not be silently dropped)", missing)
        spec["termination"] = "needs_pm"
        note = ("seed-coverage gate: seed-specified target(s) not authored — "
                + "; ".join(f"{t} [{reasons[t]}]" for t in missing))
        prev = str(spec.get("notes", "")).strip()
        spec["notes"] = f"{prev} {note}".strip() if prev else note
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
) -> dict[str, Any]:
    """Run both halves of the effectiveness gate and adjust the spec's termination."""
    noop_ids = _deterministic_noop_ids(spec)
    judgments, inconclusive = review_effectiveness(
        honey_text, spec, codebase_root, model, provider, ledger, provider_kwargs,
        docs_root=docs_root)
    return _apply_effectiveness_gate(spec, noop_ids, judgments, inconclusive)


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
        try:
            wr = call_worker(provider, model, prompt, cwd=codebase_root,
                             timeout=author_timeout, **(provider_kwargs or {}))
        except Exception as e:  # subprocess timeout, provider error, etc.
            last_exc = e
            if attempt < author_retries:
                logger.warning("specify: author call failed (%s) — retrying "
                               "(attempt %d/%d)", e, attempt + 2, author_retries + 1)
                continue
            raise
        if ledger is not None:
            ledger.record_call("specify", "specify", provider, model,
                               prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
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
    spec = _normalize_spec(spec)
    spec = _apply_anchor_not_grounded_gate(spec)

    # Effectiveness gate: a second, independent pass that refuses to present edits
    # which are anchored but do not change the reported behavior as ready. The
    # reviewer may run on a different (cheaper, tool-OFF) provider than the author.
    if review:
        spec = _review_and_gate(spec, honey_text, codebase_root,
                                review_model or model, review_provider or provider,
                                ledger, provider_kwargs, docs_root=docs_root)

    # Decisiveness gate: a verified+effective edit must be applyable even when the
    # honey also surfaced optional/policy directions (which sit in deferred[]).
    spec = _apply_decisiveness_gate(spec)

    # Seed-coverage gate (runs LAST so it has final say): a file the user named as
    # an explicit edit target must become an edit, or a ready_to_apply spec is
    # downgraded to needs_pm with the dropped target(s) reported (Defect 2 / T892).
    spec = _apply_seed_coverage_gate(spec, honey_text)

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
