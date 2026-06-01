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
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker

logger = logging.getLogger("hive.specify")

# The authoring contract doubles as the specify author's role/system prompt.
_DEFAULT_CONTRACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "recipes", "edit_spec_contract_v1.md"
)

# Structural expectations for the emitted edit-spec JSON.
_REQUIRED_KEYS = ("edits", "deferred", "gate", "termination")
_VALID_TERMINATION = {"ready_to_apply", "needs_reinvestigation", "needs_pm"}
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


def _docs_root_block(docs_root: str | None) -> str:
    """Prompt fragment telling the author about a separate design-doc tree.

    Design docs (PM-facing D/P/L specs) commonly live in a tree separate from the
    code. The author worker runs with cwd=codebase_root and would otherwise never
    see them — so a "edit the D031 design doc" direction gets mis-lowered onto the
    nearest-looking source file. This block makes the docs tree visible and tells
    the author to PREFER it when the honey's directive is about a document, and to
    emit ``file`` relative to whichever tree actually holds the target.
    """
    if not docs_root:
        return ""
    return f"""
[Design-docs root — a SEPARATE tree from the code]
{docs_root}
When the honey's fix direction targets a design document (e.g. a Markdown D/P/L
spec), the file lives under this docs root, NOT the codebase root. Open it there
(docs root + the path the honey cites) and lift `anchor_old` from the CURRENT text
byte-for-byte. Prefer the design document over any source file when the directive
is about the document. Emit `file` as the path relative to the tree that holds it.
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
                         docs_root: str | None = None) -> str:
    """Build the full prompt for the single specify author.

    The contract is the role/system prompt; the honey is the input to lower; the
    codebase_root tells the author where the LIVE code is. Anchors must be lifted
    from there byte-for-byte, never copied from the (possibly stale) honey. When
    ``docs_root`` is given, a separate design-doc tree is also made visible so a
    document-update direction is not mis-lowered onto the nearest source file.
    """
    return f"""{contract_text}

[Codebase root — read LIVE files from here]
{codebase_root}
{_docs_root_block(docs_root)}
[Input honey — lower each fix direction into the edit-spec contracted above]
Re-open every file you touch (under the codebase root, or the docs root when the
direction targets a design document) and lift `anchor_old` from the CURRENT text
byte-for-byte. Do NOT trust code quoted in the honey below.
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
For every edit decide two booleans, applying the criterion that matches the edit's kind:
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

[Output contract] Output ONLY this JSON object. No prose, no text outside the JSON.
{{
  "reviews": [
    {{ "id": "<edit id>", "effective": true, "coherent": true, "reason": "<one line>" }}
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
    """
    edits = [e for e in (spec.get("edits") or []) if isinstance(e, dict)]
    if not edits:
        return {}, False  # nothing to review

    prompt = build_review_prompt(honey_text, spec, codebase_root, docs_root)
    logger.info("Running specify effectiveness review (%d edits, independent pass)...",
                len(edits))
    try:
        wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=600,
                         **(provider_kwargs or {}))
    except Exception as e:  # subprocess timeout, provider error, etc.
        logger.warning("specify: effectiveness review worker failed: %s", e)
        return {}, True

    if ledger is not None:
        ledger.record_call("specify", "specify_review", provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "",
                           real_tokens=wr.real_tokens)

    try:
        parsed = extract_first_json(wr.stdout)
    except ValueError:
        logger.warning("specify: effectiveness review produced no parseable JSON")
        return {}, True

    reviews = parsed.get("reviews") if isinstance(parsed, dict) else None
    if not isinstance(reviews, list):
        logger.warning("specify: effectiveness review JSON missing a 'reviews' list")
        return {}, True

    judgments: dict[str, dict] = {}
    for r in reviews:
        if isinstance(r, dict) and r.get("id") is not None:
            judgments[str(r["id"])] = r
    return judgments, False


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
) -> dict[str, Any]:
    """Run the specify stage: honey + live code → edit-spec JSON.

    Calls a single author worker, extracts the first complete JSON object from its
    stdout, enforces Stage-1 invariants, runs the effectiveness gate, writes the
    spec to ``output_path`` as the SSOT JSON, and returns the parsed dict.

    The effectiveness gate (``review=True``, default) is a second, independent pass
    that downgrades a ready_to_apply spec whose edits do not actually change the
    reported behavior (see ``_review_and_gate``). It costs one extra worker call;
    pass ``review=False`` to skip it.

    Raises:
        ValueError: if the author produced no parseable JSON object.
        FileNotFoundError: if the honey or contract file is missing.
    """
    with open(honey_path, "r", encoding="utf-8") as f:
        honey_text = f.read()
    contract_text = load_contract(contract_path)
    prompt = build_specify_prompt(honey_text, contract_text, codebase_root, docs_root)

    logger.info("Running specify author (single, not fan-out)...")
    logger.debug("Prompt length: %d chars", len(prompt))

    wr = call_worker(provider, model, prompt, cwd=codebase_root, timeout=600,
                     **(provider_kwargs or {}))
    if ledger is not None:
        ledger.record_call("specify", "specify", provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "",
                           real_tokens=wr.real_tokens)

    spec = extract_first_json(wr.stdout)  # raises ValueError if no JSON found
    spec = _normalize_spec(spec)

    # Effectiveness gate: a second, independent pass that refuses to present edits
    # which are anchored but do not change the reported behavior as ready.
    if review:
        spec = _review_and_gate(spec, honey_text, codebase_root, model, provider,
                                ledger, provider_kwargs, docs_root=docs_root)

    # Decisiveness gate: a verified+effective edit must be applyable even when the
    # honey also surfaced optional/policy directions (which sit in deferred[]).
    spec = _apply_decisiveness_gate(spec)

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
