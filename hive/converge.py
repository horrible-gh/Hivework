"""CONVERGE stage — stitch scattered per-axis verdicts into ONE call path.

Pipeline position (cheap path, AFTER the per-axis JUDGE loop, BEFORE the honey):

  decompose → (retrieve → judge)*  →  CONVERGE (this)  →  render_local_honey → specify

Why this exists (N169 structural defect):
  The per-axis judge is blind across axes — each ``hive.judge.run_judge`` sees
  ONE axis's bundle and emits ONE located verdict. A defect that spans a call
  chain (endpoint → handler → db-fn → sql-key → FE) therefore comes back as N
  FRAGMENTS pointing at N different files, never stitched into the single path
  that actually executes. N169 located 5/7 axes on 5 different files, the author
  could not tell which node carried the bug, and terminated needs_reinvestigation
  with 0 edits — over and over. ``render_local_honey`` only GROUPS same-file loci;
  it cannot ORDER cross-file fragments into one executed path.

CONVERGE is the missing reconcile step on the cheap path. ONE tool-OFF model call
(deepinfra, the judge cost class) takes the located verdicts + the union of the
axes' call-chain evidence and:
  - ORDERS the fragments into a single executed path for the seed's scenario,
  - ATTRIBUTES the defect to ONE node on that path,
  - or, when a link is genuinely missing from the evidence, NAMES it — so a
    needs_reinvestigation cites the specific missing hop, not a blank re-ask.

Like judge, it is tool-OFF, single-shot, JSON-contracted, retries once on an
unparseable response, and NEVER raises: a flaky converge degrades to
``converged=False`` and the honey falls back to its ungrouped-evidence rendering.

Cost: ONE judge-class call per run, and only when there are ≥ ``min_located``
located verdicts (nothing to stitch otherwise → free skip, no model call).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.retriever import FollowupNeed, retrieve_followup

logger = logging.getLogger("hive.converge")

# Prompt-budget guards — mirror judge: the evidence is rendered COMPACT, not raw.
_MAX_EVIDENCE = 24          # distinct (file,lines) windows shown to the converger
_DATA_READ_LIMIT = 5        # max rows per data-read SELECT. NOT a fetch quota — it's a
                            # CEILING (a point lookup that matches 1 row returns 1). Small
                            # ON PURPOSE: the answer is a PRECISE row, not a table dump. The
                            # model is told to aim the WHERE at the row(s) that actually
                            # decide (e.g. for an ORDER BY, filter by the ranking predicate
                            # — the rows that would sort FIRST — not the whole sibling set),
                            # and if a read still misses, the iterative loop below lets it
                            # NARROW and read again rather than dumping rows. A small window
                            # forces a sharp condition and keeps prompt/token cost down.
_MAX_DATA_ROUNDS = 3        # iterative data-read chances: one initial read + up to two
                            # MORE — each round the converger may narrow / re-aim the query
                            # in reaction to what it saw (like reading more source when the
                            # first window misses). Bounded so a stuck model can't spin. If
                            # all three chances still can't surface the deciding row, the
                            # read CONDITION itself was likely mis-derived → the converger
                            # is told to re-derive it / redirect (missing_link) to
                            # reinvestigation rather than rule on data it never read.
_MAX_CELL = 200             # per-cell char cap when rendering live-DB rows into the honey/
                            # re-pass prompt. Row COUNT is already hard-capped (dbread
                            # LIMIT 20, glue-set), but a single wide TEXT/BLOB column under
                            # SELECT * could still bloat the prompt + token cost — so each
                            # value's rendered repr is truncated. The deciding columns the
                            # converger names are short; this only clips runaway blobs.
_EVIDENCE_CHARS = 500       # per-window char cap
_REASON_CHARS = 300         # per-verdict reason cap
_SCHEMA_LINE_CHARS = 400    # per-table char cap in the injected [DB SCHEMA] block — a
                            # very wide table's column list is clipped, not the table count

# One terse JSON-only retry (mirrors judge's lever): a reasoning model on deepinfra
# occasionally wraps the object in prose; the reparse recovers it. Recorded to the
# ledger (a real paid call) but it is still ONE logical converge pass.
_JSON_ONLY_REMINDER = (
    "\n\n[Retry] Your previous response could not be parsed as JSON. Output ONLY "
    "the single JSON object specified in the output contract above — no prose, no "
    "explanation, no markdown code fences, nothing before or after it."
)


@dataclass
class ConvergeResult:
    """The stitched single path + attributed defect for one investigation.

    ``converged`` is the gate a downstream consumer trusts; ``path`` /
    ``attributed_defect`` / ``missing_link`` are the evidence. ``raw`` keeps the
    full parsed JSON so nothing the model said is dropped.

    ``causal_check`` (N170) is the cause→symptom verification: reachability tells
    us a node is ON the executed path, but NOT that its code actually produces the
    reported symptom. ``converged`` is gated on a ``consistent`` causal check — a
    ``contradicted`` (the attributed code cannot produce the symptom under the only
    data state the scenario allows) or ``undecidable`` (the outcome depends on
    stored row state static evidence can't determine) attribution is NOT converged,
    so it is routed to reinvestigation / data-state confirmation, not to an edit.
    """

    converged: bool = False
    path: list[dict[str, Any]] = field(default_factory=list)
    attributed_defect: dict[str, Any] | None = None
    # Additional INDEPENDENT defects (N179). converge's core job is to attribute ONE node
    # on ONE call path — but some scenarios enumerate SEVERAL distinct broken outputs that
    # do NOT share a cause (e.g. "the highlight colour is wrong AND the step index is off
    # AND the status badge never flips"). Folding those into one node ships a half-fix while
    # terminating ready_to_apply. When (and only when) the scenario genuinely needs fixes at
    # multiple independent loci, the converger lists the non-primary ones here; the honey
    # renders each as its OWN edit target and specify's converge-coverage gate refuses to
    # call a spec ready that covered only some. Empty in the common single-defect case.
    additional_defects: list[dict[str, Any]] = field(default_factory=list)
    missing_link: dict[str, Any] | None = None
    causal_check: dict[str, Any] | None = None
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    # The live-DB rows actually read for the causal re-rule (N173). ``data_state_block``
    # is the auditable rendering of the executed SELECT(s) and the rows they returned —
    # the honey PASTES it verbatim so the report shows REAL data, never an assumed value.
    # ``data_state_backed`` is True iff at least one read returned ≥1 row, i.e. the
    # verdict below was ruled on FACT rather than on an assumption. ``data_state_attempted``
    # records that a configured DB read was tried even when it returned nothing / failed,
    # so the report can say so honestly instead of looking like the read never happened.
    data_state_block: str = ""
    data_state_backed: bool = False
    data_state_attempted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "path": self.path,
            "attributed_defect": self.attributed_defect,
            "additional_defects": self.additional_defects,
            "missing_link": self.missing_link,
            "causal_check": self.causal_check,
            "summary": self.summary,
            "data_state_block": self.data_state_block,
            "data_state_backed": self.data_state_backed,
            "data_state_attempted": self.data_state_attempted,
        }


def _trunc(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "…(truncated)"


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").strip().strip("/").lower()


def _aligns(a: str, b: str) -> bool:
    """Path-segment-aligned equality or suffix (handles abs↔rel, basename-degrade)."""
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or a.endswith("/" + b) or b.endswith("/" + a))


def _located(verdicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for v in verdicts if (v.get("verdict") or {}).get("located")]


def _evidence_windows(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union the axes' code windows + call-chain hops, deduped by (file, lines).

    The call-chain hops are exactly the cross-file edges the converger needs to
    ORDER the fragments into one path — they are already retrieved per axis but
    were consumed in isolation by each judge. Here they are pooled so a single
    call can see the whole chain. Capped to ``_MAX_EVIDENCE`` windows.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for b in bundles or []:
        snips = (b.get("code_snippets") or []) + (b.get("call_chain") or [])
        for s in snips:
            key = (_norm(s.get("file", "")), str(s.get("lines", "")))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= _MAX_EVIDENCE:
                return out
    return out


def _known_files(verdicts: list[dict[str, Any]], windows: list[dict[str, Any]]) -> set[str]:
    files: set[str] = set()
    for v in verdicts:
        f = _norm((v.get("verdict") or {}).get("file", ""))
        if f:
            files.add(f)
    for w in windows:
        f = _norm(w.get("file", ""))
        if f:
            files.add(f)
    return files


# ── Live-code grounding (N177) ──────────────────────────────────────────────────
# converge runs tool-OFF on the judge's pooled snippets, which are COMPACTED
# (``_EVIDENCE_CHARS``) and only as fresh/complete as each axis's retrieve. With only a
# partial view converge once FABRICATED a code mechanism — it attributed an "argument
# mismatch" to a FE call site whose live signature was actually fine, marked the
# cause→symptom check ``consistent``, and the honey then crowned that phantom as the
# primary edit target (the human PM read it as a confirmed fix). The symmetric antidote
# to the [Confirmed data state] DB block: lift the CURRENT full text at each LOCATED
# fragment's file:lines and inject it as AUTHORITATIVE code so the causal check rules on
# live source, not a truncated snippet. Deterministic, free (no model call), never raises
# — the converge analog of specify's anchor-grounding pre-flight.
_CODE_LIFT_PAD = 3            # lines of context around each fragment's cited range
_CODE_LIFT_MAX_PER = 80       # per-fragment line cap (a huge range is clamped)
_CODE_LIFT_MAX_TOTAL = 400    # total line budget across all lifts


def _parse_line_range(spec: str) -> tuple[int, int] | None:
    """Parse a ``"lo-hi"`` / ``"lo"`` fragment-line string into ``(lo, hi)`` (or None)."""
    m = re.match(r"\s*(\d+)\s*(?:-\s*(\d+))?", str(spec or ""))
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return (lo, hi) if hi >= lo else (hi, lo)


def _lift_live_code(located: list[dict[str, Any]], code_root: str | None) -> str:
    """Lift CURRENT live text at each located fragment's file:lines → authoritative block.

    Reads the real source at exactly the loci the converger is about to reason over so its
    causal check is grounded on live code (N177), not on the compacted retrieved snippets.
    Deterministic, free, never raises; an unresolvable / empty / oversized citation is
    simply skipped (it never fabricates). Total lift is bounded by ``_CODE_LIFT_MAX_TOTAL``.
    """
    if not code_root:
        return ""
    seen: set[tuple[str, int, int]] = set()
    parts: list[str] = []
    total = 0
    for v in located:
        vd = v.get("verdict") or {}
        rel = (vd.get("file") or "").replace("\\", "/").strip().lstrip("/")
        rng = _parse_line_range(vd.get("lines", ""))
        if not rel or not rng:
            continue
        lo = max(1, rng[0] - _CODE_LIFT_PAD)
        hi = rng[1] + _CODE_LIFT_PAD
        if hi - lo + 1 > _CODE_LIFT_MAX_PER:
            hi = lo + _CODE_LIFT_MAX_PER - 1
        key = (rel.lower(), lo, hi)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(os.path.join(code_root, rel), "r",
                      encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except OSError:
            continue
        clipped_hi = min(hi, len(all_lines))
        if clipped_hi < lo:
            continue
        text = "".join(all_lines[lo - 1:clipped_hi]).rstrip("\n")
        if not text.strip():
            continue
        n = text.count("\n") + 1
        if parts and total + n > _CODE_LIFT_MAX_TOTAL:
            break
        total += n
        parts.append(f"--- {rel}:{lo}-{clipped_hi} (live)\n{text}")
    return "\n".join(parts)


def build_converge_prompt(seed_text: str, located: list[dict[str, Any]],
                          unlocated: list[dict[str, Any]],
                          windows: list[dict[str, Any]],
                          data_state_block: str = "",
                          db_available: bool = False,
                          db_schema: str = "",
                          code_state_block: str = "") -> str:
    """Build the single converge prompt: fragments + evidence → one path + one node.

    ``data_state_block`` is the optional ``[Confirmed data state]`` section: on the
    re-pass after an ``undecidable`` first pass, the glue has READ the exact rows the
    converger asked for from the live DB and renders them here as AUTHORITATIVE fact.
    The model is told to rule on these real values, not on assumed ones — turning the
    undecidable causal check into a decidable (consistent / contradicted) one.

    ``db_available`` is set when a read-only DB connection IS configured for this
    codebase (N173). The converger has no tools and cannot tell whether the pipeline
    can fetch rows for it; without that signal it FABRICATED stored values (e.g.
    ``result_doc_id = 'doc123'``) and ruled ``consistent`` on the fiction, so the read
    gate downstream never fired. When this flag is set we tell the converger the read
    is available and FORBID inventing stored values — it must defer to a real read via
    ``data_reads``/``undecidable`` so the glue can fetch the rows and re-ask on fact.
    """
    frag_lines = []
    for v in located:
        vd = v.get("verdict") or {}
        frag_lines.append(
            f"- axis {v.get('axis_id', '?')}: {vd.get('file', '')}:{vd.get('lines', '')} "
            f"— {_trunc(vd.get('reason', ''), _REASON_CHARS)}")
    frags = "\n".join(frag_lines) or "(none)"

    unloc_lines = []
    for v in unlocated:
        vd = v.get("verdict") or {}
        r = _trunc(vd.get("reason", "") or "not located", _REASON_CHARS)
        unloc_lines.append(f"- axis {v.get('axis_id', '?')}: {r}")
    unlocs = "\n".join(unloc_lines)

    ev_lines = []
    for w in windows:
        via = f" via={w['via']}" if w.get("via") else ""
        ev_lines.append(f"--- {w.get('file')}:{w.get('lines')}{via}")
        ev_lines.append(_trunc(w.get("text", ""), _EVIDENCE_CHARS))
    evidence = "\n".join(ev_lines) or "(no evidence windows)"

    unloc_block = f"\n[Axes that did NOT locate (context)]\n{unlocs}\n" if unlocs else ""

    # On the re-pass, the actual rows the converger asked for have been read from the
    # live DB. They are AUTHORITATIVE — rule on them, not on assumed values.
    confirmed_block = ""
    if data_state_block.strip():
        confirmed_block = (
            "\n[Confirmed data state — ACTUAL rows read from the live DB; these are "
            "FACT, not assumptions. Re-run your cause→symptom check against THESE "
            "values and rule consistent or contradicted accordingly. Do NOT return "
            "undecidable for a field shown here.\n"
            "BUT if these rows are INSUFFICIENT — a line says the set was TRUNCATED, or "
            "the row that actually decides the verdict is NOT among them — do NOT give up "
            "and do NOT rule on what's missing. Instead emit a NEW, NARROWER "
            "causal_check.data_reads: add a condition / change the selector to home in on "
            "the deciding row (e.g. filter by the specific type/key, or read the parent "
            "first and chain to it). The pipeline will run it and ask you AGAIN with the "
            "new rows — keep narrowing until the deciding fact is in hand. The window is "
            "small by design, so make each read SHARP (aim the WHERE straight at the "
            "deciding row), not broad. Only rule once the value you need is actually "
            "present in the rows above.\n"
            "If you have already re-read a few times and STILL cannot surface the deciding "
            "row, stop narrowing the SAME way — that means the read CONDITION you derived "
            "was likely wrong (wrong table / selector / assumption about where the value "
            "lives). RE-DERIVE it from the evidence (a different table or key), or, if the "
            "value plainly is not reachable by reading here, emit a ``missing_link`` to "
            "send the investigation back to find the right place — never rule on data you "
            "could not read.]\n"
            + data_state_block.strip() + "\n")

    # When a read-only DB IS configured (db_available) AND we have not yet read it
    # (no confirmed_block on this pass), tell the converger the read exists and forbid
    # it from inventing stored values to reach a verdict (the N173 'doc123' fabrication).
    db_avail_block = ""
    if db_available and not confirmed_block:
        db_avail_block = (
            "\n[LIVE DATABASE AVAILABLE] A read-only connection to this system's live "
            "database IS configured and the pipeline CAN execute SELECT reads for you on "
            "request. Therefore:\n"
            "- You MUST NOT assume, guess, or INVENT any stored row/field value to reach a "
            "verdict (NEVER write an assumption like \"result_doc_id = 'doc123'\" or "
            "\"the head doc is approved\" for a value you cannot see in the code evidence). "
            "Inventing a stored value is a contract violation.\n"
            "- If your cause→symptom check depends on ANY stored row/field value not visible "
            "in the code evidence, you MUST set causal_check.data_dependent = true, set "
            "causal_check.verdict = \"undecidable\" on THIS pass, and "
            "emit causal_check.data_reads naming the exact table, the row selector "
            "(column=value taken from the scenario, e.g. the document id / group key), and "
            "the deciding column(s). Do NOT rule \"consistent\" or \"contradicted\" on an "
            "unread stored value. The pipeline will run your reads against the live DB and "
            "ask you again with the ACTUAL rows, where you rule on fact (KEEP "
            "data_dependent = true then). A \"consistent\" verdict flagged data_dependent "
            "but with NO data_reads — a guess about a stored value — is REJECTED.\n"
            "- Assumptions taken straight from the scenario text (e.g. \"the seed states R is "
            "approved\") are fine; assumptions about UNSEEN stored values are not — read them.\n")

    # The live DB's ACTUAL table/column names. Without this the converger guessed names
    # from whatever code was retrieved and mis-named the table (NR174: it asked for
    # ``items`` when the real table is ``workflow_sequence_items``, so the read came back
    # empty). Handing it the authoritative list makes the data_reads name real objects.
    # Live source at the located loci (N177): AUTHORITATIVE over the pooled snippets,
    # which are compacted and may be stale/partial. This is what stops converge
    # attributing a code mechanism that the real source does not actually exhibit.
    code_state = ""
    if code_state_block.strip():
        code_state = (
            "\n[Confirmed code — ACTUAL current source read live from the located loci; "
            "this is FACT, not a retrieved snippet. Run your cause→symptom check against "
            "THIS text. If your attributed defect claims a mechanism that this live source "
            "does NOT actually show — a signature/argument mismatch that is not present, a "
            "default/guard/branch that reads differently here, a value already equal to what "
            "you would change it to — that attribution is CONTRADICTED: do NOT attribute the "
            "defect to it (set causal_check.verdict=\"contradicted\" and, per [Keep hunting], "
            "point the next search elsewhere). The pooled evidence windows below may be "
            "COMPACTED or partial; where they differ from this block, THIS block wins.]\n"
            + code_state_block.strip() + "\n")

    schema_block = ""
    if db_available and db_schema.strip():
        schema_block = (
            "\n[DB SCHEMA — the live database's ACTUAL tables and columns. In "
            "causal_check.data_reads use ONLY names that appear here; never invent or "
            "ABBREVIATE a name (e.g. do NOT shorten \"workflow_sequence_items\" to "
            "\"items\"). If a name you need is NOT in this list, the deciding data is not "
            "in this DB — emit a missing_link instead of guessing a name.\n"
            "CROSS-CHECK schema-shape claims against THIS list before attributing a defect "
            "to one (N176): if the suspected cause is that a column/table is MISSING, "
            "RENAMED, or MISMATCHED, but the object IS present here, that hypothesis is "
            "CONTRADICTED — set causal_check.verdict=\"contradicted\" and look elsewhere; "
            "do NOT make a refuted schema claim the attributed defect.]\n"
            + db_schema.strip() + "\n")

    return f"""[Role] You are the CONVERGER for a Hivework investigation. Independent \
per-axis judges each localised ONE fragment of what is really a SINGLE call path \
(typically endpoint → request handler → db function → SQL key / query → frontend \
mapping). Your job is to STITCH those fragments into the one path that actually \
executes for the reported scenario, and attribute the defect to ONE node on it.

[Constraints] You have NO tools. Decide ONLY from the fragments and evidence below \
and emit the JSON immediately. Judge EXECUTION REACHABILITY: which fragments are on \
the path that actually runs for this scenario, and which are merely similar-looking \
code that is NOT on it. A located fragment can be a RED HERRING (real code, but not \
reached in this scenario) — say so by leaving it off the path. Reachability is \
NECESSARY but NOT SUFFICIENT: a node can be on the executed path yet not be what \
produces the reported symptom (step 3 below is where you check that).

[Seed is ground truth — do NOT invert it] The scenario may explicitly DECLARE an \
observed value WRONG — a worker/AI hallucination, a stale artifact, or just "this is not \
what it should be" — and state what the CORRECT value/state is instead. When it does, the \
seed's stated CORRECT value is GROUND TRUTH: the defect is that the code emits the NEGATED \
(wrong) value, and a fix must make it emit the seed's CORRECT one. NEVER write an \
attribution whose ``why`` restores the value the seed called wrong as the "intended" / \
"expected" output. (Example — seed: "the active step is painted YELLOW, but that is a \
worker hallucination; it should be BLUE." A ``why`` of "the active step is not painted \
yellow as intended" INVERTS the requirement: it re-crowns the seed-NEGATED value as the \
goal.) If your ``why`` would re-assert a seed-negated value as the intent, you have read \
the seed backwards — flip your reasoning before emitting.

[Reported scenario / seed]
{_trunc(seed_text, 2000)}
{confirmed_block}{db_avail_block}{schema_block}{code_state}
[Located fragments — each is ONE node candidate, from a different axis]
{frags}
{unloc_block}
[Pooled evidence windows (code + call-chain hops across all axes)]
{evidence}

[What to produce]
1. ORDER the on-path fragments into the single executed path for the scenario.
2. ATTRIBUTE the defect to exactly ONE node (the place a fix must change), with a \
one-line reason that explains the wrong behaviour at THAT node.
3. CAUSALLY VERIFY that attribution. Being on the executed path is not enough — you \
must show the attributed code actually PRODUCES the reported symptom. State the \
DATA-STATE assumptions the scenario forces (the concrete row/field values at each \
record the attributed code reads), then TRACE what the code OUTPUTS under those \
assumptions, and rule:
   - It reproduces the symptom under the stated assumptions → causal_check.verdict = \
"consistent".
   - Under the ONLY data state the scenario allows it does NOT (e.g. the candidate \
rows TIE on a column, so a later ORDER-BY/branch is never the discriminator and the \
claimed mis-ordering cannot occur) → verdict = "contradicted". The cause contradicts \
the symptom — do not pass it off as the defect.
   - The outcome DEPENDS on stored row state you cannot read from the static evidence \
(which row has result_doc_id set, what review status it carries, …) → verdict = \
"undecidable"; put the exact row state / fixture you would need in need_data_state, AND \
— so the pipeline can FETCH it for you and re-rule on fact — emit machine-readable \
``data_reads``: the precise table(s), the row selector (the column=value that picks the \
row, taken from the scenario, e.g. the document id), and the column(s) whose value \
decides the verdict. Name tables/columns using the EXACT names from the [DB SCHEMA] \
block above (the authoritative live-DB list); if no schema block is shown, fall back to \
the names visible in the evidence (the SQL in queries.json). Never abbreviate or invent \
a name. Do NOT guess an attribution to fill the gap. When the \
deciding row cannot be reached in one lookup (you must read a key from one table to \
find the row in the next), CHAIN the reads: give each read an ``id`` and, in a later \
read's ``where``, reference an earlier result with ``{{"from": "<that id>", "column": \
"<column to carry over>"}}`` instead of a literal — the pipeline runs the reads in order \
and feeds each result into the next (it issues plain single-table SELECTs, so express a \
join as such a chain, e.g. read the rows, then read the joined table by the id column \
they carried). Use ONLY a row selector value that actually appears in the scenario (a \
business key like the document id). Do NOT hardcode an INTERNAL key you cannot see there \
(e.g. ``sequence_id = 1``) — that is a guess that silently reads the wrong row; instead \
derive it by CHAINING from the scenario's key (read the parent table by the doc_id to get \
its internal id, then read the child by that id). When the attributed defect is an ORDER \
BY / LIMIT / "which row is \
selected first" decision, the read window is SMALL (a few rows) — so do NOT dump the \
whole sibling set hoping the winner is in it. Instead aim the WHERE at the rows that \
would RANK FIRST: filter by the ordering's LEADING predicate (the column the ORDER BY \
prioritises, e.g. ``result_doc_id IS NOT NULL`` / the not-yet-approved rows). If that \
targeted read comes back EMPTY, no row beats the tie-breaker, so the order falls through \
to the next key (e.g. sort_order) and the query already returns the EXPECTED row → the \
ordering hypothesis is CONTRADICTED. If it returns row(s), those are the actual head \
candidates — check whether they reproduce the symptom. (If you do read raw siblings and \
a TRUNCATED flag appears, re-read with the ranking predicate as the filter — that is the \
right, sharp condition.) When the query already yields the expected row, that query is \
NOT the defect even if a fork in the seed pointed at it: on a data question the live rows \
outrank the seed's framing, and the real cause lies on a DIFFERENT resolver / render path.
   - SYMPTOM DOMAIN — match the EVIDENCE you certify on to the KIND of symptom. A live \
DB / stored-value read (your ``data_reads``) can only certify a verdict about a STORED \
VALUE: which row is selected, what status/id a field holds. It can NEVER certify a \
RENDER / SHAPE / BINDING symptom — an element that is empty or missing on screen, a \
selector that does not appear, a response KEY the front-end reads under a DIFFERENT name \
(e.g. the handler emits ``module_id`` but the FE reads ``module``, so the list comes back \
empty). For those the deciding fact lives in CODE: trace the PRODUCER's emitted \
field/key to the CONSUMER's read of it and rule "consistent" ONLY if they AGREE. "The \
query now returns rows" does NOT prove "the consumer renders them" — if a competing \
fragment says the consumer reads a different key, adding rows / a UNION upstream leaves \
the symptom fully intact. Do NOT let a ``data_reads`` result that merely proves rows \
EXIST stand in for a render-layer cause→symptom check, and do NOT rule "consistent" on a \
render/binding symptom from a DB read alone.
   - DATA-DEPENDENCE FLAG — set causal_check.data_dependent = true WHENEVER the \
correctness of your ruling rests on a STORED row/field value (which row is selected, the \
status/id a field holds, whether a row exists) that is NOT written in the code evidence \
and NOT explicitly stated in the seed. When it is true you MUST also emit \
causal_check.data_reads naming the exact rows that decide it — the pipeline READS those \
rows from the live DB and re-asks you to rule on the REAL values, so your final \
consistent/contradicted is grounded on fact. A consistent (or contradicted) verdict that \
is data_dependent but carries NO data_reads — i.e. you ruled on an ASSUMED stored value \
without reading it — is INCOMPLETE and will be REJECTED (not trusted as a fix). Keep \
data_dependent = true even after you rule on the rows we returned. Set data_dependent = \
false ONLY when your ruling follows purely from the code logic plus seed-stated facts, \
with no unread stored value involved.
   - REFUTE BEFORE YOU DROP — when you leave a located fragment OFF the path, that is a \
claim it is a RED HERRING. If that fragment names a DISTINCT mechanism that could \
INDEPENDENTLY produce the reported symptom (a different file/key/branch, not a \
corroborating view of the SAME chain), you may not drop it SILENTLY: either causally \
REFUTE it (show, against live code, that it cannot produce the symptom) and say so in \
your ``trace``, or carry it as an additional INDEPENDENT defect (step 5). This is \
especially binding when your own attribution is certified only on a data read — the \
fragment you are about to discard may be the render-layer cause your DB read cannot see.
4. If — and only if — two adjacent nodes cannot be connected because a needed \
callee/symbol is NOT shown in the evidence, set converged=false and NAME the missing \
link instead of guessing.
5. MULTI-LOCUS check — is this ONE defect, or SEVERAL INDEPENDENT ones? Most scenarios \
are a single defect on a single path: attribute the one node above and leave \
``additional_defects`` EMPTY. But some reporters enumerate SEVERAL DISTINCT broken \
outputs that do NOT share a cause — e.g. "(1) the highlight colour is wrong, (2) the \
step index is off, AND (3) the status badge never flips to done" — three independent \
failures in different code, each needing its OWN fix. When (and ONLY when) the scenario \
genuinely requires fixes at MULTIPLE INDEPENDENT loci (distinct outputs, NOT corroborating \
views of one call chain), put the PRIMARY one in ``attributed_defect`` and EACH of the \
others in ``additional_defects`` with its own node/file/lines/why. Do NOT fold genuinely \
separate defects into one node, and do NOT pad ``additional_defects`` with corroborating \
context for a single defect — list a locus there only when LEAVING IT OUT would ship a \
half-fix.

[Gate] Only a "consistent" causal check is actionable downstream. When your check is \
"contradicted" or "undecidable", STILL fill attributed_defect with the node you \
suspected and causal_check with your honest reasoning — the pipeline routes it to \
reinvestigation / data-state confirmation, NOT to an edit. Do not suppress the \
finding to force a convergence.

[Keep hunting — a refutation is a LEAD, not a dead end] The reporter can only hand you \
the SYMPTOM they see on screen (e.g. "the head renders as DS"); they CANNOT tell you \
where the bug lives — DERIVING that is YOUR job, so never demand a pre-localised answer. \
When your causal check comes back "contradicted" — the suspected node is reachable but \
PROVABLY cannot produce the symptom (often the live data shows it already yields the \
EXPECTED output) — the symptom STILL has a home somewhere else. Do NOT stop at "not \
here". REASON about what MUST be true for the symptom to occur and where that path lives \
(e.g. "get_effective_head already returns the memo, yet the bar shows DS as current → \
some OTHER head/current-step resolver or the front-end render path is overriding it"), \
then ALSO emit ``missing_link`` pointing the next hunt there: name the symbols / greps / \
file_globs to search for the path that DOES produce the symptom, EXCLUDING the refuted \
node. The pipeline will fetch that neighbourhood and re-converge — turning your \
refutation into the next, better-aimed search instead of a rejection.

[Output contract] Output ONLY this JSON object. No prose outside the JSON.
{{
  "converged": true,
  "path": [
    {{ "node": "endpoint|handler|db_fn|sql_key|fe|other", "file": "<repo-relative>", "lines": "<start-end>", "symbol": "<fn/route/key name>" }}
  ],
  "attributed_defect": {{ "node": "<which node above>", "file": "<repo-relative>", "lines": "<start-end>", "why": "<one line: the wrong behaviour here>" }},
  "additional_defects": [ {{ "node": "endpoint|handler|db_fn|sql_key|fe|other", "file": "<repo-relative>", "lines": "<start-end>", "why": "<the SEPARATE wrong behaviour at this INDEPENDENT locus>" }} ],
  "causal_check": {{ "verdict": "consistent|contradicted|undecidable", "data_dependent": false, "data_state_assumptions": ["<the row/field values the scenario forces>"], "trace": "<what the attributed code outputs under those assumptions, and whether it reproduces the symptom>", "need_data_state": ["<when undecidable: the exact stored row state / fixture to confirm>"], "data_reads": [ {{ "id": "<short name for chaining, optional>", "table": "<table name from the evidence>", "where": {{ "<key column>": "<literal row selector OR {{\\"from\\": \\"<prior read id>\\", \\"column\\": \\"<column to carry over>\\"}}>" }}, "columns": ["<column(s) whose value decides the verdict>"] }} ] }},
  "missing_link": null
}}

When you cannot converge because a CODE link is missing, instead emit:
{{
  "converged": false,
  "path": [ ...the partial path you DID establish... ],
  "attributed_defect": null,
  "causal_check": null,
  "missing_link": {{ "between": ["<node A>", "<node B>"], "need": {{ "symbols": ["<callee/def to resolve>"], "greps": ["<literal/regex>"], "file_globs": ["<optional scope>"] }} }}
}}
"""


def _coerce_node(d: Any) -> dict[str, Any] | None:
    if not isinstance(d, dict):
        return None
    return {
        "node": str(d.get("node", "") or ""),
        "file": str(d.get("file", "") or ""),
        "lines": str(d.get("lines", "") or ""),
        "symbol": str(d.get("symbol", "") or ""),
    }


# The cause→symptom check verdicts we accept; anything else (typo, blank, a model
# that emitted the block but skipped the verdict) normalises to ``unverified`` so
# the gate fails CLOSED — only an explicit ``consistent`` is actionable.
_CAUSAL_VERDICTS = ("consistent", "contradicted", "undecidable")


def _coerce_causal(d: Any) -> dict[str, Any] | None:
    """Parse the converger's ``causal_check`` block, or None when absent.

    None means the converger did not perform the mandated step; the gate treats
    that as ``unverified`` (fail closed). A present-but-unrecognised verdict is
    also normalised to ``unverified`` rather than trusted.
    """
    if not isinstance(d, dict):
        return None
    verdict = str(d.get("verdict", "") or "").strip().lower()
    if verdict not in _CAUSAL_VERDICTS:
        verdict = "unverified"
    return {
        "verdict": verdict,
        # M017 data-stamp: the converger's own flag that this ruling's correctness rests
        # on a STORED row/field value not visible in the code and not stated in the seed.
        # When true the verdict must be BACKED by a real DB read (data_reads we execute);
        # the data-stamp gate demotes a ``consistent`` that is data_dependent yet unread.
        "data_dependent": bool(d.get("data_dependent", False)),
        "data_state_assumptions": [str(x) for x in (d.get("data_state_assumptions") or [])],
        "trace": str(d.get("trace", "") or ""),
        "need_data_state": [str(x) for x in (d.get("need_data_state") or [])],
        "data_reads": _coerce_data_reads(d.get("data_reads")),
    }


def _coerce_data_reads(raw: Any) -> list[dict[str, Any]]:
    """Parse the converger's machine-readable ``data_reads`` into clean read specs.

    Each spec is ``{id?, table, where: {col: val | {from, column}}, columns: [..]}`` —
    the structured form the dbread glue turns into mechanical SELECT(s). A ``where``
    VALUE may be a reference object ``{"from": "<prior read id>", "column": "<col>"}``
    instead of a literal, so reads CHAIN: a later read filters on the value(s) a prior
    read returned (resolve-a-key-then-look-it-up, the common multi-hop need). The ref is
    kept verbatim here and resolved at execution time. Lenient: drops anything without a
    table; identifier SAFETY is enforced later by hive.dbread (the anti-injection gate),
    so here we only normalise shape, never trust it.
    """
    out: list[dict[str, Any]] = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        table = str(r.get("table", "") or "").strip()
        if not table:
            continue
        where = r.get("where") if isinstance(r.get("where"), dict) else {}
        columns = [str(c) for c in (r.get("columns") or []) if str(c).strip()]
        out.append({
            "id": str(r.get("id", "") or "").strip(),
            "table": table,
            "where": {str(k): v for k, v in where.items()},
            "columns": columns,
        })
    return out


def _result_from(parsed: dict[str, Any] | None,
                 known_files: set[str]) -> ConvergeResult:
    """Parse + lightly ground the converger's JSON into a ConvergeResult.

    Anti-hallucination is SOFT here (unlike judge's hard downgrade): the converger
    legitimately attributes to a SQL-key / query file that lived only in an axis's
    call_sites, so an attributed file we can't path-align is annotated for the
    author to re-confirm rather than dropped — never invent, but don't erase a
    plausibly-correct attribution either.
    """
    if not isinstance(parsed, dict):
        return ConvergeResult(summary="converger returned no parseable JSON")

    path = [n for n in (_coerce_node(x) for x in (parsed.get("path") or [])) if n]
    converged = bool(parsed.get("converged", False))

    ad_raw = parsed.get("attributed_defect")
    attributed: dict[str, Any] | None = None
    if isinstance(ad_raw, dict) and (ad_raw.get("file") or ad_raw.get("node")):
        attributed = {
            "node": str(ad_raw.get("node", "") or ""),
            "file": str(ad_raw.get("file", "") or ""),
            "lines": str(ad_raw.get("lines", "") or ""),
            "why": str(ad_raw.get("why", "") or ""),
        }
        af = attributed["file"]
        if af and not any(_aligns(af, kf) for kf in known_files):
            attributed["ungrounded"] = True

    # Additional INDEPENDENT defects (N179) — soft-grounded exactly like the primary:
    # an entry whose file we cannot path-align is annotated ``ungrounded`` for the
    # author to re-confirm, never dropped (the converger legitimately names a locus
    # that lived only in an axis's call_sites). The primary is never duplicated here.
    additional: list[dict[str, Any]] = []
    prim_key = (_norm(attributed["file"]), str(attributed.get("lines", ""))) \
        if attributed else None
    for x in (parsed.get("additional_defects") or []):
        if not isinstance(x, dict) or not (x.get("file") or x.get("node")):
            continue
        d = {
            "node": str(x.get("node", "") or ""),
            "file": str(x.get("file", "") or ""),
            "lines": str(x.get("lines", "") or ""),
            "why": str(x.get("why", "") or ""),
        }
        if prim_key and (_norm(d["file"]), d["lines"]) == prim_key:
            continue  # same locus as the primary → not a separate defect
        if d["file"] and not any(_aligns(d["file"], kf) for kf in known_files):
            d["ungrounded"] = True
        additional.append(d)

    ml_raw = parsed.get("missing_link")
    missing: dict[str, Any] | None = None
    if isinstance(ml_raw, dict) and (ml_raw.get("between") or ml_raw.get("need")):
        need = ml_raw.get("need") if isinstance(ml_raw.get("need"), dict) else {}
        missing = {
            "between": [str(x) for x in (ml_raw.get("between") or [])],
            "need": {
                "symbols": [str(x) for x in (need.get("symbols") or [])],
                "greps": [str(x) for x in (need.get("greps") or [])],
                "file_globs": [str(x) for x in (need.get("file_globs") or [])],
            },
        }

    causal = _coerce_causal(parsed.get("causal_check"))

    # A claimed convergence with no attributed node is not a convergence.
    if converged and attributed is None:
        converged = False

    # ── Causal gate (N170): reachability is necessary, not sufficient. A converged
    # attribution is ACTIONABLE only when the cause→symptom check says the attributed
    # code actually produces the reported symptom. The converger attributed BEFORE it
    # ran this check, so it routinely emits converged=true on a node that is merely
    # reachable; we flip that to NOT-converged unless the check is ``consistent``.
    # The attribution is kept (the honey surfaces the suspected-but-refuted node) but
    # it no longer reads as a primary edit target — it routes to reinvestigation
    # (contradicted) or data-state confirmation (undecidable / unverified).
    if converged and attributed is not None:
        if causal is None:
            causal = {"verdict": "unverified", "data_dependent": False,
                      "data_state_assumptions": [],
                      "trace": "converger did not perform the cause→symptom check",
                      "need_data_state": []}
            converged = False
        elif causal["verdict"] != "consistent":
            converged = False

    summary = ""
    cv = (causal or {}).get("verdict")
    if converged and attributed:
        summary = (f"converged: defect at {attributed['file']}:{attributed['lines']} "
                   f"({attributed.get('node', '?')}) over a {len(path)}-node path "
                   f"[causal: consistent]")
    elif attributed and cv == "contradicted":
        summary = (f"not converged: causal contradiction — {attributed['file']}:"
                   f"{attributed['lines']} is reachable but cannot produce the symptom")
    elif attributed and cv in ("undecidable", "unverified"):
        summary = (f"not converged: causal check {cv} — needs data state/fixture to "
                   f"rule on {attributed['file']}:{attributed['lines']}")
    elif missing:
        summary = ("not converged: missing link between "
                   + " ↔ ".join(missing["between"]) if missing["between"]
                   else "not converged: missing link named")
    else:
        summary = "not converged"

    return ConvergeResult(converged=converged, path=path,
                          attributed_defect=attributed, additional_defects=additional,
                          missing_link=missing, causal_check=causal, summary=summary,
                          raw=parsed)


def _dedup_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup a flat window list by (file, lines), keeping first occurrence; cap size."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for w in windows:
        key = (_norm(w.get("file", "")), str(w.get("lines", "")))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(w)
        if len(out) >= _MAX_EVIDENCE:
            break
    return out


def _converge_once(seed_text: str, located: list[dict[str, Any]],
                   unlocated: list[dict[str, Any]], windows: list[dict[str, Any]],
                   known: set[str], provider: str, model: str, pk: dict[str, Any],
                   ledger, timeout: int, data_state_block: str = "",
                   db_available: bool = False, db_schema: str = "",
                   code_state_block: str = "") -> ConvergeResult:
    """One logical converge call (with a transport-level JSON-only reparse). Never raises.

    The reparse retry handles a model that wrapped the JSON in prose — it is a
    transport reparse (both attempts recorded to the ledger, like judge) and does
    NOT count against the missing-link budget the caller manages. ``data_state_block``,
    when given, injects the live-DB-confirmed rows so the re-pass rules on fact;
    ``db_available`` tells the converger a read is fetchable so it defers to it
    instead of inventing stored values (N173).
    """
    prompt = build_converge_prompt(seed_text, located, unlocated, windows,
                                   data_state_block, db_available, db_schema,
                                   code_state_block)
    attempt_prompt = prompt
    parsed: dict[str, Any] | None = None
    for attempt in range(2):
        call_id = ledger.begin_call("converge", "converge", provider, model,
                                    attempt_prompt) if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=None,
                             timeout=timeout, **pk)
        except Exception as e:  # timeout / provider error — not retried
            logger.warning("converge: worker failed: %s", e)
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            return ConvergeResult(summary=f"converge worker failed: {e}")

        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout,
                               latency_s=wr.latency_s, ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)

        try:
            parsed = extract_first_json(wr.stdout)
            break
        except ValueError:
            if attempt == 0:
                logger.warning("converge: no parseable JSON — retrying once with reminder")
                attempt_prompt = prompt + _JSON_ONLY_REMINDER
            else:
                logger.warning("converge: no parseable JSON after retry")

    return _result_from(parsed, known)


def _resolve_where(where: dict[str, Any],
                   prior: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], str]:
    """Resolve a spec's ``where`` against the rows produced by prior reads.

    A literal value passes through unchanged. A reference ``{"from": id, "column": c}``
    is replaced by the DISTINCT non-null values of column ``c`` across read ``id``'s
    rows: one value → scalar (``= ?``), several → a list (``IN (…)``), zero → the read
    cannot proceed (no upstream key to look up). Returns ``(resolved_where, skip_note)``;
    ``skip_note`` is non-empty only when a reference had no upstream values, in which
    case the caller skips the read. This is the GENERIC chaining primitive — it knows no
    table or column names, only "feed the prior result's values into the next filter".
    """
    resolved: dict[str, Any] = {}
    for col, val in where.items():
        if isinstance(val, dict) and "from" in val:
            src = prior.get(str(val.get("from")), [])
            colname = str(val.get("column", ""))
            vals: list[Any] = []
            seen: set[Any] = set()
            for row in src:
                v = row.get(colname)
                if v is not None and v not in seen:
                    seen.add(v)
                    vals.append(v)
            if not vals:
                return resolved, (f"no upstream values from "
                                  f"{val.get('from')!r}.{colname}")
            resolved[col] = vals if len(vals) > 1 else vals[0]
        else:
            resolved[col] = val
    return resolved, ""


def _introspect_schema(db_conn) -> dict[str, list[str]]:
    """Read the live DB's ``{table: [columns…]}`` once; ``{}`` on any failure.

    The single source of truth for BOTH the injected prompt block (so the converger names
    REAL objects, NR174) and the pre-execution read guard (so a hallucinated table/column
    is rejected before it becomes broken SQL). Failure degrades silently to ``{}`` — the
    converger then falls back to names from the code evidence and the guard is a no-op,
    exactly as before introspection existed.
    """
    if db_conn is None:
        return {}
    try:
        from hive.dbread import list_schema
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("converge: dbread unavailable (%s) — no schema", e)
        return {}
    try:
        return list_schema(db_conn)
    except Exception as e:
        logger.warning("converge: schema introspection failed (%s) — no schema", e)
        return {}


def _render_schema_block(schema: dict[str, list[str]]) -> str:
    """Render ``{table: [cols]}`` as the prompt's ``- table(col, col, …)`` lines."""
    lines = []
    for t in sorted(schema):
        cols = ", ".join(schema[t])
        lines.append(_trunc(f"- {t}({cols})", _SCHEMA_LINE_CHARS))
    return "\n".join(lines)


def _validate_read(spec: dict[str, Any], schema: dict[str, list[str]]) -> str:
    """Pre-execution guard: check a read's table/columns exist in the live schema.

    Returns "" when the read is safe to run (or when ``schema`` is empty — no schema means
    no basis to reject, so we degrade to running it). Otherwise returns a human note naming
    the first unknown identifier. This catches the NR174 hallucinations — a wrong table
    (``items`` for ``workflow_sequence_items``) and an invented column (``WHERE column =
    'result_doc_id'``) — BEFORE they become a SELECT, so the read is skipped honestly
    instead of failing with a SQL error or silently matching nothing.
    """
    if not schema:
        return ""
    table = (spec.get("table") or "").strip()
    if table not in schema:
        return f"unknown table {table!r} (not in live schema)"
    cols_ok = set(schema[table])
    # every named output column and every where-key must be a real column of this table
    named = list(spec.get("columns") or []) + list((spec.get("where") or {}).keys())
    for c in named:
        if c not in cols_ok:
            return f"unknown column {c!r} on table {table!r} (not in live schema)"
    return ""


def _run_data_reads(data_reads: list[dict[str, Any]], db_conn,
                    schema: dict[str, list[str]] | None = None,
                    ledger=None, axis_id: str = "converge"
                    ) -> tuple[str, bool, bool]:
    """Run the converger's ``data_reads`` against the live DB; return
    (block, any_rows, chain_broke).

    Deterministic glue — NOT a model call. Reads run IN ORDER so later ones can CHAIN on
    earlier results (a ``where`` value of ``{"from": <prior id>, "column": c}`` is
    resolved to that read's returned values — see ``_resolve_where``). This turns the
    common multi-hop need ("look up a key in table A, then use it to read table B, then
    C") into a sequence of safe single-table SELECTs via hive.dbread, with NO join logic
    and NO schema knowledge in this module.

    Every read is rendered into ``block`` for audit — the exact SELECT, then each row, or
    an explicit ``(no rows matched)`` / ``FAILED`` / ``skipped`` line so the report shows
    HONESTLY what happened (N173: never let a failed read look like a fabricated value).
    Any failure (no driver, bad identifier, connect error, an empty upstream) is logged
    and recorded, never raised: that fact just stays unconfirmed and the converge degrades
    to its static result. ``any_rows`` is True iff at least one read returned ≥1 row —
    only then is the verdict actually data-backed. ``chain_broke`` is True iff a CHAINED
    read could not run because a prior read produced NO upstream values to feed it — i.e.
    the model's own data-premise chain collapsed (the rows it expected to exist do not),
    which a coarse ``any_rows`` (satisfied by an incidental id-lookup) cannot see.
    """
    try:
        from hive.dbread import read_rows, DbReadError
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("converge: dbread unavailable (%s) — skipping data read", e)
        return "", False, False
    import time as _time
    _t0 = _time.monotonic()
    lines: list[str] = []
    any_rows = False
    chain_broke = False
    by_id: dict[str, list[dict[str, Any]]] = {}
    for idx, spec in enumerate(data_reads):
        table = spec.get("table", "")
        rid = spec.get("id") or f"read{idx}"
        # Guard FIRST: reject a hallucinated table/column against the live schema before
        # it becomes broken SQL (NR174). A skipped read is recorded honestly, never run.
        bad = _validate_read(spec, schema or {})
        if bad:
            logger.info("converge: data read %r rejected by schema guard — %s", rid, bad)
            lines.append(f"- read {rid} on {table}: skipped ({bad})")
            by_id[rid] = []
            continue
        resolved, skip = _resolve_where(spec.get("where") or {}, by_id)
        if skip:
            logger.info("converge: chained read %r skipped — %s", rid, skip)
            lines.append(f"- read {rid} on {table}: skipped ({skip})")
            by_id[rid] = []
            # A reference read with no upstream values = the model's premise chain
            # collapsed: the rows it expected to exist (e.g. a stale/non-null row to
            # feed this read) are absent in the live DB. Flag it so a "consistent"
            # ruling that rests on those absent rows does not ship (premise-refuted gate).
            chain_broke = True
            continue
        try:
            rr = read_rows(db_conn, table, columns=spec.get("columns") or None,
                           where=resolved or None, limit=_DATA_READ_LIMIT)
        except DbReadError as e:
            logger.warning("converge: data read on %r failed — skipping: %s", table, e)
            lines.append(f"- read {rid} on {table}: FAILED ({e})")
            by_id[rid] = []
            continue
        by_id[rid] = rr.rows
        lines.append(f"- query: {rr.sql}")
        if not rr.rows:
            lines.append("  -> (no rows matched)")
        else:
            any_rows = True
        for row in rr.rows:
            lines.append("  -> " + ", ".join(
                f"{k}={_trunc(repr(v), _MAX_CELL)}" for k, v in row.items()))
        # Hit the ceiling → the result set may be CLIPPED. Flag it so an ordering /
        # "which row wins" verdict is not drawn on a partial set (a silent truncation
        # would make that inference wrong, not just incomplete).
        if len(rr.rows) >= _DATA_READ_LIMIT:
            lines.append(f"  -> (NOTE: returned {_DATA_READ_LIMIT} rows = the read cap; "
                         f"the set may be TRUNCATED — do not draw an ordering/'which row "
                         f"wins' conclusion from a possibly-incomplete set)")
    block = "\n".join(lines)
    # Register this LOCAL (free, deterministic) live-DB read in the ledger so the
    # ★ undecidable → DB read ★ step shows up in the configured DB, not just model
    # calls. provider='local', mechanism='sqlite'; cost aggregate stays untouched.
    if ledger is not None:
        ledger.record_local(
            stage="db_read", axis_id=axis_id, mechanism="sqlite",
            detail=f"reads={len(data_reads)} rows={any_rows} chain_broke={chain_broke}",
            out_chars=len(block), latency_s=_time.monotonic() - _t0)
    return block, any_rows, chain_broke


def _fetch_data_state(data_reads: list[dict[str, Any]], db_conn) -> str:
    """Back-compat wrapper: run ``data_reads`` and return only the rendered block.

    Kept for callers/tests that want the auditable fact-line string; ``run_converge``
    uses :func:`_run_data_reads` directly so it can also tell whether real rows came back.
    """
    block, _, _ = _run_data_reads(data_reads, db_conn)
    return block


def _dropped_peer_guard(res: ConvergeResult, located: list[dict[str, Any]],
                        data_backed: bool) -> ConvergeResult:
    """N180: a DATA-certified ``consistent`` must not ship while a DISTINCT-locus located
    peer was dropped WITHOUT refutation.

    The failure (N180): the converger attributed the defect to one node, certified its
    cause→symptom check ``consistent`` on a LIVE DB READ (a data-state fact), and silently
    left a competing located fragment off the path — one naming a RENDER / binding-layer
    mechanism (a response key the FE reads under a different name) — neither refuted nor
    carried as an additional defect. A DB read proves rows EXIST; it cannot prove a
    render-layer symptom is resolved, so that ``consistent`` is certified on the WRONG
    evidence domain and the dropped peer may be the real cause. We demote to NOT-converged
    (stops a half-fix shipping ready_to_apply) and stamp ``dropped_peer`` so the honey
    routes it to reinvestigation (re-stitch: refute the peer against live code, or fix it).

    DELIBERATELY TIGHT (the N177 over-fire lesson): fires ONLY when the verdict is
    ``consistent`` AND was data-backed (a live read returned rows) AND a located peer sits
    at a DISTINCT file on NONE of {path, attributed, additional_defects} AND is never even
    MENTIONED in the causal trace. A consistency certified on CODE (no data read), or one
    whose trace addresses the peer, is untouched — and the [REFUTE BEFORE YOU DROP] prompt
    rule pushes the converger to MENTION why it drops a peer, which satisfies the trace
    check and keeps this guard silent in the legitimate red-herring case. Worst-case false
    fire costs one extra reinvestigation pass, never a wrong edit (fail toward re-examine).
    """
    if not (res.converged and data_backed):
        return res
    cc = res.causal_check or {}
    if cc.get("verdict") != "consistent":
        return res
    ad = res.attributed_defect or {}
    # Files the convergence already ACCOUNTS for — anything here is on the path / a named
    # target, i.e. NOT "dropped". A located peer aligning to one of these is fine.
    accounted = {_norm(ad.get("file", ""))}
    for n in res.path or []:
        accounted.add(_norm(n.get("file", "")))
    for d in res.additional_defects or []:
        accounted.add(_norm(d.get("file", "")))
    accounted.discard("")
    trace = (cc.get("trace") or "").lower()
    for v in located:
        vd = v.get("verdict") or {}
        pf = _norm(vd.get("file", ""))
        if not pf or any(_aligns(pf, a) for a in accounted):
            continue  # not located, or this fragment IS on the convergence
        base = pf.rsplit("/", 1)[-1]
        if base and base in trace:
            continue  # the converger addressed / refuted this peer in its reasoning
        # A genuinely unaddressed, distinct-locus located peer while DATA-certified.
        peer = {"axis_id": str(v.get("axis_id", "?")), "file": vd.get("file", ""),
                "lines": vd.get("lines", ""), "reason": vd.get("reason", "")}
        res.converged = False
        res.causal_check = {
            **cc, "dropped_peer": peer,
            "trace": (cc.get("trace") or "")
            + f" [N180 guard] verdict certified on a live DB read (data state) but the "
              f"competing located hypothesis at {peer['file']}:{peer['lines']} (axis "
              f"{peer['axis_id']}) was neither placed on the path nor refuted; a data read "
              f"cannot confirm a render-layer symptom — re-examine it."}
        res.summary = (
            f"not converged: data-certified consistent at "
            f"{ad.get('file', '')}:{ad.get('lines', '')} dropped an UNREFUTED competing "
            f"hypothesis at {peer['file']}:{peer['lines']} (N180 domain guard)")
        logger.info("converge: N180 guard demoted — data-certified consistent dropped "
                    "unrefuted peer %s:%s", peer["file"], peer["lines"])
        return res
    return res


def _premise_refuted_guard(res: ConvergeResult, db_available: bool,
                           chain_broke: bool) -> ConvergeResult:
    """A ``consistent`` data-dependent verdict whose OWN read chain came back with no
    upstream rows is ruled on rows the live DB PROVES are absent — demote it.

    The gap this closes (M035): the M017 data-stamp guard treats ``data_backed`` (≥1 row
    from ANY read) as "the verdict earned its stamp". But ``data_backed`` is satisfied by
    an INCIDENTAL upstream lookup (e.g. resolving the sequence id) while every DECIDING
    read — the one meant to prove the suspected stored value exists (a stale / non-null
    row) — was skipped because its parent produced NO upstream values. The converger then
    rules ``consistent`` on a premise the rows REFUTE: it suspected an SQL ordering bug
    that only bites when a stale row exists, the live rows show none exists (so the query
    already returns the expected head), yet it certifies the SQL anyway. The prompt
    already says "when the query yields the expected row, that query is NOT the defect —
    the real cause is a different resolver / render path", but a weak single-shot
    converger ignores it; this is the deterministic backstop.

    Fires ONLY when: the result converged AND a DB is configured AND the verdict is
    ``consistent`` AND the converger flagged it ``data_dependent`` AND the adopted ruling's
    read chain BROKE (``chain_broke`` — a chained read had no upstream values, i.e. the
    premise rows are absent). Demote to not-converged and stamp ``data_premise_refuted`` so
    the honey routes it to reinvestigation / the alternate (render) path.

    DELIBERATELY TIGHT (the N177 over-fire lesson): a ruling NOT flagged data_dependent is
    untouched; a data_dependent ``consistent`` whose chain resolved cleanly (the premise
    rows DID exist) is untouched; ``contradicted`` / ``undecidable`` are untouched. Worst-
    case false fire costs one extra reinvestigation pass, never a wrong edit — and only
    when the model's own declared data chain collapsed (downgrade-only, fail toward
    re-examine).
    """
    if not (res.converged and db_available and chain_broke):
        return res
    cc = res.causal_check or {}
    if cc.get("verdict") != "consistent" or not cc.get("data_dependent"):
        return res
    ad = res.attributed_defect or {}
    res.converged = False
    res.causal_check = {
        **cc, "data_premise_refuted": True,
        "trace": (cc.get("trace") or "")
        + " [premise-refuted] verdict is data_dependent (rests on a stored value) but the "
          "converger's OWN data-read chain came back with no upstream rows — the rows it "
          "relied on (e.g. a stale/non-null row) are ABSENT in the live DB, so the suspected "
          "query already yields the expected result. On a data question the live rows "
          "outrank the framing: route to the alternate (render/binding) path, do not certify "
          "this locus."}
    res.summary = (
        f"not converged: data-dependent consistent at "
        f"{ad.get('file', '')}:{ad.get('lines', '')} rests on rows the live DB proves "
        f"absent (premise refuted)")
    logger.info("converge: premise-refuted guard demoted — data-dependent consistent at "
                "%s:%s but its read chain found no upstream rows",
                ad.get("file", ""), ad.get("lines", ""))
    return res


def _data_stamp_guard(res: ConvergeResult, db_available: bool,
                      data_backed: bool) -> ConvergeResult:
    """M017 lever 2: a ``consistent`` verdict that DEPENDS on stored data must be STAMPED
    by a real DB read, never ruled on an ASSUMED value.

    The gap this closes: converge is tool-OFF, so when a symptom hinges on a stored
    row/field value (which row is selected, a status a field holds, whether a row exists)
    the converger can reason "code looks consistent" and rule ``consistent`` WITHOUT ever
    reading the DB — no ``data_reads`` named, so the existing read-loop never fires and the
    verdict ships on an assumption. The data-read machinery already turns a
    consistent-WITH-data_reads verdict into a fact-backed one (``data_backed``); this guard
    is the BACKSTOP for the model that flags its ruling data_dependent yet fails to back it
    with a read we could actually run.

    Fires ONLY when: the verdict is ``consistent`` AND a read-only DB IS configured
    (db_available) AND the converger itself flagged the ruling ``data_dependent`` AND NO
    live read backed it (``data_backed`` is False — either it named no reads, or the reads
    came back empty / failed). Then it is an UNSTAMPED assumption: demote to not-converged
    and mark ``data_unstamped`` so the honey routes it back to name + read the deciding rows
    (the read IS available — this is recoverable, not a runtime punt).

    DELIBERATELY TIGHT (the N177 over-fire lesson): a consistent verdict NOT flagged
    data_dependent (a pure code-logic ruling) is untouched; a data_dependent verdict that
    WAS backed by a real read is untouched (it earned its stamp). Worst-case false fire
    costs one extra reinvestigation pass, never a wrong edit (fail toward re-examine).
    """
    if not (res.converged and db_available):
        return res
    cc = res.causal_check or {}
    if cc.get("verdict") != "consistent" or not cc.get("data_dependent"):
        return res
    if data_backed:
        return res  # a real read returned rows that back the verdict → stamped, keep it
    ad = res.attributed_defect or {}
    res.converged = False
    res.causal_check = {
        **cc, "data_unstamped": True,
        "trace": (cc.get("trace") or "")
        + " [M017 data-stamp] verdict is data_dependent (rests on a stored value) but NO "
          "live DB read backed it — ruled on an ASSUMED value; a read-only DB IS "
          "configured, so route back to name and READ the deciding row, then re-rule on "
          "fact."}
    res.summary = (
        f"not converged: data-dependent consistent at "
        f"{ad.get('file', '')}:{ad.get('lines', '')} not backed by a live DB read "
        f"(M017 data-stamp)")
    logger.info("converge: M017 data-stamp guard demoted — data-dependent consistent at "
                "%s:%s had no backing read", ad.get("file", ""), ad.get("lines", ""))
    return res


# ── Per-locus SPLIT converge (M020 follow-up) ───────────────────────────────────
# The holistic converge asks ONE weak single-shot to do the whole stitch AND pick the
# guilty node among several competing located loci — so it wanders run to run. The split
# pass asks a NARROW, low-variance question per locus ("does THIS locus's live code
# produce the symptom?") and COMBINES the answers by deterministic elimination. The hard
# combine (which one is the cause) is then free CODE, not a model judgment, so the wobble
# at the stitch point is gone. Adopts a result ONLY on a clean elimination (exactly one
# survivor); anything else falls back to the holistic path — precision layer, never a new
# failure mode (see ConvergeSplitConfig).
_LOCUS_DATA_ROUNDS = 1   # per-locus: one initial call + at most ONE data re-ask (bounded
                         # tighter than the holistic _MAX_DATA_ROUNDS — the narrow question
                         # needs the deciding row once, not an iterative hunt).


def build_locus_prompt(seed_text: str, focal: dict[str, Any],
                       others: list[dict[str, Any]], windows: list[dict[str, Any]],
                       focal_code: str = "", data_state_block: str = "",
                       db_available: bool = False, db_schema: str = "") -> str:
    """Build the NARROW single-locus prompt: does THIS one locus produce the symptom?

    Unlike the holistic converge prompt (order the whole path + pick one of N), this asks
    a yes/no causal question about ONE located fragment. The other located loci are listed
    as CONTEXT ONLY — the model must rule on the focal locus alone. Low variance by design,
    so a cheaper model is reliable; the elimination across loci happens in CODE afterwards.
    """
    vd = focal.get("verdict") or {}
    focal_line = (f"{vd.get('file', '')}:{vd.get('lines', '')} — "
                  f"{_trunc(vd.get('reason', ''), _REASON_CHARS)}")

    other_lines = []
    for v in others:
        ovd = v.get("verdict") or {}
        other_lines.append(f"- {ovd.get('file', '')}:{ovd.get('lines', '')} "
                           f"— {_trunc(ovd.get('reason', ''), _REASON_CHARS)}")
    others_block = ("\n[Other suspected loci — CONTEXT ONLY, do NOT rule on these]\n"
                    + "\n".join(other_lines) + "\n") if other_lines else ""

    code_block = ""
    if focal_code.strip():
        code_block = ("\n[Focal locus — ACTUAL current source read live; rule on THIS "
                      "text, not on a snippet]\n" + focal_code.strip() + "\n")

    ev_lines = []
    for w in windows:
        via = f" via={w['via']}" if w.get("via") else ""
        ev_lines.append(f"--- {w.get('file')}:{w.get('lines')}{via}")
        ev_lines.append(_trunc(w.get("text", ""), _EVIDENCE_CHARS))
    evidence = "\n".join(ev_lines) or "(no extra evidence)"

    confirmed_block = ""
    if data_state_block.strip():
        confirmed_block = (
            "\n[Confirmed data state — ACTUAL rows read from the live DB; FACT, not "
            "assumptions. Rule consistent/contradicted against THESE values; do NOT return "
            "undecidable for a field shown here. If the deciding row is plainly ABSENT here "
            "(the rows you needed do not exist), the suspected stored value is not present, "
            "so this locus's code already yields the EXPECTED output → rule contradicted.]\n"
            + data_state_block.strip() + "\n")

    db_avail_block = ""
    if db_available and not confirmed_block:
        db_avail_block = (
            "\n[LIVE DATABASE AVAILABLE] A read-only DB connection IS configured. If your "
            "ruling depends on ANY stored row/field value not visible in the code, you MUST "
            "set data_dependent=true, verdict=\"undecidable\" on THIS pass, and emit "
            "data_reads naming the exact table, row selector (a business key from the "
            "scenario), and deciding column(s). NEVER invent a stored value to rule.\n")

    schema_block = ""
    if db_available and db_schema.strip():
        schema_block = ("\n[DB SCHEMA — use ONLY these table/column names in data_reads]\n"
                        + db_schema.strip() + "\n")

    return f"""[Role] You are checking ONE suspected defect locus for a Hivework \
investigation. Several independent judges each localised a fragment that MIGHT be the \
cause of the reported symptom. Your job is NOT to stitch them — it is to rule on a \
SINGLE locus: does the code at the FOCAL LOCUS below actually PRODUCE the reported \
symptom for this scenario?

[Constraints] You have NO tools. Rule ONLY on the FOCAL LOCUS. The other loci are listed \
for context so you understand the competing hypotheses, but you must NOT decide which one \
is guilty — that is combined later. Reachability is not enough: a locus can be on the \
executed path yet not be what produces the symptom.

[Seed is ground truth — do NOT invert it] If the scenario DECLARES an observed value \
WRONG and states the CORRECT one, the CORRECT value is ground truth: the defect is that \
the code emits the WRONG value. Never write reasoning that re-crowns the seed-negated \
value as the intended output.

[Symptom domain] Match the evidence to the symptom KIND. A live-DB / stored-value read \
can only certify a STORED-VALUE symptom (which row is selected, a status/id a field \
holds). It can NEVER certify a RENDER / SHAPE / BINDING symptom (an element empty on \
screen, a response key the FE reads under a different name, a colour/class). For those \
the deciding fact lives in CODE — trace the producer's emitted field/key to the \
consumer's read of it. Do not rule "consistent" on a render symptom from a DB read alone.

[Reported scenario / seed]
{_trunc(seed_text, 2000)}
{confirmed_block}{db_avail_block}{schema_block}
[FOCAL LOCUS — rule on THIS one]
{focal_line}
{code_block}{others_block}
[Pooled evidence windows (context)]
{evidence}

[What to produce] Rule whether the FOCAL LOCUS's code produces the reported symptom:
  - It DOES, under the data state the scenario forces → verdict = "consistent" (this \
locus is a cause).
  - It provably CANNOT (e.g. the live data shows it already yields the expected output, \
or the live source does not exhibit the claimed mechanism) → verdict = "contradicted".
  - The outcome DEPENDS on a stored row/field value you cannot read here → verdict = \
"undecidable"; set data_dependent=true and emit data_reads (exact table, row selector \
from the scenario, deciding columns) so the pipeline reads it and re-asks you on fact.
Set data_dependent=true WHENEVER your ruling rests on an unread stored value; false only \
when it follows purely from the code logic plus seed-stated facts.

[Output contract] Output ONLY this JSON object. No prose outside it.
{{
  "verdict": "consistent|contradicted|undecidable",
  "data_dependent": false,
  "why": "<one line: the wrong (or correct) behaviour at this locus>",
  "trace": "<what this locus's code outputs under the data state, and whether it reproduces the symptom>",
  "data_reads": [ {{ "id": "<short name, optional>", "table": "<table>", "where": {{ "<col>": "<literal OR {{\\"from\\": \\"<prior id>\\", \\"column\\": \\"<col>\\"}}>" }}, "columns": ["<deciding column(s)>"] }} ]
}}
"""


def _eval_locus_once(seed_text: str, focal: dict[str, Any],
                     others: list[dict[str, Any]], windows: list[dict[str, Any]],
                     focal_code: str, provider: str, model: str, pk: dict[str, Any],
                     ledger, timeout: int, data_state_block: str,
                     db_available: bool, db_schema: str) -> dict[str, Any] | None:
    """One narrow per-locus model call (with a JSON-only reparse). Never raises.

    Returns the parsed object (verdict / data_dependent / why / trace / data_reads) via
    :func:`_coerce_causal` plus the raw ``why``, or None when nothing parseable came back.
    """
    prompt = build_locus_prompt(seed_text, focal, others, windows, focal_code,
                                data_state_block, db_available, db_schema)
    attempt_prompt = prompt
    parsed: dict[str, Any] | None = None
    for attempt in range(2):
        call_id = ledger.begin_call("converge", "converge-locus", provider, model,
                                    attempt_prompt) if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=None,
                             timeout=timeout, **pk)
        except Exception as e:
            logger.warning("converge: locus worker failed: %s", e)
            if ledger is not None:
                ledger.finish_call(call_id, output="", latency_s=0.0, ok=False,
                                   err=str(e)[:200])
            return None
        if ledger is not None:
            ledger.finish_call(call_id, output=wr.stdout, latency_s=wr.latency_s,
                               ok=wr.exit_code == 0,
                               err=wr.stderr[:200] if wr.exit_code != 0 else "",
                               real_tokens=wr.real_tokens)
        try:
            parsed = extract_first_json(wr.stdout)
            break
        except ValueError:
            if attempt == 0:
                attempt_prompt = prompt + _JSON_ONLY_REMINDER
            else:
                logger.warning("converge: locus — no parseable JSON after retry")
    if not isinstance(parsed, dict):
        return None
    causal = _coerce_causal(parsed) or {"verdict": "unverified"}
    causal["why"] = str(parsed.get("why", "") or "")
    return causal


def _eval_locus(seed_text: str, focal: dict[str, Any], others: list[dict[str, Any]],
                windows: list[dict[str, Any]], code_root: str | None,
                provider: str, model: str, pk: dict[str, Any], ledger, timeout: int,
                db_conn, schema_map: dict[str, list[str]], db_schema: str,
                db_available: bool) -> dict[str, Any]:
    """Evaluate ONE located locus end to end: does its code produce the symptom?

    One narrow model call; if the model says the ruling is data_dependent and names
    ``data_reads`` and a DB is configured, run those reads ONCE (deterministic glue) and
    re-ask the locus on the real rows. Applies the premise-refuted rule INLINE: a
    ``consistent`` data-dependent ruling whose read chain broke (the rows it needs are
    ABSENT) is flipped to ``contradicted`` — the locus rests on rows the live DB proves
    do not exist. Returns a verdict dict; never raises (a failed call → ``unverified``).
    """
    vd = focal.get("verdict") or {}
    tag = f"{vd.get('file', '')}:{vd.get('lines', '')}"
    focal_code = _lift_live_code([focal], code_root)

    causal = _eval_locus_once(seed_text, focal, others, windows, focal_code,
                              provider, model, pk, ledger, timeout, "",
                              db_available, db_schema)
    if causal is None:
        return {"focal": focal, "verdict": "unverified", "data_dependent": False,
                "data_backed": False, "chain_broke": False, "why": "", "trace": "",
                "data_block": "", "attempted": False}

    data_block = ""
    data_backed = False
    chain_broke = False
    data_attempted = False
    reads = causal.get("data_reads") or []
    if db_conn is not None and reads and causal.get("data_dependent"):
        data_attempted = True
        block, backed, broke = _run_data_reads(reads, db_conn, schema_map,
                                               ledger=ledger, axis_id=tag or "converge")
        data_block = block
        data_backed = backed
        chain_broke = broke
        logger.info("converge: locus %s data read → rows=%s chain_broke=%s",
                    tag, backed, broke)
        # Re-ask this locus ONCE on the real rows (bounded — _LOCUS_DATA_ROUNDS).
        causal2 = _eval_locus_once(seed_text, focal, others, windows, focal_code,
                                   provider, model, pk, ledger, timeout, block,
                                   db_available, db_schema)
        if causal2 is not None:
            causal = causal2

    verdict = causal.get("verdict", "unverified")
    data_dependent = bool(causal.get("data_dependent"))
    # Premise-refuted INLINE: a consistent data-dependent ruling whose read chain broke
    # rests on rows the live DB proves absent → the locus already yields the expected
    # output, so it is NOT the cause. Flip to contradicted (downgrade-only).
    if verdict == "consistent" and data_dependent and chain_broke:
        logger.info("converge: locus %s premise-refuted (consistent but read chain broke) "
                    "→ contradicted", tag)
        verdict = "contradicted"
    # A consistent data-dependent ruling that was NEVER backed by a real read (no rows /
    # not attempted) is an unstamped assumption — not trustworthy as a survivor. Demote it
    # so it cannot win the elimination on a guess (mirrors the holistic data-stamp guard).
    elif verdict == "consistent" and data_dependent and not data_backed:
        logger.info("converge: locus %s consistent but data-dependent and unbacked "
                    "→ undecidable", tag)
        verdict = "undecidable"

    return {"focal": focal, "verdict": verdict, "data_dependent": data_dependent,
            "data_backed": data_backed, "chain_broke": chain_broke,
            "why": str(causal.get("why", "") or ""),
            "trace": str(causal.get("trace", "") or ""),
            "data_block": data_block, "attempted": data_attempted}


def _split_converge(seed_text: str, located: list[dict[str, Any]],
                    windows: list[dict[str, Any]], known: set[str],
                    code_root: str | None, max_loci: int,
                    provider: str, model: str, pk: dict[str, Any], ledger, timeout: int,
                    db_conn, schema_map: dict[str, list[str]], db_schema: str,
                    db_available: bool) -> ConvergeResult | None:
    """Per-locus elimination converge. Returns a ConvergeResult on a CLEAN elimination
    (exactly one located locus survives its cause→symptom check), else None (fall back).

    Sound-or-abstain by construction: when MORE than ``max_loci`` loci located, a subset
    evaluation cannot honestly claim "only one survives", so it abstains (None) and the
    caller runs the holistic path — which for a big set is also CHEAPER than N calls. With
    zero or several survivors it likewise abstains: a single clear winner is the only thing
    it will commit to, so it can never manufacture a worse answer than the holistic path.
    """
    if not (2 <= len(located) <= max_loci):
        logger.info("converge: split abstains — %d located locus(es) outside [2, %d]",
                    len(located), max_loci)
        return None

    results = []
    for i, focal in enumerate(located):
        others = [v for j, v in enumerate(located) if j != i]
        results.append(_eval_locus(seed_text, focal, others, windows, code_root,
                                   provider, model, pk, ledger, timeout, db_conn,
                                   schema_map, db_schema, db_available))

    survivors = [r for r in results if r["verdict"] == "consistent"]
    eliminated = [r for r in results if r["verdict"] != "consistent"]
    logger.info("converge: split evaluated %d loci → %d survivor(s) "
                "(consistent), %d eliminated", len(results), len(survivors),
                len(eliminated))

    if len(survivors) != 1:
        logger.info("converge: split inconclusive (%d survivors) — fall back to holistic",
                    len(survivors))
        return None

    win = survivors[0]
    vd = win["focal"].get("verdict") or {}
    attributed = {
        "node": "other",
        "file": vd.get("file", ""),
        "lines": vd.get("lines", ""),
        "why": win["why"] or _trunc(vd.get("reason", ""), _REASON_CHARS),
    }
    af = attributed["file"]
    if af and not any(_aligns(af, kf) for kf in known):
        attributed["ungrounded"] = True

    # Record HOW each competing locus was eliminated, in the trace — auditable, and it
    # also names the dropped peers so the holistic dropped-peer guard (which runs next on
    # this result) sees they were addressed, not silently dropped.
    elim_notes = []
    for r in eliminated:
        evd = r["focal"].get("verdict") or {}
        elim_notes.append(f"{evd.get('file', '')}:{evd.get('lines', '')}"
                          f" ({r['verdict']}: {_trunc(r['why'] or r['trace'], 120)})")
    trace = (win["trace"] or "")
    if elim_notes:
        trace += " [split elimination] eliminated competing loci: " + "; ".join(elim_notes)

    causal = {
        "verdict": "consistent",
        "data_dependent": win["data_dependent"],
        "data_state_assumptions": [],
        "trace": trace,
        "need_data_state": [],
        "data_reads": [],
    }
    res = ConvergeResult(
        converged=True,
        path=[{"node": attributed["node"], "file": attributed["file"],
               "lines": attributed["lines"], "symbol": vd.get("symbol", "") or ""}],
        attributed_defect=attributed,
        additional_defects=[],
        missing_link=None,
        causal_check=causal,
        summary=(f"converged via split elimination: 1 of {len(results)} located loci "
                 f"survived its cause→symptom check → defect at "
                 f"{attributed['file']}:{attributed['lines']}"),
        raw={"split": True, "survivors": 1, "evaluated": len(results)},
    )
    # Carry the winner's live-DB read (if any) so the honey can paste real rows.
    if win["attempted"]:
        res.data_state_attempted = True
        res.data_state_block = win["data_block"]
        res.data_state_backed = win["data_backed"]
    return res


def run_converge(*, seed_text: str, verdicts: list[dict[str, Any]],
                 bundles: list[dict[str, Any]], provider: str, model: str,
                 code_root: str | None = None, ledger=None,
                 provider_kwargs: dict | None = None, timeout: int = 180,
                 min_located: int = 2, max_calls: int = 2,
                 k: int = 6, max_hops: int = 2, db_conn=None,
                 split_enabled: bool = False, split_max_loci: int = 4,
                 split_provider: str = "", split_model: str = "") -> ConvergeResult:
    """Stitch the per-axis verdicts into one path. Tool-OFF; never raises.

    Budget (mirrors judge's retrieve→re-judge): ONE converge call, plus — ONLY when
    the converger ITSELF reports a named ``missing_link`` and ``code_root`` is given
    — ONE free local follow-up retrieve of the missing symbols/greps and ONE
    re-converge over the enriched evidence. Capped by ``max_calls`` (default 2). The
    second pass fires only on the model's own request, so the common (already-
    converged) case costs a single call. Quality lives in the EVIDENCE, not in the
    model size or a blind retry — so the lever is fetching the missing link, not
    re-asking the same question (see hive-local-retrieval-crux).

    When fewer than ``min_located`` axes located there is nothing to stitch and the
    whole stage is SKIPPED (free, no model spend).
    """
    located = _located(verdicts)
    if len(located) < min_located:
        return ConvergeResult(
            summary=f"skipped: {len(located)} located verdict(s) < min_located={min_located}")

    unlocated = [v for v in verdicts if v not in located]
    windows = _evidence_windows(bundles)
    known = _known_files(verdicts, windows)

    # Live-code grounding (N177): lift the CURRENT source at the located loci so the
    # causal check rules on real code, not on the compacted retrieved snippets. Free,
    # deterministic; empty when no code_root is given (degrades to the snippet-only path).
    code_state_block = _lift_live_code(located, code_root)
    if code_state_block:
        logger.info("converge: live-code grounding lifted %d located locus block(s)",
                    code_state_block.count("--- "))

    # Tool-OFF single-shot (mirrors judge): no file/shell access, decide on the bundle.
    pk = dict(provider_kwargs or {})
    pk.setdefault("available_tools", [])

    db_available = db_conn is not None
    # Introspect the live schema ONCE: feeds BOTH the prompt block (name real objects,
    # NR174) and the pre-execution read guard (reject hallucinated table/columns).
    schema_map = _introspect_schema(db_conn)
    db_schema = _render_schema_block(schema_map)

    # ── Per-locus SPLIT pass (M020, opt-in): ask one narrow cause→symptom question per
    # located locus and combine by deterministic elimination. On a CLEAN elimination
    # (exactly one survivor) it returns the converged result and we skip the holistic
    # stitch entirely (its data loop / missing-link re-pass below no-op for a split result
    # — empty data_reads, converged, no missing_link). On anything ambiguous it returns
    # None and we fall back to the holistic converge unchanged. Precision layer, not a new
    # failure mode (see ConvergeSplitConfig / _split_converge).
    split_res = None
    if split_enabled:
        sprov = split_provider or provider
        smodel = split_model or model
        logger.info("converge: split pass (%s/%s, max_loci=%d) over %d located locus(es)",
                    sprov, smodel, split_max_loci, len(located))
        split_res = _split_converge(seed_text, located, windows, known, code_root,
                                    split_max_loci, sprov, smodel, pk, ledger, timeout,
                                    db_conn, schema_map, db_schema, db_available)

    if split_res is not None:
        res = split_res
    else:
        res = _converge_once(seed_text, located, unlocated, windows, known,
                             provider, model, pk, ledger, timeout,
                             db_available=db_available, db_schema=db_schema,
                             code_state_block=code_state_block)
    logger.info("converge: %s", res.summary)

    # ── Data-state read (N172/N173): the verdict hinges on a STORED row value static
    # evidence can't determine, and the converger named the exact rows in
    # causal_check.data_reads. If a read-only DB connection is configured for this
    # codebase, FETCH those rows (deterministic glue, NOT a model call) and re-converge
    # on FACT. This runs BEFORE the missing-link re-pass on purpose: in N172 the first
    # pass was undecidable AND named a missing link, and the link re-pass rationalised a
    # band-aid to "consistent" on an ASSUMED data state.
    #
    # The gate is "a DB is configured AND the converger named data_reads" — NOT only the
    # undecidable verdict (N173): when told the DB is available (db_available) the
    # converger is instructed to emit data_reads instead of inventing a value, but a
    # flaky model may still claim "consistent" while ALSO listing the reads it relied on.
    # Reading those rows and re-ruling on FACT is strictly better than trusting the
    # claim, so whenever real reads are named we execute them — turning any
    # assumption-backed verdict into a data-backed one. Read-only by construction; any
    # read failure degrades to the static path (the verdict stays as-is, reported honestly).
    data_ruled = False
    data_backed = False
    data_attempted = False
    # Whether the ADOPTED ruling's data-read chain collapsed (a chained read found no
    # upstream rows → the premise rows the verdict rests on are absent). Captured only
    # at the round we actually adopt, so an earlier broken round the model later repaired
    # does not taint the final ruling.
    data_chain_broke = False
    block_parts: list[str] = []
    seen_sigs: set[str] = set()
    pending = (res.causal_check or {}).get("data_reads") or []
    rounds = 0
    while db_conn is not None and pending and rounds < _MAX_DATA_ROUNDS:
        sig = repr(pending)
        if sig in seen_sigs:
            # The model re-asked for the SAME read — no new angle, stop rather than spin.
            logger.info("converge: data read repeated unchanged — ending read loop")
            break
        seen_sigs.add(sig)
        data_attempted = True
        rounds += 1
        block, backed, chain_broke = _run_data_reads(pending, db_conn, schema_map,
                                                     ledger=ledger, axis_id="converge")
        block_parts.append(block)
        data_backed = data_backed or backed
        logger.info("converge: data read round %d → %d row-set(s), rows=%s, chain_broke=%s",
                    rounds, len(pending), backed, chain_broke)
        combined = "\n\n".join(p for p in block_parts if p)
        res2 = _converge_once(seed_text, located, unlocated, windows, known,
                              provider, model, pk, ledger, timeout,
                              data_state_block=combined, db_available=db_available,
                              db_schema=db_schema, code_state_block=code_state_block)
        logger.info("converge: data re-pass %d → %s", rounds, res2.summary)
        v2 = (res2.causal_check or {}).get("verdict")
        if v2 in ("consistent", "contradicted") and backed:
            # Fact-grounded RULING (rows were ACTUALLY read) wins over an assumed
            # verdict — done. Gated on ``backed``: a verdict claimed over an EMPTY
            # read is not fact-grounded (N173 fabrication risk), so it does not count
            # as a ruling — it falls through to the "failed to strengthen" branch.
            res = res2
            data_ruled = True
            data_chain_broke = chain_broke
            break
        # The re-pass could NOT rule on fact: the read came back empty/failed, or the
        # model still cannot decide. Per N174 #2 this is a FAILED CONFIRMATION, not a
        # convergence CANCELLATION — "data unavailable" must never DOWNGRADE the prior
        # static result. So we do NOT overwrite a converged ``res`` with this weaker
        # re-pass, and we never let an UNBACKED re-pass PROMOTE to converged (a
        # "consistent" claimed on rows we could not read is the N173 fiction). We adopt
        # the re-pass ONLY to carry its latest reasoning forward when NEITHER side is a
        # trustworthy convergence — and LOOP if it NAMED a different/narrower read (the
        # agentic "add a condition and read again" step: the first set was TRUNCATED, or
        # the deciding row wasn't in the window). The honey still shows exactly what was
        # read, so a kept-static verdict is reported as "data confirmation attempted but
        # unavailable", never as a fabricated value.
        if not res.converged and not res2.converged:
            res = res2
        pending = (res2.causal_check or {}).get("data_reads") or []
    data_block = "\n\n".join(p for p in block_parts if p)

    # ── Conditional second pass: the principled "more" — fetch the lead the model
    # named, then re-converge ONCE. Not a blind retry and not a bigger model: a
    # converger that already stitched the path is trusted as-is; one that NAMED where
    # to look next is exactly what a free local follow-up can unblock. Two cases emit a
    # ``missing_link`` lead: (1) a missing CODE hop it couldn't resolve, and (2) a
    # data-backed CONTRADICTION — the suspected node provably can't produce the symptom,
    # so the symptom's real home is elsewhere and the model points the hunt there
    # (NR173: don't dead-end at "not here" and demand the reporter pre-localise the fix;
    # the refutation IS the next, better-aimed search). So we re-hunt on a named lead
    # REGARDLESS of data_ruled — a data-ruled contradiction is precisely when we redirect.
    if (not res.converged and res.missing_link
            and code_root and max_calls > 1):
        need_d = res.missing_link.get("need") or {}
        symbols = [str(s) for s in (need_d.get("symbols") or [])]
        greps = [str(g) for g in (need_d.get("greps") or [])]
        if symbols or greps:
            logger.info("converge: missing link %s — fetching %d symbol(s)/%d grep(s), "
                        "re-converging", res.missing_link.get("between"),
                        len(symbols), len(greps))
            need = FollowupNeed(axis_id="CONVERGE", symbols=symbols, greps=greps,
                                file_globs=[str(g) for g in (need_d.get("file_globs") or [])])
            try:
                fu = retrieve_followup(need, code_root, k=k, max_hops=max_hops)
            except Exception as e:  # local retrieve must never crash converge
                logger.warning("converge: follow-up retrieve failed: %s", e)
                fu = None
            if fu:
                extra = list(fu.get("seeds") or []) + list(fu.get("call_chain") or [])
                # Put the freshly-fetched link FIRST so it survives the window cap.
                merged = _dedup_windows(extra + windows)
                known2 = known | {_norm(w.get("file", "")) for w in extra if w.get("file")}
                # Re-lift live code at the merged loci (the follow-up may add new files).
                res2 = _converge_once(seed_text, located, unlocated, merged, known2,
                                      provider, model, pk, ledger, timeout,
                                      data_state_block=data_block,
                                      db_available=db_available, db_schema=db_schema,
                                      code_state_block=_lift_live_code(located, code_root)
                                      or code_state_block)
                logger.info("converge: re-pass → %s", res2.summary)
                # Adopt the re-pass only if it actually converged; otherwise keep the
                # first result, which at least named the missing link for the author.
                if res2.converged:
                    res = res2

    # When the split pass produced this result, the holistic data loop above no-op'd, so
    # the local data_* vars are still their False defaults. Reflect the WINNER's actual
    # read state (captured per-locus inside the split) so the trailing guards see the truth
    # — in particular the data-stamp guard must NOT demote a data-dependent split verdict
    # that WAS backed by a real read. The premise-refutation was already applied per-locus,
    # so data_chain_broke stays False (do not double-fire the premise-refuted guard).
    if split_res is not None:
        data_backed = res.data_state_backed
        data_attempted = res.data_state_attempted
        data_block = res.data_state_block
        data_chain_broke = False

    # ── Dropped-peer domain guard (N180): a ``consistent`` certified on a live DB read
    # must not stay converged while a DISTINCT-locus located peer was dropped without
    # refutation — a data read cannot vouch for a render/binding-layer cause. Runs on the
    # FINAL result (after any data-read / missing-link re-pass); demotes to not-converged
    # and stamps the peer so the honey routes it to reinvestigation. Tight by design.
    res = _dropped_peer_guard(res, located, data_backed)

    # ── Premise-refuted gate (M035): a data-dependent ``consistent`` whose adopted read
    # chain BROKE (a chained read had no upstream rows) rests on rows the live DB proves
    # absent — the suspected query already yields the expected result, so the real cause is
    # elsewhere (render/alternate path). Demotes to not-converged. No-op unless the chain
    # actually collapsed; closes the M017 gap where an incidental id-lookup satisfied
    # ``data_backed`` while every deciding read was skipped. Tight by design.
    res = _premise_refuted_guard(res, db_available, data_chain_broke)

    # ── Data-stamp gate (M017 lever 2): a ``consistent`` verdict the converger flagged
    # ``data_dependent`` must be BACKED by a real DB read, never ruled on an assumed stored
    # value. Runs on the FINAL result; demotes an unstamped data-dependent consistent to
    # not-converged and marks ``data_unstamped`` so the honey routes it back to read the
    # deciding row. No-op when no DB is configured, when the ruling is pure code-logic
    # (not data_dependent), or when a real read already backed it. Tight by design.
    res = _data_stamp_guard(res, db_available, data_backed)

    # Carry the live-DB read onto whichever result we return so the honey can PASTE the
    # real rows (or honestly report that the read was attempted but returned nothing).
    if data_attempted:
        res.data_state_attempted = True
        res.data_state_block = data_block
        res.data_state_backed = data_backed
    return res
