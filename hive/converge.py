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
from dataclasses import dataclass, field
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.retriever import FollowupNeed, retrieve_followup

logger = logging.getLogger("hive.converge")

# Prompt-budget guards — mirror judge: the evidence is rendered COMPACT, not raw.
_MAX_EVIDENCE = 24          # distinct (file,lines) windows shown to the converger
_EVIDENCE_CHARS = 500       # per-window char cap
_REASON_CHARS = 300         # per-verdict reason cap

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
    missing_link: dict[str, Any] | None = None
    causal_check: dict[str, Any] | None = None
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "path": self.path,
            "attributed_defect": self.attributed_defect,
            "missing_link": self.missing_link,
            "causal_check": self.causal_check,
            "summary": self.summary,
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


def build_converge_prompt(seed_text: str, located: list[dict[str, Any]],
                          unlocated: list[dict[str, Any]],
                          windows: list[dict[str, Any]],
                          data_state_block: str = "") -> str:
    """Build the single converge prompt: fragments + evidence → one path + one node.

    ``data_state_block`` is the optional ``[Confirmed data state]`` section: on the
    re-pass after an ``undecidable`` first pass, the glue has READ the exact rows the
    converger asked for from the live DB and renders them here as AUTHORITATIVE fact.
    The model is told to rule on these real values, not on assumed ones — turning the
    undecidable causal check into a decidable (consistent / contradicted) one.
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
            "undecidable for a field shown here.]\n" + data_state_block.strip() + "\n")

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

[Reported scenario / seed]
{_trunc(seed_text, 2000)}
{confirmed_block}
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
decides the verdict. Use the EXACT table/column names visible in the evidence (the SQL \
in queries.json names them). Do NOT guess an attribution to fill the gap.
4. If — and only if — two adjacent nodes cannot be connected because a needed \
callee/symbol is NOT shown in the evidence, set converged=false and NAME the missing \
link instead of guessing.

[Gate] Only a "consistent" causal check is actionable downstream. When your check is \
"contradicted" or "undecidable", STILL fill attributed_defect with the node you \
suspected and causal_check with your honest reasoning — the pipeline routes it to \
reinvestigation / data-state confirmation, NOT to an edit. Do not suppress the \
finding to force a convergence.

[Output contract] Output ONLY this JSON object. No prose outside the JSON.
{{
  "converged": true,
  "path": [
    {{ "node": "endpoint|handler|db_fn|sql_key|fe|other", "file": "<repo-relative>", "lines": "<start-end>", "symbol": "<fn/route/key name>" }}
  ],
  "attributed_defect": {{ "node": "<which node above>", "file": "<repo-relative>", "lines": "<start-end>", "why": "<one line: the wrong behaviour here>" }},
  "causal_check": {{ "verdict": "consistent|contradicted|undecidable", "data_state_assumptions": ["<the row/field values the scenario forces>"], "trace": "<what the attributed code outputs under those assumptions, and whether it reproduces the symptom>", "need_data_state": ["<when undecidable: the exact stored row state / fixture to confirm>"], "data_reads": [ {{ "table": "<table name from the evidence>", "where": {{ "<key column>": "<row selector value from the scenario>" }}, "columns": ["<column(s) whose value decides the verdict>"] }} ] }},
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
        "data_state_assumptions": [str(x) for x in (d.get("data_state_assumptions") or [])],
        "trace": str(d.get("trace", "") or ""),
        "need_data_state": [str(x) for x in (d.get("need_data_state") or [])],
        "data_reads": _coerce_data_reads(d.get("data_reads")),
    }


def _coerce_data_reads(raw: Any) -> list[dict[str, Any]]:
    """Parse the converger's machine-readable ``data_reads`` into clean read specs.

    Each spec is ``{table, where: {col: val}, columns: [..]}`` — the structured form
    the dbread glue turns into one mechanical SELECT. Lenient: drops anything without a
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
            causal = {"verdict": "unverified", "data_state_assumptions": [],
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
                          attributed_defect=attributed, missing_link=missing,
                          causal_check=causal, summary=summary, raw=parsed)


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
                   ledger, timeout: int, data_state_block: str = "") -> ConvergeResult:
    """One logical converge call (with a transport-level JSON-only reparse). Never raises.

    The reparse retry handles a model that wrapped the JSON in prose — it is a
    transport reparse (both attempts recorded to the ledger, like judge) and does
    NOT count against the missing-link budget the caller manages. ``data_state_block``,
    when given, injects the live-DB-confirmed rows so the re-pass rules on fact.
    """
    prompt = build_converge_prompt(seed_text, located, unlocated, windows, data_state_block)
    attempt_prompt = prompt
    parsed: dict[str, Any] | None = None
    for attempt in range(2):
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=None,
                             timeout=timeout, **pk)
        except Exception as e:  # timeout / provider error — not retried
            logger.warning("converge: worker failed: %s", e)
            return ConvergeResult(summary=f"converge worker failed: {e}")

        if ledger is not None:
            ledger.record_call("converge", "converge", provider, model,
                               prompt=attempt_prompt, output=wr.stdout,
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


def _fetch_data_state(data_reads: list[dict[str, Any]], db_conn) -> str:
    """Run the converger's ``data_reads`` against the live DB; render rows as fact lines.

    Deterministic glue — NOT a model call. Each spec becomes ONE read-only SELECT via
    hive.dbread (read-only by construction, identifier-validated). Any failure (no
    driver, bad identifier, connect error) is logged and skipped, never raised: a
    failed read just means that fact stays unconfirmed and the converge degrades to its
    static result. Returns "" when nothing could be read.
    """
    try:
        from hive.dbread import read_rows, DbReadError
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("converge: dbread unavailable (%s) — skipping data read", e)
        return ""
    lines: list[str] = []
    for spec in data_reads:
        table = spec.get("table", "")
        try:
            rr = read_rows(db_conn, table, columns=spec.get("columns") or None,
                           where=spec.get("where") or None, limit=20)
        except DbReadError as e:
            logger.warning("converge: data read on %r failed — skipping: %s", table, e)
            continue
        lines.append(f"- query: {rr.sql}")
        if not rr.rows:
            lines.append("  -> (no rows matched)")
        for row in rr.rows:
            lines.append("  -> " + ", ".join(f"{k}={v!r}" for k, v in row.items()))
    return "\n".join(lines)


def run_converge(*, seed_text: str, verdicts: list[dict[str, Any]],
                 bundles: list[dict[str, Any]], provider: str, model: str,
                 code_root: str | None = None, ledger=None,
                 provider_kwargs: dict | None = None, timeout: int = 180,
                 min_located: int = 2, max_calls: int = 2,
                 k: int = 6, max_hops: int = 2, db_conn=None) -> ConvergeResult:
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

    # Tool-OFF single-shot (mirrors judge): no file/shell access, decide on the bundle.
    pk = dict(provider_kwargs or {})
    pk.setdefault("available_tools", [])

    res = _converge_once(seed_text, located, unlocated, windows, known,
                         provider, model, pk, ledger, timeout)
    logger.info("converge: %s", res.summary)

    # ── Data-state read (N172): the converger came back UNDECIDABLE because the
    # verdict hinges on a STORED row value static evidence can't determine — and it
    # named the exact rows in causal_check.data_reads. If a read-only DB connection is
    # configured for this codebase, FETCH those rows (deterministic glue, NOT a model
    # call) and re-converge on FACT. This runs BEFORE the missing-link re-pass on
    # purpose: in N172 the first pass was undecidable AND named a missing link, and the
    # link re-pass rationalised a band-aid to "consistent" on an ASSUMED data state.
    # Ruling on the real rows instead either confirms the cause (consistent) or refutes
    # it (contradicted) — both beat the guess, and a contradiction correctly stops the
    # band-aid. Read-only by construction; any read failure degrades to the static path.
    data_ruled = False
    cc = res.causal_check or {}
    if (not res.converged and db_conn is not None
            and cc.get("verdict") == "undecidable" and cc.get("data_reads")):
        block = _fetch_data_state(cc.get("data_reads") or [], db_conn)
        if block:
            logger.info("converge: undecidable → read live DB for %d row-set(s), "
                        "re-converging on fact", len(cc.get("data_reads") or []))
            res2 = _converge_once(seed_text, located, unlocated, windows, known,
                                  provider, model, pk, ledger, timeout,
                                  data_state_block=block)
            logger.info("converge: data re-pass → %s", res2.summary)
            v2 = (res2.causal_check or {}).get("verdict")
            # Adopt the fact-grounded re-pass on any real RULING (consistent OR
            # contradicted) — fact wins over the undecidable guess either way. A
            # still-undecidable re-pass (the read didn't cover the deciding field)
            # keeps the original and lets the missing-link path try.
            if v2 in ("consistent", "contradicted"):
                res = res2
                data_ruled = True

    # ── Conditional second pass: the principled "more" — fetch the link the model
    # said it lacked, then re-converge ONCE. Not a blind retry and not a bigger
    # model: a converger that already stitched the path is trusted as-is; one that
    # NAMED a missing hop is exactly what a free local follow-up can unblock.
    if (not res.converged and not data_ruled and res.missing_link
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
                res2 = _converge_once(seed_text, located, unlocated, merged, known2,
                                      provider, model, pk, ledger, timeout)
                logger.info("converge: re-pass → %s", res2.summary)
                # Adopt the re-pass only if it actually converged; otherwise keep the
                # first result, which at least named the missing link for the author.
                if res2.converged:
                    res = res2
    return res
