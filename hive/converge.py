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

# ── Adversarial LENS refutation (swarm best-of-N at converge) ─────────────────────
# converge's causal_check is best-of-1: ONE cheap single-shot rules consistent/contradicted
# over compacted evidence. It WOBBLES at depth (right call chain, wrong node — store.py vs
# process_service) and is blind to OMISSION/SHADOW it was never shown (a dead/legacy route a
# sibling overrides, a field the fix leaves missing). judge already de-risks its own noise
# with best-of-N UNION voting; converge had no equivalent. This is it — pointed the OTHER
# way: when converge ships an ACTIONABLE ``consistent`` attribution, N INDEPENDENT refuters
# (the cheap swarm tier, e.g. gpt-oss-120b), EACH through a DISTINCT lens, try to REFUTE it.
# A majority refutation DEMOTES converged→False so a wobbly/omission/shadow attribution never
# ships as a fix — it routes to reinvestigation instead. Runs ONCE on the FINAL adopted
# attribution (NOT the data-read / missing-link inner re-passes), so the added cost is
# EXACTLY ``len(lenses)`` swarm calls per actionable converge, and ZERO when the lens set is
# empty (opt-in) or the verdict was not an actionable ``consistent``. Kill: HIVE_NO_LENS_REFUTE.
_LENS_MAX_EVIDENCE = 12        # evidence windows shown to a refuter (tighter than converge's 24)

# The distinct refutation angles. Each maps to one confirmed converge failure class, so a
# panel of all three covers shadow + omission + wobble. A lens NOT in this map still runs
# with a generic "refute on <name> grounds" instruction (config may add bespoke lenses).
_LENS_DEFINITIONS: dict[str, str] = {
    "datasource-liveness": (
        "Is the attributed locus actually ON the LIVE executed path for THIS scenario, or is "
        "it shadowed / dead / legacy code that the real live route overrides? If a DIFFERENT "
        "module, registration, or handler serves this request FIRST (so this locus never runs "
        "for the reported case), the attribution is REFUTED — name the live path that wins."),
    "omission": (
        "Would correcting ONLY this locus FULLY remove the symptom, or is something that "
        "SHOULD exist still MISSING elsewhere on the path — an absent field/key, branch, "
        "registration, or handler the edit here does not add? A defect of OMISSION cannot be "
        "fixed by changing a node that is present; if the real gap is a missing element the "
        "attributed edit would not create, REFUTE and say what is absent."),
    "reproduction": (
        "Under the CONCRETE data/scenario state the seed forces, does the attributed code "
        "ACTUALLY produce the reported symptom, and would correcting it remove the symptom — "
        "by the mechanism visible in the live code, NOT a paraphrase of the symptom? If the "
        "code already yields the EXPECTED output under the only state the scenario allows "
        "(e.g. the rows tie so a later branch never decides), the cause contradicts the "
        "symptom — REFUTE."),
}


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
    # Deterministic route -> response-producer proof carried from retriever evidence.
    # Unlike ``path`` (model-authored), this is free local grounding and is safe for
    # specify's on-path gate to consume.
    winning_path: list[dict[str, Any]] = field(default_factory=list)
    # Adversarial LENS refutation panel result (swarm best-of-N). Empty when the panel
    # did not run (disabled, or the verdict was not an actionable ``consistent``). When it
    # DID run it records ``{lenses, votes, refuted_votes, of, threshold, verdict}``; a
    # ``verdict == "refuted"`` means a majority of the distinct lenses broke the attribution
    # and ``converged`` was demoted to False (routed to reinvestigation, not an edit).
    lens_check: dict[str, Any] = field(default_factory=dict)

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
            "winning_path": self.winning_path,
            "lens_check": self.lens_check,
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


_FE_EXTS = (".vue", ".jsx", ".tsx", ".svelte", ".ts", ".js", ".mjs", ".cjs")


def _is_fe_file(path: str) -> bool:
    return _norm(path).endswith(_FE_EXTS)


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


def _winning_http_path_nodes(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ordered, deterministic winning HTTP path nodes from bundle evidence."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for bundle in bundles or []:
        if not isinstance(bundle, dict):
            continue
        seq = (bundle.get("code_snippets") or []) + (bundle.get("call_chain") or [])
        for window in seq:
            if not isinstance(window, dict):
                continue
            via = window.get("via")
            if via not in ("http-binding", "http-producer"):
                continue
            if window.get("ambiguous") or not window.get("winning", True):
                continue
            url = str(window.get("url", "") or "").rstrip("/")
            if not url:
                continue
            grouped.setdefault(url, []).append(window)

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for url in sorted(grouped):
        windows = sorted(
            grouped[url],
            key=lambda w: (
                0 if w.get("via") == "http-binding" else 1,
                int(w.get("path_depth", 0) or 0),
                _norm(w.get("file", "")),
            ),
        )
        for window in windows:
            if window.get("via") == "http-binding":
                for client_file in window.get("client_files") or []:
                    client_node = {
                        "url": url,
                        "verb": str(window.get("verb", "") or "").upper(),
                        "role": "client",
                        "file": str(client_file),
                        "lines": "",
                        "symbol": "",
                        "depth": -1,
                    }
                    client_key = (url, _norm(client_node["file"]), "", "client")
                    if client_node["file"] and client_key not in seen:
                        seen.add(client_key)
                        out.append(client_node)
            role = "handler" if window.get("via") == "http-binding" else (
                "producer" if window.get("producer") else "response-call")
            node = {
                "url": url,
                "verb": str(window.get("verb", "") or "").upper(),
                "role": role,
                "file": str(window.get("file", "") or ""),
                "lines": str(window.get("lines", "") or ""),
                "symbol": str(window.get("symbol", "") or ""),
                "depth": int(window.get("path_depth", 0) or 0),
            }
            key = (url, _norm(node["file"]), node["lines"], role)
            if not node["file"] or key in seen:
                continue
            seen.add(key)
            out.append(node)
    return out


def _winning_producer_loci(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift the deepest proven response producer per URL into converge's located set."""
    out: list[dict[str, Any]] = []
    by_url: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_url.setdefault(str(node.get("url", "")), []).append(node)
    for url, path in by_url.items():
        producers = [node for node in path if node.get("role") == "producer"]
        candidates = producers or [
            node for node in path if node.get("role") == "response-call"
        ]
        if not candidates:
            continue
        node = max(candidates, key=lambda item: int(item.get("depth", 0) or 0))
        out.append({
            "axis_id": f"HTTP_WINNING_PATH:{url}",
            "title": f"winning HTTP response producer for {url}",
            "verdict": {
                "located": True,
                "file": node.get("file", ""),
                "lines": node.get("lines", ""),
                "reason": (
                    f"Deterministic winning request-path grounding: {url} reaches "
                    f"this response producer through the first registered handler."
                ),
                "symbol": node.get("symbol", ""),
                "via": "http-winning-path",
            },
        })
    return out


# ── Omission nominator: winning-path gap (① negative-space probe) ────────────────
# converge attributes among the LOCATED fragments and reasons over the UNION of what the
# per-axis judges retrieved. It therefore cannot, by construction, attribute to a node NO
# axis located — the located set can never reveal what is ABSENT from it. The deterministic
# ``winning_path`` (route→producer proof) is the should-exist set: any winning-path node whose
# file no located/known fragment covers is a live hop converge never saw — the omission/
# coverage gap. Diffing winning_path AGAINST located names that uncovered node, which the
# existing free scoped re-retrieve then fetches so a re-converge can reach it. Free,
# deterministic, never raises. (The deepest winning producer is already LIFTED into located
# upstream, so in practice this surfaces uncovered HANDLER / response-call hops — exactly the
# nodes the located-only stitch is blind to.) Kill via HIVE_NO_OMISSION_LEAD at the call site.
_OMISSION_ROLE_RANK = {"producer": 0, "response-call": 1, "handler": 2}


def _winning_path_omission(winning_path: list[dict[str, Any]],
                           located: list[dict[str, Any]],
                           known: set[str]) -> dict[str, Any] | None:
    """Name the deepest winning-path node uncovered by any located/known fragment.

    Returns a ``missing_link``-shaped dict ``{between, need:{symbols, greps, file_globs}}``
    scoping a re-retrieve to that node, or None when the winning path is empty or every
    node is already covered. Pure, deterministic, never raises.
    """
    if not winning_path:
        return None
    covered: set[str] = set()
    for v in located or []:
        f = _norm((v.get("verdict") or {}).get("file", ""))
        if f:
            covered.add(f)
    for k in known or set():
        nk = _norm(k)
        if nk:
            covered.add(nk)

    cands: list[dict[str, Any]] = []
    for node in winning_path:
        role = node.get("role", "")
        if role not in _OMISSION_ROLE_RANK:   # skip client nodes — not a server-side cause site
            continue
        nf = _norm(node.get("file", ""))
        if not nf or any(_aligns(nf, c) for c in covered):
            continue
        cands.append(node)
    if not cands:
        return None
    node = sorted(cands, key=lambda n: (_OMISSION_ROLE_RANK.get(n.get("role", ""), 9),
                                        -int(n.get("depth", 0) or 0)))[0]
    sym = str(node.get("symbol", "") or "").strip()
    url = str(node.get("url", "") or "")
    seg = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    greps = [g for g in (sym, seg) if g]
    return {
        "between": ["winning-path", str(node.get("role", "node") or "node")],
        "need": {
            "symbols": [sym] if sym else [],
            "greps": greps,
            "file_globs": [str(node.get("file", "") or "")],
        },
    }


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


# ── FE→BE HTTP-edge grounding (N183) ────────────────────────────────────────────
# The per-axis judges localise the two ends of an HTTP round-trip in DIFFERENT axes —
# a front-end response-mapping (``NewRequirementModal.vue`` maps ``resp.projects``) and
# the back-end getter that serves it (``process_service.get_projects_with_modules``) —
# but the edge BETWEEN them is an HTTP request, NOT a function call, so it appears in no
# call-chain. With no call edge to follow, the holistic stitch reports a ``missing_link``
# (FE mapping ↔ BE getter) and returns 0 edits — exactly the N183 dead-end. The fix is the
# same family as the retriever's binding resolver: deterministically match the FE fetch-URL
# literal to the BE route whose path it hits, and HAND converge that resolved edge as FACT
# so it can stitch across the boundary instead of declaring the link missing. Pure literal
# harvest (only real URLs / routes from real source), fail-open (no match → nothing added,
# converge behaves exactly as before), zero model cost — never SYNTHESISES a join.
_HTTP_BRIDGE_MAX = 8           # max resolved edges rendered into the prompt
_HTTP_BRIDGE_MAX_FILES = 16    # max in-scope files read for URL literals


def _http_binding_bridges(located: list[dict[str, Any]],
                          windows: list[dict[str, Any]],
                          code_root: str | None) -> str:
    """Resolve FE fetch-URL literals to their BE route handlers across the pooled
    fragments → rendered ``FE client ↔ BE route`` edge lines (or "" when none/no root).

    Reuses the retriever's deterministic binding machinery on the union of the evidence
    windows + the located fragments' in-scope files. Never raises; a resolver/import
    failure degrades silently to "" (the snippet-only path, exactly as before).

    Set ``HIVE_NO_HTTP_BRIDGE=1`` to disable (A/B isolation / kill-switch) — the stage
    then behaves exactly as it did before this grounding was added.
    """
    if not code_root or os.environ.get("HIVE_NO_HTTP_BRIDGE"):
        return ""
    try:
        from hive.retriever import _read_text, _resolve_http_bindings
    except Exception:  # pragma: no cover - import guard
        return ""
    # Distinct in-scope files (located loci + evidence windows). Read each file's FULL
    # text once so a fetch-URL literal is seen regardless of which window happened to
    # capture it — the keyword windows routinely fall in the GAP around the fetch line
    # (N183), and a located fragment that DOES cover it carries no text on a verdict stub.
    files: list[str] = []
    for v in located or []:
        rel = ((v.get("verdict") or {}).get("file") or "").replace("\\", "/")
        if rel and rel not in files:
            files.append(rel)
    for w in windows or []:
        rel = (w.get("file") or "").replace("\\", "/")
        if rel and rel not in files:
            files.append(rel)
    if not files:
        return ""
    pool: list[dict[str, Any]] = []
    for rel in files[:_HTTP_BRIDGE_MAX_FILES]:
        text = _read_text(code_root, rel)
        if text:
            pool.append({"file": rel, "lines": "", "text": text})
    if not pool:
        return ""
    try:
        bindings = _resolve_http_bindings(pool, code_root)
    except Exception as e:  # pragma: no cover - defensive; grounding never blocks
        logger.warning("converge: http-binding bridge failed: %s", e)
        return ""
    if not bindings:
        return ""
    lines: list[str] = []
    for b in bindings[:_HTTP_BRIDGE_MAX]:
        callees = ", ".join(b.get("callees", []))
        amb = " (AMBIGUOUS — several handlers serve this path; treat each as a candidate)" \
            if b.get("ambiguous") else ""
        lines.append(
            f"- FE client `{b['url']}`"
            + (f" (callees: {callees})" if callees else "")
            + f" → BE {b['verb'].upper()} {b['full_path']} at {b['file']}:{b['lines']}{amb}")
    return "\n".join(lines)


def _coerce_refuted_peers(raw: Any) -> list[dict[str, str]]:
    """Normalize explicit peer refutations; malformed entries are ignored."""
    out: list[dict[str, str]] = []
    for peer in raw if isinstance(raw, list) else []:
        if not isinstance(peer, dict):
            continue
        out.append({
            "file": str(peer.get("file", "") or ""),
            "lines": str(peer.get("lines", "") or ""),
            "why_not": str(peer.get("why_not", "") or ""),
        })
    return out


def _fragment_fact_cards(located: list[dict[str, Any]],
                         windows: list[dict[str, Any]],
                         bundles: list[dict[str, Any]],
                         code_root: str | None) -> str:
    """Render compact deterministic provenance cards for each located fragment.

    The cards expose structure the pipeline already computed: field producers, resolved
    HTTP bindings, call-chain reachability, and whether the cited locus was lifted from
    live source. Malformed evidence is skipped and the builder never raises.
    """
    if not isinstance(located, list) or not located:
        return ""
    try:
        all_windows: list[dict[str, Any]] = [
            w for w in (windows if isinstance(windows, list) else [])
            if isinstance(w, dict)
        ]
        chains: list[list[dict[str, Any]]] = []
        for bundle in bundles if isinstance(bundles, list) else []:
            if not isinstance(bundle, dict):
                continue
            seq = [
                w for w in ((bundle.get("code_snippets") or [])
                            + (bundle.get("call_chain") or []))
                if isinstance(w, dict)
            ]
            all_windows.extend(seq)
            if seq:
                chains.append(seq)

        cards: list[str] = []
        for v in located:
            if not isinstance(v, dict):
                continue
            vd = v.get("verdict") if isinstance(v.get("verdict"), dict) else {}
            file = str(vd.get("file", "") or "")
            lines = str(vd.get("lines", "") or "")
            if not file:
                continue
            nf = _norm(file)
            fields: set[str] = set()
            routes: set[str] = set()
            reachable: set[str] = set()

            for w in all_windows:
                wf = _norm(w.get("file", ""))
                if not _aligns(nf, wf):
                    continue
                if w.get("via") == "field-producer":
                    field_name = str(w.get("field", "") or "").strip()
                    if field_name:
                        fields.add(field_name)
                if w.get("via") == "http-binding":
                    text = str(w.get("text", "") or "")
                    route = str(w.get("full_path", "") or w.get("url", "") or "").strip()
                    if not route:
                        m = re.search(r"(?i)\b(?:GET|POST|PUT|PATCH|DELETE)\s+(/[^\s<]+)", text)
                        route = m.group(1) if m else ""
                    if route:
                        routes.add(route.rstrip("/"))

            for seq in chains:
                for i, w in enumerate(seq):
                    if not _aligns(nf, w.get("file", "")):
                        continue
                    if i > 0:
                        prev = seq[i - 1]
                        label = str(prev.get("symbol", "") or "").strip()
                        prev_file = str(prev.get("file", "") or "").strip()
                        if prev_file:
                            reachable.add(f"{prev_file}"
                                          + (f" {label}" if label else ""))
                    elif w.get("via") == "call-chain":
                        label = str(w.get("symbol", "") or "").strip()
                        reachable.add(label or "call-chain evidence")

            live = bool(_lift_live_code([v], code_root))
            cards.extend([
                f"- axis {v.get('axis_id', '?')} / {file}:{lines}",
                "    produces FE-bound field(s): "
                + (", ".join(sorted(fields)) if fields else "(none)"),
                "    bound to HTTP route(s):     "
                + (", ".join(sorted(routes)) if routes else "(none)"),
                "    reachable from:             "
                + (", ".join(sorted(reachable))
                   if reachable else "(not linked to any other located fragment)"),
                f"    live-code confirmed:        {'yes' if live else 'no'}",
            ])
        if not cards:
            return ""
        return (
            "[Fragment facts — deterministic annotations computed by the pipeline; "
            "treat as FACT]\n" + "\n".join(cards))
    except Exception as e:  # deterministic prompt scaffolding must never block converge
        logger.warning("converge: fragment fact-card build failed: %s", e)
        return ""


def build_converge_prompt(seed_text: str, located: list[dict[str, Any]],
                          unlocated: list[dict[str, Any]],
                          windows: list[dict[str, Any]],
                          data_state_block: str = "",
                          db_available: bool = False,
                          db_schema: str = "",
                          code_state_block: str = "",
                          http_binding_block: str = "",
                          fragment_fact_block: str = "",
                          refuted_block: str = "") -> str:
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
    any_design_change = False
    for v in located:
        vd = v.get("verdict") or {}
        dc = str(vd.get("type", "")).strip().lower() == "design_change"
        any_design_change = any_design_change or dc
        tag = " [DESIGN-CHANGE site]" if dc else ""
        frag_lines.append(
            f"- axis {v.get('axis_id', '?')}: {vd.get('file', '')}:{vd.get('lines', '')}{tag} "
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

    # FE→BE HTTP edges resolved deterministically from the source (N183): the request
    # boundary that no call-chain hop spans. Handing converge these as FACT lets it stitch
    # an FE response-mapping to the BE getter that serves it instead of reporting the pair
    # as a missing_link. Authoritative: a pair bridged here is NOT a missing link.
    http_bindings = ""
    if http_binding_block.strip():
        http_bindings = (
            "\n[HTTP request edges — REAL FE→BE bindings resolved from the source (a FE "
            "fetch-URL literal matched to the BE route that serves that path). These are "
            "FACT, not inference: each line is an EDGE on the executed path that crosses the "
            "client→server boundary via an HTTP request (which appears in NO call-chain hop, "
            "so you would otherwise miss it). USE them to ORDER fragments across the FE/BE "
            "boundary: a front-end mapping and the back-end getter joined by an edge here are "
            "ON THE SAME PATH. Do NOT emit a missing_link for a FE↔BE pair already bridged "
            "below — the link is established; stitch it.]\n"
            + http_binding_block.strip() + "\n")

    fragment_facts = ""
    if fragment_fact_block.strip():
        fragment_facts = (
            "\n" + fragment_fact_block.strip() + "\n"
            "Use these annotations to compare candidates: a fragment that produces the "
            "FE-bound symptom field, or that is reachable on the executed path, outranks "
            "a lexically-similar fragment that produces nothing and links to nothing.\n")

    # Refuted-node exclusion (M035): a PRIOR pass attributed the defect to a node whose
    # cause→symptom check came back ``contradicted`` — it provably cannot produce the
    # symptom — but the model did NOT name where to look next (no missing_link). The
    # symptom still has a home; this is the deterministic redirect the prompt's [Keep
    # hunting] asks for, made explicit. The refuted node has been REMOVED from the
    # located fragments above; this block names it so the converger does not re-attribute
    # there. It must attribute to a DIFFERENT remaining fragment that CAN produce the
    # symptom (e.g. the front-end render/binding locus a sibling axis localised — a
    # render symptom can never be produced by the contradicted backend node), or, if none
    # can, emit a missing_link naming the producing path — never re-crown a refuted node.
    refuted_excl = ""
    if refuted_block.strip():
        refuted_excl = (
            "\n[REFUTED nodes — EXCLUDED candidates. A prior cause→symptom check PROVED "
            "each of these cannot produce the reported symptom. They are NOT in the "
            "located-fragment list above and you MUST NOT attribute the defect to any of "
            "them. Re-stitch the executed path and attribute to a DIFFERENT remaining "
            "fragment that CAN produce the symptom. A render/shape/binding symptom in "
            "particular cannot originate in a contradicted backend/query node — prefer "
            "the front-end render or response-binding locus. If NO remaining fragment can "
            "produce it, emit a missing_link naming where the producing path lives; do "
            "NOT re-attribute to a refuted node below.]\n"
            + refuted_block.strip() + "\n")

    # Design-change carve-out (M037): when a judge tagged a fragment [DESIGN-CHANGE site],
    # the converger must not refute it with a spec-conformance argument ("the code matches
    # its own design, so nothing is wrong") — that is the exact T905 FE-visual-axis miss.
    # The reporter's declared expectation is ground truth, so a faithful-to-design site CAN
    # be the node a change lands on; it is checked the SAME way (does it EMIT the rejected
    # behaviour?), not against its own spec.
    design_change_note = ""
    if any_design_change:
        design_change_note = (
            "\n[DESIGN-CHANGE sites among the fragments] One or more located fragments are "
            "tagged [DESIGN-CHANGE site]: a judge ruled the code there FAITHFULLY implements "
            "its own design/spec, yet the reporter declared the resulting behaviour wrong or "
            "unwanted, so the SITE still must change. Treat such a fragment as a LEGITIMATE "
            "attribution target — do NOT refute it merely because 'the code matches its own "
            "design definition'. A design-change site produces the rejected on-screen result "
            "BY DESIGN; the reporter's stated expectation is ground truth, so that site CAN be "
            "the node a change must land on. Apply the SAME cause→symptom check (does this "
            "site EMIT the rejected behaviour?), never a spec-conformance check.\n")

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
{design_change_note}
[Reported scenario / seed]
{_trunc(seed_text, 2000)}
{confirmed_block}{db_avail_block}{schema_block}{code_state}{http_bindings}
[Located fragments — each is ONE node candidate, from a different axis]
{frags}
{fragment_facts}{refuted_excl}
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
assumptions. For a ``consistent`` verdict, also state a one-line COUNTERFACTUAL: if \
the code AT THE ATTRIBUTED LOCUS were corrected, the reported symptom would disappear \
BECAUSE of the concrete mechanism visible in the live code/evidence — not because the \
seed wishes it so, and not as a paraphrase of the symptom. Then rule:
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
   - REFUTE BEFORE YOU DROP — MANDATORY for EVERY located fragment at a DISTINCT file \
that you leave off the path. You may not drop it silently: either list it in \
``causal_check.refuted_peers`` with a concrete evidence-grounded ``why_not`` explaining \
why it cannot independently produce the symptom (or why it is only a corroborating view \
of the SAME chain), or carry it as an additional INDEPENDENT defect (step 5). This applies \
in every symptom domain, not only data-backed rulings. A data read is especially unable \
to refute a render/binding peer it cannot observe.
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
half-fix. A strong tell for INDEPENDENCE: the broken outputs span DIFFERENT mechanisms / \
layers (a wrong BACK-END value, a wrong FRONT-END colour or visual state, a status/badge \
that never flips) or are produced by DIFFERENT response fields — outputs at different \
layers or carried by different fields CANNOT all be the same call chain, so each is its \
own root even though ONE reporter listed them together. Do NOT assume a multi-symptom \
report reduces to a single shared cause: when several enumerated symptoms each have their \
own producer among the fragments, attribute the primary and carry the rest as independent \
defects rather than refuting them as "the same chain".

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
  "causal_check": {{ "verdict": "consistent|contradicted|undecidable", "data_dependent": false, "data_state_assumptions": ["<the row/field values the scenario forces>"], "trace": "<what the attributed code outputs under those assumptions, and whether it reproduces the symptom>", "counterfactual": "<when consistent: correcting the attributed locus removes the symptom because...>", "refuted_peers": [ {{ "file": "<other located fragment>", "lines": "<start-end>", "why_not": "<why it cannot independently produce the symptom, or is the same chain>" }} ], "need_data_state": ["<when undecidable: the exact stored row state / fixture to confirm>"], "data_reads": [ {{ "id": "<short name for chaining, optional>", "table": "<table name from the evidence>", "where": {{ "<key column>": "<literal row selector OR {{\\"from\\": \\"<prior read id>\\", \\"column\\": \\"<column to carry over>\\"}}>" }}, "columns": ["<column(s) whose value decides the verdict>"] }} ] }},
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


def _coerce_string_list(raw: Any) -> list[str]:
    """Return a string list for model fields that are contractually arrays."""
    return [str(x) for x in raw] if isinstance(raw, list) else []


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
        "data_state_assumptions": _coerce_string_list(d.get("data_state_assumptions")),
        "trace": str(d.get("trace", "") or ""),
        "counterfactual": str(d.get("counterfactual", "") or ""),
        "refuted_peers": _coerce_refuted_peers(d.get("refuted_peers")),
        "need_data_state": _coerce_string_list(d.get("need_data_state")),
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


def _render_refuted_block(res: "ConvergeResult") -> str:
    """Render the refuted attributed node (+ any model-listed refuted peers) as the
    exclusion lines for a redirect re-stitch (M035). Each line is ``file:lines — why``
    drawn from the contradicted causal check, so the re-pass knows EXACTLY which loci a
    prior pass proved cannot produce the symptom. Deterministic; never raises."""
    causal = res.causal_check or {}
    lines: list[str] = []
    ad = res.attributed_defect or {}
    if ad.get("file"):
        why = (causal.get("trace") or ad.get("why")
               or "cause→symptom check contradicted").strip()
        lines.append(f"- {ad.get('file')}:{ad.get('lines', '')} — "
                     f"{_trunc(why, _REASON_CHARS)}")
    seen = {(_norm(ad.get("file", "")), str(ad.get("lines", "")))}
    for peer in causal.get("refuted_peers") or []:
        pf = str(peer.get("file", "") or "")
        if not pf:
            continue
        key = (_norm(pf), str(peer.get("lines", "")))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {pf}:{peer.get('lines', '')} — "
                     f"{_trunc(str(peer.get('why_not', '') or ''), _REASON_CHARS)}")
    return "\n".join(lines)


def _converge_once(seed_text: str, located: list[dict[str, Any]],
                   unlocated: list[dict[str, Any]], windows: list[dict[str, Any]],
                   known: set[str], provider: str, model: str, pk: dict[str, Any],
                   ledger, timeout: int, data_state_block: str = "",
                   db_available: bool = False, db_schema: str = "",
                   code_state_block: str = "",
                   http_binding_block: str = "",
                   fragment_fact_block: str = "",
                   refuted_block: str = "") -> ConvergeResult:
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
                                   code_state_block, http_binding_block,
                                   fragment_fact_block, refuted_block)
    attempt_prompt = prompt
    parsed: dict[str, Any] | None = None
    for attempt in range(2):
        call_id = ledger.begin_call("converge", "converge", provider, model,
                                    attempt_prompt) if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=None,
                             timeout=timeout,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **pk)
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


_HTTP_WINNING_AXIS_PREFIX = "HTTP_WINNING_PATH:"


def _attributed_winning_urls(located: list[dict[str, Any]],
                             ad: dict[str, Any] | None) -> set[str]:
    """URLs whose lifted winning-path producer aligns with the attributed locus.

    Empty when the attributed defect is not on any grounded HTTP path — the caller then
    fails OPEN (keeps the legacy file-only peer behaviour) so this never touches non-HTTP
    bugs. Built only from the synthetic ``HTTP_WINNING_PATH:`` loci already in ``located``.
    """
    ad_file = _norm((ad or {}).get("file", "")) if isinstance(ad, dict) else ""
    if not ad_file:
        return set()
    urls: set[str] = set()
    for item in located if isinstance(located, list) else []:
        if not isinstance(item, dict):
            continue
        axis = str(item.get("axis_id", "") or "")
        if not axis.startswith(_HTTP_WINNING_AXIS_PREFIX):
            continue
        vf = _norm((item.get("verdict") or {}).get("file", ""))
        if vf and _aligns(vf, ad_file):
            urls.add(axis[len(_HTTP_WINNING_AXIS_PREFIX):])
    return urls


def _is_offpath_synthetic_peer(v: dict[str, Any], attributed_urls: set[str]) -> bool:
    """True for a winning-path synthetic peer on a DIFFERENT URL than the attributed locus.

    Such a peer is cross-endpoint noise (e.g. an auth/JWT producer harvested from an
    UNRELATED request path), not a competing locus for THIS symptom. Filters ONLY the
    synthetic ``HTTP_WINNING_PATH:`` peers; real judge-axis peers are untouched. Fail-open:
    only filters when the attributed URL set is known AND the peer's URL resolves and differs.
    """
    axis = str((v or {}).get("axis_id", "") or "")
    if not axis.startswith(_HTTP_WINNING_AXIS_PREFIX):
        return False
    if not attributed_urls:
        return False
    peer_url = axis[len(_HTTP_WINNING_AXIS_PREFIX):]
    return bool(peer_url) and peer_url not in attributed_urls


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
    # P0 may already have demoted this exact peer. Preserve the historical N180 stamp
    # for data-backed cases instead of making guard ordering erase existing diagnostics.
    p0_peer = (res.causal_check or {}).get("unrefuted_peer") \
        if isinstance(res.causal_check, dict) else None
    if not data_backed or (not res.converged and not isinstance(p0_peer, dict)):
        return res
    cc = res.causal_check or {}
    if cc.get("verdict") != "consistent":
        return res
    ad = res.attributed_defect or {}
    if not res.converged and isinstance(p0_peer, dict):
        res.causal_check = {**cc, "dropped_peer": p0_peer}
        res.summary = (
            f"not converged: data-certified consistent at "
            f"{ad.get('file', '')}:{ad.get('lines', '')} dropped an UNREFUTED competing "
            f"hypothesis at {p0_peer.get('file', '')}:{p0_peer.get('lines', '')} "
            f"(N180 domain guard)")
        return res
    # Files the convergence already ACCOUNTS for — anything here is on the path / a named
    # target, i.e. NOT "dropped". A located peer aligning to one of these is fine.
    accounted = {_norm(ad.get("file", ""))}
    for n in res.path or []:
        accounted.add(_norm(n.get("file", "")))
    for d in res.additional_defects or []:
        accounted.add(_norm(d.get("file", "")))
    accounted.discard("")
    trace = (cc.get("trace") or "").lower()
    attributed_urls = _attributed_winning_urls(located, ad)
    for v in located:
        if _is_offpath_synthetic_peer(v, attributed_urls):
            continue  # cross-endpoint winning-path noise, not a competing locus for this URL
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


def _counterfactual_complete_guard(res: ConvergeResult,
                                   located: list[dict[str, Any]]) -> ConvergeResult:
    """Require a grounded counterfactual and explicit accounting for every distinct peer.

    This is the proactive, domain-agnostic form of the N180 lesson: a ``consistent``
    verdict is not earned by reachability or row existence alone. It must explain why
    correcting the attributed locus removes the symptom, and it must refute or retain
    every other located file. The check is deliberately demote-only and tight: trace
    basename mentions count as accounting, malformed input degrades without raising, and
    the worst case is one extra reinvestigation pass. Disabled by
    ``HIVE_NO_COUNTERFACTUAL``.
    """
    if os.environ.get("HIVE_NO_COUNTERFACTUAL") or not res.converged:
        return res
    if not isinstance(res.causal_check, dict):
        res.converged = False
        res.causal_check = {
            "verdict": "unverified",
            "counterfactual_incomplete": True,
            "trace": "[counterfactual] malformed causal_check; certification was not earned.",
        }
        res.summary = "not converged: malformed causal_check"
        return res
    cc = res.causal_check
    if cc.get("verdict") != "consistent":
        return res
    ad = res.attributed_defect if isinstance(res.attributed_defect, dict) else {}

    if not str(cc.get("counterfactual", "") or "").strip():
        res.converged = False
        res.causal_check = {
            **cc,
            "counterfactual_incomplete": True,
            "trace": str(cc.get("trace", "") or "")
            + " [counterfactual] consistent verdict supplied no cause-to-symptom "
              "counterfactual for the attributed locus; certification was not earned.",
        }
        res.summary = (
            f"not converged: consistent attribution at {ad.get('file', '')}:"
            f"{ad.get('lines', '')} has no grounded counterfactual")
        logger.info("converge: counterfactual facet demoted — missing counterfactual at %s:%s",
                    ad.get("file", ""), ad.get("lines", ""))
        return res

    accounted: set[str] = set()
    for node in [ad, *(res.path or []), *(res.additional_defects or [])]:
        if isinstance(node, dict):
            f = _norm(node.get("file", ""))
            if f:
                accounted.add(f)
    for peer in cc.get("refuted_peers", []) if isinstance(
            cc.get("refuted_peers"), list) else []:
        if isinstance(peer, dict):
            f = _norm(peer.get("file", ""))
            if f:
                accounted.add(f)

    trace = str(cc.get("trace", "") or "").lower()
    attributed_urls = _attributed_winning_urls(located, ad)
    for v in located if isinstance(located, list) else []:
        if not isinstance(v, dict):
            continue
        if _is_offpath_synthetic_peer(v, attributed_urls):
            continue  # cross-endpoint winning-path noise, not a competing locus for this URL
        vd = v.get("verdict") if isinstance(v.get("verdict"), dict) else {}
        pf = _norm(vd.get("file", ""))
        if not pf or any(_aligns(pf, a) for a in accounted):
            continue
        base = pf.rsplit("/", 1)[-1]
        if base and base in trace:
            continue
        peer = {
            "axis_id": str(v.get("axis_id", "?")),
            "file": str(vd.get("file", "") or ""),
            "lines": str(vd.get("lines", "") or ""),
            "reason": str(vd.get("reason", "") or ""),
        }
        res.converged = False
        res.causal_check = {
            **cc,
            "unrefuted_peer": peer,
            "trace": str(cc.get("trace", "") or "")
            + f" [counterfactual] competing located hypothesis at {peer['file']}:"
              f"{peer['lines']} (axis {peer['axis_id']}) was neither included nor "
              "causally refuted; re-examine before certifying one locus.",
        }
        res.summary = (
            f"not converged: consistent attribution at {ad.get('file', '')}:"
            f"{ad.get('lines', '')} left an unrefuted peer at "
            f"{peer['file']}:{peer['lines']}")
        logger.info("converge: counterfactual facet demoted — unrefuted peer %s:%s",
                    peer["file"], peer["lines"])
        return res
    return res


_TRACE_TOKEN_STOP = {
    "and", "async", "await", "class", "def", "else", "false", "for", "from",
    "if", "import", "in", "is", "none", "not", "or", "return", "self", "true",
    "with",
}


def _salient_locus_tokens(attributed: dict[str, Any],
                          code_root: str | None) -> set[str]:
    """Extract conservative function/key/field tokens from the attributed live locus."""
    if not code_root or not isinstance(attributed, dict):
        return set()
    focal = {"verdict": {
        "located": True,
        "file": attributed.get("file", ""),
        "lines": attributed.get("lines", ""),
    }}
    lifted = _lift_live_code([focal], code_root)
    if not lifted:
        return set()
    tokens: set[str] = set()
    for pat in (
        r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"""['"]([A-Za-z_][A-Za-z0-9_]{3,})['"]""",
        r"\b([A-Z][A-Z0-9_]{3,})\b",
        r"\b([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]{2,})\b",
    ):
        for token in re.findall(pat, lifted):
            low = token.lower()
            if low not in _TRACE_TOKEN_STOP:
                tokens.add(low)
    return tokens


def _trace_grounding_guard(res: ConvergeResult,
                           code_root: str | None) -> ConvergeResult:
    """Demote generic ``consistent`` prose that never references its attributed locus.

    A basename mention is sufficient. When live source is available, one salient
    function/SQL-key/field token from the cited lines is also sufficient. The facet is
    deliberately demote-only and permissive about terse real traces; it fires only on
    total absence of a locus reference. Disabled by ``HIVE_NO_TRACE_GROUNDING``.
    """
    if os.environ.get("HIVE_NO_TRACE_GROUNDING") or not res.converged:
        return res
    cc = res.causal_check if isinstance(res.causal_check, dict) else {}
    if cc.get("verdict") != "consistent":
        return res
    ad = res.attributed_defect if isinstance(res.attributed_defect, dict) else {}
    file = _norm(ad.get("file", ""))
    combined = (
        str(cc.get("trace", "") or "") + " "
        + str(cc.get("counterfactual", "") or "")
    ).lower()
    base = file.rsplit("/", 1)[-1] if file else ""
    tokens = _salient_locus_tokens(ad, code_root)
    if (base and base in combined) or any(
            re.search(rf"\b{re.escape(token)}\b", combined) for token in tokens):
        return res

    res.converged = False
    res.causal_check = {
        **cc,
        "trace_ungrounded": {
            "file": str(ad.get("file", "") or ""),
            "lines": str(ad.get("lines", "") or ""),
        },
        "trace": str(cc.get("trace", "") or "")
        + " [trace-grounding] consistent reasoning referenced neither the attributed "
          "file nor any salient symbol from its live cited source; certification is "
          "generic and must be re-examined.",
    }
    res.summary = (
        f"not converged: consistent attribution at {ad.get('file', '')}:"
        f"{ad.get('lines', '')} has an ungrounded causal trace")
    logger.info("converge: trace-grounding facet demoted generic attribution at %s:%s",
                ad.get("file", ""), ad.get("lines", ""))
    return res


def _evidence_sufficiency_guard(
        res: ConvergeResult,
        located: list[dict[str, Any]],
        windows: list[dict[str, Any]],
        *,
        min_located: int,
        data_backed: bool) -> ConvergeResult:
    """Abstain at the location floor when no positive provenance grounds the choice.

    This facet fires only for a ``consistent`` result with exactly ``min_located``
    candidates, no field-producer, HTTP-binding, or call-chain evidence touching a
    located file, and no live data backing. It is demote-only, costs at worst one extra
    reinvestigation pass, and is disabled by ``HIVE_NO_SUFFICIENCY_GATE``.
    """
    if (os.environ.get("HIVE_NO_SUFFICIENCY_GATE") or not res.converged
            or data_backed or not isinstance(located, list)
            or len(located) != min_located):
        return res
    cc = res.causal_check if isinstance(res.causal_check, dict) else {}
    if cc.get("verdict") != "consistent":
        return res
    located_files = {
        _norm((v.get("verdict") or {}).get("file", ""))
        for v in located if isinstance(v, dict) and isinstance(v.get("verdict"), dict)
    }
    located_files.discard("")
    grounded = False
    for w in windows if isinstance(windows, list) else []:
        if not isinstance(w, dict):
            continue
        via = str(w.get("via", "") or "")
        if via not in ("field-producer", "http-binding", "call-chain"):
            continue
        wf = _norm(w.get("file", ""))
        if via == "http-binding" or any(_aligns(wf, lf) for lf in located_files):
            grounded = True
            break
    if grounded:
        return res

    ad = res.attributed_defect if isinstance(res.attributed_defect, dict) else {}
    res.converged = False
    res.causal_check = {
        **cc,
        "low_confidence": True,
        "trace": str(cc.get("trace", "") or "")
        + " [evidence-sufficiency] attribution was chosen at the minimum located-fragment "
          "floor with no field-producer, HTTP-binding, call-chain, or live-data grounding; "
          "reinvestigate rather than certify a thin guess.",
    }
    res.summary = (
        f"not converged: consistent attribution at {ad.get('file', '')}:"
        f"{ad.get('lines', '')} has insufficient positive grounding")
    logger.info("converge: sufficiency facet demoted thin floor-level attribution")
    return res


def _attribution_stability_guard(
        res: ConvergeResult,
        comparison: ConvergeResult | None) -> ConvergeResult:
    """Demote when two already-produced attribution routes disagree on the file.

    No model call is made here. The facet only compares a holistic result and an
    independently available split result; absent/partial comparisons are a no-op.
    Disagreement stamps ``attribution_unstable`` and routes to reinvestigation. Disabled
    by ``HIVE_NO_STABILITY_CHECK``.
    """
    if (os.environ.get("HIVE_NO_STABILITY_CHECK") or not res.converged
            or not isinstance(comparison, ConvergeResult)
            or not comparison.converged):
        return res
    ad = res.attributed_defect if isinstance(res.attributed_defect, dict) else {}
    other = comparison.attributed_defect \
        if isinstance(comparison.attributed_defect, dict) else {}
    file = str(ad.get("file", "") or "")
    other_file = str(other.get("file", "") or "")
    if not file or not other_file or _aligns(file, other_file):
        return res
    cc = res.causal_check if isinstance(res.causal_check, dict) else {}
    res.converged = False
    res.causal_check = {
        **cc,
        "attribution_unstable": {
            "selected": {"file": file, "lines": str(ad.get("lines", "") or "")},
            "comparison": {
                "file": other_file,
                "lines": str(other.get("lines", "") or ""),
            },
        },
        "trace": str(cc.get("trace", "") or "")
        + f" [stability] independent attribution routes disagree: {file} versus "
          f"{other_file}; reinvestigate rather than ship a contested locus.",
    }
    res.summary = (
        f"not converged: attribution unstable between {file} and {other_file}")
    logger.info("converge: stability facet demoted disagreement %s versus %s",
                file, other_file)
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


# ── HTTP-bound datasource provenance (M036 module-selector off-path) ───────────
# Field-provenance handles snake_case response fields that the FE reads directly, but it
# deliberately ignores short bare fields like ``module``. The module-selector miss is the
# sibling shape: a FE collection variable (``currentModules``) is gated on non-empty,
# populated from an HTTP endpoint (``/api/v1/projects``), and the endpoint's datasource
# hardcodes the relevant field empty (``'' AS module``). The retriever already grounds the
# executed HTTP path; this guard makes converge respect that path when a lexically-similar
# route (``list_modules``) is an off-path decoy.
_HTTP_DS_URL_RE = re.compile(
    r"""(?P<callee>[A-Za-z_$][\w.$]*)\s*(?:<[^>(){}]*>)?\s*\(\s*[`'"]\s*(?P<path>/[A-Za-z0-9_./:{}-]*)""")
_HTTP_DS_BINDING_RE = re.compile(r"client\s+(?P<url>/[A-Za-z0-9_./:{}-]+)")
_HTTP_DS_CALL_RE = re.compile(r"(?:\.|\b)([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
_HTTP_DS_DEF_RE = re.compile(r"\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_HTTP_DS_SKIP = {
    "array", "bool", "dict", "enumerate", "get", "isinstance", "jsonresponse", "len",
    "list", "open", "range", "return", "set", "sorted", "str", "tuple",
}
# Property accesses on an assignment RHS: ``.field`` / ``?.field`` / ``["field"]`` /
# ``['field']``. Matching EVERY access (not just the first dotted token) keeps this
# codebase-agnostic — ``res.modules``, ``resp.data.modules`` and ``payload["modules"]`` all
# surface the field rather than a wrapper object.
_HTTP_DS_PROP_RE = re.compile(
    r"""(?:\?\.|\.)\s*([A-Za-z_][A-Za-z0-9_]*)\b|\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]""")


def _field_stems(field: str) -> set[str]:
    """A field token plus its singular form — the FE payload property is usually plural while
    the DB row field is singular (``modules`` → ``module``; ``categories`` → ``category``)."""
    stems = {field}
    if field.endswith("ies") and len(field) > 3:
        stems.add(field[:-3] + "y")
    if field.endswith("s") and len(field) > 1:
        stems.add(field[:-1])
    return stems


def _camel_tail(name: str) -> str:
    """Last camelCase/snake_case segment, lowercased: a gated variable usually names the
    collection it holds (``currentModules`` → ``modules``, ``allowed_projects`` → ``projects``).
    A fallback field source when the assignment RHS is indirect (``x = someLocal``)."""
    parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", name)
    return parts[-1].lower() if parts else ""


def _http_ds_fe_edges(windows: list[dict[str, Any]],
                      code_root: str | None = None) -> list[dict[str, Any]]:
    """Extract FE gated collection variables and the response field/URL that feeds them.

    Tight, structural signal only: a FE file must show a non-empty length gate for a local
    collection variable AND contain an HTTP URL literal. Candidate response fields come from
    the variable's assignments (any property access on the RHS, codebase-agnostic) or, as a
    fallback, the variable's own name tail. No seed text parsing; no invented field names.
    """
    by_file: dict[str, str] = {}
    for w in windows or []:
        f = w.get("file", "")
        if not _is_fe_file(f):
            continue
        by_file[f] = by_file.get(f, "") + "\n" + (w.get("text") or "")
    if code_root:
        # The gate, fetch, and assignment often sit in three small windows with gaps
        # between them. Read only FE files already present in evidence, mirroring the
        # HTTP bridge's in-scope full-file grounding.
        for f in list(by_file):
            rel = (f or "").replace("\\", "/")
            abspath = os.path.join(code_root, rel)
            try:
                with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
                    live = fh.read()
            except OSError:
                continue
            if live:
                by_file[f] = by_file[f] + "\n" + live

    edges: list[dict[str, Any]] = []
    for f, text in by_file.items():
        gates = set()
        for pat in (
            r"""v-if\s*=\s*["'][^"']*\b([A-Za-z_$][\w$]*)\s*(?:\.value)?\.length\s*>\s*0""",
            r"""\bif\s*\(\s*([A-Za-z_$][\w$]*)\s*(?:\.value)?\.length\s*>\s*0""",
        ):
            gates.update(m.group(1) for m in re.finditer(pat, text))
        if not gates:
            continue
        urls = sorted({m.group("path").rstrip("/")
                       for m in _HTTP_DS_URL_RE.finditer(text)})
        if not urls:
            continue
        for var in sorted(gates):
            # Candidate response fields, codebase-agnostic: every property access on the RHS
            # of any assignment to the gated var — ``res.modules``, ``resp.data.modules``,
            # ``payload["modules"]``, ``x?.modules ?? []`` all surface the field rather than a
            # wrapper. Fall back to the variable's own name tail (``currentModules`` →
            # ``modules``) when the RHS is indirect (``x = someLocal``). FE payload is plural,
            # the DB row often singular, so each candidate carries its singular stem too. A
            # wrong candidate is harmless: the guard fires only when a LOCATED datasource
            # literally hardcodes one of these empty, so extra fields simply never match.
            fields: list[str] = []
            for am in re.finditer(
                    rf"""\b{re.escape(var)}\b\s*(?:\.value)?\s*=\s*(?P<rhs>[^;\n]+)""", text):
                for pm in _HTTP_DS_PROP_RE.finditer(am.group("rhs")):
                    fields.append(pm.group(1) or pm.group(2))
            tail = _camel_tail(var)
            if tail:
                fields.append(tail)
            fields = [x for x in fields if x]
            if not fields:
                continue
            stems: set[str] = set()
            for fld in fields:
                stems |= _field_stems(fld)
            edges.append({"file": f, "var": var, "field": fields[0],
                          "stems": stems, "urls": urls})
    return edges


def _http_ds_binding_url(w: dict[str, Any]) -> str:
    text = w.get("text") or ""
    m = _HTTP_DS_BINDING_RE.search(text)
    if m:
        return m.group("url").rstrip("/")
    return str(w.get("url") or "").rstrip("/")


def _http_ds_def_name(w: dict[str, Any]) -> str:
    sym = str(w.get("symbol") or "")
    if sym:
        return sym
    m = _HTTP_DS_DEF_RE.search(w.get("text") or "")
    return m.group(1) if m else ""


def _http_ds_calls(text: str) -> set[str]:
    out = set()
    for m in _HTTP_DS_CALL_RE.finditer(text or ""):
        name = m.group(1)
        if name.lower() not in _HTTP_DS_SKIP:
            out.add(name)
    return out


def _http_ds_reachable(binding: dict[str, Any],
                       windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reachability-lite from a resolved HTTP handler through same-name def windows."""
    by_def: dict[str, list[dict[str, Any]]] = {}
    for w in windows or []:
        name = _http_ds_def_name(w)
        if name:
            by_def.setdefault(name, []).append(w)

    out: list[dict[str, Any]] = [binding]
    seen_win = {(_norm(binding.get("file", "")), str(binding.get("lines", "")))}
    seen_sym: set[str] = set()
    queue = list(_http_ds_calls(binding.get("text") or ""))
    while queue and len(seen_sym) < 64:
        sym = queue.pop(0)
        if sym in seen_sym:
            continue
        seen_sym.add(sym)
        for w in by_def.get(sym, []):
            key = (_norm(w.get("file", "")), str(w.get("lines", "")))
            if key in seen_win:
                continue
            seen_win.add(key)
            out.append(w)
            for nxt in sorted(_http_ds_calls(w.get("text") or "")):
                if nxt not in seen_sym:
                    queue.append(nxt)
    return out


def _http_ds_empty_field(text: str, stems: set[str]) -> str | None:
    """Return the field whose datasource evidence hardcodes/omits it as empty."""
    for field in sorted(stems, key=lambda x: (len(x), x)):
        q = re.escape(field)
        if re.search(rf"""(?i)(?:''|"")\s+AS\s+{q}\b""", text or ""):
            return field
        if re.search(rf"""(?i)['"]{q}['"]\s*:\s*(?:''|""|\[\s*\])""", text or ""):
            return field
        if re.search(rf"""(?i)\b{q}\b\s*=\s*(?:''|""|\[\s*\])""", text or ""):
            return field
    return None


def _read_locus_text(code_root: str | None, file: str, lines: str) -> str:
    """Read live source at a located locus's ``file:lines`` (best-effort, "" on failure)."""
    if not code_root or not file:
        return ""
    rng = _parse_line_range(lines)
    rel = (file or "").replace("\\", "/").lstrip("./")
    try:
        with open(os.path.join(code_root, rel), "r",
                  encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError:
        return ""
    if not rng:
        return "".join(all_lines)
    lo, hi = rng
    return "".join(all_lines[max(1, lo) - 1:min(len(all_lines), hi)])


def _winningpath_ds_targets(edges: list[dict[str, Any]],
                            located: list[dict[str, Any]],
                            code_root: str | None) -> list[dict[str, Any]]:
    """Fallback datasource targets: the deterministic winning-path PRODUCER for a gated
    FE field's URL, used only when no explicit empty-literal datasource window was found.

    The live producer of a symptom field is often an OMISSION (the response simply never
    carries the field — e.g. ``SELECT * FROM projects`` with no module column) rather than
    an explicit ``'' AS field``. :func:`_http_ds_empty_field` is blind to omission, so the
    older guard could only re-point to a handler that HARDCODES the field empty — which on a
    shadowed route is the dead path (M036: store.py ``'' AS module`` vs the live
    db/projects.py that omits modules entirely). The registration-order-aware winning-path
    producer (already lifted into ``located`` as ``HTTP_WINNING_PATH:<url>``) is the
    deterministic producer of the URL the gated FE variable is filled from; re-pointing there
    is structural grounding, not seed parsing or a guess. Fires only when (a) the gated FE
    edge's URL has such a producer locus AND (b) that producer's live code does NOT mention
    the field (true omission) — if it emits the field, the emptiness is elsewhere and we
    abstain. The apply-side red→green backstop is the final check on any re-point.
    """
    wp_by_url: dict[str, dict[str, Any]] = {}
    for v in located or []:
        ax = str(v.get("axis_id", "") or "")
        if ax.startswith(_HTTP_WINNING_AXIS_PREFIX):
            wp_by_url[ax[len(_HTTP_WINNING_AXIS_PREFIX):].rstrip("/")] = v
    if not wp_by_url:
        return []
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        stems = {s for s in (edge.get("stems") or []) if s}
        for url in edge.get("urls", []):
            lv = wp_by_url.get(str(url).rstrip("/"))
            if not lv:
                continue
            vd = lv.get("verdict") or {}
            key = (_norm(vd.get("file", "")), str(vd.get("lines", "")))
            if key in seen:
                continue
            ptext = _read_locus_text(code_root, vd.get("file", ""), vd.get("lines", ""))
            if ptext and stems and any(
                    re.search(rf"\b{re.escape(s)}\b", ptext, re.I) for s in stems):
                continue  # producer DOES emit the field → emptiness is not here; abstain
            seen.add(key)
            targets.append({
                "edge": edge, "binding": {"url": str(url).rstrip("/")},
                "window": {"text": ""}, "located": lv,
                "field": edge.get("field") or (sorted(stems)[0] if stems else ""),
                "reachable_files": set(), "via_winning_path": True,
            })
    return targets


def _clear_resolved_peer(cc: dict[str, Any] | None,
                         ad: dict[str, Any] | None) -> dict[str, Any]:
    """Drop a now-RESOLVED unrefuted/dropped peer once the attribution lands on it.

    The dropped-peer / counterfactual guards may stamp a competing peer (stored as
    ``unrefuted_peer`` / ``dropped_peer``) before a later provenance guard re-points the
    attribution ONTO that very peer. Leaving the stamp makes the honey read "a peer was left
    unrefuted" about the file we just selected. Removed ONLY when the peer aligns with the
    final attribution, so every genuinely-dropped peer stays intact.
    """
    if not isinstance(cc, dict):
        return cc or {}
    af = _norm((ad or {}).get("file", ""))
    if not af:
        return cc
    out = dict(cc)
    for key in ("unrefuted_peer", "dropped_peer"):
        peer = out.get(key)
        if isinstance(peer, dict) and _aligns(af, peer.get("file", "")):
            out.pop(key, None)
    return out


def _http_datasource_provenance_guard(res: ConvergeResult,
                                      located: list[dict[str, Any]],
                                      windows: list[dict[str, Any]],
                                      code_root: str | None = None) -> ConvergeResult:
    """Re-point off-path HTTP decoys to the datasource that empties a gated FE field.

    Fires only when all grounding lines up:
      • FE evidence shows a collection variable gated by ``length > 0`` and assigned from
        a response field (e.g. ``currentModules`` ← ``modules``);
      • EITHER a real HTTP binding's call-chain reaches a non-FE window that hardcodes the
        field empty (``'' AS module`` / empty collection) at a LOCATED file [explicit case],
      • OR the gated field's URL has a deterministic winning-path PRODUCER locus (lifted into
        ``located`` as ``HTTP_WINNING_PATH:<url>``) whose live code OMITS the field entirely
        [omission case, :func:`_winningpath_ds_targets`]. The explicit scan is blind to
        omission and to registration-order shadowing, so on a shadowed route it would re-point
        to the dead handler (M036: store.py ``'' AS module`` vs the live db/projects.py that
        omits modules); the winning-path producer is the registration-order-aware live source.

    No located/winning-path datasource → no-op. Ambiguous/no structural FE signal → no-op.
    Disabled by ``HIVE_NO_HTTP_DATASOURCE_PROVENANCE``. This is a re-point/confirm guard
    only; it never invents an edit target, and the apply-side red→green backstop is the final
    execution check on any re-point.
    """
    if os.environ.get("HIVE_NO_HTTP_DATASOURCE_PROVENANCE"):
        return res
    edges = _http_ds_fe_edges(windows, code_root)
    if not edges:
        return res
    # http-binding windows drive the explicit-empty datasource scan. Their ABSENCE no longer
    # short-circuits the guard: the winning-path producer fallback below grounds on the
    # deterministic HTTP_WINNING_PATH loci in ``located``, which exist independently of a
    # binding window surviving the bundle cap.
    bindings = [w for w in windows if w.get("via") == "http-binding"]

    located_by_file: dict[str, dict[str, Any]] = {}
    for v in located or []:
        f = _norm((v.get("verdict") or {}).get("file", ""))
        if f:
            located_by_file[f] = v

    targets: list[dict[str, Any]] = []
    for edge in edges:
        edge_urls = {u.rstrip("/") for u in edge.get("urls", [])}
        for b in bindings:
            if _http_ds_binding_url(b) not in edge_urls:
                continue
            reachable = _http_ds_reachable(b, windows)
            reachable_files = {_norm(w.get("file", "")) for w in reachable}
            for w in reachable:
                f = _norm(w.get("file", ""))
                if not f or _is_fe_file(f):
                    continue
                empty_field = _http_ds_empty_field(w.get("text") or "",
                                                   set(edge.get("stems") or []))
                if not empty_field:
                    continue
                lv = next((v for lf, v in located_by_file.items() if _aligns(f, lf)), None)
                if not lv:
                    continue
                targets.append({"edge": edge, "binding": b, "window": w,
                                "located": lv, "field": empty_field,
                                "reachable_files": reachable_files})
    if not targets:
        # No explicit empty-literal datasource found. Fall back to the deterministic
        # winning-path producer for a gated FE field's URL (handles the OMISSION case the
        # explicit-empty scan is blind to, and prefers the live registration-order handler
        # over a shadowed one). Fail-open: empty → original no-op behaviour.
        targets = _winningpath_ds_targets(edges, located, code_root)
    if not targets:
        return res

    # Prefer the deepest reachable constant site (a datasource) over wrappers with the
    # same name, then stable-sort by file. In the M036 chain this picks store.py over
    # db.py/process_service.py and ignores off-path list_modules.
    targets.sort(key=lambda t: (
        0 if _http_ds_empty_field(t["window"].get("text") or "", {t["field"]}) else 1,
        _norm((t["located"].get("verdict") or {}).get("file", "")),
    ))
    target = targets[0]
    tvd = target["located"].get("verdict") or {}
    tf = _norm(tvd.get("file", ""))
    ad = res.attributed_defect or {}
    cf = _norm(ad.get("file", ""))
    if not cf:
        return res

    if _aligns(cf, tf):
        if not res.converged:
            res.converged = True
            cc = dict(res.causal_check or {})
            cc["http_datasource_provenance_confirmed"] = {
                "file": ad.get("file", ""), "lines": ad.get("lines", ""),
                "field": target["field"], "url": _http_ds_binding_url(target["binding"]),
                "var": target["edge"].get("var", ""),
            }
            cc["trace"] = (cc.get("trace") or "") + (
                f" [http-datasource-provenance] attribution {ad.get('file', '')}:"
                f"{ad.get('lines', '')} is the HTTP-bound datasource that does not produce a "
                f"non-empty {target['field']} for gated FE variable {target['edge'].get('var', '')}.")
            res.causal_check = cc
            res.summary = (
                f"converged (HTTP datasource provenance): defect at {ad.get('file', '')}:"
                f"{ad.get('lines', '')} empties the field feeding the gated FE variable")
        res.causal_check = _clear_resolved_peer(res.causal_check, ad)
        return res

    old = {"file": ad.get("file", ""), "lines": ad.get("lines", "")}
    res.attributed_defect = {
        "node": "http-bound-datasource",
        "file": tvd.get("file", ""),
        "lines": tvd.get("lines", ""),
        "why": (f"the FE gates {target['edge'].get('var', '')} on non-empty values from "
                f"{_http_ds_binding_url(target['binding'])}, and this HTTP-bound datasource "
                f"does not produce a non-empty {target['field']}; the prior attribution "
                f"{old['file']}:{old['lines']} is not the datasource producing that empty "
                f"field for the executed endpoint"),
    }
    cc = dict(res.causal_check or {})
    cc["http_datasource_provenance_repointed"] = {
        "from": old,
        "to": {"file": tvd.get("file", ""), "lines": tvd.get("lines", "")},
        "field": target["field"],
        "url": _http_ds_binding_url(target["binding"]),
        "var": target["edge"].get("var", ""),
    }
    cc["trace"] = (cc.get("trace") or "") + (
        f" [http-datasource-provenance] FE variable {target['edge'].get('var', '')} is "
        f"gated on non-empty data and is filled from {_http_ds_binding_url(target['binding'])}; "
        f"the resolved HTTP path reaches {tvd.get('file', '')}:{tvd.get('lines', '')}, "
        f"which does not produce a non-empty {target['field']}. Re-pointed from {old['file']}:"
        f"{old['lines']} to the executed datasource and refuted the off-path/non-datasource "
        f"attribution.")
    res.causal_check = _clear_resolved_peer(cc, res.attributed_defect)
    res.converged = True
    res.summary = (
        f"converged (HTTP datasource re-point): defect at {tvd.get('file', '')}:"
        f"{tvd.get('lines', '')} empties {target['field']} for the endpoint feeding "
        f"{target['edge'].get('var', '')} (was mis-attributed to {old['file']}:"
        f"{old['lines']})")
    logger.info("converge: HTTP datasource guard re-pointed attribution %s:%s -> %s:%s%s",
                old["file"], old["lines"], tvd.get("file", ""), tvd.get("lines", ""),
                " (winning-path producer)" if target.get("via_winning_path") else "")
    return res


def _field_provenance_reaim(res: ConvergeResult,
                            prod_fields: dict[str, set[str]],
                            prod_text: dict[str, str],
                            prod_loc: dict[str, tuple[str, str]]) -> ConvergeResult:
    """Block a NAME-DECOY attribution when the FE-bound field's producer exists but was not
    located (M037 — the negative-space twin of :func:`_field_provenance_guard`).

    field-producer grounding resolved a snake_case field the FE reads to the server code
    that FILLS it, yet NO judge located that producer. The converger then anchored on a
    file that merely SHARES a concept token with the symptom (``head`` → the head-route
    serializer) but does NOT produce the field and is NOT on its production path — a name
    decoy (T905: attributed ``workflow_head_routes.py``; the strip actually reads
    ``workflow_head_type`` produced in ``documents.py``). We do not RE-POINT to the
    unlocated producer (that would invent an unvetted edit target); we DEMOTE the
    convergence and emit a ``missing_link`` aimed at the producer so the run re-investigates
    THERE instead of locking the decoy. Left untouched when the attribution itself produces
    an FE-bound field, lies on a producer's production path, or a lead is already named.
    Disabled by ``HIVE_NO_FIELD_PROVENANCE`` / ``HIVE_NO_FIELD_PROVENANCE_REAIM``.
    """
    if os.environ.get("HIVE_NO_FIELD_PROVENANCE_REAIM"):
        return res
    ad = res.attributed_defect or {}
    c = _norm(ad.get("file", ""))
    if not c:
        return res
    if res.missing_link:                 # a better-aimed lead already exists — don't clobber
        return res
    if c in prod_fields:                 # C itself produces an FE-bound field → not a decoy
        return res
    c_stem = c.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    for txt in prod_text.values():       # C read by a producer (on its path) → upstream, leave
        if c_stem and c_stem in (txt or "").lower():
            return res
    # A node ON the converger's winning path is, by definition, live executed code for THIS
    # request — it cannot be a name-decoy, which the docstring defines as a file OFF the
    # production path matched only on a shared concept token. An attribution there that does
    # not PRODUCE the field is an OMISSION site (the live datasource simply never adds it —
    # M036: db/projects.py ``SELECT * FROM projects`` omits ``modules``), which is the correct
    # fix target. Re-aiming away would chase a richer OFF-path producer that is often the
    # lexical decoy itself (M036: ``module_id`` from the /modules list route). The positive
    # winning-path re-point is the datasource guard's job; here we only refuse to DEMOTE a
    # winning-path attribution as a decoy. (HIVE_NO_FIELD_PROVENANCE_REAIM still disables all.)
    wp_files = {n.get("file", "") for n in (res.winning_path or []) if n.get("file")}
    if any(_aligns(c, wf) for wf in wp_files):
        logger.info("converge: field-provenance re-aim ABSTAINS — attribution %s is on the "
                    "winning path (live executed datasource = omission site, not a name-decoy)",
                    ad.get("file", ""))
        return res
    # Richest producer P (most FE-bound fields) is the re-aim target.
    P = sorted(prod_fields.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]
    pfile, plines = prod_loc.get(P, (P, ""))
    fields = sorted(x for x in prod_fields.get(P, set()) if x)
    old = {"file": ad.get("file", ""), "lines": ad.get("lines", "")}
    res.converged = False
    res.missing_link = {
        "between": ["fe-binding", "field-producer"],
        "need": {
            "symbols": fields[:4],
            "greps": fields[:4],
            "file_globs": [pfile] if pfile else [],
        },
    }
    cc = dict(res.causal_check or {})
    cc["field_provenance_reaimed"] = {
        "from": old, "toward": {"file": pfile, "lines": plines}, "fields": fields}
    cc["trace"] = (cc.get("trace") or "") + (
        f" [field-provenance re-aim] the symptom is a wrong FE-bound field value "
        f"({', '.join(fields)}); that field is PRODUCED at {pfile}:{plines} (field-producer "
        f"grounding), but the attribution {old['file']}:{old['lines']} neither produces it "
        f"nor lies on its production path — a name-decoy. Demoted and re-aimed at the "
        f"producer (not certified).")
    res.causal_check = cc
    res.summary = (
        f"not converged (field-provenance re-aim): {old['file']}:{old['lines']} is a "
        f"name-decoy; the FE-bound field is produced at {pfile}:{plines} (not located) — "
        f"routed to reinvestigation toward the producer")
    logger.info("converge: field-provenance guard re-aimed name-decoy %s:%s → producer "
                "%s:%s (demoted, missing_link)", old["file"], old["lines"], pfile, plines)
    return res


def _field_provenance_guard(res: ConvergeResult,
                            located: list[dict[str, Any]],
                            fp_windows: list[dict[str, Any]]) -> ConvergeResult:
    """Anchor the attribution to the code that PRODUCES the FE-bound field the symptom is
    about (field-producer provenance).

    The gap this closes (head off-by-one, M035 §4): the symptom is a wrong value in a
    response field the FE reads (``workflow_head_type``). field-producer grounding (the
    retriever ``via=field-producer`` windows) deterministically locates the code that
    FILLS that field (``out["workflow_head_type"] = …`` in documents.py, right at the
    ``NON_HEAD_TYPES`` bug). But a weak single-shot converger anchored on the seed and
    "data-certified" by an INCIDENTAL SQL read attributed the defect to a sibling query
    merely NAMED for the concept (``get_effective_head``) that the FE never reads, OR a
    correct producer attribution got demoted by a domain guard over an unrelated peer.
    Data existence ≠ causal link (the N170/N180 lesson). This guard uses the STRUCTURAL
    fact — "field F is produced HERE" — to anchor the attribution, NOT any seed
    natural-language parsing (the N177 over-fire trap).

    The producer of interest P is the LOCATED candidate filling the MOST FE-bound fields
    (the judge already vetted it as a defect site, and field-count picks the symptom's
    field-rich producer over an incidental single-field one — e.g. documents.py's whole
    ``workflow_head_*`` family over pipeline_service's lone ``in_progress``). Then:
      • attribution already AT P → ASSERT converged (restore a demotion: the grounded
        producer of the symptom field outranks a consistency dropped over an unrelated
        peer);
      • attribution at a DIFFERENT file that produces none of P's fields AND is not on
        P's production path → RE-POINT to P;
      • attribution at some other producer, or on P's path (the legitimate "bug is
        downstream of the producer" case) → left untouched.

    Disabled by ``HIVE_NO_FIELD_PROVENANCE``. Short fields (<8 chars, e.g. ``module``)
    are never harvested by field-producer, so field-poor symptoms never reach this guard.
    Operates on the FULL (uncapped) field-producer evidence — the pooled ``windows`` cap
    can drop the symptom producer, so the caller passes facts straight from the bundles.
    """
    if os.environ.get("HIVE_NO_FIELD_PROVENANCE"):
        return res
    prod_fields: dict[str, set[str]] = {}
    prod_text: dict[str, str] = {}
    prod_loc: dict[str, tuple[str, str]] = {}
    for w in fp_windows or []:
        if w.get("via") != "field-producer":
            continue
        f = _norm(w.get("file", ""))
        if not f:
            continue
        prod_fields.setdefault(f, set()).add(str(w.get("field", "")))
        prod_text[f] = prod_text.get(f, "") + "\n" + (w.get("text") or "")
        prod_loc.setdefault(f, (str(w.get("file", "")), str(w.get("lines", ""))))
    if not prod_fields:
        return res
    # P = located producer filling the MOST FE-bound fields (vetted + symptom-rich).
    prod_located = [v for v in located
                    if _norm((v.get("verdict") or {}).get("file", "")) in prod_fields]
    if not prod_located:
        # The symptom's FE-bound field HAS a known producer, but no judge located it. A
        # converged attribution to a file that neither produces that field nor lies on its
        # production path is a NAME-DECOY — matched on a shared concept token, not on the
        # executed FE→BE binding (the T905 ``workflow_head_routes.py`` miss). Don't certify
        # it by default; demote and name the real producer as a lead so the run re-aims at
        # it. We never RE-POINT to the unlocated producer (that would invent an unvetted
        # edit target) — only emit the missing_link.
        return _field_provenance_reaim(res, prod_fields, prod_text, prod_loc)

    def _rank(v: dict[str, Any]) -> tuple[int, str]:
        f = _norm((v.get("verdict") or {}).get("file", ""))
        return (-len(prod_fields.get(f, set())), f)

    target = sorted(prod_located, key=_rank)[0]
    tvd = target.get("verdict") or {}
    tf = _norm(tvd.get("file", ""))
    fields = sorted(prod_fields.get(tf, set()))
    ad = res.attributed_defect or {}
    c = _norm(ad.get("file", ""))
    if not c:
        return res

    # (job 2) attribution already AT the symptom field's producer → assert convergence.
    if _aligns(c, tf):
        if not res.converged:
            res.converged = True
            cc = dict(res.causal_check or {})
            cc["field_provenance_confirmed"] = {"file": ad.get("file", ""),
                                                "lines": ad.get("lines", ""), "fields": fields}
            cc["trace"] = (cc.get("trace") or "") + (
                f" [field-provenance] attribution {ad.get('file', '')}:{ad.get('lines', '')} "
                f"is the producer of the FE-bound symptom field(s) {', '.join(fields)} — "
                f"convergence asserted over a demotion on an unrelated peer.")
            res.causal_check = cc
            res.summary = (
                f"converged (field-provenance): defect at {ad.get('file', '')}:"
                f"{ad.get('lines', '')} produces the FE-bound field the symptom is about")
            logger.info("converge: field-provenance guard confirmed producer attribution "
                        "%s:%s (converged)", ad.get("file", ""), ad.get("lines", ""))
        return res

    # attribution is elsewhere:
    if c in prod_fields:
        return res  # a DIFFERENT field's producer — do not arbitrate producer-vs-producer
    # reachability-lite: if C is what P READS (C on P's production path), C may be the real
    # upstream cause → stay silent. Proxy: C's file stem in P's producer text, or both on
    # the converged path.
    c_stem = c.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if c_stem and c_stem in prod_text.get(tf, "").lower():
        return res
    path_files = {_norm(n.get("file", "")) for n in (res.path or [])}
    if c in path_files and tf in path_files:
        return res

    # (job 1) RE-POINT a disconnected decoy to the field's real producer.
    old = {"file": ad.get("file", ""), "lines": ad.get("lines", "")}
    res.attributed_defect = {
        "node": "field-producer",
        "file": tvd.get("file", ""),
        "lines": tvd.get("lines", ""),
        "why": (f"fills the FE-bound response field(s) {', '.join(fields)} the symptom is "
                f"about; the prior attribution {old['file']}:{old['lines']} neither produces "
                f"that field nor lies on its production path"),
    }
    cc = dict(res.causal_check or {})
    cc["field_provenance_repointed"] = {
        "from": old, "to": {"file": tvd.get("file", ""), "lines": tvd.get("lines", "")},
        "fields": fields}
    cc["trace"] = (cc.get("trace") or "") + (
        f" [field-provenance] the symptom is a wrong FE-bound field value; that field is "
        f"PRODUCED at {tvd.get('file', '')}:{tvd.get('lines', '')} (field-producer "
        f"grounding), while {old['file']}:{old['lines']} does not produce it and is not on "
        f"its production path — re-pointed attribution to the producer.")
    res.causal_check = cc
    res.converged = True
    res.summary = (
        f"converged (field-provenance re-point): defect at {tvd.get('file', '')}:"
        f"{tvd.get('lines', '')} produces the FE-bound field the symptom is about "
        f"(was mis-attributed to {old['file']}:{old['lines']})")
    logger.info("converge: field-provenance guard re-pointed attribution %s:%s → %s:%s",
                old["file"], old["lines"], tvd.get("file", ""), tvd.get("lines", ""))
    return res


def _multi_root_coverage_guard(res: ConvergeResult,
                               located: list[dict[str, Any]],
                               fp_windows: list[dict[str, Any]]) -> ConvergeResult:
    """Surface INDEPENDENT roots that a single-path stitch collapsed (M037 / N179 reinforce).

    A multi-mechanism scenario — e.g. a wrong BE value AND a wrong FE colour AND a status
    badge that never flips — needs fixes at SEVERAL independent loci. The converger, built to
    attribute ONE node on ONE path, sometimes folds them into a single defect and refutes the
    peers as "the same chain", shipping a half-fix. This guard uses field-producer grounding
    as a STRUCTURAL independence test: a located peer that PRODUCES a different FE-bound field
    than the attributed locus (disjoint field sets, different files) cannot be a corroborating
    view of the SAME chain — it is a distinct output with its own root. Such a vetted, dropped
    peer is promoted into ``additional_defects`` so specify's converge-coverage gate refuses to
    call the spec ready until each independent root is fixed.

    Conservative: fires only on a CONVERGED result where BOTH the attribution and the peer are
    judge-LOCATED field-producers with disjoint, non-empty field sets. Never demotes the
    primary — it only ADDS missed roots. Disabled by ``HIVE_NO_MULTI_ROOT``.
    """
    if os.environ.get("HIVE_NO_MULTI_ROOT") or not res.converged:
        return res
    ad = res.attributed_defect or {}
    af = _norm(ad.get("file", ""))
    if not af:
        return res
    prod_fields: dict[str, set[str]] = {}
    for w in fp_windows or []:
        if w.get("via") != "field-producer":
            continue
        f = _norm(w.get("file", ""))
        fld = str(w.get("field", "")).strip()
        if not f or not fld:
            continue
        prod_fields.setdefault(f, set()).add(fld)
    if af not in prod_fields:
        return res                       # attribution is not a field-producer → don't guess
    a_fields = prod_fields[af]
    located_by_file: dict[str, dict[str, Any]] = {}
    for v in located or []:
        f = _norm((v.get("verdict") or {}).get("file", ""))
        if f:
            located_by_file.setdefault(f, v)
    existing = {(_norm(d.get("file", "")), str(d.get("lines", "")))
                for d in (res.additional_defects or [])}
    existing.add((af, str(ad.get("lines", ""))))
    added: list[dict[str, Any]] = []
    for pf, pflds in sorted(prod_fields.items()):
        if pf == af or pf not in located_by_file:
            continue                     # only promote a VETTED (judge-located) producer
        if a_fields & pflds:
            continue                     # shares a field with the attribution → same output
        vd = located_by_file[pf].get("verdict") or {}
        key = (pf, str(vd.get("lines", "")))
        if key in existing:
            continue
        existing.add(key)
        added.append({
            "node": "field-producer",
            "file": vd.get("file", ""),
            "lines": vd.get("lines", ""),
            "why": (f"independent root: produces the distinct FE-bound field(s) "
                    f"{', '.join(sorted(pflds))} — a SEPARATE output from the primary's "
                    f"{', '.join(sorted(a_fields))}, so it cannot be the same chain and "
                    f"needs its own fix"),
        })
    if not added:
        return res
    res.additional_defects = list(res.additional_defects or []) + added
    cc = dict(res.causal_check or {})
    cc["multi_root_candidates"] = [{"file": d["file"], "lines": d["lines"]} for d in added]
    cc["trace"] = (cc.get("trace") or "") + (
        f" [multi-root] {len(added)} located peer(s) produce DISTINCT FE-bound field(s) from "
        f"the attributed locus — promoted to additional_defects as independent roots so the "
        f"fix covers every reported output, not just one.")
    res.causal_check = cc
    logger.info("converge: multi-root guard promoted %d independent field-producer root(s) "
                "to additional_defects", len(added))
    return res


def _causal_provenance_arbiter(
        res: ConvergeResult,
        located: list[dict[str, Any]],
        *,
        fp_windows: list[dict[str, Any]],
        http_ds_windows: list[dict[str, Any]],
        data_backed: bool,
        data_chain_broke: bool,
        db_available: bool,
        code_root: str | None,
        windows: list[dict[str, Any]],
        min_located: int,
        split_origin: bool = False,
        stability_comparison: ConvergeResult | None = None) -> ConvergeResult:
    """Apply all causal/provenance decisions through one fail-closed entry point.

    Precedence is explicit. Demotion facets first collect negative evidence: incomplete
    counterfactual/peer accounting, refuted data premise, missing data stamp, and generic
    trace grounding. Positive deterministic provenance has the final decision: the
    HTTP-bound datasource may confirm/re-point, then the richer field-producer grounding
    runs last and may override a prior demotion exactly as before. Every facet retains its
    own tight firing condition and kill-switch. Any unexpected error demotes rather than
    escaping converge.

    ``split_origin`` marks a result produced by the per-locus SPLIT pass. Its narrow,
    deterministic per-locus elimination IS positive grounding (stronger than a single
    field-producer/HTTP-binding window), so the evidence-sufficiency facet — which abstains
    only on a thin, wholly UNGROUNDED floor-level guess — must not fire on it. The reactive
    guards (dropped-peer / premise / data-stamp) and the prompt-side facets still run, so a
    split result is held to the same causal bar a holistic one is.
    """
    try:
        res = _counterfactual_complete_guard(res, located)
        res = _dropped_peer_guard(res, located, data_backed)
        res = _premise_refuted_guard(res, db_available, data_chain_broke)
        res = _data_stamp_guard(res, db_available, data_backed)
        res = _trace_grounding_guard(res, code_root)
        if not split_origin:
            res = _evidence_sufficiency_guard(
                res, located, windows, min_located=min_located, data_backed=data_backed)
        res = _attribution_stability_guard(res, stability_comparison)
        res = _http_datasource_provenance_guard(
            res, located, http_ds_windows, code_root)
        res = _field_provenance_guard(res, located, fp_windows)
        # Last: with the attribution settled, surface any INDEPENDENT roots (distinct
        # FE-bound field producers) the single-path stitch collapsed (N179 reinforcement).
        res = _multi_root_coverage_guard(res, located, fp_windows)
        return res
    except Exception as e:
        logger.warning("converge: causal provenance arbiter failed closed: %s", e)
        cc = res.causal_check if isinstance(res.causal_check, dict) else {}
        res.converged = False
        res.causal_check = {
            **cc,
            "trace": str(cc.get("trace", "") or "")
            + " [causal-provenance-arbiter] deterministic verification failed; "
              "re-examine rather than certifying.",
        }
        res.summary = "not converged: causal provenance verification failed"
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
                             timeout=timeout,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **pk)
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
        "counterfactual": (
            f"correcting {attributed['file']}:{attributed['lines']} would remove the "
            f"symptom because this was the only locus whose narrow cause-to-symptom "
            f"check remained consistent"),
        "refuted_peers": [
            {
                "file": (r["focal"].get("verdict") or {}).get("file", ""),
                "lines": (r["focal"].get("verdict") or {}).get("lines", ""),
                "why_not": r["why"] or r["trace"] or
                           f"narrow check ruled {r['verdict']}",
            }
            for r in eliminated
        ],
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


def _build_lens_prompt(seed_text: str, attributed: dict[str, Any],
                       causal_check: dict[str, Any] | None, lens: str, lens_desc: str,
                       code_state_block: str, evidence: str,
                       design_change_site: bool = False,
                       winning_path_site: bool = False) -> str:
    """One adversarial refutation prompt: break the attribution through ONE lens.

    ``design_change_site`` carries the converger's DESIGN-CHANGE carve-out into the
    refuter (M037/T905). Without it the adversarial panel re-introduced the exact
    design-match trap the design_change verdict class exists to prevent: a refuter
    breaks a correct, on-path attribution with "the code matches its own design, so
    nothing is wrong" — shaving a legitimate FE design-change site (the
    workflowViewState miss). The carve-out forbids that one refutation reason; the
    shadow / omission / reproduction failure classes still refute normally.
    """
    cc = causal_check or {}
    winning_path_note = ""
    if winning_path_site:
        winning_path_note = (
            "\n[WINNING-PATH grounding — READ FIRST] The attributed locus is on the "
            "converge's registration-order-resolved WINNING PATH — the LIVE handler/"
            "datasource actually executed for THIS request. A sibling that merely LOOKS "
            "wired to the same route (e.g. a get_X_with_Y helper, or a same-named handler "
            "in another router/module) may be SHADOWED / dead by registration order; the "
            "winning-path resolution is STRONGER evidence than reading which function "
            "appears connected. Do NOT refute by claiming a different sibling serves the "
            "request or produces the field — that is the exact shadowed-decoy trap. Refute "
            "ONLY with positive evidence that THIS locus is off the executed path, or a "
            "concrete omission/reproduction failure of THIS locus itself.\n")
    design_change_note = ""
    if design_change_site:
        design_change_note = (
            "\n[DESIGN-CHANGE carve-out — READ FIRST] A located fragment here is a "
            "DESIGN-CHANGE site: a judge ruled the code FAITHFULLY implements its own "
            "design/spec, yet the reporter declared the RESULTING on-screen behaviour "
            "wrong or unwanted. The reporter's stated expectation is GROUND TRUTH. You "
            "must NOT refute this attribution merely because 'the code matches its own "
            "design / spec / state definition' or 'it already produces its designed "
            "output' — a design-change site produces the rejected result BY DESIGN, so "
            "spec-conformance is NOT a valid refutation, and 'expected output' means the "
            "REPORTER's expectation, not the code's designed output. Refute ONLY if the "
            "locus is shadowed / dead / off the live path, the fix would be incomplete (a "
            "real OMISSION elsewhere on the path), or it genuinely cannot influence the "
            "reported behaviour at all.\n")
    return f"""[Role] You are a REFUTER auditing a Hivework converge result. Another model \
attributed a reported defect to ONE code locus and ruled it the cause. Your ONLY job is to \
try to PROVE that attribution WRONG, strictly through the {lens} lens. You are adversarial: \
unless YOUR lens positively confirms the attribution holds, you REFUTE it. Default to \
refuted=true when uncertain — a false "survives" ships a wrong fix to a human; a false \
"refuted" only costs one more re-hunt. You have NO tools; decide from the evidence below.
{winning_path_note}{design_change_note}
[The {lens} lens] {lens_desc}

[Reported scenario / seed]
{_trunc(seed_text, 1500)}

[Attributed defect — the claim you must try to break]
{attributed.get('file', '')}:{attributed.get('lines', '')} — {attributed.get('why', '')}

[The converger's own causal reasoning]
trace: {_trunc(cc.get('trace', '') or '(none)', 600)}
counterfactual: {_trunc(cc.get('counterfactual', '') or '(none)', 400)}

[Confirmed live code at the attributed locus]
{code_state_block.strip() or '(none lifted)'}

[Pooled evidence windows]
{evidence}

[Output contract] Output ONLY this JSON object — no prose, no fences, nothing else:
{{ "refuted": true, "why": "<one line: the concrete {lens} reason the attribution does NOT survive — or, if it does survive, set refuted=false and say why it holds>" }}
"""


def _lens_refute_once(prompt: str, provider: str, model: str, pk: dict[str, Any],
                      ledger, timeout: int, lens: str) -> dict[str, Any] | None:
    """One refuter model call (with a JSON-only reparse). Never raises; None on no JSON.

    Recorded to the ledger under the ``swarm`` role so the cost lands in the cheap-tier
    accounting (these are swarm-tier 120b calls, not the premium converge call).
    """
    attempt_prompt = prompt
    parsed: dict[str, Any] | None = None
    for attempt in range(2):
        call_id = ledger.begin_call("swarm", f"lens:{lens}", provider, model,
                                    attempt_prompt) if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=None, timeout=timeout,
                             on_start=(lambda: ledger.mark_running(call_id))
                             if (ledger is not None and call_id is not None) else None,
                             **pk)
        except Exception as e:
            logger.warning("converge: lens %s worker failed: %s", lens, e)
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
                logger.warning("converge: lens %s — no parseable JSON after retry", lens)
    return parsed if isinstance(parsed, dict) else None


def _lens_refute(res: ConvergeResult, seed_text: str, located: list[dict[str, Any]],
                 windows: list[dict[str, Any]], code_state_block: str,
                 lenses: list[str], provider: str, model: str, pk: dict[str, Any],
                 ledger, timeout: int, min_refute: int) -> ConvergeResult:
    """Adversarial best-of-N lens refutation of an ACTIONABLE converge attribution.

    Assumes the caller already gated on ``res`` being a converged, causal-``consistent``
    attribution with a file. Runs ONE refuter per lens; a refuter that fails to parse
    ABSTAINS (counts as "survives") so the panel can only demote on a REAL majority — never
    on a flaky call. A majority (``min_refute`` votes, or simple majority when 0) demotes
    ``converged`` to False and stamps ``res.lens_check``. Never raises.
    """
    attributed = res.attributed_defect or {}
    # Design-change carve-out (M037/T905): mirror the converger's own carve-out into the
    # adversarial panel. When a located fragment is a DESIGN-CHANGE site, no refuter may
    # break the attribution on spec-conformance grounds. Same trigger the converger uses
    # (any located fragment tagged design_change) — verdict.type lives under ["verdict"].
    design_change_site = any(
        str((v.get("verdict") or {}).get("type", "")).strip().lower() == "design_change"
        for v in (located or []))
    if design_change_site:
        logger.info("converge: lens panel — DESIGN-CHANGE carve-out active "
                    "(refuters forbidden from spec-conformance refutation)")
    # Winning-path carve-out: when the attribution is on the converge's registration-order-
    # resolved winning path, the cheap refuters must not break it by citing a shadowed sibling
    # as "the real handler/producer" (M036: the dead get_projects_with_modules chain refuting
    # the live db/projects.py). The converge knows the resolved live path; the 120b refuter
    # does not. Same alignment test as the field-provenance abstain. Liveness/shadowing
    # refutation is softened on these loci; omission/reproduction with positive evidence still
    # bites.
    wp_files = {n.get("file", "") for n in (res.winning_path or []) if n.get("file")}
    af = (attributed or {}).get("file", "")
    winning_path_site = bool(af) and any(_aligns(af, wf) for wf in wp_files)
    if winning_path_site:
        logger.info("converge: lens panel — WINNING-PATH carve-out active for %s "
                    "(refuters told not to cite shadowed siblings as the live path)", af)
    ev_lines: list[str] = []
    for w in (windows or [])[:_LENS_MAX_EVIDENCE]:
        ev_lines.append(f"--- {w.get('file')}:{w.get('lines')}")
        ev_lines.append(_trunc(w.get("text", ""), _EVIDENCE_CHARS))
    evidence = "\n".join(ev_lines) or "(no evidence windows)"

    votes: list[dict[str, Any]] = []
    for lens in lenses:
        desc = _LENS_DEFINITIONS.get(lens) or \
            f"Try to refute the attribution on {lens} grounds; default to refuted when unsure."
        prompt = _build_lens_prompt(seed_text, attributed, res.causal_check, lens, desc,
                                    code_state_block, evidence, design_change_site,
                                    winning_path_site)
        parsed = _lens_refute_once(prompt, provider, model, pk, ledger, timeout, lens)
        refuted = bool(parsed.get("refuted")) if isinstance(parsed, dict) else False
        why = str(parsed.get("why", "") or "") if isinstance(parsed, dict) else ""
        votes.append({"lens": lens, "refuted": refuted, "why": why,
                      "answered": parsed is not None})

    n = len(votes)
    refutes = [v for v in votes if v["refuted"]]
    threshold = min_refute if (min_refute and min_refute > 0) else (n // 2 + 1)
    demoted = n > 0 and len(refutes) >= threshold
    # Winning-path override (M036): the converge's registration-order resolution is STRONGER
    # evidence of which datasource is live than a cheap 120b refuter panel — which is routinely
    # fooled by a shadowed look-alike sibling (the dead get_projects_with_modules chain refuted
    # the live db/projects.py 3/3, all three citing that sibling as "the real path"). When the
    # attribution is on the winning path it CANNOT be demoted by the panel: the structural
    # grounding outranks the adversarial vote. Prompt-level softening alone proved insufficient
    # against the shadowing confusion (the refuters kept refuting), so the override is
    # structural. Votes are still recorded for visibility, and a winning-path node remains
    # subject to the omission / multi-root guards and the apply-side red→green backstop — so a
    # genuinely incomplete fix is still caught, just not by this shadow-blind panel.
    wp_override = demoted and winning_path_site
    if wp_override:
        logger.info("converge: lens panel demote OVERRIDDEN — %s is the winning-path live "
                    "datasource; %d/%d refutation(s) (shadowed-sibling confusion) do not "
                    "outrank registration-order grounding — held converged", af,
                    len(refutes), n)
        demoted = False
    # Borderline = the outcome would FLIP if a single lens had voted the other way (the
    # run-to-run wobble seen live on a genuinely-incomplete attribution). Surfaced so a
    # marginal demote/survive is visible rather than reading as a confident verdict.
    borderline = (not wp_override) and n > 0 and len(refutes) in (threshold - 1, threshold)
    res.lens_check = {
        "lenses": [v["lens"] for v in votes],
        "votes": votes,
        "refuted_votes": len(refutes),
        "of": n,
        "threshold": threshold,
        "borderline": borderline,
        "winning_path_override": wp_override,
        "verdict": "refuted" if demoted else "survived",
    }
    if borderline:
        logger.info("converge: lens panel BORDERLINE (%d/%d refuted, threshold %d) — "
                    "one vote from flipping", len(refutes), n, threshold)
    if demoted:
        res.converged = False
        # Neutralize the causal verdict too: a refuted attribution must NOT keep a
        # ``consistent`` stamp. The honey writes ``causal_verdict`` into its attribution
        # marker INDEPENDENTLY of ``converged``, and downstream gates (e.g. specify's
        # _apply_layer_consistency_gate) key off ``causal_verdict == "consistent"`` alone —
        # leaving it consistent would let the refuted locus still be trusted, bypassing this
        # demotion. The lenses proved the code does not produce the symptom, which is exactly
        # this codebase's definition of ``contradicted``; stamp it so the refutation is
        # honored end to end. ``lens_refuted`` marks WHY (vs a model-authored contradiction).
        if isinstance(res.causal_check, dict):
            res.causal_check["verdict"] = "contradicted"
            res.causal_check["lens_refuted"] = True
        reasons = "; ".join(f"[{v['lens']}] {v['why']}" for v in refutes if v["why"])
        res.summary = _trunc(
            (res.summary or "") + f" | LENS-REFUTED {len(refutes)}/{n}: " + reasons, 1000)
        logger.info("converge: lens panel REFUTED attribution %s:%s (%d/%d ≥ %d) — "
                    "demoting converged→False (routed to reinvestigation)",
                    attributed.get("file"), attributed.get("lines"),
                    len(refutes), n, threshold)
    else:
        logger.info("converge: lens panel — attribution SURVIVED (%d/%d refuted, need %d)",
                    len(refutes), n, threshold)
    return res


def _apply_omission_lead(res: ConvergeResult, *, seed_text: str,
                         winning_path: list[dict[str, Any]], located: list[dict[str, Any]],
                         unlocated: list[dict[str, Any]], windows: list[dict[str, Any]],
                         known: set[str], bundles: list[dict[str, Any]],
                         code_root: str | None, provider: str, model: str,
                         pk: dict[str, Any], ledger, timeout: int, max_calls: int,
                         k: int, max_hops: int, data_block: str, db_available: bool,
                         db_schema: str, code_state_block: str, http_binding_block: str,
                         fragment_fact_block: str) -> ConvergeResult:
    """Name the uncovered winning-path node as a missing_link lead, then (when a code_root is
    given) run ONE free scoped re-retrieve + re-converge over it. The negative-space probe the
    located-only stitch structurally cannot do (① winning_path − located). Adopts only on a
    real convergence; else leaves the named lead. Caller gates on ``not converged and not
    missing_link``. Reused by BOTH the pre-arbiter fallback and the post-lens redirect.
    """
    omission = _winning_path_omission(winning_path, located, known)
    if not omission:
        return res
    res.missing_link = omission
    need_d = omission.get("need") or {}
    logger.info("converge: omission nominator — winning-path node %s uncovered by any "
                "located fragment; synthesized missing_link lead", need_d.get("file_globs"))
    if not (code_root and max_calls > 1):
        return res
    need = FollowupNeed(
        axis_id="CONVERGE_OMISSION",
        symbols=[str(s) for s in (need_d.get("symbols") or [])],
        greps=[str(g) for g in (need_d.get("greps") or [])],
        file_globs=[str(g) for g in (need_d.get("file_globs") or [])])
    try:
        fu = retrieve_followup(need, code_root, k=k, max_hops=max_hops)
    except Exception as e:  # local retrieve must never crash converge
        logger.warning("converge: omission follow-up retrieve failed: %s", e)
        return res
    if not fu:
        return res
    extra = list(fu.get("seeds") or []) + list(fu.get("call_chain") or [])
    merged = _dedup_windows(extra + windows)
    known2 = known | {_norm(w.get("file", "")) for w in extra if w.get("file")}
    res2 = _converge_once(seed_text, located, unlocated, merged, known2,
                          provider, model, pk, ledger, timeout,
                          data_state_block=data_block,
                          db_available=db_available, db_schema=db_schema,
                          code_state_block=_lift_live_code(located, code_root)
                          or code_state_block,
                          http_binding_block=_http_binding_bridges(
                              located, merged, code_root) or http_binding_block,
                          fragment_fact_block=_fragment_fact_cards(
                              located, merged, bundles, code_root) or fragment_fact_block)
    logger.info("converge: omission re-pass → %s", res2.summary)
    if res2.converged:
        res2.winning_path = winning_path
        res = res2
    return res


def run_converge(*, seed_text: str, verdicts: list[dict[str, Any]],
                 bundles: list[dict[str, Any]], provider: str, model: str,
                 code_root: str | None = None, ledger=None,
                 provider_kwargs: dict | None = None, timeout: int = 180,
                 min_located: int = 2, max_calls: int = 2,
                 k: int = 6, max_hops: int = 2, db_conn=None,
                 split_enabled: bool = False, split_max_loci: int = 4,
                 split_provider: str = "", split_model: str = "",
                 lens_lenses: list[str] | None = None, lens_provider: str = "",
                 lens_model: str = "", lens_min_refute: int = 0) -> ConvergeResult:
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
    winning_path = _winning_http_path_nodes(bundles)
    located = _located(verdicts)
    for lifted in _winning_producer_loci(winning_path):
        lf = (lifted.get("verdict") or {}).get("file", "")
        ll = (lifted.get("verdict") or {}).get("lines", "")
        if not any(
            _aligns(lf, (item.get("verdict") or {}).get("file", ""))
            and str((item.get("verdict") or {}).get("lines", "")) == str(ll)
            for item in located
        ):
            located.append(lifted)
    if winning_path:
        logger.info("converge: winning HTTP path grounded %d node(s), lifted %d "
                    "response producer locus/loci",
                    len(winning_path),
                    sum(1 for item in located
                        if str(item.get("axis_id", "")).startswith("HTTP_WINNING_PATH:")))
    if len(located) < min_located:
        return ConvergeResult(
            summary=f"skipped: {len(located)} located verdict(s) < min_located={min_located}",
            winning_path=winning_path)

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

    # FE→BE HTTP-edge grounding (N183): resolve fetch-URL literals to the BE routes they
    # hit so converge can stitch a FE mapping to its BE getter across the request boundary
    # instead of reporting a missing_link. Free, deterministic, fail-open (empty → no-op).
    http_binding_block = _http_binding_bridges(located, windows, code_root)
    if http_binding_block:
        logger.info("converge: HTTP-edge grounding resolved %d FE→BE binding(s)",
                    http_binding_block.count("- FE client"))

    fragment_fact_block = _fragment_fact_cards(located, windows, bundles, code_root)

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
                             code_state_block=code_state_block,
                             http_binding_block=http_binding_block,
                             fragment_fact_block=fragment_fact_block)
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
                              db_schema=db_schema, code_state_block=code_state_block,
                              http_binding_block=http_binding_block,
                              fragment_fact_block=fragment_fact_block)
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
                # Recompute the FE→BE bridge over the MERGED windows: the follow-up may
                # have just fetched the BE route file that completes the missing link.
                res2 = _converge_once(seed_text, located, unlocated, merged, known2,
                                      provider, model, pk, ledger, timeout,
                                      data_state_block=data_block,
                                      db_available=db_available, db_schema=db_schema,
                                      code_state_block=_lift_live_code(located, code_root)
                                      or code_state_block,
                                      http_binding_block=_http_binding_bridges(
                                          located, merged, code_root) or http_binding_block,
                                      fragment_fact_block=_fragment_fact_cards(
                                          located, merged, bundles, code_root)
                                      or fragment_fact_block)
                logger.info("converge: re-pass → %s", res2.summary)
                # Adopt the re-pass only if it actually converged; otherwise keep the
                # first result, which at least named the missing link for the author.
                if res2.converged:
                    res = res2

    # ── Redirect re-stitch on a LEADLESS contradiction (M035). The missing-link re-pass
    # above only fires when the model NAMED where to look next. A flaky converger instead
    # rules ``contradicted`` (the suspected node provably can't produce the symptom) and
    # leaves missing_link null — dead-ending at "not here", so the honey ships the refuted
    # node and specify, finding no bug there, defers → needs_reinvestigation (the punt the
    # operator's first principle forbids). When a LOCATED front-end render/binding fragment
    # distinct from the refuted node exists, the symptom's home is almost certainly there
    # (a contradicted backend/query node can NEVER produce a render symptom — the prompt's
    # own SYMPTOM DOMAIN rule). So we deterministically RE-STITCH with the refuted node
    # EXCLUDED, forcing the converger to attribute among the remaining fragments — the
    # redirect the prompt's [Keep hunting] asks for, made to fire even when the model
    # forgot the lead. No new retrieve (free re-ask over the same evidence); ONE converge
    # call; adopt only if it actually converges. Gated tightly so a plain leadless
    # contradiction WITH no FE alternative still dead-ends honestly (unchanged). Kill via
    # HIVE_NO_REDIRECT_RESTITCH.
    if (not res.converged and not res.missing_link and max_calls > 1
            and not os.environ.get("HIVE_NO_REDIRECT_RESTITCH")
            and (res.causal_check or {}).get("verdict") == "contradicted"
            and res.attributed_defect):
        rfile = _norm((res.attributed_defect or {}).get("file", ""))
        fe_targets = [v for v in located
                      if _is_fe_file((v.get("verdict") or {}).get("file", ""))
                      and not _aligns((v.get("verdict") or {}).get("file", ""), rfile)]
        if rfile and fe_targets:
            remaining = [v for v in located
                         if not _aligns((v.get("verdict") or {}).get("file", ""), rfile)]
            refuted_block = _render_refuted_block(res)
            logger.info("converge: leadless contradiction at %s, but %d located FE "
                        "render locus/loci exist — re-stitching with the refuted node "
                        "excluded", rfile, len(fe_targets))
            res2 = _converge_once(seed_text, remaining, unlocated, windows, known,
                                  provider, model, pk, ledger, timeout,
                                  data_state_block=data_block,
                                  db_available=db_available, db_schema=db_schema,
                                  code_state_block=code_state_block,
                                  http_binding_block=http_binding_block,
                                  fragment_fact_block=fragment_fact_block,
                                  refuted_block=refuted_block)
            logger.info("converge: redirect re-stitch → %s", res2.summary)
            if res2.converged:
                res = res2

    # ── Deterministic OMISSION nominator (winning-path gap, ①). LAST deterministic fallback,
    # after the model's own missing-link re-pass AND the M035 contradiction-redirect have both
    # had their turn. A still-unconverged result with NO named lead may simply be BLIND to the
    # real node: the proven winning HTTP path runs through a hop NO axis located, so the cause
    # cannot be among the located fragments converge reasoned over (the located set never
    # reveals what is absent from it). Diff winning_path (should-exist) against located/known,
    # name the uncovered live node as a missing_link, then run the SAME free scoped re-retrieve
    # over it and re-converge — the negative-space probe the located-only stitch structurally
    # cannot do. Adopts only on a real convergence; else leaves the named lead for honey /
    # reinvestigation (better-aimed than a blank re-ask). Kill via HIVE_NO_OMISSION_LEAD.
    if (not res.converged and not res.missing_link
            and not os.environ.get("HIVE_NO_OMISSION_LEAD")):
        res = _apply_omission_lead(
            res, seed_text=seed_text, winning_path=winning_path, located=located,
            unlocated=unlocated, windows=windows, known=known, bundles=bundles,
            code_root=code_root, provider=provider, model=model, pk=pk, ledger=ledger,
            timeout=timeout, max_calls=max_calls, k=k, max_hops=max_hops,
            data_block=data_block, db_available=db_available, db_schema=db_schema,
            code_state_block=code_state_block, http_binding_block=http_binding_block,
            fragment_fact_block=fragment_fact_block)

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

    http_ds_windows = [s for b in (bundles or [])
                       for s in ((b.get("code_snippets") or []) + (b.get("call_chain") or []))
                       if isinstance(s, dict)]
    fp_windows = [s for b in (bundles or [])
                  for s in ((b.get("code_snippets") or []) + (b.get("call_chain") or []))
                  if isinstance(s, dict) and s.get("via") == "field-producer"]

    # Attach the winning path BEFORE the arbiter so the provenance guards can read it: the
    # field-provenance re-aim uses it to refuse demoting a winning-path (live-executed)
    # attribution as a name-decoy (the M036 omission case). The guards mutate ``res`` in
    # place, so an early assignment is visible throughout the arbiter; the post-arbiter
    # assignment below is kept as a defensive restore.
    res.winning_path = winning_path
    # One causal/provenance decision surface. The arbiter preserves every existing stamp
    # and kill-switch while making precedence explicit and housing P0/P4 as facets.
    res = _causal_provenance_arbiter(
        res, located,
        fp_windows=fp_windows,
        http_ds_windows=http_ds_windows,
        data_backed=data_backed,
        data_chain_broke=data_chain_broke,
        db_available=db_available,
        code_root=code_root,
        windows=http_ds_windows,
        min_located=min_located,
        split_origin=split_res is not None,
        stability_comparison=split_res if split_res is not res else None)

    # Carry the live-DB read onto whichever result we return so the honey can PASTE the
    # real rows (or honestly report that the read was attempted but returned nothing).
    if data_attempted:
        res.data_state_attempted = True
        res.data_state_block = data_block
        res.data_state_backed = data_backed
    res.winning_path = winning_path

    # ── Adversarial LENS refutation (swarm best-of-N) — the converge analog of judge's
    # best-of-N, pointed at REFUTATION. Runs ONCE on the FINAL adopted attribution and ONLY
    # when it is actionable (converged + causal ``consistent`` + an attributed file). N
    # independent refuters, each through a DISTINCT lens, try to break it; a majority demotes
    # converged→False so a wobbly / omission / shadow attribution routes to reinvestigation
    # instead of shipping as a fix. Added cost = exactly ``len(lenses)`` swarm calls; zero
    # when the lens set is empty (opt-in) or the verdict was not consistent. Kill-switch env
    # HIVE_NO_LENS_REFUTE. Never crashes converge — a failing panel keeps the attribution.
    lenses = [str(x) for x in (lens_lenses or []) if str(x).strip()]
    if (lenses and not os.environ.get("HIVE_NO_LENS_REFUTE")
            and res.converged
            and (res.causal_check or {}).get("verdict") == "consistent"
            and (res.attributed_defect or {}).get("file")):
        lp = lens_provider or provider
        lm = lens_model or model
        logger.info("converge: lens refutation panel (%s/%s) — %d lens(es) over "
                    "attribution %s:%s", lp, lm, len(lenses),
                    (res.attributed_defect or {}).get("file"),
                    (res.attributed_defect or {}).get("lines"))
        try:
            res = _lens_refute(res, seed_text, located, windows, code_state_block,
                               lenses, lp, lm, pk, ledger, timeout, lens_min_refute)
        except Exception as e:  # the adversarial layer must never crash converge
            logger.warning("converge: lens panel failed (kept attribution): %s", e)

    # ── Lever 1: a lens refutation is a RE-AIM, not a dead end. When the panel DEMOTED the
    # attribution, EXCLUDE the refuted locus and (a) re-stitch among the remaining located
    # fragments (the M035 contradiction-redirect pattern), then (b) if still unstitched, fall
    # to the omission nominator (① winning-path gap) — so "blocked a wrong fix" becomes
    # "blocked AND re-pointed" instead of dead-ending at an honest-NR. Bounded: at most ONE
    # redirect converge call + the omission nominator's one scoped re-retrieve, and ONLY when
    # the panel demoted. The redirect attribution is NOT re-lensed (bounded — no recursion).
    # The refuted locus is excluded only on a MAJORITY refutation (lens_min_refute governs the
    # panel), so a correct attribution is not cheaply discarded. Kill via HIVE_NO_LENS_REDIRECT.
    if (res.lens_check.get("verdict") == "refuted" and not res.converged and max_calls > 1
            and not os.environ.get("HIVE_NO_LENS_REDIRECT")):
        rfile = _norm((res.attributed_defect or {}).get("file", ""))
        remaining = [v for v in located
                     if not _aligns((v.get("verdict") or {}).get("file", ""), rfile)]
        if rfile and remaining:
            refuted_block = _render_refuted_block(res)
            logger.info("converge: lens-refuted %s — redirect re-stitch with it EXCLUDED "
                        "(%d remaining located)", rfile, len(remaining))
            res2 = _converge_once(seed_text, remaining, unlocated, windows, known,
                                  provider, model, pk, ledger, timeout,
                                  data_state_block=data_block, db_available=db_available,
                                  db_schema=db_schema, code_state_block=code_state_block,
                                  http_binding_block=http_binding_block,
                                  fragment_fact_block=fragment_fact_block,
                                  refuted_block=refuted_block)
            logger.info("converge: lens redirect re-stitch → %s", res2.summary)
            if res2.converged:
                res2.winning_path = winning_path
                res2.lens_check = res.lens_check  # carry the refutation record forward
                res = res2
        if not res.converged and not res.missing_link:
            res = _apply_omission_lead(
                res, seed_text=seed_text, winning_path=winning_path, located=located,
                unlocated=unlocated, windows=windows, known=known, bundles=bundles,
                code_root=code_root, provider=provider, model=model, pk=pk, ledger=ledger,
                timeout=timeout, max_calls=max_calls, k=k, max_hops=max_hops,
                data_block=data_block, db_available=db_available, db_schema=db_schema,
                code_state_block=code_state_block, http_binding_block=http_binding_block,
                fragment_fact_block=fragment_fact_block)

    return res
