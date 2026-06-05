"""JUDGE stage — the one model call that turns a local FIND bundle into a verdict.

Pipeline position (retrieval redesign, M004):

  retrieve (local, free)  →  JUDGE (this)  →  [retrieve_followup (free)]  →  re-JUDGE

The crux experiment (M004 §6) showed drone *quality* lived in JUDGE + a shallow
keyword FIND, not in open-ended multi-turn agency: a capable judge, handed the
local bundle, can both render a verdict AND name the one callee/symbol it still
needs resolved. This module wires that — replacing the harness's *simulated*
``FollowupNeed`` (smoke/retrieval_experiment) with a real model call.

Budget (M004 §4, surfaced in hive.config.json ``judge``):
  - call 1 emits BOTH a preliminary verdict and an optional ``need``;
  - if ``max_calls_per_axis >= 2`` and the judge asked for a follow-up, we run
    one free ``retrieve_followup`` and spend a second call to RE-judge the
    merged bundle for the final verdict.
  So ``max_calls_per_axis=1`` costs one call (verdict only), ``=2`` costs the
  documented "1 judge + ≤1 re-judge". The cap is enforced here, not hidden in a
  counter — operators dial it in config.

Cost gate (the queen-1 vs judge-7 asymmetry fix): the second call is the only
reducible spend, so a free LOCAL grounding check decides whether it is worth it.
If call 1 already returned a ``located`` verdict whose file is *in the evidence
bundle*, we trust it and skip the re-judge — the common success path drops from
2 calls/axis to 1. Conversely, a ``located`` verdict citing a file the judge was
never shown is a hallucination (the judge is tool-less); it is downgraded to
``located=false`` and, if budget allows, the re-judge fires precisely there. So
locality both lowers cost on grounded axes and raises quality on hallucinated
ones — see ``_verdict_is_grounded``.

Like specify's effectiveness review, JUDGE never raises: a flaky/unparseable
model response degrades to ``located=false`` rather than crashing the pipeline.
"""
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from hive.parse import extract_first_json
from hive.providers import call_worker
from hive.retriever import FollowupNeed, retrieve_followup

logger = logging.getLogger("hive.judge")

# Prompt-budget guards — the whole redesign exists to protect the token budget,
# so the bundle is rendered COMPACT into the judge prompt, not dumped raw.
_MAX_SNIPPETS = 14
_SNIPPET_CHARS = 700
_GIT_CHARS = 400
_DESIGN_CHARS = 400
# Seed-naming signal: the judge can only NAME a follow-up against symbols it can
# SEE. A bare file list strips exactly that, so we surface a bounded sample of
# the keyword-hit LINES — declarations first (the nameable namespace), then a
# few call-expression hits for vocabulary.
_CALL_SITE_DEFS = 30
_CALL_SITE_OTHER = 12
_CALL_SITE_TEXT = 120

# When the model returns prose, fenced, or garbled output that even the repair
# pass in ``extract_first_json`` cannot salvage, one terse "JSON only" nudge
# usually recovers a clean object. This retry lives INSIDE the logical judge
# call: it is a real, paid model call (recorded to the ledger) but it does NOT
# advance the per-axis budget — ``max_calls_per_axis`` gates the expensive
# re-judge follow-up, not a transport-level reparse of the same verdict.
_JSON_ONLY_REMINDER = (
    "\n\n[Retry] Your previous response could not be parsed as JSON. Output ONLY "
    "the single JSON object specified in the output contract above — no prose, no "
    "explanation, no markdown code fences, nothing before or after it."
)


@dataclass
class JudgeVerdict:
    """The judge's localisation of the bug for one axis.

    ``located`` is the only field a downstream consumer must trust as a gate;
    ``file``/``lines``/``reason`` are the evidence. ``raw`` keeps the full parsed
    JSON so nothing the model said is silently dropped.
    """

    axis_id: str
    located: bool = False
    file: str = ""
    lines: str = ""
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _trunc(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "…(truncated)"


def summarize_bundle(bundle: dict[str, Any], *, include_call_sites: bool = True) -> str:
    """Render a FIND bundle into a compact, judge-readable text block.

    Code windows are the primary evidence; call_sites are reduced to a distinct
    file list (vocabulary the judge can NAME a follow-up against) rather than
    dumped line-by-line.
    """
    parts: list[str] = []
    snips = (bundle.get("code_snippets") or []) + (bundle.get("call_chain") or [])
    parts.append(f"## CODE WINDOWS ({len(snips)})")
    for s in snips[:_MAX_SNIPPETS]:
        via = f" via={s['via']}" if s.get("via") else ""
        hits = f" hits={s.get('hits')}" if s.get("hits") else ""
        parts.append(f"--- {s.get('file')}:{s.get('lines')}{via}{hits}")
        parts.append(_trunc(s.get("text", ""), _SNIPPET_CHARS))

    gh = bundle.get("git_history") or []
    if gh:
        parts.append(f"\n## GIT HISTORY ({len(gh)})")
        for g in gh:
            line = (g.get("blame", "") + "\n" + g.get("log", "")).strip()
            if line:
                parts.append(f"--- {g.get('file', '')}")
                parts.append(_trunc(line, _GIT_CHARS))

    de = bundle.get("design_excerpts") or []
    if de:
        parts.append(f"\n## DESIGN EXCERPTS ({len(de)})")
        for d in de:
            parts.append(f"--- {d.get('doc')}:{d.get('lines')} topics={d.get('topics')}")
            parts.append(_trunc(d.get("text", ""), _DESIGN_CHARS))

    if include_call_sites:
        sites = bundle.get("call_sites") or []
        freq: dict[str, int] = {}            # hits per file → prefer hot files
        for h in sites:
            freq[h.get("file", "")] = freq.get(h.get("file", ""), 0) + 1
        seen: set[tuple] = set()
        defs: list[tuple] = []
        others: list[tuple] = []
        for h in sites:
            key = (h.get("file"), h.get("line"))
            if key in seen:
                continue
            seen.add(key)
            txt = (h.get("text") or "").strip()
            row = (freq.get(h.get("file", ""), 0), h.get("file"), h.get("line"), txt)
            (defs if txt.startswith(("def ", "async def ", "class ")) else others).append(row)
        defs.sort(key=lambda r: -r[0])
        others.sort(key=lambda r: -r[0])
        picked = defs[:_CALL_SITE_DEFS] + others[:_CALL_SITE_OTHER]
        if picked:
            parts.append(
                f"\n## CANDIDATE SYMBOLS / KEYWORD-HIT LINES "
                f"({len(defs)} declarations, {len(others)} other hits; showing {len(picked)})")
            for _, f, ln, txt in picked:
                parts.append(f"{f}:{ln}: {_trunc(txt, _CALL_SITE_TEXT)}")

    return "\n".join(parts)


def _verdict_contract(want_need: bool) -> str:
    need_block = (
        ',\n  "need": { "symbols": ["<callee/def names to resolve>"], '
        '"greps": ["<literal/regex to locate>"], '
        '"file_globs": ["<optional narrowed scope>"] }'
        if want_need else ""
    )
    need_guidance = (
        "\nIf — and only if — you cannot yet point at the exact buggy lines because a "
        "callee/symbol referenced in the windows is not shown, fill ``need`` with the "
        "names/patterns to resolve (≤4 each). If the bug is already visible, set located=true "
        "and leave need empty.\n"
        if want_need else ""
    )
    return f"""{need_guidance}
[Output contract] Output ONLY this JSON object. No prose outside the JSON.
{{
  "verdict": {{ "located": true, "file": "<repo-relative path>", "lines": "<start-end>", "reason": "<one line: why THIS is the bug>" }}{need_block}
}}"""


def build_judge_prompt(axis_id: str, symptom: str, bundle_text: str,
                       *, want_need: bool, code_root: str = "",
                       seed_axis: bool = False) -> str:
    """Build the JUDGE prompt for one axis. ``want_need`` enables follow-up asks.

    The judge rules on the SUPPLIED bundle — it has no tools and must not try to
    read files or run commands (the bundle is the substitute for agentic FIND;
    re-exploring defeats the redesign and blows latency). When it lacks a piece,
    it asks for it via ``need`` (a bounded re-search), it does not go fetch it.

    ``seed_axis`` marks an axis the seed itself mandated (e.g. the injected
    SEED_ANCHOR): the ruling mandate below is stated for every axis, but a
    seed-mandated one gets an extra line because dismissing it is the costliest
    miss (N170: axis E "does get_pending call the wrong SQL key?" was waved off as
    a "simple lookup request" and never confirmed or refuted).
    """
    seed_line = (
        "\nThis axis is MANDATED BY THE SEED — the user specifically asked it to be "
        "investigated. A dismissal here is a defect: you MUST return an explicit "
        "confirm or refute.\n" if seed_axis else "")
    return f"""[Role] You are the JUDGE for Hivework axis "{axis_id}". You are given a \
locally-retrieved evidence bundle (keyword windows, call-chain hops, git history) and \
the reported symptom. Your job is to localise the bug: name the exact file and line range \
that must change. Judge EXECUTION REACHABILITY — that the lines actually run on the \
symptom's path — not merely that matching text exists.

[Constraints] You have NO tools. Do NOT attempt to read files or run commands. Decide \
ONLY from the evidence below and emit the JSON immediately. If a referenced callee/symbol \
you need is not shown, ask for it in ``need`` rather than trying to fetch it.

[Ruling mandate] The axis brief is a HYPOTHESIS about where/whether this symptom's \
defect lives — RULE on it. Return exactly one of:
  - CONFIRM: located=true, naming the exact buggy file/lines, OR
  - REFUTE: located=false with a concrete, evidence-based reason this code path is \
correct / not the cause (cite what in the evidence disproves the hypothesis).
You may NOT decline to rule. A brief phrased as a question ("does X call the wrong \
key?", "is the ORDER BY wrong?") still demands a confirm/refute answer about the \
symptom — do NOT dismiss it as "merely an informational/lookup request", "not itself \
a code defect", or "no source-line change needed" to sidestep judging. "located=false" \
must mean "refuted, because <evidence>", never "this was not a real question."{seed_line}

[Symptom / axis brief]
{symptom}

[Local evidence bundle]
{bundle_text}
{_verdict_contract(want_need)}
"""


def _call_and_parse(provider: str, model: str, prompt: str, *, cwd: str,
                    axis_id: str, stage: str, ledger=None,
                    timeout: int = 180, provider_kwargs: dict | None = None,
                    retry_on_unparseable: bool = True,
                    ) -> dict[str, Any] | None:
    """Call the model, record to ledger, return parsed JSON (or None). Never raises.

    On UNPARSEABLE output, retries once with a terse JSON-only reminder appended
    (see ``_JSON_ONLY_REMINDER``): both attempts are recorded to the ledger as
    'judge' calls, but the retry does not advance the caller's per-axis budget.
    A worker-level failure (timeout / provider error) is NOT retried — it returns
    ``None`` immediately, since a re-call is unlikely to fix transport breakage.
    """
    attempt_prompt = prompt
    attempts = 2 if retry_on_unparseable else 1
    for attempt in range(attempts):
        call_id = ledger.begin_call("judge", axis_id, provider, model, attempt_prompt) \
            if ledger is not None else None
        try:
            wr = call_worker(provider, model, attempt_prompt, cwd=cwd, timeout=timeout,
                             **(provider_kwargs or {}))
        except Exception as e:  # timeout, provider error, etc.
            logger.warning("judge: %s worker failed for %s: %s", stage, axis_id, e)
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
            return extract_first_json(wr.stdout)
        except ValueError:
            if attempt + 1 < attempts:
                logger.warning("judge: %s produced no parseable JSON for %s — "
                               "retrying once with JSON-only reminder", stage, axis_id)
                attempt_prompt = prompt + _JSON_ONLY_REMINDER
            else:
                logger.warning("judge: %s produced no parseable JSON for %s", stage, axis_id)
    return None


def _verdict_from(parsed: dict[str, Any] | None, axis_id: str) -> JudgeVerdict:
    if not isinstance(parsed, dict):
        return JudgeVerdict(axis_id=axis_id)
    v = parsed.get("verdict")
    if not isinstance(v, dict):
        return JudgeVerdict(axis_id=axis_id, raw=parsed)
    return JudgeVerdict(
        axis_id=axis_id,
        located=bool(v.get("located", False)),
        file=str(v.get("file", "")),
        lines=str(v.get("lines", "")),
        reason=str(v.get("reason", "")),
        raw=parsed,
    )


def _need_from(parsed: dict[str, Any] | None, axis_id: str,
               default_globs: list[str]) -> FollowupNeed | None:
    """Build a FollowupNeed from the judge's ``need`` block, or None if empty."""
    if not isinstance(parsed, dict):
        return None
    n = parsed.get("need")
    if not isinstance(n, dict):
        return None
    symbols = [str(s) for s in (n.get("symbols") or []) if str(s).strip()]
    greps = [str(g) for g in (n.get("greps") or []) if str(g).strip()]
    if not symbols and not greps:
        return None  # judge asked for nothing → no follow-up
    globs = [str(g) for g in (n.get("file_globs") or []) if str(g).strip()]
    return FollowupNeed(axis_id=axis_id, symbols=symbols, greps=greps,
                        file_globs=globs or list(default_globs))


def _norm_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip().strip("/").lower()


def _bundle_files(bundle: dict[str, Any]) -> set[str]:
    """Every file the judge was actually SHOWN in this bundle.

    The judge is tool-less and rules only on the bundle, so a verdict can be
    *grounded* only in a file that appears here. A cited file absent from this
    set was invented (hallucinated), not localised.
    """
    files: set[str] = set()
    for key in ("code_snippets", "call_chain", "call_sites"):
        for s in bundle.get(key) or []:
            f = _norm_path(s.get("file", ""))
            if f:
                files.add(f)
    for d in bundle.get("design_excerpts") or []:
        f = _norm_path(d.get("doc", ""))
        if f:
            files.add(f)
    for f in (bundle.get("stats") or {}).get("ranked_files") or []:
        nf = _norm_path(f)
        if nf:
            files.add(nf)
    return files


def _path_aligns(a: str, b: str) -> bool:
    """Path-segment-aligned equality or suffix (handles abs↔rel, basename-degrade)."""
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def _verdict_is_grounded(verdict: "JudgeVerdict", bundle: dict[str, Any]) -> bool:
    """Is a *located* verdict's cited file one the judge was actually shown?

    Purely from the bundle (no disk/model) — deterministic and free. Used both as
    a COST gate (a grounded localisation needs no second opinion → skip re-judge)
    and an anti-hallucination gate (an ungrounded ``located`` is downgraded).
    """
    if not verdict.located:
        return False
    vf = _norm_path(verdict.file)
    if not vf:
        return False
    return any(_path_aligns(vf, bf) for bf in _bundle_files(bundle))


def _merge_followup(bundle: dict[str, Any], fu: dict[str, Any]) -> dict[str, Any]:
    """Merge follow-up seeds/call-chain into the first-pass bundle (re-judge input)."""
    merged = dict(bundle)
    merged["code_snippets"] = (list(bundle.get("code_snippets") or [])
                               + list(fu.get("seeds") or [])
                               + list(fu.get("call_chain") or []))
    return merged


# Dismissal phrases: a located=false verdict whose reason rejects the QUESTION
# ("this isn't a real defect / just a lookup") rather than REFUTING the hypothesis
# with evidence. We don't flip the bit (the model may be right that it's unlocated)
# — we SURFACE it (free, deterministic) so a non-ruling is visible to the operator
# and the honey, and never silently accepted (N170 axis E). Matched on a
# dash-normalised, lowercased reason.
_DISMISSAL_MARKERS = (
    "informational request", "informational query", "not a code defect",
    "not a defect", "no source-line change", "no source line change",
    "no code change", "does not require a source", "not require a source-line",
    "merely a lookup", "merely an informational", "simply a lookup",
    "just a lookup", "just a query", "not itself a defect",
)


def _is_dismissal(reason: str) -> bool:
    """True when an unlocated reason DISMISSES the question instead of refuting it."""
    norm = (reason or "").lower()
    for dash in ("‐", "‑", "‒", "–", "—"):
        norm = norm.replace(dash, "-")
    return any(m in norm for m in _DISMISSAL_MARKERS)


def _cites_seed_file(cited: str, seed_files: set[str]) -> bool:
    """True when a verdict's cited file is one of the seed's named (on-disk) targets.

    Each axis runs its OWN retrieve, so a seed-named file present in one axis's
    bundle can be ABSENT from another's. A verdict citing such a file is then
    wrongly downgraded as 'absent from evidence' even though the file is real and
    seed-relevant (T892 axis A cited workflowViewState.ts — a seed target — and
    was dropped while axis C had it). Seed files are disk-confirmed upstream, so a
    citation to one is grounded, not a hallucination.
    """
    vf = _norm_path(cited)
    return bool(vf) and any(_path_aligns(vf, _norm_path(s)) for s in seed_files)


def run_judge(*, plan_bundle: dict[str, Any], symptom: str, axis_globs: list[str],
              code_root: str, provider: str, model: str, judge_cfg,
              ledger=None, provider_kwargs: dict | None = None,
              k: int = 6, max_hops: int = 2, timeout: int = 180,
              seed_files: set[str] | None = None,
              seed_axis: bool = False) -> dict[str, Any]:
    """End-to-end JUDGE for one axis: verdict (+ optional one re-judge after follow-up).

    Returns a comb dict::

        {axis_id, calls_made, verdict, need, followup_bundle, history[]}

    ``calls_made`` and the presence of ``followup_bundle`` reflect the budget in
    ``judge_cfg`` (``max_calls_per_axis``). The function never raises; on a flaky
    model it returns ``located=false`` with whatever it parsed.
    """
    axis_id = str(plan_bundle.get("axis_id", "?"))
    max_calls = max(1, int(getattr(judge_cfg, "max_calls_per_axis", 2)))
    want_need = max_calls >= 2
    history: list[dict[str, Any]] = []

    # The judge is single-shot and tool-less: it rules on the bundle, never
    # re-explores (that defeats the redesign and blows latency). Default to no
    # tools unless a caller deliberately overrides.
    pk = dict(provider_kwargs or {})
    pk.setdefault("available_tools", [])

    # ── Call 1: verdict + (optional) need, on the first-pass bundle.
    bundle_text = summarize_bundle(plan_bundle)
    prompt1 = build_judge_prompt(axis_id, symptom, bundle_text,
                                 want_need=want_need, code_root=code_root,
                                 seed_axis=seed_axis)
    parsed1 = _call_and_parse(provider, model, prompt1, cwd=code_root,
                              axis_id=axis_id, stage="judge1", ledger=ledger,
                              provider_kwargs=pk, timeout=timeout)
    calls_made = 1
    verdict = _verdict_from(parsed1, axis_id)
    need = _need_from(parsed1, axis_id, axis_globs) if want_need else None
    grounded1 = _verdict_is_grounded(verdict, plan_bundle)
    history.append({"stage": "judge1", "parsed": parsed1})

    # ── Cost gate (the asymmetry fix): the re-judge is the only reducible call.
    # Spend it only when it can change the answer — i.e. the judge asked for a
    # follow-up AND we do NOT already hold a grounded localisation. A located
    # verdict whose file is in the evidence is trusted as-is (saves the call); an
    # ungrounded or unlocated one is exactly what the follow-up exists to fix.
    check_bundle = plan_bundle
    followup_bundle = None
    if (want_need and need is not None and calls_made < max_calls
            and not (verdict.located and grounded1)):
        followup_bundle = retrieve_followup(need, code_root, k=k, max_hops=max_hops)
        merged = _merge_followup(plan_bundle, followup_bundle)
        prompt2 = build_judge_prompt(axis_id, symptom, summarize_bundle(merged),
                                     want_need=False, code_root=code_root,
                                     seed_axis=seed_axis)
        parsed2 = _call_and_parse(provider, model, prompt2, cwd=code_root,
                                  axis_id=axis_id, stage="judge2", ledger=ledger,
                                  provider_kwargs=pk, timeout=timeout)
        calls_made += 1
        history.append({"stage": "judge2", "parsed": parsed2})
        # The re-judge is the FINAL verdict — but never let a flaky re-judge
        # erase a good first verdict: only override if the re-judge parsed.
        v2 = _verdict_from(parsed2, axis_id)
        if v2.raw:
            verdict = v2
            check_bundle = merged

    # ── Anti-hallucination downgrade: the judge has no tools, so a ``located``
    # verdict citing a file it was never shown is invented, not localised. Never
    # emit it as located — downgrade to located=false with a flagged reason.
    # EXCEPTION: a seed-named target is real and disk-confirmed upstream, so a
    # citation to it is grounded even when this axis's own bundle didn't window it
    # (Defect 4a — seed files aren't guaranteed in every axis's retrieve).
    if verdict.located and not _verdict_is_grounded(verdict, check_bundle):
        if seed_files and _cites_seed_file(verdict.file, seed_files):
            logger.info("judge: [%s] cites seed-named file %s absent from this "
                        "axis's bundle — grounded via seed target (kept located)",
                        axis_id, verdict.file)
        else:
            logger.info("judge: [%s] verdict cites %s absent from evidence — downgraded",
                        axis_id, verdict.file)
            verdict = JudgeVerdict(
                axis_id=axis_id, located=False, file=verdict.file, lines=verdict.lines,
                reason="ungrounded (cited file absent from evidence bundle): "
                       + (verdict.reason or ""),
                raw=verdict.raw)

    # ── Dismissal backstop (N170 axis E): an unlocated verdict whose reason rejects
    # the QUESTION ("informational request, not a code defect") instead of REFUTING
    # the hypothesis is not a ruling. Free + deterministic: we don't flip the bit
    # (it may genuinely be unlocated) — we FLAG it so the non-ruling is visible in
    # the report/honey and not silently accepted, escalated for a seed-mandated axis.
    if not verdict.located and _is_dismissal(verdict.reason):
        logger.warning("judge: [%s]%s unlocated verdict DISMISSED the question rather "
                       "than refuting it (reason=%r) — flagged as an unruled dismissal",
                       axis_id, " SEED-MANDATED" if seed_axis else "", verdict.reason)
        verdict = JudgeVerdict(
            axis_id=axis_id, located=False, file=verdict.file, lines=verdict.lines,
            reason="[unruled-dismissal: question waved off, not refuted with evidence] "
                   + (verdict.reason or ""),
            raw=verdict.raw)

    return {
        "axis_id": axis_id,
        "calls_made": calls_made,
        "verdict": verdict,
        "need": need,
        "followup_bundle": followup_bundle,
        "history": history,
    }


def _merge_vote_followups(combs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pool every vote's follow-up windows into ONE bundle (deduped), or None.

    A candidate located by vote *j* may cite a file only vote *j*'s follow-up
    retrieve windowed; converge pools evidence across axes but per axis it gets a
    single bundle. So we union the follow-up seeds/call-chain from all votes here
    — every candidate then has its supporting window in the evidence converge sees,
    not just the representative vote's. Deduped by (file, lines); ``None`` when no
    vote ran a follow-up (mirrors run_judge's ``followup_bundle is None``).
    """
    seeds: list[dict[str, Any]] = []
    call_chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    any_fu = False
    for c in combs:
        fu = c.get("followup_bundle")
        if not fu:
            continue
        any_fu = True
        for dst, key in (("seeds", "seeds"), ("call_chain", "call_chain")):
            target = seeds if dst == "seeds" else call_chain
            for s in fu.get(key) or []:
                sig = (_norm_path(s.get("file", "")), str(s.get("lines", "")))
                if sig in seen:
                    continue
                seen.add(sig)
                target.append(s)
    if not any_fu:
        return None
    return {"seeds": seeds, "call_chain": call_chain, "stats": {}}


def run_judge_votes(*, votes: int, **judge_kwargs) -> dict[str, Any]:
    """Best-of-N JUDGE for one axis: N INDEPENDENT verdicts, UNIONed (M010 §5).

    The judge is noisy — in the equal-budget A/B a real locus was sometimes hit by
    only 1 of N passes (axis A: 1/7), and the cheap-but-many-passes model won *per
    dollar* precisely because more votes eventually caught those rare hits. This
    wires that into the operating pipeline: each pass is a full, independent
    :func:`run_judge` (same re-judge budget — exactly the harness that validated
    the win, ``smoke/judge_ab_budget.py``); the located loci are then **unioned,
    not voted by majority**. Majority would discard the 1/N rare-but-correct hit —
    the very thing N passes exist to catch — so every distinct located ``(file,
    lines)`` survives as a candidate and the DOWNSTREAM converge causal gate, not a
    vote count, controls precision. N raises recall; converge pays for it.

    Each vote is SINGLE-SHOT when voting (``votes > 1``): the per-vote re-judge
    (``max_calls_per_axis=2``) is REDUNDANT under best-of-N — the union across N
    votes already supplies the "look again" the re-judge gives within one vote. A
    live N176 A/B confirmed single-shot voting matches re-judge voting on recall at
    ~half the calls, so we force ``max_calls_per_axis=1`` inside voting and leave
    the re-judge to the ``votes=1`` single-judgment path (config's value still
    governs that). Net: ``votes > 1`` ⇒ a DETERMINISTIC ``axes × votes`` call count.

    ``votes <= 1`` degrades EXACTLY to one ``run_judge`` (same calls, same verdict,
    a one-element candidate list, untouched follow-up bundle) — the feature is
    opt-in and zero-cost at N=1.

    Returns the representative pass's ``run_judge`` comb dict (so existing callers
    keep reading ``verdict``/``calls_made``/``need``/``followup_bundle``), with:
      - ``calls_made``    summed across all votes,
      - ``votes``         N actually attempted,
      - ``located_votes`` how many passes located,
      - ``candidates``    distinct located ``JudgeVerdict`` (the union), most-voted
                          file first — the fragments converge should reason over,
      - ``file_tally``    ``{normalised file: vote count}`` for logging/report,
      - ``followup_bundle`` the POOLED follow-up windows of all votes (N>1).
    The representative ``verdict`` is a located verdict from the most-voted file
    (ties → earliest pass); when no pass located, the first pass's unlocated
    verdict — so its refute/dismissal reason still reaches the honey.
    """
    n = max(1, int(votes))
    # Voting ⇒ single-shot per vote: drop the redundant per-vote re-judge so the
    # union (not a second call) does the "look again". Leaves votes=1 untouched.
    if n > 1:
        jc = judge_kwargs.get("judge_cfg")
        if jc is not None and int(getattr(jc, "max_calls_per_axis", 1)) > 1:
            judge_kwargs = dict(judge_kwargs)
            judge_kwargs["judge_cfg"] = dataclasses.replace(jc, max_calls_per_axis=1)
    combs = [run_judge(**judge_kwargs) for _ in range(n)]
    located = [c["verdict"] for c in combs if c["verdict"].located]

    # Vote tally by normalised file (first-seen order → stable tie-breaking).
    tally: dict[str, int] = {}
    order: list[str] = []
    for v in located:
        f = _norm_path(v.file)
        if f not in tally:
            tally[f] = 0
            order.append(f)
        tally[f] += 1

    # Union of distinct located loci, deduped by (file, lines), ranked by the
    # file's vote count then first-seen order. This is the set converge filters.
    seen: set[tuple[str, str]] = set()
    candidates: list[JudgeVerdict] = []
    for v in sorted(located, key=lambda v: (-tally[_norm_path(v.file)],
                                            order.index(_norm_path(v.file)))):
        key = (_norm_path(v.file), (v.lines or "").strip())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(v)

    # Representative comb: a located pass on the most-voted file, else the first
    # pass (keeps its unlocated reason). candidates[0] is already the top file.
    if candidates:
        top = _norm_path(candidates[0].file)
        rep = next(c for c in combs
                   if c["verdict"].located and _norm_path(c["verdict"].file) == top)
    else:
        rep = combs[0]

    out = dict(rep)
    out["calls_made"] = sum(c["calls_made"] for c in combs)
    out["votes"] = n
    out["located_votes"] = len(located)
    out["candidates"] = candidates
    out["file_tally"] = tally
    # Only re-pool when there was actually more than one vote — at N=1 keep the
    # single pass's own bundle untouched (exact run_judge equivalence).
    if n > 1:
        out["followup_bundle"] = _merge_vote_followups(combs)
    return out
