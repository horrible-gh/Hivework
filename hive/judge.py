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

Like specify's effectiveness review, JUDGE never raises: a flaky/unparseable
model response degrades to ``located=false`` rather than crashing the pipeline.
"""
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
                       *, want_need: bool, code_root: str = "") -> str:
    """Build the JUDGE prompt for one axis. ``want_need`` enables follow-up asks.

    The judge rules on the SUPPLIED bundle — it has no tools and must not try to
    read files or run commands (the bundle is the substitute for agentic FIND;
    re-exploring defeats the redesign and blows latency). When it lacks a piece,
    it asks for it via ``need`` (a bounded re-search), it does not go fetch it.
    """
    return f"""[Role] You are the JUDGE for Hivework axis "{axis_id}". You are given a \
locally-retrieved evidence bundle (keyword windows, call-chain hops, git history) and \
the reported symptom. Your job is to localise the bug: name the exact file and line range \
that must change. Judge EXECUTION REACHABILITY — that the lines actually run on the \
symptom's path — not merely that matching text exists.

[Constraints] You have NO tools. Do NOT attempt to read files or run commands. Decide \
ONLY from the evidence below and emit the JSON immediately. If a referenced callee/symbol \
you need is not shown, ask for it in ``need`` rather than trying to fetch it.

[Symptom / axis brief]
{symptom}

[Local evidence bundle]
{bundle_text}
{_verdict_contract(want_need)}
"""


def _call_and_parse(provider: str, model: str, prompt: str, *, cwd: str,
                    axis_id: str, stage: str, ledger=None,
                    timeout: int = 180, provider_kwargs: dict | None = None,
                    ) -> dict[str, Any] | None:
    """Call the model, record to ledger, return parsed JSON (or None). Never raises."""
    try:
        wr = call_worker(provider, model, prompt, cwd=cwd, timeout=timeout,
                         **(provider_kwargs or {}))
    except Exception as e:  # timeout, provider error, etc.
        logger.warning("judge: %s worker failed for %s: %s", stage, axis_id, e)
        return None

    if ledger is not None:
        ledger.record_call("judge", axis_id, provider, model,
                           prompt=prompt, output=wr.stdout, latency_s=wr.latency_s,
                           ok=wr.exit_code == 0,
                           err=wr.stderr[:200] if wr.exit_code != 0 else "")

    try:
        return extract_first_json(wr.stdout)
    except ValueError:
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


def _merge_followup(bundle: dict[str, Any], fu: dict[str, Any]) -> dict[str, Any]:
    """Merge follow-up seeds/call-chain into the first-pass bundle (re-judge input)."""
    merged = dict(bundle)
    merged["code_snippets"] = (list(bundle.get("code_snippets") or [])
                               + list(fu.get("seeds") or [])
                               + list(fu.get("call_chain") or []))
    return merged


def run_judge(*, plan_bundle: dict[str, Any], symptom: str, axis_globs: list[str],
              code_root: str, provider: str, model: str, judge_cfg,
              ledger=None, provider_kwargs: dict | None = None,
              k: int = 6, max_hops: int = 2, timeout: int = 180) -> dict[str, Any]:
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
                                 want_need=want_need, code_root=code_root)
    parsed1 = _call_and_parse(provider, model, prompt1, cwd=code_root,
                              axis_id=axis_id, stage="judge1", ledger=ledger,
                              provider_kwargs=pk, timeout=timeout)
    calls_made = 1
    verdict = _verdict_from(parsed1, axis_id)
    need = _need_from(parsed1, axis_id, axis_globs) if want_need else None
    history.append({"stage": "judge1", "parsed": parsed1})

    followup_bundle = None
    # ── Optional Call 2: re-judge the merged bundle, if budget + a need exist.
    if want_need and need is not None and calls_made < max_calls:
        followup_bundle = retrieve_followup(need, code_root, k=k, max_hops=max_hops)
        merged = _merge_followup(plan_bundle, followup_bundle)
        prompt2 = build_judge_prompt(axis_id, symptom, summarize_bundle(merged),
                                     want_need=False, code_root=code_root)
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

    return {
        "axis_id": axis_id,
        "calls_made": calls_made,
        "verdict": verdict,
        "need": need,
        "followup_bundle": followup_bundle,
        "history": history,
    }
