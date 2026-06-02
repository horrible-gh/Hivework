"""Queen→retrieve bridge — lower a decompose *task* into a retriever ``SearchPlan``.

The decompose stage (queen) emits axes as ``{id, title, brief, depends_on,
[search_plan]}``. The local retriever needs a ``SearchPlan{keywords, file_globs,
doc_topics}`` to run its zero-cost FIND. This module is the missing spine between
them — the seam where the swarm drones used to sit (M004 redesign).

Hybrid strategy (A+B), chosen so the bridge never adds a model call:

  A. If the queen emitted a per-axis ``search_plan`` (extended decompose
     contract), use it — the queen has the domain judgement to name precise
     keywords (``type_code``, ``ORDER BY``) a text scraper would miss.
  B. Deterministically extract keywords/globs/topics from the brief text and
     MERGE with (A), so a thin or absent queen plan still yields a usable plan.
     The queen call quality varies run-to-run; the local layer guarantees a floor.

Zero model cost — pure text extraction. The judge's follow-up loop is the
model-side safety net for whatever the seed still misses (M004 §4); this bridge
only has to window the ENTRY region reliably, not localise the bug itself.
"""
from __future__ import annotations

import re
from typing import Any

from hive.retriever import SearchPlan

# ── Source-tree roots: a one-slash token is only a path if it is rooted here (or
#    carries an extension / wildcard / depth ≥2). Kills prose like "head/lookup".
_SRC_ROOTS = frozenset({
    "server", "client", "src", "app", "lib", "tests", "test", "packages",
    "modules", "api", "web", "frontend", "backend", "core",
})

# Multi-word SQL phrases are unambiguous signal (case-insensitive). Single SQL
# words ("WHERE", "SELECT") are only taken when they appear UPPERCASE in the
# brief — lowercase "where branches" is English noise, not a query token.
_SQL_PHRASES = (
    "order by", "group by", "is null", "is not null", "left join",
    "inner join", "outer join", "having",
)
_SQL_WORDS = ("SELECT", "WHERE", "LIMIT", "OFFSET", "DISTINCT", "NULL",
              "INSERT", "UPDATE", "DELETE", "JOIN")

# Design-doc reference codes (D030, TR872, DB004, M004 …) → doc_topics, not keywords.
_DOC_CODE_RE = re.compile(r"\b(?:D|R|M|TR|DB|DOC|FR|NFR)\d{2,4}[A-Za-z]?\b")

# A path-ish token: at least one slash, made of path chars.
_PATH_RE = re.compile(r"[A-Za-z0-9_*][A-Za-z0-9_.*\-]*(?:/[A-Za-z0-9_.*\-]+)+")
# Quoted literal (single/double/backtick), 2-40 chars — a literal grep string.
_QUOTED_RE = re.compile(r"""['"`]([^'"`\n]{2,40})['"`]""")
# Bare identifier candidate (incl. dotted member access).
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,6}$")

# Generic words that survive the code-ish filter only by accident — drop them.
_KW_STOP = frozenset({
    "fastapi", "vue", "vite", "json", "html", "http", "https", "url", "api",
    "sha", "shas",
})


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive de-dupe (keeps first spelling seen)."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        it = (it or "").strip().strip(".,;:)('\"")
        if not it:
            continue
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _norm_glob(p: str) -> str:
    """Normalise a raw path token into a ripgrep glob.

    Wildcards and extensioned files pass through; a bare directory becomes a
    recursive ``dir/**/*`` so its whole subtree is in scope. Runs of ``/`` are
    collapsed: an over-escaped Windows path (``C:\\\\…`` → ``C:\\…``) would
    otherwise become a doubled-slash glob ``C://…`` that matches nothing (T890).
    """
    p = p.strip().strip(".,;:)('\"").replace("\\", "/").rstrip("/")
    p = re.sub(r"/{2,}", "/", p)
    if not p:
        return ""
    if "*" in p or _EXT_RE.search(p):
        return p
    return p + "/**/*"


_WORDISH_SEG_RE = re.compile(r"[a-z][a-z0-9_]+")


def _looks_like_path(tok: str) -> bool:
    """Is this slash-bearing token actually a repo path (not prose like and/or,
    nor a slash-joined list of doc codes like ``D0xx/R0xx/TR863``)?"""
    if "*" in tok or _EXT_RE.search(tok):
        return True
    segs = tok.split("/")
    # A real path has at least one lowercase word-ish directory segment; a run of
    # doc codes (uppercase+digits) or prose initials has none.
    if not any(_WORDISH_SEG_RE.fullmatch(s) for s in segs):
        return False
    if tok.count("/") >= 2:
        return True
    return segs[0].lower() in _SRC_ROOTS


def extract_globs(text: str) -> list[str]:
    """Pull repo-relative path scopes out of free-text brief prose."""
    out: list[str] = []
    for m in _PATH_RE.finditer(text or ""):
        tok = m.group(0)
        if _looks_like_path(tok):
            g = _norm_glob(tok)
            if g:
                out.append(g)
    return _dedupe(out)


def _is_codeish(tok: str) -> bool:
    """True for snake_case / dotted / mixedCase identifiers — i.e. things a grep
    over source would hit, as opposed to Title-case English ("Provide")."""
    if len(tok) < 3:
        return False
    if "_" in tok or "." in tok:
        return True
    has_lower = any(c.islower() for c in tok)
    has_inner_upper = any(c.isupper() for c in tok[1:])
    return has_lower and has_inner_upper


def extract_keywords(text: str) -> list[str]:
    """Derive blind grep keywords from a brief: SQL tokens, quoted literals, and
    code-ish identifiers. Ordered strongest-signal first, then de-duped."""
    text = text or ""
    low = text.lower()
    sql: list[str] = []
    for ph in _SQL_PHRASES:
        if ph in low:
            sql.append(ph.upper())
    for w in _SQL_WORDS:
        if re.search(rf"\b{w}\b", text):  # case-SENSITIVE: only uppercase mentions
            sql.append(w)

    quoted = [q.strip() for q in _QUOTED_RE.findall(text) if q.strip()]

    idents: list[str] = []
    for m in _IDENT_RE.finditer(text):
        tok = m.group(0)
        if not _is_codeish(tok):
            continue
        if tok.lower() in _KW_STOP:
            continue
        # UPPER_SNAKE tokens in a brief are almost always axis-id cross-refs
        # ("use the files from FE_RENDER_PATH") or macros, not grep seeds.
        if "_" in tok and tok.replace("_", "").isupper():
            continue
        idents.append(tok)
        if "." in tok:  # also offer the trailing member name on its own
            tail = tok.rsplit(".", 1)[-1]
            if len(tail) >= 3 and tail.lower() not in _KW_STOP:
                idents.append(tail)

    return _dedupe(sql + quoted + idents)


def extract_doc_topics(text: str) -> list[str]:
    """Design-doc reference codes + multi-word quoted phrases → design-grep topics."""
    codes = _DOC_CODE_RE.findall(text or "")
    phrases = [q.strip() for q in _QUOTED_RE.findall(text or "")
               if " " in q.strip() and len(q.strip()) <= 40]
    return _dedupe(codes + phrases)


# ── Visibility-class symptom probe (N176) ──────────────────────────────────────
# A "X is not visible / is disabled / does not render" symptom is produced by the
# component TEMPLATE's conditional-render branch (a v-if/v-show that evaluates false,
# an unpopulated v-for option list, a :disabled bind) — NOT by the data/endpoint
# layer. But such a symptom is usually WORDED about data ("the module isn't accepted"),
# so the blind keyword extractor hunts only the data path and the template branch never
# enters the bundle: N176 had two runs conclude "the endpoint already accepts it / the
# module is bound" with 0 edits, while the real cause was the selector never rendering.
# The deterministic fix (NOT a queen-prompt plea): when the SEED carries a visibility
# symptom, add the template conditional-render directives as keywords so that wherever a
# front-end template file IS in a plan's scope, the branch governing the element's
# visibility is retrieved and judged. Where no template file is in scope these tokens
# match nothing — an honest no-op, never a fabricated front-end finding.
_VISIBILITY_SYMPTOM_RE = re.compile(
    r"not\s+(?:visible|render(?:ed|ing)?|showing|shown|display(?:ed|ing)?|"
    r"appear(?:ing|s)?)"
    r"|(?:isn'?t|aren'?t|doesn'?t|don'?t|won'?t|can'?t|no longer)\s+"
    r"(?:see|show|shown|render|rendered|appear|appears|display|displayed|visible)"
    r"|(?:greyed|grayed)\s*out|\bnot\s+enabled\b|\bgreyed\b|\bdisabled\b|\bhidden\b"
    r"|\binvisible\b"
    r"|안\s*보|보이지\s*않|표시되지\s*않|렌더(?:링)?\s*(?:안|되지\s*않)"
    r"|나타나지\s*않|노출되지\s*않|비활성",
    re.IGNORECASE,
)

# Low-noise template directives: these rarely false-match outside a front-end template,
# so adding them to a backend-scoped plan is a harmless no-op. v-for is included because
# an empty option list ("the selector shows nothing to pick") is the same symptom class.
_VISIBILITY_KEYWORDS = ("v-if", "v-show", "v-else-if", "v-else", "v-for", ":disabled")


def is_visibility_symptom(seed_text: str) -> bool:
    """True when the seed describes a 'not visible / disabled / not rendered' symptom.

    Deterministic. A false positive only ever causes the low-noise directive keywords
    below to be added (a no-op where no template is in scope), so the detector is allowed
    to be generous rather than risk missing the symptom class N176 flagged.
    """
    return bool(_VISIBILITY_SYMPTOM_RE.search(seed_text or ""))


def with_visibility_probe(plan: SearchPlan) -> SearchPlan:
    """Return a copy of ``plan`` with the template conditional-render directives added.

    Idempotent and order-preserving: the probe keywords are appended AFTER the plan's own
    keywords (the queen's precise terms still rank first) and de-duped case-insensitively.
    Free and deterministic; the caller gates this on :func:`is_visibility_symptom`.
    """
    have = {k.lower() for k in plan.keywords}
    extra = [kw for kw in _VISIBILITY_KEYWORDS if kw.lower() not in have]
    if not extra:
        return plan
    return SearchPlan(axis_id=plan.axis_id, keywords=plan.keywords + extra,
                      file_globs=plan.file_globs, doc_topics=plan.doc_topics)


def task_to_searchplan(task: dict[str, Any], *,
                       default_globs: list[str] | None = None,
                       max_keywords: int = 14) -> SearchPlan:
    """Lower one decompose task into a ``SearchPlan`` (hybrid queen ∪ local).

    Args:
        task: a decompose task dict ``{id|axis_id, title, brief, [search_plan]}``.
        default_globs: scope to fall back to when neither the queen nor the brief
            named any path (e.g. the recipe's repo-wide globs).
        max_keywords: cap on emitted keywords (ripgrep is free, but the judge
            bundle is capped, so an unbounded keyword list buys nothing).
    """
    axis_id = str(task.get("id") or task.get("axis_id") or "?")
    brief = "\n".join(str(task.get(k, "")) for k in ("title", "brief")).strip()

    qp = task.get("search_plan") or {}
    if not isinstance(qp, dict):
        qp = {}
    q_kw = [str(x) for x in (qp.get("keywords") or [])]
    q_globs = [str(x) for x in (qp.get("file_globs") or [])]
    q_topics = [str(x) for x in (qp.get("doc_topics") or [])]

    # A (queen) first so its precise terms win the de-dupe; B (local) augments.
    keywords = _dedupe(q_kw + extract_keywords(brief))[:max_keywords]
    globs = _dedupe([_norm_glob(g) for g in q_globs] + extract_globs(brief))
    if not globs:
        globs = list(default_globs or [])
    topics = _dedupe(q_topics + extract_doc_topics(brief))

    return SearchPlan(axis_id=axis_id, keywords=keywords,
                      file_globs=globs, doc_topics=topics)
