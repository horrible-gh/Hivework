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
