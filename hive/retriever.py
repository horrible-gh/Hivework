"""Local retriever — FIND-stage replacement for swarm drones (M004 redesign).

Given an axis *search plan* ``{keywords, file_globs, doc_topics}``, produce an
evidence *bundle* using purely local, zero-cost tools:

    ripgrep  +  ±k file read  +  git blame/log  +  design-doc grep

NO model calls happen here. The bundle is what a single downstream JUDGE call
will consume in place of a self-driven multi-turn copilot drone (2-6 min,
per-internal-turn billing). See 110_memo/M004 for the cost rationale.

Bundle shape::

    {
      "axis_id": str,
      "code_snippets":   [{"file","lines","text","hits":[kw...]}],
      "call_sites":      [{"file","line","text","keyword"}],   # raw grep hits
      "git_history":     [{"file","blame","log"}],
      "design_excerpts": [{"doc","lines","text","topic"}],
      "stats": {...},
    }
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

# identifier immediately followed by "(" — a call site (optionally x.method()).
_CALL_RE = re.compile(r"(?:\.|\b)([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
# generic builtins / stdlib noise not worth a def-lookup hop.
_CALL_SKIP = frozenset({
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "range", "open", "super", "isinstance", "getattr", "setattr",
    "hasattr", "format", "join", "append", "get", "items", "keys", "values",
    "split", "strip", "replace", "enumerate", "zip", "sorted", "type",
})


@dataclass
class SearchPlan:
    """What the queen/반장 emits per axis (blind to the answer)."""

    axis_id: str
    keywords: list[str]
    file_globs: list[str]
    doc_topics: list[str] = field(default_factory=list)


@dataclass
class FollowupNeed:
    """A JUDGE-directed re-search request (M004 §2 step-3 ``need:[...]``).

    The crux experiment (§6) showed every residual MISS is a call-chain hop in a
    low-keyword-density region the *blind* density-seeded follow never windowed
    — the follow mechanism works, but its SEED is starved. The fix (§4
    correction) is to let the JUDGE, having seen the first bundle (e.g. the entry
    handler), NAME the callee/symbol it wants resolved. This dataclass carries
    that named seed back to the retriever for ONE bounded local re-search.

    Fields:
        symbols: callee/def names to resolve — grep ``def <symbol>`` then follow.
        greps:   literal/regex patterns to locate (e.g. ``db_docs.create(``);
                 their windows seed the follow.
        file_globs: scope (reuse the axis globs unless the judge narrows them).
    """

    axis_id: str
    symbols: list[str] = field(default_factory=list)
    greps: list[str] = field(default_factory=list)
    file_globs: list[str] = field(default_factory=list)


def _run(cmd: list[str], cwd: str, timeout: int = 30) -> str:
    """Run a local command, return stdout ('' on failure). Never raises."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return p.stdout or ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _norm_glob(g: str, root: str) -> str:
    """Make a search-plan glob matchable by ``rg -g``.

    ``rg -g`` matches its pattern against paths RELATIVE to the search root. The
    queen routinely emits ABSOLUTE globs (e.g. ``C:/…/210_design/D031_*.md``);
    rg never matches an absolute pattern against the relative paths it walks, so
    every glob silently excludes everything — the whole FIND returns 0 hits and
    the JUDGE ends up ruling on an empty bundle (observed: investigate_e2e run 1).

    Fix: an absolute glob *under* ``root`` becomes root-relative; an absolute
    glob *outside* ``root`` (un-relativizable) degrades to its basename pattern,
    which ``rg -g`` matches at any depth. Already-relative globs pass through.

    Separator hygiene: the queen periodically over-escapes a Windows path
    (``C:\\\\…`` → the parsed value ``C:\\…`` with doubled backslashes), which
    ``\\``→``/`` turns into ``C://…`` — a doubled-slash glob ``rg`` matches against
    NOTHING (observed: T890, D031 never retrieved). Collapse runs of ``/`` so the
    glob is matchable. (Drive paths only — no UNC ``//host`` in scope here.)
    """
    gn = re.sub(r"/{2,}", "/", g.replace("\\", "/"))
    rn = re.sub(r"/{2,}", "/", root.replace("\\", "/")).rstrip("/")
    if gn.lower().startswith(rn.lower() + "/"):
        return gn[len(rn) + 1:]
    if re.match(r"^[A-Za-z]:/", gn) or gn.startswith("/"):
        return gn.rsplit("/", 1)[-1]
    return gn


def _abs_under(glob: str, root: str | None) -> bool:
    """True when ``glob`` is an absolute path nested under ``root``.

    Used to route the queen's globs to the tree they actually point at: a glob
    under ``docs_root`` must be probed/searched against the docs tree, not the
    code tree (separator-normalized, case-insensitive for Windows drives).
    """
    if not root:
        return False
    g = re.sub(r"/{2,}", "/", glob.replace("\\", "/"))
    r = re.sub(r"/{2,}", "/", root.replace("\\", "/")).rstrip("/")
    return g.lower().startswith(r.lower() + "/")


def _rel_under_docs(glob: str, docs_root: str | None) -> str | None:
    """Recognise a *relative* glob the queen rooted ABOVE ``docs_root``.

    The queen emits design-doc targets relative to the workspace root — an
    ancestor of both ``code_root`` and ``docs_root`` — so they arrive carrying
    one or more of ``docs_root``'s own trailing path segments as their leading
    segments (e.g. ``Documents/projects/FlowGate/210_design/**``). Such a glob is
    NOT absolute, so :func:`_abs_under` never routes it to the docs channel; it
    leaks to the code tree, matches nothing, and the design SSOT is silently
    dropped.

    The leading overlap can be more than one segment: how many depends on how
    DEEP ``docs_root`` is.

      * Shallow ``docs_root`` ending in ``/Documents`` → the glob's single leading
        ``Documents`` overlaps (N165 D030_CHECK/DESIGN_SSOT).
      * Deep ``docs_root`` ending in ``/Documents/projects/FlowGate`` (the launcher
        default for a per-project docs tree) → the glob's leading
        ``Documents/projects/FlowGate`` *all three* overlap. The original
        single-basename strip ("FlowGate" only) never matched a glob whose first
        segment is "Documents", so every doc glob leaked to code and
        ``design_excerpts`` was permanently empty (observed T891).

    Strip the LONGEST leading run of glob segments that is a contiguous SUFFIX of
    ``docs_root``'s path, and return the remainder made docs-root-relative (e.g.
    ``210_design/**``) so ``rg -g`` matches it under ``docs_root``; or ``None`` if
    no such overlap exists (a real code glob is left code-side). Longest-first is
    the more specific, safer match; the 1-segment case is exactly the prior
    behaviour. Absolute globs are left to :func:`_abs_under`.
    """
    if not docs_root:
        return None
    g = re.sub(r"/{2,}", "/", glob.replace("\\", "/")).lstrip("/")
    if re.match(r"^[A-Za-z]:/", g):           # absolute — _abs_under's job
        return None
    root_segs = [s for s in re.sub(r"/{2,}", "/", docs_root.replace("\\", "/"))
                 .rstrip("/").split("/") if s]
    g_segs = [s for s in g.split("/") if s]
    if not root_segs or not g_segs:
        return None
    # Largest j where the first j glob segments equal the last j docs_root
    # segments (case-insensitive for Windows). Bounded so the remainder stays
    # non-empty — a glob that is ONLY the overlap names the dir, not a target.
    for j in range(min(len(root_segs), len(g_segs) - 1), 0, -1):
        if [s.lower() for s in g_segs[:j]] == [s.lower() for s in root_segs[-j:]]:
            return "/".join(g_segs[j:])
    return None


def _partition_globs(globs: list[str], code_root: str,
                     docs_root: str | None) -> tuple[list[str], list[str]]:
    """Split the axis globs into (code-tree globs, docs-tree globs).

    The queen routinely lowers a design-doc target into ``file_globs`` pointing at
    the docs tree, in two shapes:

      * ABSOLUTE under the docs tree (e.g. ``C:/…/Documents/…/D031_*.md``) — routed
        by :func:`_abs_under`; or
      * RELATIVE, rooted at the docs tree's PARENT (e.g.
        ``Documents/…/D031_*.md``) — routed by :func:`_rel_under_docs`, which also
        rewrites it docs-root-relative so it actually matches there (N165).

    Either way the glob matches nothing under ``code_root``, so validating/searching
    it there silently drops the doc and the judge rules on a bundle that never
    contained it (T890: 0/3 axes located). Docs-tree globs go to the docs channel;
    everything else stays code-side.
    """
    code_g: list[str] = []
    doc_g: list[str] = []
    for g in globs:
        if _abs_under(g, docs_root) and not _abs_under(g, code_root):
            doc_g.append(g)
            continue
        rel = _rel_under_docs(g, docs_root)
        if rel is not None:
            doc_g.append(rel)
            continue
        code_g.append(g)
    return code_g, doc_g


def _count_glob_files(g: str, root: str, cap: int) -> int:
    """How many files does this glob actually match under ``root``?

    Uses ``rg --files -g`` (the same glob engine the real search uses, via the
    same :func:`_norm_glob`), counting up to ``cap + 1`` lines so an enormous
    over-glob is cheap to detect (we never need the exact count past the cap).
    """
    out = _run(["rg", "--files", "-g", _norm_glob(g, root), "."], cwd=root)
    n = 0
    for _ in out.splitlines():
        n += 1
        if n > cap:
            break
    return n


def _validate_globs(globs: list[str], root: str,
                    overbroad_files: int = 2000) -> tuple[list[str], dict[str, Any]]:
    """Drop garbage / over-broad queen globs by probing the real tree (free).

    The queen emits globs *blind* to the repo, so two failure modes corrupt the
    bundle (M004 follow-up findings):

      * **Garbage** — a non-path the queen mistook for one, e.g. git-log fields
        ``message/author/date`` lowered to ``message/author/date/**/*``. It is
        structurally indistinguishable from a real ``a/b/c`` path, so no text
        filter catches it — but it matches **0 files**, so existence does.
      * **Over-glob** — a ``**/*``-ish scope matching thousands of files. The
        bundle is snippet-bounded, but *which* snippets survive then depends on
        noise, not the axis. If a narrower usable glob exists, the over-glob only
        dilutes — drop it.

    Partition by match count and keep the usable ones; fall back to over-broad
    globs only when nothing narrower exists, and to whole-tree (``[]``) only when
    every glob is garbage (a guaranteed-empty bundle is the worse outcome).

    Pure-local and deterministic — no model call. Returns ``(kept, diag)``.
    """
    usable: list[str] = []
    overbroad: list[str] = []
    empty: list[str] = []
    counts: dict[str, int] = {}
    for g in globs:
        n = _count_glob_files(g, root, overbroad_files)
        counts[g] = n
        if n == 0:
            empty.append(g)
        elif n > overbroad_files:
            overbroad.append(g)
        else:
            usable.append(g)

    if usable:
        kept = usable
    elif overbroad:
        kept = overbroad  # no narrow scope — keep over-glob; density ranks it
    else:
        kept = []         # all garbage — search whole tree over guaranteed-empty
    diag = {"kept": kept, "dropped_empty": empty,
            "dropped_overbroad": overbroad if usable else [],
            "counts": counts}
    return kept, diag


def _widen_globs(globs: list[str]) -> list[str]:
    """Drop each glob's file-EXTENSION constraint, keeping its directory scope.

    The queen emits ``file_globs`` blind to the target stack, so she routinely
    names the wrong extensions for a tree: e.g. ``client/**/*.js`` + ``*.tsx`` for
    a Vue 3 + TypeScript app whose component layer is ``.vue``/``.ts`` (T889
    forensics). :func:`_validate_globs` cannot catch this — a stray ``.js`` build
    artefact makes the glob match >0 files, so it is not dropped as garbage — yet
    the keyword search inside that scope never reaches the real ``.vue`` site and
    the axis comes back with 0 code snippets.

    Widening replaces the extension token of each glob's final segment with a
    wildcard (``client/**/*.js`` → ``client/**/*``; ``*.tsx`` → ``*``;
    ``a/b/Foo.vue`` → ``a/b/Foo*``), so a re-search keeps the directory locality
    the queen got right while dropping the extension she got wrong. Deterministic,
    free, and only ever run as a fallback (see :func:`retrieve`). Order-preserving
    and de-duplicated.
    """
    widened: list[str] = []
    for g in globs:
        head, sep, tail = g.replace("\\", "/").rpartition("/")
        if "." not in tail:
            wide_tail = tail            # already extensionless — leave it alone
        else:
            base = tail.split(".", 1)[0]   # "*" from "*.js"; "Foo" from "Foo.vue"
            wide_tail = base if base.endswith("*") else (base + "*" if base else "*")
        wg = head + sep + wide_tail
        if wg not in widened:
            widened.append(wg)
    return widened


# A "structured key→value" file (a SQL/JSON query map, a one-locale-per-line i18n
# file) packs a whole record onto ONE long line. Both the ±k line WINDOW and the
# keyword-cooccurrence CLUSTERING break there: neighbouring lines are unrelated
# records, and in an all-SQL file EVERY line carries SELECT/WHERE/JOIN so dozens
# tie at max coverage and the greedy pick is decided by line ORDER, not by the
# target. (T892: ``get_pending_head_by_group`` sat on a 586-char line; the same
# file N168 located at queries.json:117-129 came back located=False because the
# ±6-line window centred elsewhere and the 2000-char cap truncated the key out.)
# When a file's matched lines are this long, snippet PER LINE instead — each
# record is isolated and ranked by its own keyword coverage. Deterministic, free.
_DENSE_LINE_CHARS = 200
_DENSE_MAX_LINES = 6      # per dense file: surface up to this many record-lines
_DENSE_LINE_CAP = 1500    # chars kept per surfaced record-line


def _dense_line_snippets(code_root: str, relpath: str,
                         hits: list[tuple[int, str]], max_lines: int,
                         line_cap: int) -> list[dict[str, Any]]:
    """Per-line snippets for a one-record-per-line file (SQL/JSON query map, i18n).

    Ranks the matched lines by DISTINCT-keyword coverage (ties → lowest line) and
    surfaces each WHOLE line as its own snippet, so a dense SQL/JSON record is
    handed to the judge isolated instead of averaged into a ±k window of unrelated
    neighbours (or truncated out of it). This is what makes anchor location on a
    dense single-line file deterministic given the same hits (Defect 1 / T892).
    """
    abspath = os.path.join(code_root, relpath)
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError:
        return []
    kws_at: dict[int, set[str]] = defaultdict(set)
    for ln, kw in hits:
        kws_at[ln].add(kw)
    ranked_lines = sorted(kws_at.keys(),
                          key=lambda ln: (len(kws_at[ln]), -ln), reverse=True)
    snips: list[dict[str, Any]] = []
    for ln in ranked_lines[:max_lines]:
        if 1 <= ln <= len(all_lines):
            text = all_lines[ln - 1].rstrip("\n")[:line_cap]
            snips.append({"file": relpath, "lines": f"{ln}-{ln}",
                          "text": text, "hits": sorted(kws_at[ln])})
    return snips


def _scan_code(keywords: list[str], globs: list[str], code_root: str,
               k: int, top_files: int) -> dict[str, Any]:
    """ripgrep keywords → rank files → window densest clusters (retrieve steps 1-3).

    Factored out so :func:`retrieve` can run it a second time with widened globs
    when the first pass returns 0 snippets (extension-blind fallback, T889).
    """
    call_sites: list[dict[str, Any]] = []
    file_hits: dict[str, set[str]] = defaultdict(set)       # file -> keywords
    file_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)  # (line,kw)
    file_maxlen: dict[str, int] = defaultdict(int)          # file -> longest hit line
    for kw in keywords:
        for h in _ripgrep(kw, globs, code_root):
            h["keyword"] = kw
            call_sites.append(h)
            file_hits[h["file"]].add(kw)
            file_lines[h["file"]].append((h["line"], kw))
            file_maxlen[h["file"]] = max(file_maxlen[h["file"]], len(h.get("text", "")))

    hit_count: dict[str, int] = defaultdict(int)
    for h in call_sites:
        hit_count[h["file"]] += 1
    ranked = sorted(
        file_hits.keys(),
        key=lambda f: (len(file_hits[f]), hit_count[f]),
        reverse=True,
    )

    raw_snips: list[dict[str, Any]] = []
    # Dense per-line snippets are kept OUT of _merge_windows: in a query map the
    # records sit on consecutive lines, so merging would re-collapse them into the
    # whole-file window the dense path exists to avoid (Defect 1/T892).
    dense_snips: list[dict[str, Any]] = []
    densest_line: dict[str, int] = {}  # file -> centroid of its top cluster
    for f in ranked[:top_files]:
        # Dense one-record-per-line file (SQL/JSON map): snippet per line so the
        # target record is isolated, not averaged into a ±k window (Defect 1/T892).
        if file_maxlen[f] >= _DENSE_LINE_CHARS:
            dense = _dense_line_snippets(code_root, f, file_lines[f],
                                         _DENSE_MAX_LINES, _DENSE_LINE_CAP)
            if dense:
                densest_line[f] = int(dense[0]["lines"].split("-")[0])
                dense_snips.extend(dense)
                continue
        clusters = _cluster_lines(file_lines[f], k, max_clusters=2)
        if clusters:
            densest_line[f] = (clusters[0]["lo"] + clusters[0]["hi"]) // 2
        for c in clusters:
            w = _read_window(code_root, f, (c["lo"] + c["hi"]) // 2, k)
            raw_snips.append({
                "file": f, "lines": w["lines"], "text": w["text"],
                "hits": c["kws"],
            })
    return {
        "call_sites": call_sites,
        "file_hits": file_hits,
        "file_lines": file_lines,
        "ranked": ranked,
        "densest_line": densest_line,
        "code_snippets": _merge_windows(raw_snips) + dense_snips,
    }


def _ripgrep(keyword: str, globs: list[str], root: str,
             max_hits: int = 40) -> list[dict[str, Any]]:
    """ripgrep one keyword constrained to globs. Returns [{file,line,text}]."""
    cmd = ["rg", "--no-heading", "-n", "-i", "--max-count", str(max_hits)]
    for g in globs:
        cmd += ["-g", _norm_glob(g, root)]
    cmd += ["-e", keyword, "."]
    out = _run(cmd, cwd=root)
    hits: list[dict[str, Any]] = []
    for line in out.splitlines():
        # format: relpath:lineno:text
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        relpath, lineno, text = parts
        if not lineno.isdigit():
            continue
        hits.append({
            "file": relpath.replace("\\", "/"),
            "line": int(lineno),
            "text": text.strip()[:300],
        })
    return hits


def _read_window(root: str, relpath: str, line: int, k: int) -> dict[str, Any]:
    """Read ±k lines around `line` (1-based). Returns {lines,text}."""
    abspath = os.path.join(root, relpath)
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return {"lines": f"{line}", "text": ""}
    lo = max(0, line - 1 - k)
    hi = min(len(all_lines), line + k)
    text = "".join(all_lines[lo:hi])
    return {"lines": f"{lo + 1}-{hi}", "text": text[:2000]}


def _read_def_body(root: str, relpath: str, line: int,
                   max_lines: int = 60) -> dict[str, Any]:
    """Read a whole def/class body starting at ``line`` (1-based).

    A fixed ±k window centred on a ``def`` reaches only the signature, not the
    bug that lives deeper in the body — exactly why the judge-directed follow-up
    must hand the re-judge the *function*, not a slice (M004 §6: store.py bug at
    1446 sat below the def±6 window). We read from the def line until the first
    later non-blank line whose indent is ≤ the def's indent (a sibling def/class
    or module-level statement ends the body), capped at ``max_lines``.

    Falls back to a small forward window if ``line`` is not actually a def.
    """
    abspath = os.path.join(root, relpath)
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return {"lines": f"{line}", "text": ""}
    idx = line - 1
    if not (0 <= idx < len(all_lines)):
        return {"lines": f"{line}", "text": ""}
    def_indent = len(all_lines[idx]) - len(all_lines[idx].lstrip())
    end = min(len(all_lines), idx + max_lines)

    # 1. Walk the (possibly multi-line) signature to its end. A multi-line
    #    signature's closing ``)`` sits at the def's OWN indent (``) -> dict:``),
    #    so naive indent-dedent detection stops INSIDE the signature and never
    #    reaches the body (observed: get_document → _parse_doc_workflow at +10 was
    #    never read). Track paren depth from the def line until balanced and the
    #    body-opening ``:`` is seen.
    sig_end = idx
    depth = 0
    seen_paren = False
    for j in range(idx, end):
        code = all_lines[j].split("#", 1)[0]
        depth += code.count("(") - code.count(")")
        if "(" in code:
            seen_paren = True
        sig_end = j
        if (seen_paren and depth <= 0 and ":" in code) or \
           (not seen_paren and ":" in code):
            break

    # 2. Read the body: lines indented deeper than the def, blank lines included,
    #    until a line dedents to ≤ the def's indent (a sibling/module statement).
    hi = sig_end + 1
    for j in range(sig_end + 1, end):
        ln = all_lines[j]
        if not ln.strip():
            hi = j + 1
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= def_indent:
            break
        hi = j + 1
    text = "".join(all_lines[idx:hi])
    return {"lines": f"{line}-{hi}", "text": text[:2400]}


def _merge_windows(snips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse overlapping ±k windows in the same file into one snippet."""
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in snips:
        by_file[s["file"]].append(s)
    merged: list[dict[str, Any]] = []
    for f, items in by_file.items():
        def lo_of(it: dict[str, Any]) -> int:
            return int(it["lines"].split("-")[0])
        items.sort(key=lo_of)
        cur = items[0]
        for nxt in items[1:]:
            cur_hi = int(cur["lines"].split("-")[1])
            nxt_lo = lo_of(nxt)
            if nxt_lo <= cur_hi + 1:  # overlap/adjacent → merge
                nxt_hi = int(nxt["lines"].split("-")[1])
                new_hi = max(cur_hi, nxt_hi)
                cur = {
                    "file": f,
                    "lines": f"{lo_of(cur)}-{new_hi}",
                    "text": cur["text"] if len(cur["text"]) >= len(nxt["text"]) else nxt["text"],
                    "hits": sorted(set(cur.get("hits", [])) | set(nxt.get("hits", []))),
                }
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    return merged


def _cluster_lines(hits: list[tuple[int, str]], k: int,
                   max_clusters: int) -> list[dict[str, Any]]:
    """Find the fixed-width windows where the MOST distinct keywords co-occur.

    ``hits`` = list of (line, keyword). For each hit line we consider the window
    [center-k, center+k] and score it by how many DISTINCT keywords hit inside.
    Windows are picked greedily, highest coverage first, skipping any that
    overlap an already-picked one. Tight co-occurrence — not single-keyword
    frequency — is the bug signal: a 7-line region with SELECT+type+WHERE+
    documents beats 500 scattered lone "SELECT"s.

    (Chained-cluster windowing failed: in an all-SQL file every line chained
    into one whole-file cluster; first-hit windowing missed bugs late in big
    files. Both are M004 §6 findings.)
    """
    if not hits:
        return []
    lines = sorted({ln for ln, _ in hits})
    kws_at: dict[int, set[str]] = defaultdict(set)
    for ln, kw in hits:
        kws_at[ln].add(kw)

    candidates = []
    for center in lines:
        lo, hi = center - k, center + k
        kws: set[str] = set()
        for ln in lines:
            if lo <= ln <= hi:
                kws |= kws_at[ln]
            elif ln > hi:
                break
        candidates.append({"center": center, "lo": lo, "hi": hi, "kws": kws})

    candidates.sort(key=lambda c: len(c["kws"]), reverse=True)
    picked: list[dict[str, Any]] = []
    for c in candidates:
        if any(not (c["hi"] < p["lo"] or c["lo"] > p["hi"]) for p in picked):
            continue  # overlaps an already-picked window
        picked.append(c)
        if len(picked) >= max_clusters:
            break
    return [{"lo": max(1, p["lo"]), "hi": p["hi"], "kws": sorted(p["kws"])}
            for p in picked]


def _git_history(root: str, relpath: str, line: int, k: int) -> dict[str, Any]:
    """git blame around hit + recent log for the file. Mechanical, free."""
    lo = max(1, line - k)
    hi = line + k
    blame = _run(
        ["git", "--no-pager", "blame", "-L", f"{lo},{hi}", "--", relpath],
        cwd=root,
    )
    log = _run(
        ["git", "--no-pager", "log", "-n", "5", "--pretty=format:%h %ad %s",
         "--date=short", "--", relpath],
        cwd=root,
    )
    return {
        "file": relpath,
        "blame": blame.strip()[:1500],
        "log": log.strip()[:800],
    }


def _follow_calls(snippets: list[dict[str, Any]], globs: list[str], root: str,
                  k: int, max_hops: int) -> list[dict[str, Any]]:
    """1-2 hop call-chain follow — the cheap local stand-in for a drone's
    adaptive cross-file tracing (M004 §2 'import/call 1-2 hops').

    For each symbol *called* in the current snippets, grep ``def <symbol>`` (or
    SQL ``CREATE``-style not handled) within the globs and pull the definition
    window. Then repeat once on the newly fetched defs (hop 2). Bounded and
    deterministic: no model, no open-ended exploration. This is what closes the
    residual ~20% of misses (store.py/db.py reached via a callee name) that pure
    keyword-density retrieval can't see — M004 §6 finding.
    """
    follows: list[dict[str, Any]] = []
    seen_syms: set[str] = set()
    seen_defs: set[str] = set()
    frontier = snippets
    for _hop in range(max_hops):
        next_frontier: list[dict[str, Any]] = []
        for s in frontier:
            for m in _CALL_RE.finditer(s.get("text", "")):
                name = m.group(1)
                if name in seen_syms or name in _CALL_SKIP:
                    continue
                seen_syms.add(name)
                for h in _ripgrep(rf"def {name}\b", globs, root, max_hits=2):
                    key = f"{h['file']}:{h['line']}"
                    if key in seen_defs:
                        continue
                    seen_defs.add(key)
                    # Read the whole def BODY, not a ±k window: a hit here is always
                    # a ``def`` line, and the next hop's calls (and the bug) live in
                    # the body below the signature — a fixed window centred on the
                    # def reaches only the signature (M004 §6; the delegation hop
                    # get_document → _parse_doc_workflow sits ~10 lines below its def).
                    w = _read_def_body(root, h["file"], h["line"])
                    snip = {"file": h["file"], "lines": w["lines"],
                            "text": w["text"], "symbol": name,
                            "via": "call-chain"}
                    follows.append(snip)
                    next_frontier.append(snip)
        if not next_frontier:
            break
        frontier = next_frontier
    return follows


# A label / i18n reference inside a snippet:  callee('KEY')  or  callee("KEY").
# The KEY's RENDERED text is what tells two otherwise-identical sibling elements
# apart (T891: two `<div class="wf-undecided">` distinguishable only by
# getLabel('R')->"요건정의" vs t('...undecided')->"미정"). The raw snippet shows
# only the call, so a judge must GUESS what it renders to — and guesses
# inconsistently across rounds, anchoring the wrong sibling.
_REF_RE = re.compile(r"""([A-Za-z_$][\w.$]*)\s*\(\s*(['"])([^'"\n]{1,120})\2""")

# Callees whose string argument is a label / i18n KEY worth resolving. Matched
# against the callee's LAST dotted segment, case-insensitively. Kept tight on
# purpose: resolving every ``f('x')`` would flood the bundle with the noise the
# whole local-FIND budget exists to keep out — we add ONLY the discriminator.
_LABEL_GETTERS = frozenset({
    "t", "$t", "tc", "$tc", "te", "i18n", "translate", "tr", "trans",
    "getlabel", "label", "gettext", "msg", "message", "getname",
    "displayname", "title", "caption", "text", "name",
})


def _looks_like_label_ref(callee: str, key: str) -> bool:
    """Worth resolving? A known label/i18n getter, or a dotted i18n-style key."""
    if callee.split(".")[-1].lower() in _LABEL_GETTERS:
        return True
    return key.count(".") >= 1 and " " not in key


def _extract_value(line: str, key: str) -> str | None:
    """If ``line`` defines ``key`` as a quoted string value, return that value.

    Matches ``KEY: "value"`` / ``'KEY' => 'value'`` / ``KEY = "value"`` shapes —
    the common label-map / locale-file definition forms across JS/TS/JSON/Vue.
    """
    m = re.search(
        r"['\"]?" + re.escape(key) + r"['\"]?\s*(?::|=>|=)\s*(['\"])(.+?)\1",
        line)
    return m.group(2) if m else None


def _resolve_key(key: str, roots: list[str], max_vals: int = 4) -> list[dict[str, Any]]:
    """Resolve a label/i18n KEY to its string value(s) by local grep (free).

    Finds ``KEY : "value"`` definitions and collects the DISTINCT values. One
    value = confident resolution; several = ambiguous, returned as candidates (the
    judge still sees the options instead of guessing); none = nothing attached
    (never invent a value — that would be the hallucination we are avoiding).
    """
    pat = r"['\"]?" + re.escape(key) + r"['\"]?\s*(?::|=>|=)"
    seen: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root:
            continue
        for h in _ripgrep(pat, [], root, max_hits=20):
            val = _extract_value(h["text"], key)
            if val and val not in seen:
                seen[val] = {"value": val, "source": f"{h['file']}:{h['line']}"}
            if len(seen) >= max_vals:
                break
    return list(seen.values())


def _resolve_discriminators(snippets: list[dict[str, Any]], code_root: str,
                            docs_root: str | None) -> int:
    """Attach resolved label/i18n values to snippets (T891 disambiguation).

    Mutates each snippet in place, adding a ``resolved`` list when it contains
    label/i18n references that resolve locally. Returns the total number of
    references resolved (for stats). Pure-local, deterministic, zero model cost —
    this is the cheap local READ the operator otherwise had to do by hand.
    """
    roots = [code_root, docs_root]
    cache: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for s in snippets:
        refs: list[dict[str, Any]] = []
        seen_in_snip: set[str] = set()
        for m in _REF_RE.finditer(s.get("text", "")):
            callee, key = m.group(1), m.group(3)
            if not _looks_like_label_ref(callee, key):
                continue
            ref_str = f"{callee}('{key}')"
            if ref_str in seen_in_snip:
                continue
            seen_in_snip.add(ref_str)
            if key not in cache:
                cache[key] = _resolve_key(key, roots)
            vals = cache[key]
            if vals:
                refs.append({
                    "ref": ref_str, "key": key,
                    "values": [v["value"] for v in vals],
                    "sources": [v["source"] for v in vals],
                    "ambiguous": len(vals) > 1,
                })
        if refs:
            s["resolved"] = refs
            total += len(refs)
    return total


# ── HTTP call-binding edge (FE field → fetch URL literal → backend route → handler).
# The crux miss (N177): a UI symptom's data source is reachable only by matching a
# fetch URL *literal* in the client to the server route that fills it — a
# cross-language string join that keyword/density retrieval never builds. So the
# judge grounds on a lexically-similar but WRONG handler (e.g. ``get_effective_head``
# — literally named "head") and misses the real source (``_parse_doc_workflow``,
# reached only via ``/api/v1/documents/detail`` → ``@router.get("/detail")``). This
# resolves that one edge: deterministic, free, language-neutral; same pattern as
# :func:`_follow_calls` (symbol→def) and :func:`_resolve_discriminators` (key→value).

# A client HTTP call: a fetch/axios/get-style callee taking a URL-path literal
# (quote or backtick). Captures the STATIC leading path (up to the first query
# ``?``, interpolation ``${``, or closing quote) — the dynamic tail is a path param.
_HTTP_CALL_RE = re.compile(
    r"""(?P<callee>[A-Za-z_$][\w.$]*)\s*(?:<[^>(){}]*>)?\s*\(\s*[`'"]\s*(?P<path>/[A-Za-z0-9_./:{}-]*)""")

# Callee's LAST dotted segment that marks an HTTP request (so we don't treat every
# function taking a "/x" string as a fetch). Tight on purpose — same discipline as
# ``_LABEL_GETTERS``: resolving every ``f('/x')`` would flood the bundle.
_HTTP_CALLEES = frozenset({
    "get", "post", "put", "patch", "delete", "del", "head", "options",
    "request", "fetch", "query", "mutate", "send",
    "getrequest", "postrequest", "putrequest", "patchrequest", "deleterequest",
    "getjson", "postjson", "httpget", "httppost",
})

# A backend route declaration: ``@router.get("/path")`` / ``@app.route('/path')`` —
# a DECORATOR (FastAPI / Flask). The leading ``@`` is what tells a server route apart
# from a CLIENT fetch call (``axios.get('/api/…')``), which otherwise matches the same
# ``.get("/…")`` shape and would be miscollected as a route. (Narrow on purpose:
# non-decorator routers like Express ``router.get('/p', cb)`` are out of scope until a
# stack needs them — start with the FastAPI/Flask form FlowGate uses.)
_ROUTE_DECL_RE = re.compile(
    r"""@\s*[A-Za-z_][\w.]*\.(?P<verb>get|post|put|patch|delete|route|head|options)\s*\(\s*(['"])(?P<route>/[^'"]*)\2""",
    re.IGNORECASE)

# A router's OWN declared path prefix: ``APIRouter(prefix="/documents")`` /
# ``Blueprint(..., url_prefix="/x")``. Two routes can declare the SAME decorator
# tail (``/detail``) in different routers; their full paths differ only by this
# prefix (FlowGate: ``/documents`` vs ``/api/v1`` → only the former completes
# ``/api/v1/documents/detail``). Reading it disambiguates the collision
# deterministically — it is the router's own construct, not the outer mount tree.
_ROUTER_PREFIX_RE = re.compile(
    r"""(?:APIRouter|Blueprint|Router)\s*\([^)]*?(?:url_)?prefix\s*=\s*(['"])(?P<prefix>/[^'"]*)\1""")


def _path_segs(p: str) -> list[str]:
    """Split a URL/route path into non-empty segments (strip leading/trailing /)."""
    return [s for s in p.strip("/").split("/") if s]


def _route_suffix_match(url_segs: list[str],
                        route_segs: list[str]) -> tuple[bool, int, int]:
    """Does ``route_segs`` segment-align as a SUFFIX of ``url_segs``?

    The client sends a full path (``/api/v1/documents/detail``) while a backend
    decorator usually carries only its router-relative tail (``/detail``) — the
    mount prefix is assembled elsewhere (``include_router(prefix=...)``), often in
    another file. Matching the decorator path as a segment-suffix of the URL,
    with a ``{param}``/``:param``/``*`` route segment matching any one URL segment,
    resolves the edge without parsing the whole mount tree.

    Returns ``(matched, literal_segs, param_segs)`` — the latter two rank
    specificity (a literal match beats a param match, FastAPI's own precedence).
    """
    if not route_segs or len(route_segs) > len(url_segs):
        return (False, 0, 0)
    tail = url_segs[len(url_segs) - len(route_segs):]
    lit = par = 0
    for u, r in zip(tail, route_segs):
        if (r.startswith("{") and r.endswith("}")) or r.startswith(":") or r == "*":
            par += 1
        elif r.lower() == u.lower():
            lit += 1
        else:
            return (False, 0, 0)
    return (True, lit, par)


def _read_def_below(root: str, relpath: str, line: int,
                    search: int = 25) -> dict[str, Any]:
    """Read the handler ``def`` body that follows a route decorator at ``line``.

    A route decorator (and any stacked decorators like ``@require_permission``)
    sits above the handler ``def``; scan downward to the first ``def``/``async def``
    and read its whole body (:func:`_read_def_body`). Falls back to a small window
    at the decorator if no def is found within ``search`` lines.
    """
    abspath = os.path.join(root, relpath)
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError:
        return {"lines": str(line), "text": ""}
    for j in range(line - 1, min(len(all_lines), line - 1 + search)):
        st = all_lines[j].lstrip()
        if st.startswith(("def ", "async def ")):
            return _read_def_body(root, relpath, j + 1)
    return _read_window(root, relpath, line, 8)


def _router_prefix(code_root: str, relpath: str) -> str | None:
    """The router's own declared path prefix for ``relpath``, or None.

    Reads the file and matches the first ``APIRouter(prefix=...)`` /
    ``Blueprint(url_prefix=...)`` literal. A non-literal prefix (f-string, var) is
    left unresolved — better than guessing.
    """
    abspath = os.path.join(code_root, relpath)
    try:
        with open(abspath, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
    except OSError:
        return None
    m = _ROUTER_PREFIX_RE.search(txt)
    return m.group("prefix") if m else None


def _literal_mount_prefix(node: ast.expr) -> str | None:
    """Return the statically visible path in an include_router prefix.

    Plain string literals are exact. For f-strings, dynamic fields are omitted
    and only literal path segments survive (``f"{ROOT}/api/v1"`` → ``/api/v1``).
    Other expressions stay unresolved rather than being guessed.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
        if value and not value.startswith("/"):
            return None
    elif isinstance(node, ast.JoinedStr):
        value = "".join(
            part.value for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        slash = value.find("/")
        value = value[slash:] if slash >= 0 else ""
    else:
        return None
    return re.sub(r"/{2,}", "/", value).rstrip("/")


# Dirs never holding app source — pruned from the import-resolution walk so a
# vendored copy of a module (``.venv/.../project_settings.py``) cannot create a
# false suffix-match ambiguity that silently disables the mount-prefix fold.
_WALK_PRUNE_DIRS = frozenset({
    ".venv", "venv", "site-packages", "node_modules", "dist", "build",
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".apply_backups",
})


def _list_py_files(code_root: str) -> set[str]:
    """All ``.py`` file relpaths under ``code_root`` (forward slashes), vendored/
    hidden dirs pruned.

    Needed because the python source root is often a SUBDIR of the codebase root
    (FlowGate runs from ``server/``, so ``from modules...`` resolves to
    ``server/modules/...``): resolving imports purely against ``code_root`` would
    miss every such file.
    """
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(code_root):
        dirnames[:] = [d for d in dirnames if d not in _WALK_PRUNE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fn), code_root)
                out.add(rel.replace(os.sep, "/"))
    return out


def _imported_router_path(importer: str, node: ast.ImportFrom,
                          py_files: set[str]) -> str | None:
    """Resolve ``from module import router`` to one local Python file.

    Exact resolution (module path rooted directly at the codebase root) wins.
    On a miss, an ABSOLUTE import is retried as a unique path SUFFIX — the python
    source root is a subdir (``server/``) so ``from modules...`` lives at
    ``server/modules/...``. A relative import is already rooted at ``code_root``,
    and a non-unique suffix is left unresolved rather than guessed.
    """
    module_parts = node.module.split(".") if node.module else []
    importer = importer.replace("\\", "/")
    if importer.startswith("./"):
        importer = importer[2:]
    importer_parts = importer.split("/")[:-1]
    if node.level:
        climb = node.level - 1
        if climb > len(importer_parts):
            return None
        base = importer_parts[:len(importer_parts) - climb]
    else:
        base = []
    relbase = "/".join(base + module_parts)
    if not relbase:
        return None
    candidates = [f"{relbase}.py", f"{relbase}/__init__.py"]
    exact = [c for c in candidates if c in py_files]
    if exact:
        return exact[0] if len(exact) == 1 else None
    if node.level:                       # relative import: no source-root subdir to span
        return None
    matches = {
        rel for rel in py_files
        for c in candidates
        if rel == c or rel.endswith("/" + c)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _mount_metadata(code_root: str) -> dict[str, dict[str, Any]]:
    """Map router source files to static mount prefixes and registration order.

    Candidate wiring files come from ripgrep. Within each file, AST links
    ``from ... import router as alias`` to ``include_router(alias, prefix=...)``.
    The source line of ``include_router`` is the deterministic FastAPI registration
    order within that app assembly: when two routes have the same method/path,
    Starlette dispatches the first registered route. Unparseable files, dynamic
    expressions, missing targets, and conflicting mounts are ignored so route
    collection falls back to ambiguity instead of guessing a winner.
    """
    hits = _ripgrep(r"\binclude_router\s*\(", [], code_root, max_hits=400)
    wiring_files = sorted({h["file"] for h in hits})
    if not wiring_files:                  # no FastAPI mounts → skip the tree walk
        return {}
    py_files = _list_py_files(code_root)
    found: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for relpath in wiring_files:
        try:
            with open(os.path.join(code_root, relpath), "r",
                      encoding="utf-8", errors="replace") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError, ValueError):
            continue

        aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _imported_router_path(relpath, node, py_files)
            if not target:
                continue
            for name in node.names:
                if name.name == "router" and name.asname:
                    aliases[name.asname] = target

        calls = sorted(
            (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
            key=lambda node: getattr(node, "lineno", 0),
        )
        for node in calls:
            func = node.func
            if not (isinstance(func, ast.Attribute)
                    and func.attr == "include_router"
                    and node.args
                    and isinstance(node.args[0], ast.Name)):
                continue
            target = aliases.get(node.args[0].id)
            if not target:
                continue
            prefix_arg = next(
                (kw.value for kw in node.keywords if kw.arg == "prefix"), None)
            prefix = (_literal_mount_prefix(prefix_arg)
                      if prefix_arg is not None else "")
            if prefix is None:
                continue
            meta = {
                "prefix": prefix,
                "wiring_file": relpath.replace("\\", "/"),
                "order": int(getattr(node, "lineno", 0) or 0),
            }
            if target in found and found[target] != meta:
                conflicts.add(target)
            else:
                found[target] = meta
    return {path: meta for path, meta in found.items()
            if path not in conflicts}


def _mount_prefixes(code_root: str) -> dict[str, str]:
    """Backward-compatible prefix-only view of :func:`_mount_metadata`."""
    return {path: str(meta.get("prefix", ""))
            for path, meta in _mount_metadata(code_root).items()}


def _collect_routes(code_root: str, max_hits: int = 400) -> list[dict[str, Any]]:
    """Grep all backend route declarations once (free, bounded, deterministic).

    Each route's match path is its outer ``include_router`` mount prefix + the
    router's own declared ``prefix`` + the decorator tail. This preserves the
    previous suffix-only fallback whenever either static prefix is unavailable.
    """
    routes: list[dict[str, Any]] = []
    prefix_cache: dict[str, str | None] = {}
    mount_metadata = _mount_metadata(code_root)
    pat = r"@\s*[A-Za-z_][\w.]*\.(get|post|put|patch|delete|route|head|options)\s*\(\s*['\"]/"
    for h in _ripgrep(pat, [], code_root, max_hits=max_hits):
        m = _ROUTE_DECL_RE.search(h.get("text", ""))
        if not m:
            continue
        f = h["file"]
        if f not in prefix_cache:
            prefix_cache[f] = _router_prefix(code_root, f)
        prefix = prefix_cache[f]
        rel = f[2:] if f.startswith("./") else f
        mount_meta = mount_metadata.get(rel, {})
        mount = mount_meta.get("prefix")
        route = m.group("route")
        full = "".join(
            part.rstrip("/") for part in (mount, prefix) if part
        ) + route
        routes.append({
            "file": f, "line": h["line"],
            "verb": m.group("verb").lower(), "route": route,
            "full_path": full, "segs": _path_segs(full),
            "mount_file": mount_meta.get("wiring_file", ""),
            "mount_order": mount_meta.get("order"),
        })
    return routes


def _resolve_http_bindings(snippets: list[dict[str, Any]], code_root: str,
                           max_urls: int = 12,
                           max_candidates: int = 3) -> list[dict[str, Any]]:
    """Resolve client fetch-URL literals to their backend route handlers (N177).

    For each HTTP call URL literal in the snippets, find the backend route whose
    declared path segment-matches the URL (most-specific wins; literal beats
    param), and attach that handler's def body. When several equally-specific
    routes match the same URL the binding is ``ambiguous`` and ALL are surfaced —
    same "show candidates, never guess" rule as :func:`_resolve_key`. Distinct
    handlers feeding one client path is itself a duplicated-source signal worth a
    look. Pure-local, deterministic, zero model cost.
    """
    urls: dict[str, set[str]] = defaultdict(set)
    url_files: dict[str, set[str]] = defaultdict(set)
    for s in snippets:
        for m in _HTTP_CALL_RE.finditer(s.get("text", "")):
            if m.group("callee").split(".")[-1].lower() not in _HTTP_CALLEES:
                continue
            path = m.group("path").rstrip("/")
            if path.count("/") < 1 or len(_path_segs(path)) < 1:
                continue
            urls[path].add(m.group("callee"))
            if s.get("file"):
                url_files[path].add(str(s["file"]))
    if not urls:
        return []

    routes = _collect_routes(code_root)
    if not routes:
        return []

    bindings: list[dict[str, Any]] = []
    for path in sorted(urls)[:max_urls]:
        usegs = _path_segs(path)
        inferred_verbs = {
            verb for callee in urls[path]
            for verb in [_http_verb_from_callee(callee)]
            if verb
        }
        scored: list[tuple[int, int, int, dict[str, Any]]] = []
        for r in routes:
            if inferred_verbs and r["verb"] not in inferred_verbs:
                continue
            ok, lit, par = _route_suffix_match(usegs, r["segs"])
            if ok:
                scored.append((len(r["segs"]), lit, -par, r))
        if not scored:
            continue
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        top = scored[0][:3]
        # Keep routes tying the top specificity. When every tied route is mounted by
        # the SAME app assembly, registration order proves the FastAPI/Starlette winner:
        # first include_router wins for an identical method/path. Otherwise preserve the
        # old ambiguity behavior rather than comparing unrelated app files.
        tied = [r for (a, b, c, r) in scored if (a, b, c) == top]
        mount_files = {r.get("mount_file") for r in tied if r.get("mount_file")}
        ordered = bool(tied) and len(mount_files) == 1 and all(
            isinstance(r.get("mount_order"), int) for r in tied
        )
        shadowed: list[dict[str, Any]] = []
        if len(tied) > 1 and ordered:
            tied.sort(key=lambda r: (r["mount_order"], r["line"], r["file"]))
            shadowed = tied[1:]
            cands = tied[:1]
        else:
            cands = tied[:max_candidates]
        for r in cands:
            w = _read_def_below(code_root, r["file"], r["line"])
            def_match = re.search(
                r"\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                w["text"],
            )
            full = r.get("full_path", r["route"])
            header = (f"# RESOLVED BINDING (hive): {r['verb'].upper()} {full} "
                      f"← client {path}\n")
            bindings.append({
                "url": path, "route": r["route"], "full_path": full,
                "verb": r["verb"], "file": r["file"], "lines": w["lines"],
                "symbol": def_match.group(1) if def_match else "",
                "text": header + w["text"],
                "via": "http-binding", "callees": sorted(urls[path]),
                "client_files": sorted(url_files[path]),
                "ambiguous": len(cands) > 1,
                "winning": len(cands) == 1,
                "mount_file": r.get("mount_file", ""),
                "mount_order": r.get("mount_order"),
                "shadowed": [
                    {
                        "file": s["file"], "line": s["line"],
                        "full_path": s.get("full_path", s["route"]),
                        "mount_order": s.get("mount_order"),
                    }
                    for s in shadowed
                ],
            })
    return bindings


def _http_verb_from_callee(callee: str) -> str:
    """Infer an HTTP verb from common client helper names, or ``""`` if generic."""
    tail = str(callee or "").split(".")[-1].lower()
    for verb in ("get", "post", "put", "patch", "delete", "head", "options"):
        if tail == verb or tail.startswith(verb):
            return verb
    if tail == "del":
        return "delete"
    return ""


_RESPONSE_STORE_CALLS = frozenset({
    "_fetch_all", "_fetch_one", "execute", "fetchall", "fetchone",
})


def _response_dependency_refs(text: str) -> list[tuple[str, str]]:
    """Return callees that contribute to a function's returned response value.

    This is a narrow AST data-flow walk: start at every ``return`` expression, follow
    local assignments referenced by that expression, and collect calls encountered on
    those value paths. Incidental calls such as authentication/logging are excluded.
    """
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return []
    fn = next(
        (node for node in ast.walk(tree)
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if fn is None:
        return []

    assigned: dict[str, ast.AST] = {}
    returns: list[ast.AST] = []
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned[target.id] = value
        elif isinstance(node, ast.Return) and node.value is not None:
            returns.append(node.value)

    calls: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    def visit_value(node: ast.AST) -> None:
        if isinstance(node, ast.Name) and node.id in assigned and node.id not in seen_names:
            seen_names.add(node.id)
            visit_value(assigned[node.id])
            return
        if isinstance(node, ast.Call):
            func = node.func
            qualifier = ""
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
                if isinstance(func.value, ast.Name):
                    qualifier = func.value.id
            ref = (qualifier, name)
            if name and ref not in calls:
                calls.append(ref)
        for child in ast.iter_child_nodes(node):
            visit_value(child)

    for value in returns:
        visit_value(value)
    return calls


def _response_dependency_calls(text: str) -> list[str]:
    """Compatibility name-only view used by focused unit tests/debugging."""
    return [name for _, name in _response_dependency_refs(text)]


def _import_alias_paths(code_root: str, relpath: str,
                        py_files: set[str] | None = None) -> dict[str, str]:
    """Resolve imported module/function aliases in one Python source file."""
    text = _read_text(code_root, relpath)
    if not text:
        return {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    py_files = py_files if py_files is not None else _list_py_files(code_root)
    out: dict[str, str] = {}

    def unique(candidates: list[str]) -> str | None:
        matches = {
            rel for rel in py_files
            for candidate in candidates
            if rel == candidate or rel.endswith("/" + candidate)
        }
        return next(iter(matches)) if len(matches) == 1 else None

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.replace(".", "/")
            for name in node.names:
                alias = name.asname or name.name
                target = unique([
                    f"{module}/{name.name}.py",
                    f"{module}/{name.name}/__init__.py",
                ])
                if target is None:
                    target = unique([f"{module}.py", f"{module}/__init__.py"])
                if target:
                    out[alias] = target
        elif isinstance(node, ast.Import):
            for name in node.names:
                alias = name.asname or name.name.split(".")[0]
                module = name.name.replace(".", "/")
                target = unique([f"{module}.py", f"{module}/__init__.py"])
                if target:
                    out[alias] = target
    return out


def _is_response_producer(text: str) -> bool:
    """Whether a returned value is assembled/read at this function."""
    if re.search(r"(?is)\bSELECT\b.+\bFROM\b", text or ""):
        return True
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, (ast.Dict, ast.ListComp, ast.DictComp))
        for node in ast.walk(tree)
    )


def _resolve_http_producer_paths(bindings: list[dict[str, Any]], code_root: str,
                                 max_hops: int = 4,
                                 max_defs_per_call: int = 2) -> list[dict[str, Any]]:
    """Trace each proven winning HTTP handler to its response-value producers.

    Unlike the generic call-chain follower, this follows only calls that feed a
    ``return`` value and carries the URL/handler identity on every hop. That makes the
    result a deterministic winning-request-path proof instead of an unordered bag of
    same-named functions.
    """
    out: list[dict[str, Any]] = []
    py_files = _list_py_files(code_root)
    for binding in bindings or []:
        if binding.get("ambiguous") or not binding.get("winning", True):
            continue
        url = str(binding.get("url", "") or "")
        handler_file = str(binding.get("file", "") or "")
        frontier = [(binding, 0)]
        seen_defs: set[str] = {f"{handler_file}:{binding.get('lines', '')}"}
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            text = str(current.get("text", "") or "")
            # A SQL-returning function is the concrete row producer. Do not descend
            # into generic store helpers such as _fetch_all, which are shared plumbing.
            if depth > 0 and re.search(r"(?is)\bSELECT\b.+\bFROM\b", text):
                continue
            aliases = _import_alias_paths(
                code_root, str(current.get("file", "") or ""), py_files)
            for qualifier, symbol in _response_dependency_refs(text):
                if symbol in _RESPONSE_STORE_CALLS:
                    continue
                target = aliases.get(qualifier or symbol)
                globs = [target] if target else []
                hits = _ripgrep(rf"def {re.escape(symbol)}\b", globs, code_root,
                                max_hits=max_defs_per_call + 1)
                if not hits or len(hits) > max_defs_per_call:
                    continue
                for h in hits[:max_defs_per_call]:
                    key = f"{h['file']}:{h['line']}"
                    if key in seen_defs:
                        continue
                    seen_defs.add(key)
                    w = _read_def_body(code_root, h["file"], h["line"])
                    if not w["text"]:
                        continue
                    node = {
                        "url": url,
                        "verb": binding.get("verb", ""),
                        "full_path": binding.get("full_path", ""),
                        "handler_file": handler_file,
                        "file": h["file"],
                        "lines": w["lines"],
                        "text": (
                            f"# WINNING HTTP RESPONSE PATH (hive): {url} "
                            f"handler {handler_file} -> {symbol}\n" + w["text"]
                        ),
                        "symbol": symbol,
                        "via": "http-producer",
                        "winning": True,
                        "path_depth": depth + 1,
                        "producer": _is_response_producer(w["text"]),
                    }
                    out.append(node)
                    frontier.append((node, depth + 1))
    return out


def _covered_ranges(snippets: list[dict[str, Any]], rel: str) -> list[tuple[int, int]]:
    """The (lo, hi) line ranges already windowed for ``rel`` in the bundle."""
    ranges: list[tuple[int, int]] = []
    for s in snippets:
        if (s.get("file") or "").replace("\\", "/") != rel:
            continue
        m = re.match(r"(\d+)(?:-(\d+))?", str(s.get("lines", "")).strip())
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        ranges.append((lo, hi))
    return ranges


def _harvest_inscope_fetch_urls(snippets: list[dict[str, Any]], code_root: str,
                                k: int = 4, max_files: int = 8,
                                max_urls_per_file: int = 6) -> list[dict[str, Any]]:
    """Pull HTTP-call URL literals from files ALREADY in the bundle into scope.

    ``_resolve_http_bindings`` can only cross the FE→BE boundary for a fetch URL that
    a keyword window happened to capture. When the symptom is "UI variable X stays
    empty", keyword density clusters on the RENDER / assignment of X, and the
    ``getRequest('/api/v1/projects')`` that ACTUALLY feeds X frequently sits in a GAP
    BETWEEN those windows (observed: NewRequirementModal's fetch at line 242 fell
    between the ``module`` windows at 228±k and 265±k). The producer chain — route
    handler → service → store query — then never enters the bundle, so the converger
    sees only the symptom-side red herrings and cannot reach the real defect (a store
    query that hardcodes the empty field).

    For each distinct file that already contributed a snippet, grep its HTTP-call URL
    literals and add a small ±k window around any whose line is NOT already covered by
    an in-scope window. The downstream :func:`_resolve_http_bindings` then resolves the
    newly-surfaced URL to its backend, and the call-follow unrolls the rest of the
    producer chain (the existing machinery already reaches the leaf store query once
    the URL is present — this is purely the missing UPSTREAM step).

    Pure-local, deterministic, zero model cost; never invents (only real lines from
    files already in scope); bounded by ``max_files`` / ``max_urls_per_file``. This is
    GROUNDING — it adds context, it gates nothing; a spurious harvest is at worst a
    real-but-unused fetch window (mild noise), never a wrong edit.
    """
    seen_files: list[str] = []
    for s in snippets:
        rel = (s.get("file") or "").replace("\\", "/")
        if rel and rel not in seen_files:
            seen_files.append(rel)

    out: list[dict[str, Any]] = []
    for rel in seen_files[:max_files]:
        text = _read_text(code_root, rel)
        if not text:
            continue
        covered = _covered_ranges(snippets, rel)
        added_lines: list[int] = []
        added = 0
        for i, line in enumerate(text.splitlines(), start=1):
            if added >= max_urls_per_file:
                break
            m = _HTTP_CALL_RE.search(line)
            if not m:
                continue
            if m.group("callee").split(".")[-1].lower() not in _HTTP_CALLEES:
                continue
            if any(lo <= i <= hi for lo, hi in covered):
                continue                          # the fetch is already in scope
            if any(abs(i - j) <= k for j in added_lines):
                continue                          # already harvested an adjacent fetch
            w = _read_window(code_root, rel, i, k)
            if not w["text"]:
                continue
            out.append({"file": rel, "lines": w["lines"], "text": w["text"],
                        "via": "fetch-harvest"})
            added_lines.append(i)
            added += 1
    return out


# ── FIELD-PRODUCER grounding (cross-language response-field provenance) ──────────
# A "screen shows the wrong X" symptom binds a FE field to a backend value; the bug
# lives in the SERVER code that PRODUCES that field, but keyword density clusters on
# the render side and the converger grounds on a lexically-similar DECOY. The head
# strip is the canonical case: the field the FE actually reads is ``workflow_head_type``
# (filled by ``out["workflow_head_type"] = head_type`` in documents.py, right under the
# ``not in NON_HEAD_TYPES`` exclusion that IS the bug), yet a SQL helper merely *named*
# ``get_effective_head`` — never used by the FE — outscores the real producer on the
# word "head". :func:`_resolve_http_bindings` crosses the REQUEST-PATH boundary (URL →
# route); it does NOT trace a RESPONSE FIELD back to the code that fills it. This does.
#
# Discriminator (codebase-agnostic): a snake_case identifier appearing in FE code is
# almost certainly a backend response field — FE locals are camelCase, so a snake_case
# token at the FE↔BE seam (``doc.value?.workflow_head_type``) is the field name as the
# server serialized it. For each such field in the bundle, grep the tree for its
# PRODUCING site — a quoted key followed by ``:`` (serialization) or by ``]``/``=``
# (dict-subscript assignment ``out["field"] = …``) — and surface a window around it.
# Reads (``x.field``, ``.get("field")``) and type-decls (``field?: T``) lack that
# quote+[:=] shape, so FE reads and backend locals self-filter; only genuine serialized
# fields surface — the producer PATTERN is the gate, no separate gate needed.
#
# Pure GROUNDING, mirrors _harvest_inscope_fetch_urls / _resolve_http_bindings: real
# lines only, never invents, gates NOTHING. A spurious fire is at worst a real-but-
# unused producer window (mild noise), never a wrong edit. Self-limiting: a field
# nobody serializes yields nothing, and a field serialized in too many places is
# dropped as a non-discriminating common key (e.g. ``project_id``).

# A snake_case identifier: lowercase, at least one ``_`` boundary (so ≥2 segments).
# camelCase FE locals and single bare words (``id``, ``name``) never match.
_SNAKE_FIELD_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")

# HARVEST only from front-end source: a snake_case token in a FE file is the
# discriminator (FE locals are camelCase, so snake_case = a backend response field
# read at the seam). Harvesting from server ``.py``/``.json``/``.sql`` instead pulls in
# SQL query names (``get_effective_head``), DB columns (``sort_order``) and library
# params — none of which are FE-bound response fields — and floods the producer search
# with noise (observed: m035 reconverge harvested 26 such non-fields). Restricting to
# FE extensions makes the snake_case signal mean what it is supposed to mean.
_FE_EXTS = (".vue", ".jsx", ".tsx", ".svelte", ".ts", ".js", ".mjs", ".cjs")

# Search everything for a field's producer EXCEPT test trees and vendored deps. A test
# asserting ``parsed["field"] == "X"`` is a consumer, not a producer; a vendored
# ``site-packages`` match (``return_value`` in httpcore) is never our field. Both would
# otherwise pollute the hit count (``rg -g`` applies these as include/exclude rules).
_PRODUCER_SEARCH_GLOBS = [
    "*", "!**/tests/**", "!**/test/**", "!**/__tests__/**",
    "!**/test_*.*", "!**/*_test.*", "!**/*.spec.*", "!**/*.test.*",
    "!**/.venv/**", "!**/venv/**", "!**/site-packages/**",
    "!**/node_modules/**", "!**/dist/**", "!**/build/**", "!**/.git/**",
]


def _resolve_field_producers(snippets: list[dict[str, Any]], code_root: str,
                             min_len: int = 8, max_fields: int = 10,
                             max_producers_per_field: int = 3,
                             max_total_hits: int = 6, k: int = 8) -> list[dict[str, Any]]:
    """Surface the backend code that PRODUCES a snake_case response field the FE reads.

    See the section header above for the why. Harvest snake_case field tokens from the
    bundle (most-mentioned first — the symptom field recurs), then for each grep the
    tree for its producing site and window it in. Pure-local, deterministic, never
    invents; bounded by ``max_fields`` / ``max_producers_per_field`` / ``max_total_hits``.
    """
    freq: dict[str, int] = defaultdict(int)
    for s in snippets:
        if not (s.get("file") or "").lower().endswith(_FE_EXTS):
            continue  # snake_case is only a "response field" signal in FE source
        for m in _SNAKE_FIELD_RE.finditer(s.get("text", "")):
            tok = m.group(1)
            if len(tok) >= min_len:
                freq[tok] += 1
    if not freq:
        return []
    fields = sorted(freq, key=lambda t: (-freq[t], t))[:max_fields]

    out: list[dict[str, Any]] = []
    for field in fields:
        # Coarse grep (rg's Rust engine has no lookahead): a quoted key followed by
        # ``:`` (serialization) or by ``]``/``=`` (dict-subscript assignment). Exclude
        # test trees up front — a test ASSERTING a field is not a producer and would
        # otherwise flood the count. The ``==`` reads this coarse pattern still admits
        # are dropped by the precise Python re-check below.
        pat = rf"""["']{re.escape(field)}["']\s*\]?\s*[:=]"""
        confirm = re.compile(rf"""["']{re.escape(field)}["']\s*\]?\s*(?::|=(?!=))""")
        raw = _ripgrep(pat, _PRODUCER_SEARCH_GLOBS, code_root, max_hits=max_total_hits + 4)
        # precise filter: a SINGLE ``=`` (assignment) or ``:`` (serialization), never
        # ``==`` (an equality read) — the producer vs. consumer distinction.
        hits = [h for h in raw if confirm.search(h.get("text", ""))]
        if not hits or len(hits) > max_total_hits:
            continue  # 0 producers, or a common key serialized everywhere → not a pinpoint
        added_by_file: dict[str, list[int]] = defaultdict(list)
        added = 0
        for h in hits:
            if added >= max_producers_per_field:
                break
            rel, ln = h["file"], h["line"]
            if rel.startswith("./"):
                rel = rel[2:]  # rg prefixes paths with ``./`` — match bundle convention
            if any(lo <= ln <= hi for lo, hi in _covered_ranges(snippets, rel)):
                continue  # producer already in scope
            if any(abs(ln - j) <= k for j in added_by_file[rel]):
                continue  # adjacent producer in this file already surfaced
            w = _read_window(code_root, rel, ln, k)
            if not w["text"]:
                continue
            header = (f"# RESOLVED FIELD-PRODUCER (hive): backend fills response field "
                      f"{field!r} here ← the FE reads this field\n")
            out.append({"file": rel, "lines": w["lines"], "text": header + w["text"],
                        "via": "field-producer", "field": field})
            added_by_file[rel].append(ln)
            added += 1
    return out


# ── PEER-IMPLEMENTATION grounding (sibling-pattern resolver).
# The fix for a class of defects is not in the buggy file's own numbers but in how
# the codebase ALREADY solves the same concern in a SIBLING (z-index/stacking: a
# toast sits behind a modal not because its z-index is low — it is 2000 — but
# because a sibling overlay escapes its stacking context via ``<Teleport to="body">``
# and the toast does not). Keyword/density retrieval can't see this: the discriminating
# fact is the ASYMMETRY between two files, not text inside one. Same family as
# :func:`_resolve_http_bindings` (cross-file join, deterministic, never invents).
#
# DISCIPLINE — fires ONLY at the necessary moment, by construction:
#   1. signal gate   — a file ALREADY in the bundle must carry the concern's signal
#                      (so the axis is already looking at it); else stays silent.
#   2. asymmetry gate — its same-dir sibling overlays must handle the concern
#                      INCONSISTENTLY; if they all match the target, nothing is shown.
#   3. never invents  — only real sibling lines are lifted (cf. :func:`_resolve_key`).
# The downside of a spurious fire is a real-but-irrelevant sibling snippet (mild
# noise), never a fabricated mismatch.
#
# GENERAL mechanism, TIGHT enumerable registry: add a concern (each needing a
# concrete greppable signal + an asymmetry rule) ONE at a time — same growth path as
# ``_HTTP_CALLEES`` (started with FlowGate's FastAPI form, Express deferred). Today
# the registry holds exactly one concern: stacking/layering.

_PEER_EXTS = frozenset({".vue", ".jsx", ".tsx", ".js", ".ts", ".svelte"})

# Stacking signals. ``z-?index`` matches CSS ``z-index:`` and JS ``zIndex:`` (case-
# insensitive). Teleport/portal is how an overlay ESCAPES its parent stacking context.
_STK_TELEPORT_RE = re.compile(r"<\s*(?:teleport|portal)\b|createportal\b", re.IGNORECASE)
_STK_POSITION_RE = re.compile(r"position\s*:\s*(fixed|absolute|sticky)\b", re.IGNORECASE)
_STK_ZINDEX_RE = re.compile(r"z-?index\s*:\s*(\d+)", re.IGNORECASE)


@dataclass
class _PeerConcern:
    """One registered concern for :func:`_resolve_peer_patterns`.

    ``profile`` reads a file's text → an approach dict carrying ``applies`` (signal
    gate). ``asymmetric`` decides whether target vs sibling approaches disagree
    enough to surface. ``render`` produces the prompt block. Adding a concern is
    adding one of these — the resolver loop is concern-agnostic.
    """

    name: str
    profile: Callable[[str], dict[str, Any]]
    asymmetric: Callable[[dict[str, Any], list[dict[str, Any]]], bool]
    render: Callable[[str, dict[str, Any], list[tuple[str, dict[str, Any]]]], str]


def _stacking_profile(text: str) -> dict[str, Any]:
    """A file's approach to stacking/layering. ``applies`` = it is a layered overlay."""
    pos = _STK_POSITION_RE.search(text)
    zs = [int(m.group(1)) for m in _STK_ZINDEX_RE.finditer(text)]
    return {
        "applies": bool(pos) or bool(zs),     # an overlay (positioned and/or z-indexed)
        "teleports": bool(_STK_TELEPORT_RE.search(text)),
        "zindex": max(zs) if zs else None,
        "position": pos.group(1).lower() if pos else None,
    }


def _stacking_asymmetric(t: dict[str, Any], sibs: list[dict[str, Any]]) -> bool:
    """Fire only when overlay siblings DISAGREE with the target on context-escape.

    Teleport/portal usage is the discriminating, actionable signal (a raw z-index
    delta is noisy — many legitimate values coexist). If every comparable sibling
    escapes the same way the target does, there is nothing to show → silent.
    """
    return any(s["teleports"] != t["teleports"] for s in sibs)


def _stacking_render(rel: str, tprof: dict[str, Any],
                     sib_profs: list[tuple[str, dict[str, Any]]]) -> str:
    """Render the target-vs-siblings stacking comparison for the judge/author."""
    def _line(name: str, p: dict[str, Any]) -> str:
        bits = [f"teleport={'YES' if p['teleports'] else 'NO'}"]
        if p.get("zindex") is not None:
            bits.append(f"z-index={p['zindex']}")
        if p.get("position"):
            bits.append(f"position:{p['position']}")
        return f"{name}: " + ", ".join(bits)

    out = [
        "# PEER PATTERN (hive): stacking/layering — this overlay and its sibling "
        "overlays escape their stacking context INCONSISTENTLY. A high z-index alone "
        "does not lift an element above a sibling drawn in a higher stacking context; "
        "matching how the siblings escape (e.g. <Teleport to=\"body\">) is usually the "
        "real fix, not bumping the number. Compare:",
        _line(rel + "  <-- target", tprof),
    ]
    out += [_line(sib, p) for sib, p in sib_profs]
    return "\n".join(out)


_PEER_CONCERNS: list[_PeerConcern] = [
    _PeerConcern("stacking", _stacking_profile, _stacking_asymmetric, _stacking_render),
]


def _read_text(root: str, relpath: str) -> str:
    """Read a whole file's text ('' on failure). For peer-profile comparison."""
    try:
        with open(os.path.join(root, relpath), "r",
                  encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _peer_files(code_root: str, rel_target: str) -> list[str]:
    """Same-directory siblings of ``rel_target`` in the component-class extensions."""
    rel_target = rel_target.replace("\\", "/")
    d = os.path.dirname(rel_target)
    abs_d = os.path.join(code_root, d) if d else code_root
    sibs: list[str] = []
    try:
        for name in sorted(os.listdir(abs_d)):
            rel = f"{d}/{name}" if d else name
            if rel == rel_target:
                continue
            if os.path.splitext(name)[1].lower() in _PEER_EXTS:
                sibs.append(rel)
    except OSError:
        pass
    return sibs


def _resolve_peer_patterns(snippets: list[dict[str, Any]], code_root: str,
                           max_targets: int = 3,
                           max_siblings: int = 12) -> list[dict[str, Any]]:
    """Surface how SIBLINGS already solve a concern this file handles differently.

    For each component file in the bundle that carries a registered concern's signal
    (gate 1), compare its approach to its same-dir sibling overlays; emit a block
    ONLY when they disagree (gate 2). Pure-local, deterministic, zero model cost;
    never fabricates (only real sibling lines). See the registry note above.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for s in snippets:
        rel = (s.get("file") or "").replace("\\", "/")
        if not rel or rel in seen:
            continue
        if os.path.splitext(rel)[1].lower() not in _PEER_EXTS:
            continue
        target_text = _read_text(code_root, rel)
        if not target_text:
            continue
        seen.add(rel)
        for concern in _PEER_CONCERNS:
            tprof = concern.profile(target_text)
            if not tprof.get("applies"):
                continue                                  # gate 1: signal absent
            sib_profs: list[tuple[str, dict[str, Any]]] = []
            for sib in _peer_files(code_root, rel)[:max_siblings]:
                sp = concern.profile(_read_text(code_root, sib))
                if sp.get("applies"):                     # only comparable overlays
                    sib_profs.append((sib, sp))
            if not sib_profs:
                continue
            if not concern.asymmetric(tprof, [p for _, p in sib_profs]):
                continue                                  # gate 2: siblings agree
            out.append({
                "file": rel, "lines": s.get("lines", "1-1"),
                "text": concern.render(rel, tprof, sib_profs),
                "via": "peer-pattern", "concern": concern.name,
                "siblings": [sib for sib, _ in sib_profs],
            })
            break                                         # one concern per target
        if len(out) >= max_targets:
            break
    return out


def retrieve(plan: SearchPlan, code_root: str, docs_root: str | None = None,
             k: int = 6, top_files: int = 8,
             blame_files: int = 3, max_hops: int = 2,
             overbroad_files: int = 2000) -> dict[str, Any]:
    """Run the full local FIND for one axis. Pure local, zero model cost.

    Args:
        plan: the axis search plan (keywords/globs/doc_topics).
        code_root: target codebase root.
        docs_root: design docs root (for doc_topics grep). Optional.
        k: ±lines read around each hit.
        top_files: max distinct files to pull code windows from.
        blame_files: max files to run git blame/log on.
        overbroad_files: a glob matching more than this many files is treated as
            over-broad and dropped when a narrower glob exists.
    """
    # 0. Route each glob to the tree it actually points at, THEN validate the
    #    code-tree globs against the real tree (free): drop garbage (0-match) and
    #    over-broad globs so the snippet budget windows the axis, not noise.
    #    Whole-tree fallback only if every glob is garbage. Docs-tree globs are
    #    handled by the design-doc channel below (step 5), not searched in code.
    code_globs, doc_globs = _partition_globs(plan.file_globs, code_root, docs_root)
    globs, glob_diag = _validate_globs(code_globs, code_root, overbroad_files)

    # 1-3. ripgrep keywords → rank files → window densest clusters.
    scan = _scan_code(plan.keywords, globs, code_root, k, top_files)

    # 3a. Extension-blind fallback (T889): the queen named extensions the tree
    #     does not use (``*.js``/``*.tsx`` for a Vue 3 + TS app whose sites are
    #     ``.vue``/``.ts``), so the scope was right but every snippet missed. When
    #     the first pass found NOTHING, drop the extension constraint and re-search
    #     ONCE. Deterministic & free; never touches the queen's (non-deterministic)
    #     decompose. Adopt the wider result only if it actually recovers snippets.
    widen_diag: dict[str, Any] | None = None
    if not scan["code_snippets"] and globs:
        wide_globs, wide_glob_diag = _validate_globs(
            _widen_globs(globs), code_root, overbroad_files)
        if set(wide_globs) != set(globs):
            rescan = _scan_code(plan.keywords, wide_globs, code_root, k, top_files)
            widen_diag = {"widened_globs": wide_globs, "from": globs,
                          "recovered": bool(rescan["code_snippets"]),
                          "validation": wide_glob_diag}
            if rescan["code_snippets"]:
                scan, globs = rescan, wide_globs

    call_sites = scan["call_sites"]
    file_hits = scan["file_hits"]
    file_lines = scan["file_lines"]
    ranked = scan["ranked"]
    densest_line = scan["densest_line"]
    code_snippets = scan["code_snippets"]

    # 3b. follow call-chains out of the code snippets (1-2 hops, local & free).
    call_chain = _follow_calls(code_snippets, globs, code_root,
                               k, max_hops) if max_hops > 0 else []

    # 3c. resolve label/i18n references so near-identical sibling elements are
    #     told apart by MEANING, not by the judge guessing what a call renders to
    #     (T891: getLabel('R')->"요건정의" vs t('...undecided')->"미정"). The cheap
    #     local READ the operator otherwise had to do by hand. Annotates snippets
    #     (and any call-chain windows) with a `resolved` field in place.
    resolved_refs = _resolve_discriminators(
        code_snippets + call_chain, code_root, docs_root)

    # 3d. resolve HTTP call-binding edges (N177): match each client fetch-URL
    #     literal to the backend route handler that fills it — the cross-language
    #     string join keyword retrieval can't build, so the judge otherwise grounds
    #     on a lexically-similar but wrong handler. Append the resolved handlers to
    #     call_chain so the judge consumes them with the rest of the evidence.
    # 3d-pre: harvest fetch-URL literals from files ALREADY in the bundle so the
    #     boundary resolver below sees the fetch that FEEDS the symptom even when
    #     keyword density landed BETWEEN the windows and missed the fetch line. Without
    #     this, a "UI variable stays empty" symptom strands the converger on the
    #     render/assignment side and the producer chain (route → service → store query)
    #     never enters the bundle. Pure grounding; gates nothing.
    fetch_harvest = _harvest_inscope_fetch_urls(code_snippets + call_chain, code_root)
    http_bindings = _resolve_http_bindings(
        code_snippets + call_chain + fetch_harvest, code_root)
    # Follow calls OUT of the resolved handlers (whole-tree, since the handler is
    # server-side while the axis globs are usually client-side): a route handler
    # commonly DELEGATES (get_document_rpc → get_document → _parse_doc_workflow),
    # so the real source is one or two hops past the handler the URL points at.
    binding_follow = (_follow_calls(http_bindings, [], code_root, k, max_hops)
                      if http_bindings and max_hops > 0 else [])
    http_producers = _resolve_http_producer_paths(http_bindings, code_root)
    call_chain = (call_chain + http_bindings + http_producers
                  + binding_follow + fetch_harvest)

    # 3d-2. field-producer grounding (N183 round-2): trace a snake_case RESPONSE FIELD
    #     the FE reads back to the backend code that FILLS it. http-binding above
    #     crosses the URL→route boundary but never field→producer, so a "screen shows
    #     wrong X" symptom strands the converger on a lexically-similar decoy (the head
    #     strip: FE reads ``workflow_head_type``, filled in documents.py, but a SQL
    #     helper merely *named* get_effective_head — unused by the FE — outscores it on
    #     "head"). Pure grounding; rides in call_chain so the judge/converger consume it
    #     with the rest of the evidence (no judge.py change).
    field_producers = _resolve_field_producers(code_snippets + call_chain, code_root)
    call_chain = call_chain + field_producers

    # 3e. peer-implementation grounding: when a component file in the bundle handles a
    #     registered concern (today: stacking/layering) DIFFERENTLY from its same-dir
    #     sibling overlays, surface the asymmetry — the discriminating fact is between
    #     two files (toast lacks <Teleport> a sibling uses), invisible to keyword
    #     retrieval. Self-gating: silent unless a signal-bearing file is present AND its
    #     siblings disagree. Rides in call_chain so the judge consumes it (no judge.py change).
    peer_patterns = _resolve_peer_patterns(code_snippets + call_chain, code_root)
    call_chain = call_chain + peer_patterns

    # 4. git blame/log on the highest-ranked files, around their densest region.
    git_history = []
    for f in ranked[:blame_files]:
        line = densest_line.get(f, file_lines[f][0][0] if file_lines[f] else 1)
        git_history.append(_git_history(code_root, f, line, k))

    # 5. design-doc grep (doc_topics) → excerpts. Rank docs by topic-coverage
    #    and cap to top_files, same density discipline as code — otherwise a
    #    generic topic ("head") floods the bundle with hundreds of md hits and
    #    blows the token budget the whole redesign exists to protect.
    #    The grep scope is the queen's explicit docs-tree globs when she gave any
    #    (so a "edit D031" axis actually pulls D031), else the whole docs tree.
    #    With explicit doc targets we also grep the axis KEYWORDS, not just the
    #    (often generic) doc_topics, so the relevant section lands in the bundle.
    design_excerpts: list[dict[str, Any]] = []
    doc_scope = doc_globs or ["*.md"]
    doc_terms = list(plan.doc_topics)
    if doc_globs:
        doc_terms += [kw for kw in plan.keywords if kw not in doc_terms]
    if not doc_terms:
        doc_terms = list(plan.keywords)
    doc_k = max(k, 12)  # docs need a wider window to capture the enclosing heading
    if docs_root and doc_terms and (plan.doc_topics or doc_globs):
        doc_topics_hit: dict[str, set[str]] = defaultdict(set)
        doc_first_line: dict[str, dict[str, int]] = defaultdict(dict)
        doc_hit_count: dict[str, int] = defaultdict(int)
        for topic in doc_terms:
            for h in _ripgrep(topic, doc_scope, docs_root, max_hits=20):
                doc_topics_hit[h["file"]].add(topic)
                doc_first_line[h["file"]].setdefault(topic, h["line"])
                doc_hit_count[h["file"]] += 1
        ranked_docs = sorted(
            doc_topics_hit.keys(),
            key=lambda d: (len(doc_topics_hit[d]), doc_hit_count[d]),
            reverse=True,
        )
        raw_doc_snips: list[dict[str, Any]] = []
        for d in ranked_docs[:top_files]:
            for topic, ln in doc_first_line[d].items():
                w = _read_window(docs_root, d, ln, doc_k)
                raw_doc_snips.append({
                    "file": d, "lines": w["lines"], "text": w["text"],
                    "hits": [topic],
                })
        design_excerpts = [
            {"doc": s["file"], "lines": s["lines"], "text": s["text"],
             "topics": s.get("hits", [])}
            for s in _merge_windows(raw_doc_snips)
        ]

    return {
        "axis_id": plan.axis_id,
        "code_snippets": code_snippets,
        "call_chain": call_chain,
        "call_sites": call_sites,
        "http_bindings": http_bindings,
        "http_producers": http_producers,
        "peer_patterns": peer_patterns,
        "git_history": git_history,
        "design_excerpts": design_excerpts,
        "stats": {
            "raw_hits": len(call_sites),
            "files_hit": len(file_hits),
            "files_windowed": min(len(ranked), top_files),
            "snippets": len(code_snippets),
            "call_chain": len(call_chain),
            "http_bindings": len(http_bindings),
            "http_bindings_ambiguous": sum(1 for b in http_bindings if b.get("ambiguous")),
            "http_producers": len(http_producers),
            "fetch_harvest": len(fetch_harvest),
            "field_producers": len(field_producers),
            "peer_patterns": len(peer_patterns),
            "resolved_refs": resolved_refs,
            "design_excerpts": len(design_excerpts),
            "ranked_files": ranked[:top_files],
            "globs_used": globs,
            "glob_validation": glob_diag,
            "glob_widening": widen_diag,
        },
    }


def retrieve_followup(need: FollowupNeed, code_root: str,
                      k: int = 6, max_hops: int = 2,
                      max_per_seed: int = 10) -> dict[str, Any]:
    """One bounded, JUDGE-directed local re-search (M004 §2 step-3).

    Unlike :func:`retrieve` (whose follow is seeded by keyword *density* and so
    starves on low-density call sites), the seed here is whatever the judge
    NAMED in its ``need`` — so the very call-chain hops that density retrieval
    misses become reachable. Still pure-local and zero model cost; the model
    cost is the judge call that *produces* the need, not this re-search.

    Returns a supplemental bundle ``{axis_id, seeds[], call_chain[], stats}``
    meant to be merged with the first-pass bundle before the re-judge. Stays
    within the §4 budget of ≤1 follow-up per axis (≤2 model calls total).
    """
    globs, glob_diag = _validate_globs(need.file_globs, code_root)
    seeds: list[dict[str, Any]] = []
    seen_defs: set[str] = set()

    def _add(snip: dict[str, Any]) -> None:
        key = f"{snip['file']}:{snip['lines']}"
        if key not in seen_defs:
            seen_defs.add(key)
            seeds.append(snip)

    # 1. resolve each named symbol → its full def/class BODY (not a ±k slice —
    #    the bug the judge is chasing usually lives below the signature).
    for sym in need.symbols:
        for h in _ripgrep(rf"def {sym}\b", globs, code_root,
                          max_hits=max_per_seed):
            w = _read_def_body(code_root, h["file"], h["line"])
            _add({"file": h["file"], "lines": w["lines"], "text": w["text"],
                  "symbol": sym, "via": "need-symbol"})

    # 2. locate each requested grep pattern → body window if the hit is a def,
    #    else a ±k window around the hit line.
    for pat in need.greps:
        for h in _ripgrep(pat, globs, code_root, max_hits=max_per_seed):
            is_def = h["text"].lstrip().startswith(("def ", "async def "))
            w = (_read_def_body(code_root, h["file"], h["line"]) if is_def
                 else _read_window(code_root, h["file"], h["line"], k))
            _add({"file": h["file"], "lines": w["lines"], "text": w["text"],
                  "symbol": pat, "via": "need-grep"})

    seeds = _merge_windows(seeds)

    # 3. follow call-chains OUT of the judge-named seeds (now well-fed).
    call_chain = _follow_calls(seeds, globs, code_root, k, max_hops) \
        if max_hops > 0 else []

    return {
        "axis_id": need.axis_id,
        "seeds": seeds,
        "call_chain": call_chain,
        "stats": {
            "symbols": len(need.symbols),
            "greps": len(need.greps),
            "seeds": len(seeds),
            "call_chain": len(call_chain),
            "globs_used": globs,
            "glob_validation": glob_diag,
        },
    }
