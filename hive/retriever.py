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

import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

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
    hi = idx + 1
    for j in range(idx + 1, min(len(all_lines), idx + max_lines)):
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
                    w = _read_window(root, h["file"], h["line"], k)
                    snip = {"file": h["file"], "lines": w["lines"],
                            "text": w["text"], "symbol": name,
                            "via": "call-chain"}
                    follows.append(snip)
                    next_frontier.append(snip)
        if not next_frontier:
            break
        frontier = next_frontier
    return follows


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
    # 0. validate the queen's globs against the real tree (free): drop garbage
    #    (0-match) and over-broad globs so the snippet budget windows the axis,
    #    not noise. Whole-tree fallback only if every glob is garbage.
    globs, glob_diag = _validate_globs(plan.file_globs, code_root, overbroad_files)

    # 1. ripgrep every keyword across globs → raw call_sites, keep ALL hit
    #    lines per file (not just first) so dense late regions stay reachable.
    call_sites: list[dict[str, Any]] = []
    file_hits: dict[str, set[str]] = defaultdict(set)       # file -> keywords
    file_lines: dict[str, list[tuple[int, str]]] = defaultdict(list)  # (line,kw)
    for kw in plan.keywords:
        for h in _ripgrep(kw, globs, code_root):
            h["keyword"] = kw
            call_sites.append(h)
            file_hits[h["file"]].add(kw)
            file_lines[h["file"]].append((h["line"], kw))

    # 2. rank files by keyword-coverage (distinct keywords) then total hits.
    hit_count: dict[str, int] = defaultdict(int)
    for h in call_sites:
        hit_count[h["file"]] += 1
    ranked = sorted(
        file_hits.keys(),
        key=lambda f: (len(file_hits[f]), hit_count[f]),
        reverse=True,
    )

    # 3. for top files, window the densest keyword CLUSTERS (top 2 per file).
    raw_snips: list[dict[str, Any]] = []
    densest_line: dict[str, int] = {}  # file -> centroid of its top cluster
    for f in ranked[:top_files]:
        clusters = _cluster_lines(file_lines[f], k, max_clusters=2)
        if clusters:
            densest_line[f] = (clusters[0]["lo"] + clusters[0]["hi"]) // 2
        for c in clusters:
            w = _read_window(code_root, f, (c["lo"] + c["hi"]) // 2, k)
            raw_snips.append({
                "file": f, "lines": w["lines"], "text": w["text"],
                "hits": c["kws"],
            })
    code_snippets = _merge_windows(raw_snips)

    # 3b. follow call-chains out of the code snippets (1-2 hops, local & free).
    call_chain = _follow_calls(code_snippets, globs, code_root,
                               k, max_hops) if max_hops > 0 else []

    # 4. git blame/log on the highest-ranked files, around their densest region.
    git_history = []
    for f in ranked[:blame_files]:
        line = densest_line.get(f, file_lines[f][0][0] if file_lines[f] else 1)
        git_history.append(_git_history(code_root, f, line, k))

    # 5. design-doc grep (doc_topics) → excerpts. Rank docs by topic-coverage
    #    and cap to top_files, same density discipline as code — otherwise a
    #    generic topic ("head") floods the bundle with hundreds of md hits and
    #    blows the token budget the whole redesign exists to protect.
    design_excerpts: list[dict[str, Any]] = []
    if docs_root and plan.doc_topics:
        doc_topics_hit: dict[str, set[str]] = defaultdict(set)
        doc_first_line: dict[str, dict[str, int]] = defaultdict(dict)
        doc_hit_count: dict[str, int] = defaultdict(int)
        for topic in plan.doc_topics:
            for h in _ripgrep(topic, ["*.md"], docs_root, max_hits=20):
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
                w = _read_window(docs_root, d, ln, k)
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
        "git_history": git_history,
        "design_excerpts": design_excerpts,
        "stats": {
            "raw_hits": len(call_sites),
            "files_hit": len(file_hits),
            "files_windowed": min(len(ranked), top_files),
            "snippets": len(code_snippets),
            "call_chain": len(call_chain),
            "design_excerpts": len(design_excerpts),
            "ranked_files": ranked[:top_files],
            "globs_used": globs,
            "glob_validation": glob_diag,
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
