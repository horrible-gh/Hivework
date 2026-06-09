"""Hook A (M028): discover the backend root-cause locus via the code-map and
inject it as a front axis the reading loop judges.

The queen is blind to the call graph, so it name-matches a backend file and
misses the real gate/producer one hop away (project_settings, process_service,
documents.py.workflow_head_type) — unfixable by model or prompt (measured).
Here the READING stage queries a static code-map (hive.codemap, M027) to reach
that locus and adds it as a CODEMAP_BE_ROOT axis, mirroring the existing
SEED_ANCHOR injection so it is judged and not truncated.

Depends only on the hive.codemap M027 API; degrades to a no-op when codemap is
absent (e.g. before GPT's module lands) or finds nothing. Pure, deterministic,
no model.
"""
import json
import logging
import os
import re

logger = logging.getLogger("hive.be_root")

try:                                  # graceful until hive/codemap.py exists
    from hive import codemap as _codemap
except Exception:                     # pragma: no cover
    _codemap = None

_GENERIC_TAIL = frozenset({"create", "list", "get", "update", "delete", "new",
                           "edit", "save", "fetch", "all", "index"})
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_BE_ROOT_CAP = 6


def _endpoint_tail(ep):
    segs = [s for s in ep.split("/")
            if s and not s.startswith("{") and len(s) >= 4
            and s.lower() not in {"api", "v1", "v2", "flowgate"}]
    # prefer a non-generic last segment
    for s in reversed(segs):
        if s.lower() not in _GENERIC_TAIL:
            return s
    return segs[-1] if segs else ""


def _path_segs(p):
    """Path → significant segments (drop {params}, empty, leading context)."""
    return [s for s in str(p).strip("/").split("/")
            if s and not s.startswith("{")]


def _live_handler_for(cm, code_root, endpoint):
    """endpoint string → live handler_file via EXACT path match + liveness.

    Tail/substring matching was too loose — '/api/v1/projects' caught every
    '/api/v1/projects/{id}/files/...' sub-route and picked the wrong handler
    (L2 M036: project_settings missed, noise injected). Match by full path
    segments instead: an exact route wins; only then fall back to a suffix
    overlap. Among ties, a LIVE route wins, then the earliest-registered.
    """
    ep = _path_segs(endpoint)
    if not ep:
        return None
    try:
        rows = cm.route_table(code_root)
    except Exception:
        return None
    scored = []
    for r in rows:
        rt = _path_segs(r.get("path", ""))
        if not rt:
            continue
        exact = rt == ep
        suffix = (not exact and
                  (rt[-len(ep):] == ep if len(ep) <= len(rt)
                   else ep[-len(rt):] == rt))
        if exact or suffix:
            scored.append((exact, bool(r.get("live")),
                           -int(r.get("register_index", 999)), r))
    if not scored:
        return None, None
    scored.sort(key=lambda t: t[:3], reverse=True)   # exact > live > earliest
    r = scored[0][3]
    # codemap.handler_callees needs the handler_func (AST locates that body) —
    # passing "" finds no callees and the hop to the service is lost.
    return r.get("handler_file"), r.get("handler_func", "")


def discover_be_root_globs(seed_text, leaves, code_root, cm=None):
    """Return backend file paths traced from FE axes + seed, or [] (no-op safe)."""
    cm = cm or _codemap
    if cm is None or not code_root:
        return []
    fe_globs, found = [], set()
    for t in (leaves or []):
        sp = t.get("search_plan") or {}
        fe_globs += [g for g in (sp.get("file_globs") or [])
                     if str(g).lower().endswith((".vue", ".ts", ".tsx", ".js", ".jsx"))
                     and "*" not in str(g)]   # concrete files only; '**' globs are
    # over-broad (the queen emits client/src/**/*.vue) and carry no locating signal
    fe_globs = list(dict.fromkeys(fe_globs))
    try:
        # action-forward: FE file → endpoints → live handler → its services
        for g in fe_globs[:8]:
            for ep in (cm.endpoints_in_file(code_root, g) or []):
                hf, hfunc = _live_handler_for(cm, code_root, ep)
                if hf:
                    found.add(hf)
                    for c in (cm.handler_callees(code_root, hf, hfunc) or []):
                        if c.get("module_file"):
                            found.add(c["module_file"])
        # field-backward: snake fields the FE reads → backend producer
        for g in fe_globs[:8]:
            try:
                with open(os.path.join(code_root, g), encoding="utf-8",
                          errors="replace") as f:
                    fe_txt = f.read()
            except OSError:
                continue
            for fld in set(_SNAKE_RE.findall(fe_txt)):
                if fld.count("_") >= 2 or len(fld) >= 12:
                    for p in (cm.field_producers(code_root, fld) or [])[:3]:
                        if p.get("file"):
                            found.add(p["file"])
        # seed-concept: snake_case tokens written in the seed → symbol defs
        for tok in set(_SNAKE_RE.findall(seed_text or "")):
            for s in (cm.find_symbol(code_root, tok) or [])[:2]:
                if s.get("file"):
                    found.add(s["file"])
    except Exception as e:
        logger.warning("be-root discovery skipped (%s)", e)
        return []
    fe_set = {g.replace("\\", "/") for g in fe_globs}
    out = sorted(p.replace("\\", "/").removeprefix("./") for p in found
                 if str(p).endswith(".py") and p.replace("\\", "/") not in fe_set)
    return out[:_BE_ROOT_CAP]


_PICK_PROMPT = """A user reported this symptom in a Vue(frontend)+FastAPI(Python) app:
SYMPTOM: {seed}

Candidate frontend files the investigator surfaced, with the API endpoints each
calls and notable data fields each reads. ONE is the real entry point for the
symptom — the control/screen the user means (weigh ALL symptom cues, not keyword
overlap with one phrase):

{cands}

Reply with ONLY this JSON (no prose): the single most relevant file, and the
single API endpoint OR data field through which the backend produces the symptom:
{{"file": "<one path above>", "endpoint": "<one /api/... or empty>", "field": "<one snake_case field or empty>"}}"""


def _gather_candidates(leaves, code_root, cm):
    """FE files the queen named → each with its endpoints + read fields. Free."""
    files = []
    for t in (leaves or []):
        for g in (t.get("search_plan") or {}).get("file_globs", []):
            g = str(g)
            if g.lower().endswith((".vue", ".ts", ".tsx", ".js", ".jsx")) and "*" not in g:
                files.append(g)
    files = list(dict.fromkeys(files))[:12]
    cands = []
    for f in files:
        eps = (cm.endpoints_in_file(code_root, f) or [])[:5]
        try:
            with open(os.path.join(code_root, f), encoding="utf-8", errors="replace") as fh:
                fields = sorted({x for x in _SNAKE_RE.findall(fh.read())
                                 if x.count("_") >= 2})[:6]
        except OSError:
            fields = []
        cands.append({"file": f, "endpoints": eps, "fields": fields})
    return cands


def _trace_pick(pick, code_root, cm):
    """One picked node {file,endpoint,field} → backend .py paths. Free, no noise."""
    found = set()
    ep = str(pick.get("endpoint") or "").strip()
    if ep:
        hf, hfunc = _live_handler_for(cm, code_root, ep)
        if hf:
            found.add(hf)
            for c in (cm.handler_callees(code_root, hf, hfunc) or []):
                if c.get("module_file"):
                    found.add(c["module_file"])
    fld = str(pick.get("field") or "").strip()
    if fld:
        for p in (cm.field_producers(code_root, fld) or [])[:4]:
            if p.get("file"):
                found.add(p["file"])
    return sorted(p.replace("\\", "/").removeprefix("./") for p in found
                  if str(p).endswith(".py"))[:_BE_ROOT_CAP]


def pick_relevant_node(seed_text, candidates, call_fn):
    """Model picks the symptom-relevant node (the one model-tractable step)."""
    block = "\n".join(f"- {c['file']}\n    endpoints: {c['endpoints']}\n    "
                      f"fields: {c['fields']}" for c in candidates)
    out = call_fn(_PICK_PROMPT.format(seed=seed_text, cands=block)) or ""
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {}


def be_root_axis(seed_text, leaves, code_root, cm=None, call_fn=None):
    """Build a CODEMAP_BE_ROOT front axis, or None when nothing grounds.

    With ``call_fn`` (a model worker), uses the role-separation method — the model
    picks the symptom-relevant FE node, the code-map traces only THAT one (no
    fan-out noise; measured 2/3 on hard cases). Without it, falls back to the broad
    free trace (no model). The pick is the single model call the method adds.
    """
    cm = cm or _codemap
    if cm is None or not code_root:
        return None
    if call_fn is not None:
        cands = _gather_candidates(leaves, code_root, cm)
        if cands:
            globs = _trace_pick(pick_relevant_node(seed_text, cands, call_fn),
                                code_root, cm)
        else:
            globs = []
    else:
        globs = discover_be_root_globs(seed_text, leaves, code_root, cm)
    if not globs:
        return None
    logger.info("Hook A: code-map traced backend root candidates → %s", globs)
    return {
        "id": "CODEMAP_BE_ROOT",
        "title": "Backend root-cause traced from FE/seed via code-map",
        "brief": ("Statically traced from the frontend symptom file's API call / "
                  "read field, and the seed's symbols, to the LIVE backend handler "
                  "and the service or producer it reaches (shadowed handlers "
                  "excluded). Confirm the gate or data source HERE — this is the "
                  "root the surface FE axis only hints at."),
        "depends_on": [],
        "coverage_risk": "ok",
        "search_plan": {"keywords": [], "file_globs": globs, "doc_topics": []},
    }
