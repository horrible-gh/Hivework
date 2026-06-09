"""INTERIM stub of hive/codemap.py (M027 API) — for testing the integration
(Hook A/B, M028) BEFORE GPT's real codemap.py lands. Reuses the proven
provenance kernel. NOT for merge — GPT's hive/codemap.py replaces this; the
integration imports `hive.codemap` and tests monkeypatch it with this stub.

Same API contract as M027 §1, best-effort, never raises.
"""
import os
import re
import subprocess

_ROUTE_METHODS = "get|post|put|patch|delete"
_SERVICE_CALL_RE = re.compile(r"\b([a-z][a-z0-9]*_service)\.([a-z_][a-z0-9_]*)",
                              re.IGNORECASE)
_GENERIC_SEG = frozenset({"create", "list", "get", "update", "delete", "new",
                          "edit", "save", "fetch", "all", "index", "api",
                          "v1", "v2", "flowgate"})


def _rg(pattern, root, globs=None, max_hits=40):
    cmd = ["rg", "--no-heading", "-n", "-i", "--max-count", str(max_hits)]
    for g in (globs or []):
        cmd += ["-g", g]
    cmd += ["-e", pattern, "."]
    try:
        out = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    hits = []
    for line in out.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            hits.append({"file": parts[0].replace("\\", "/").removeprefix("./"),
                         "line": int(parts[1]), "text": parts[2].strip()[:300]})
    return hits


def _files(root):
    try:
        p = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=30)
        if p.returncode == 0 and p.stdout.strip():
            return [l.strip().replace("\\", "/") for l in p.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    skip = {".git", "node_modules", "__pycache__", ".venv", "dist", "build"}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in skip]
        rel = os.path.relpath(dp, root).replace("\\", "/")
        for f in fn:
            out.append(f if rel == "." else f"{rel}/{f}")
    return out


def _router_order(code_root):
    main_py = os.path.join(code_root, "server", "routers", "main.py")
    try:
        with open(main_py, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except OSError:
        return {}
    var2mod = {m.group(2): m.group(1).split(".")[-1]
               for m in re.finditer(r"from\s+([\w.]+)\s+import\s+router\s+as\s+(\w+)", txt)}
    order, idx = {}, 0
    for m in re.finditer(r"^\s*app\.include_router\(\s*(\w+)", txt, re.MULTILINE):
        mod = var2mod.get(m.group(1))
        if mod and mod not in order:
            order[mod] = idx
            idx += 1
    return order


def _mod(f):
    return os.path.splitext(os.path.basename(f))[0]


def endpoints_in_file(code_root, rel_path):
    try:
        with open(os.path.join(code_root, rel_path), encoding="utf-8",
                  errors="replace") as f:
            txt = f.read()
    except OSError:
        return []
    eps = set()
    for s in re.findall(r"""['"`]([^'"`\s]{2,120})['"`]""", txt):
        if "/api/" in s or "/flowgate/" in s:
            eps.add(s)
    return sorted(eps)


def _handler_files_for(endpoint, code_root):
    """Route files defining a decorator matching endpoint's distinctive tail."""
    segs = [s for s in endpoint.split("/")
            if s and not s.startswith("{") and s.lower() not in _GENERIC_SEG
            and len(s) >= 4]
    if not segs:
        return []
    tail = segs[-1]
    hits = _rg(rf"router\.(?:{_ROUTE_METHODS})\([\"'][^\"']*{re.escape(tail)}",
               code_root, ["*.py"], max_hits=20)
    return list(dict.fromkeys(h["file"] for h in hits))


def route_table(code_root):
    """Coarse route table w/ liveness via include_router order (stub)."""
    order = _router_order(code_root)
    hits = _rg(rf"@\w*router\w*\.(?:{_ROUTE_METHODS})\(", code_root, ["*routes*.py", "*router*.py"],
               max_hits=2000)
    rows = []
    for h in hits:
        m = re.search(rf"\.({_ROUTE_METHODS})\(\s*[\"']([^\"']+)", h["text"], re.IGNORECASE)
        if not m:
            continue
        rows.append({"path": m.group(2), "methods": [m.group(1).upper()],
                     "handler_file": h["file"], "handler_func": "",
                     "register_index": order.get(_mod(h["file"]), 999)})
    # liveness: same (method, path-tail) -> lowest register_index wins
    seen = {}
    for r in sorted(rows, key=lambda r: r["register_index"]):
        key = (r["methods"][0], r["path"].rstrip("/").split("/")[-1])
        r["live"] = key not in seen
        seen[key] = True
    return rows


def handler_callees(code_root, handler_file, handler_func=""):
    try:
        with open(os.path.join(code_root, handler_file), encoding="utf-8",
                  errors="replace") as f:
            body = f.read()
    except OSError:
        return []
    out = []
    for svc, fn in set(_SERVICE_CALL_RE.findall(body)):
        mf = None
        for f in _files(code_root):
            if os.path.basename(f) == f"{svc.lower()}.py":
                mf = f
                break
        out.append({"name": f"{svc}.{fn}", "module_file": mf})
    return out


def field_producers(code_root, field):
    hits = _rg(rf"{re.escape(field)}[\"']?\s*\]?\s*[:=]", code_root,
               ["server/**/*.py"], max_hits=12)
    return [{"file": h["file"], "line": h["line"]} for h in hits
            if h["file"].endswith(".py") and "/test" not in h["file"].lower()]


def find_symbol(code_root, name):
    hits = _rg(rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b|^\s*class\s+{re.escape(name)}\b",
               code_root, ["server/**/*.py"], max_hits=20)
    out = []
    for h in hits:
        kind = "class" if "class " in h["text"] else "def"
        out.append({"file": h["file"], "line": h["line"], "kind": kind})
    return out


# convenience used by integration: endpoint -> live handler file + its services
def trace_endpoint_to_be(code_root, endpoint):
    cand = _handler_files_for(endpoint, code_root)
    if not cand:
        return []
    order = _router_order(code_root)
    live = min(cand, key=lambda f: order.get(_mod(f), 999))
    out = [live]
    for c in handler_callees(code_root, live):
        if c["module_file"]:
            out.append(c["module_file"])
    return list(dict.fromkeys(out))
