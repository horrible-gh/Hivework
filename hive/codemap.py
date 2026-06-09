"""Deterministic static code map for Python backends and frontend API calls.

The public functions in this module are deliberately best-effort.  They use
only local source text and the Python standard library, and return an empty
list when a repository cannot be inspected.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any


_HTTP_METHODS = {
    "get", "post", "put", "patch", "delete", "options", "head", "trace",
    "websocket",
}
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".next", "coverage", "site-packages",
}
_SOURCE_ROOTS = {"server", "src", "app", "backend"}
_FRONTEND_EXTS = {".vue", ".ts", ".tsx", ".js", ".jsx", ".svelte"}


@dataclass
class _Route:
    router: str
    path: str
    methods: list[str]
    handler_func: str
    line: int


@dataclass
class _Include:
    owner: str
    target: list[str]
    prefix: str
    line: int


@dataclass
class _Module:
    rel: str
    module_name: str
    tree: ast.Module
    constants: dict[str, str] = field(default_factory=dict)
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    routers: dict[str, str] = field(default_factory=dict)
    apps: set[str] = field(default_factory=set)
    routes: list[_Route] = field(default_factory=list)
    includes: list[_Include] = field(default_factory=list)


@dataclass
class _Index:
    root: str
    files: list[str]
    modules: dict[str, _Module]
    module_paths: dict[str, str]


def _is_test_path(rel: str) -> bool:
    parts = rel.replace("\\", "/").lower().split("/")
    name = parts[-1] if parts else ""
    return (
        any(part in {"test", "tests", "__tests__"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.py", ".spec.py"))
    )


def _source_files(code_root: str) -> list[str]:
    if not code_root or not os.path.isdir(code_root):
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", code_root, "ls-files"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if proc.returncode == 0:
            tracked = sorted({
                line.strip().replace("\\", "/")
                for line in proc.stdout.splitlines()
                if line.strip()
            })
            if tracked:
                return tracked
    except (OSError, subprocess.SubprocessError):
        pass

    found: list[str] = []
    try:
        for dirpath, dirnames, filenames in os.walk(code_root):
            dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
            rel_dir = os.path.relpath(dirpath, code_root).replace("\\", "/")
            for filename in filenames:
                rel = filename if rel_dir == "." else f"{rel_dir}/{filename}"
                found.append(rel)
    except OSError:
        return []
    return sorted(set(found))


def _read(code_root: str, rel: str) -> str:
    try:
        with open(
            os.path.join(code_root, rel), "r", encoding="utf-8", errors="replace"
        ) as handle:
            return handle.read()
    except OSError:
        return ""


def _module_names(rel: str) -> list[str]:
    path = rel.replace("\\", "/")
    if not path.endswith(".py"):
        return []
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    names: list[str] = []
    for start in range(len(parts)):
        if start == 0 or parts[start - 1] in _SOURCE_ROOTS:
            name = ".".join(parts[start:])
            if name:
                names.append(name)
    if parts and parts[0] in _SOURCE_ROOTS:
        names.append(".".join(parts[1:]))
    return list(dict.fromkeys(name for name in names if name))


def _primary_module_name(rel: str) -> str:
    names = _module_names(rel)
    if not names:
        return ""
    without_source_root = [
        name for name in names if name.split(".", 1)[0] not in _SOURCE_ROOTS
    ]
    return without_source_root[0] if without_source_root else names[0]


def _string_value(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left, constants)
        right = _string_value(node.right, constants)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                resolved = _string_value(value.value, constants)
                pieces.append(resolved or "")
        return "".join(pieces)
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _call_name(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + [node.attr]
    return []


def _absolute_import_module(
    importer: _Module, module: str | None, level: int
) -> str:
    if level <= 0:
        return module or ""
    package = importer.module_name.split(".")
    if not importer.rel.endswith("/__init__.py"):
        package = package[:-1]
    keep = max(0, len(package) - level + 1)
    base = package[:keep]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _resolve_module_path(module: str, paths: dict[str, str]) -> str | None:
    if not module:
        return None
    if module in paths:
        return paths[module]
    suffix = "." + module
    candidates = [
        (name.count("."), rel)
        for name, rel in paths.items()
        if name.endswith(suffix)
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _parse_module(
    code_root: str, rel: str, module_paths: dict[str, str]
) -> _Module | None:
    text = _read(code_root, rel)
    if not text:
        return None
    try:
        tree = ast.parse(text, filename=rel)
    except (SyntaxError, ValueError):
        return None

    info = _Module(rel=rel, module_name=_primary_module_name(rel), tree=tree)

    pending_constants: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    pending_constants[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                pending_constants[node.target.id] = node.value
    for _ in range(len(pending_constants) + 1):
        changed = False
        for name, value_node in pending_constants.items():
            value = _string_value(value_node, info.constants)
            if value is not None and info.constants.get(name) != value:
                info.constants[name] = value
                changed = True
        if not changed:
            break

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_module_path(alias.name, module_paths)
                if target:
                    local = alias.asname or alias.name.split(".")[0]
                    info.imports[local] = (target, None)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_module(info, node.module, node.level)
            base_rel = _resolve_module_path(base, module_paths)
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                submodule = f"{base}.{alias.name}" if base else alias.name
                sub_rel = _resolve_module_path(submodule, module_paths)
                if sub_rel:
                    info.imports[local] = (sub_rel, None)
                elif base_rel:
                    info.imports[local] = (base_rel, alias.name)

        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if isinstance(value, ast.Call):
            called = _call_name(value.func)
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if called and called[-1] == "APIRouter":
                    info.routers[target.id] = (
                        _string_value(_keyword(value, "prefix"), info.constants) or ""
                    )
                elif called and called[-1] == "FastAPI":
                    info.apps.add(target.id)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                called = _call_name(decorator.func)
                if len(called) < 2:
                    continue
                method = called[-1].lower()
                router = called[-2]
                if router not in info.routers and router not in info.apps:
                    continue
                if method in _HTTP_METHODS:
                    methods = ["WEBSOCKET" if method == "websocket" else method.upper()]
                elif method == "api_route":
                    methods_node = _keyword(decorator, "methods")
                    methods = _string_list(methods_node, info.constants) or ["GET"]
                else:
                    continue
                path_node = decorator.args[0] if decorator.args else _keyword(
                    decorator, "path"
                )
                path = _string_value(path_node, info.constants)
                if path is None:
                    continue
                info.routes.append(_Route(
                    router=router,
                    path=path,
                    methods=methods,
                    handler_func=node.name,
                    line=getattr(decorator, "lineno", node.lineno),
                ))

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            called = _call_name(call.func)
            if len(called) >= 2 and called[-1] == "include_router" and call.args:
                prefix = _string_value(_keyword(call, "prefix"), info.constants) or ""
                info.includes.append(_Include(
                    owner=called[-2],
                    target=_call_name(call.args[0]),
                    prefix=prefix,
                    line=node.lineno,
                ))
    return info


def _string_list(
    node: ast.AST | None, constants: dict[str, str]
) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_string_value(item, constants) for item in node.elts]
        return [value.upper() for value in values if value]
    value = _string_value(node, constants)
    return [value.upper()] if value else []


def _build_index(code_root: str) -> _Index:
    root = os.path.abspath(code_root)
    files = _source_files(root)
    py_files = [
        rel for rel in files if rel.endswith(".py") and not _is_test_path(rel)
    ]
    module_paths: dict[str, str] = {}
    for rel in py_files:
        for name in _module_names(rel):
            module_paths.setdefault(name, rel)
    modules: dict[str, _Module] = {}
    for rel in py_files:
        info = _parse_module(root, rel, module_paths)
        if info:
            modules[rel] = info
    return _Index(root=root, files=files, modules=modules, module_paths=module_paths)


def _resolve_ref(
    index: _Index,
    module_rel: str,
    chain: list[str],
    seen: set[tuple[str, tuple[str, ...]]] | None = None,
) -> tuple[str, str] | None:
    if not chain or module_rel not in index.modules:
        return None
    marker = (module_rel, tuple(chain))
    seen = seen or set()
    if marker in seen:
        return None
    seen.add(marker)

    info = index.modules[module_rel]
    first, rest = chain[0], chain[1:]
    if first in info.routers and not rest:
        return module_rel, first
    imported = info.imports.get(first)
    if not imported:
        return None
    target_rel, symbol = imported
    target_chain = ([symbol] if symbol else []) + rest
    if not target_chain:
        return None
    return _resolve_ref(index, target_rel, target_chain, seen)


def _join_path(*parts: str) -> str:
    segments: list[str] = []
    for part in parts:
        if not part:
            continue
        segments.extend(segment for segment in part.split("/") if segment)
    return "/" + "/".join(segments) if segments else "/"


def _expand_router(
    index: _Index,
    module_rel: str,
    router_name: str,
    prefix: str,
    register_index: int,
    stack: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    marker = (module_rel, router_name)
    if marker in stack:
        return []
    info = index.modules.get(module_rel)
    if not info or router_name not in info.routers:
        return []

    nested_stack = set(stack)
    nested_stack.add(marker)
    router_prefix = _join_path(prefix, info.routers[router_name])
    rows: list[dict[str, Any]] = []
    for route in sorted(info.routes, key=lambda item: item.line):
        if route.router != router_name:
            continue
        rows.append({
            "path": _join_path(router_prefix, route.path),
            "methods": route.methods,
            "handler_file": module_rel,
            "handler_func": route.handler_func,
            "live": True,
            "register_index": register_index,
            "_line": route.line,
        })

    for include in sorted(info.includes, key=lambda item: item.line):
        if include.owner != router_name:
            continue
        resolved = _resolve_ref(index, module_rel, include.target)
        if not resolved:
            continue
        child_rel, child_router = resolved
        rows.extend(_expand_router(
            index,
            child_rel,
            child_router,
            _join_path(router_prefix, include.prefix),
            register_index,
            nested_stack,
        ))
    return rows


def route_table(code_root: str) -> list[dict]:
    """Return FastAPI routes, registration order, and deterministic liveness."""
    try:
        index = _build_index(code_root)
        root_events: list[tuple[str, int, str, Any]] = []
        for rel, info in sorted(index.modules.items()):
            for include in info.includes:
                if include.owner in info.apps:
                    root_events.append((rel, include.line, "include", include))
            for route in info.routes:
                if route.router in info.apps:
                    root_events.append((rel, route.line, "route", route))
        root_events.sort(key=lambda item: (item[0], item[1]))

        rows: list[dict[str, Any]] = []
        if root_events:
            for register_index, (rel, _, kind, event) in enumerate(root_events):
                if kind == "route":
                    route = event
                    rows.append({
                        "path": _join_path(route.path),
                        "methods": route.methods,
                        "handler_file": rel,
                        "handler_func": route.handler_func,
                        "live": True,
                        "register_index": register_index,
                        "_line": route.line,
                    })
                    continue
                include = event
                resolved = _resolve_ref(index, rel, include.target)
                if resolved:
                    module_rel, router_name = resolved
                    rows.extend(_expand_router(
                        index,
                        module_rel,
                        router_name,
                        include.prefix,
                        register_index,
                        set(),
                    ))
        else:
            register_index = 0
            for rel, info in sorted(index.modules.items()):
                for router_name in sorted(info.routers):
                    rows.extend(_expand_router(
                        index, rel, router_name, "", register_index, set()
                    ))
                    register_index += 1

        rows.sort(key=lambda row: (
            row["register_index"], row["_line"], row["handler_file"],
            row["handler_func"],
        ))
        seen: set[tuple[str, str]] = set()
        for row in rows:
            keys = [(method.upper(), row["path"]) for method in row["methods"]]
            row["live"] = all(key not in seen for key in keys)
            seen.update(keys)
            row.pop("_line", None)
        return rows
    except Exception:
        return []


def _function_node(tree: ast.Module, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def handler_callees(
    code_root: str, handler_file: str, handler_func: str
) -> list[dict]:
    """Return repository module-function calls made by one handler body."""
    try:
        index = _build_index(code_root)
        rel = handler_file.replace("\\", "/").removeprefix("./")
        info = index.modules.get(rel)
        if not info:
            return []
        func = _function_node(info.tree, handler_func)
        if func is None:
            return []

        results: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            called = _call_name(node.func)
            if len(called) < 2:
                continue
            imported = info.imports.get(called[0])
            if not imported:
                continue
            target_rel, symbol = imported
            if symbol and len(called) > 2:
                submodule = _resolve_module_path(
                    f"{index.modules[target_rel].module_name}.{symbol}",
                    index.module_paths,
                )
                if submodule:
                    target_rel = submodule
            name = ".".join(called)
            key = (name, target_rel)
            if key not in seen:
                seen.add(key)
                results.append({"name": name, "module_file": target_rel})
        return results
    except Exception:
        return []


def field_producers(code_root: str, field: str) -> list[dict]:
    """Return non-test Python lines that assign or serialize ``field``."""
    try:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field or ""):
            return []
        patterns = (
            re.compile(
                rf"""\[\s*["']{re.escape(field)}["']\s*\]\s*=(?!=)"""
            ),
            re.compile(rf"""["']{re.escape(field)}["']\s*:"""),
            re.compile(
                rf"""^\s*{re.escape(field)}\s*(?::[^=]+)?=(?!=)"""
            ),
        )
        rows: list[dict[str, Any]] = []
        for rel in _source_files(code_root):
            if not rel.endswith(".py") or _is_test_path(rel):
                continue
            text = _read(code_root, rel)
            for line_no, line in enumerate(text.splitlines(), 1):
                if any(pattern.search(line) for pattern in patterns):
                    rows.append({"file": rel, "line": line_no})
        return rows
    except Exception:
        return []


def find_symbol(code_root: str, name: str) -> list[dict]:
    """Return Python definition locations for a function, class, or module value."""
    try:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            return []
        rows: list[dict[str, Any]] = []
        for rel in _source_files(code_root):
            if not rel.endswith(".py") or _is_test_path(rel):
                continue
            text = _read(code_root, rel)
            if not text:
                continue
            try:
                tree = ast.parse(text, filename=rel)
            except (SyntaxError, ValueError):
                rows.extend(_fallback_symbol_rows(rel, text, name))
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == name:
                        rows.append({
                            "file": rel, "line": node.lineno, "kind": "function",
                        })
                elif isinstance(node, ast.ClassDef) and node.name == name:
                    rows.append({
                        "file": rel, "line": node.lineno, "kind": "class",
                    })
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name
                    for target in node.targets
                ):
                    rows.append({
                        "file": rel, "line": node.lineno, "kind": "variable",
                    })
                elif (
                    isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == name
                ):
                    rows.append({
                        "file": rel, "line": node.lineno, "kind": "variable",
                    })
        return sorted(
            { (row["file"], row["line"], row["kind"]): row for row in rows }.values(),
            key=lambda row: (row["file"], row["line"], row["kind"]),
        )
    except Exception:
        return []


def _fallback_symbol_rows(rel: str, text: str, name: str) -> list[dict[str, Any]]:
    escaped = re.escape(name)
    patterns = (
        ("function", re.compile(rf"^\s*(?:async\s+)?def\s+{escaped}\s*\(")),
        ("class", re.compile(rf"^\s*class\s+{escaped}\b")),
        ("variable", re.compile(rf"^{escaped}\s*(?::[^=]+)?=(?!=)")),
    )
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, pattern in patterns:
            if pattern.search(line):
                rows.append({"file": rel, "line": line_no, "kind": kind})
                break
    return rows


def endpoints_in_file(code_root: str, rel_path: str) -> list[str]:
    """Extract quoted frontend endpoint strings containing /api/ or /flowgate/."""
    try:
        rel = rel_path.replace("\\", "/").removeprefix("./")
        if os.path.splitext(rel)[1].lower() not in _FRONTEND_EXTS:
            return []
        text = _read(code_root, rel)
        if not text:
            return []
        endpoints: list[str] = []
        for match in re.finditer(r"""(["'`])([^"'`\r\n]+)\1""", text):
            value = match.group(2)
            if "/api/" in value or "/flowgate/" in value:
                endpoints.append(value)
        return list(dict.fromkeys(endpoints))
    except Exception:
        return []
