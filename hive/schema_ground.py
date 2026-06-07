"""Deterministic migration-schema grounding and generated-test INSERT checks.

The parser intentionally covers the DDL shapes used by ordinary migration files
without trying to be a complete SQL grammar. Anything it cannot prove is skipped:
grounding and validation are both fail-open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import os
import re
from typing import Any, Iterable


_IDENT = r'(?:[A-Za-z_][A-Za-z0-9_$]*|"(?:""|[^"])+"|`(?:``|[^`])+`|\[[^\]]+\])'
_QUALIFIED_IDENT = rf"{_IDENT}(?:\s*\.\s*{_IDENT})*"
_CREATE_TABLE_RE = re.compile(
    rf"^\s*CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    rf"(?P<table>{_QUALIFIED_IDENT})\s*\(",
    re.IGNORECASE | re.DOTALL,
)
_ALTER_TABLE_RE = re.compile(
    rf"^\s*ALTER\s+TABLE\s+(?:ONLY\s+)?(?P<table>{_QUALIFIED_IDENT})\s+(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_UNIQUE_INDEX_RE = re.compile(
    rf"^\s*CREATE\s+UNIQUE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?{_QUALIFIED_IDENT}"
    rf"\s+ON\s+(?P<table>{_QUALIFIED_IDENT})\s*\((?P<cols>.*)\)",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_RE = re.compile(
    rf"\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+(?P<table>{_QUALIFIED_IDENT})"
    rf"\s*\((?P<cols>[^)]*)\)\s*VALUES\s*",
    re.IGNORECASE | re.DOTALL,
)
_TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__)/"
    r"|(?:^|/)test_[^/]+$"
    r"|(?:^|/)[^/]+_test\.[^./]+$"
    r"|\.spec\.[^./]+$"
)
_UNKNOWN = object()


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    referenced_table: str
    referenced_columns: tuple[str, ...]


@dataclass
class TableConstraints:
    name: str
    columns: list[str] = field(default_factory=list)
    unique: list[tuple[str, ...]] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    not_null: set[str] = field(default_factory=set)
    primary_key: tuple[str, ...] = ()
    defaults: set[str] = field(default_factory=set)
    generated: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)


Schema = dict[str, TableConstraints]


@dataclass(frozen=True)
class _InsertRow:
    edit_id: str
    file: str
    scope: str
    table: str
    values: dict[str, Any]


def _unquote_identifier(value: str) -> str:
    value = value.strip()
    if len(value) >= 2:
        if value[0] == value[-1] == '"':
            return value[1:-1].replace('""', '"')
        if value[0] == value[-1] == "`":
            return value[1:-1].replace("``", "`")
        if value[0] == "[" and value[-1] == "]":
            return value[1:-1]
    return value


def _identifier_parts(value: str) -> list[str]:
    return [_unquote_identifier(part) for part in re.findall(_IDENT, value)]


def _table_key(value: str) -> str:
    parts = _identifier_parts(value)
    return (parts[-1] if parts else value.strip()).lower()


def _column_name(value: str) -> str | None:
    m = re.match(rf"^\s*(?P<name>{_IDENT})", value, re.DOTALL)
    return _unquote_identifier(m.group("name")).lower() if m else None


def _strip_comments(sql: str) -> str:
    out: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(sql):
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < len(sql) else ""
        if quote:
            out.append(ch)
            if ch == quote:
                if nxt == quote:
                    out.append(nxt)
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "-" and nxt == "-":
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(sql) and sql[i:i + 2] != "*/":
                if sql[i] in "\r\n":
                    out.append(sql[i])
                i += 1
            i = min(len(sql), i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if quote:
            if ch == quote:
                if nxt == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _iter_statements(sql: str) -> Iterable[str]:
    for statement in _split_top_level(_strip_comments(sql), ";"):
        if statement.strip():
            yield statement.strip()


def _matching_paren(text: str, opening: int) -> int | None:
    depth = 0
    quote: str | None = None
    i = opening
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if quote:
            if ch == quote:
                if nxt == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _column_list(text: str) -> tuple[str, ...]:
    cols: list[str] = []
    for item in _split_top_level(text):
        match = re.fullmatch(
            rf"\s*(?P<name>{_IDENT})(?:\s+(?:ASC|DESC))?\s*",
            item, re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return ()
        cols.append(_unquote_identifier(match.group("name")).lower())
    return tuple(cols)


def _append_unique(table: TableConstraints, columns: tuple[str, ...]) -> None:
    if columns and columns not in table.unique:
        table.unique.append(columns)


def _append_fk(table: TableConstraints, fk: ForeignKey) -> None:
    if fk.columns and fk.referenced_columns and fk not in table.foreign_keys:
        table.foreign_keys.append(fk)


def _parse_table_constraint(table: TableConstraints, clause: str) -> bool:
    body = re.sub(
        rf"^\s*CONSTRAINT\s+{_IDENT}\s+", "", clause, count=1,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    m = re.match(r"^(?:UNIQUE|PRIMARY\s+KEY)\s*\(", body, re.IGNORECASE)
    if m:
        end = _matching_paren(body, body.find("(", m.start()))
        if end is None:
            return True
        cols = _column_list(body[body.find("(") + 1:end])
        if re.match(r"^PRIMARY\s+KEY", body, re.IGNORECASE):
            table.primary_key = cols
        else:
            _append_unique(table, cols)
        return True

    m = re.match(r"^FOREIGN\s+KEY\s*\(", body, re.IGNORECASE)
    if m:
        local_end = _matching_paren(body, body.find("("))
        if local_end is None:
            return True
        ref = re.search(
            rf"\bREFERENCES\s+(?P<table>{_QUALIFIED_IDENT})\s*\(",
            body[local_end + 1:], re.IGNORECASE | re.DOTALL,
        )
        if not ref:
            return True
        ref_open = local_end + 1 + ref.end() - 1
        ref_end = _matching_paren(body, ref_open)
        if ref_end is None:
            return True
        _append_fk(table, ForeignKey(
            columns=_column_list(body[body.find("(") + 1:local_end]),
            referenced_table=_table_key(ref.group("table")),
            referenced_columns=_column_list(body[ref_open + 1:ref_end]),
        ))
        return True
    return False


def _parse_column(table: TableConstraints, clause: str) -> None:
    match = re.match(rf"^\s*(?P<name>{_IDENT})\s+(?P<body>.+)$",
                     clause, re.IGNORECASE | re.DOTALL)
    if not match:
        return
    name = _unquote_identifier(match.group("name")).lower()
    if name in {"constraint", "unique", "primary", "foreign", "check", "exclude"}:
        return
    body = match.group("body")
    if name not in table.columns:
        table.columns.append(name)
    if re.search(r"\bNOT\s+NULL\b", body, re.IGNORECASE):
        table.not_null.add(name)
    if re.search(r"\bDEFAULT\b", body, re.IGNORECASE):
        table.defaults.add(name)
    if re.search(r"\bGENERATED\b|\bIDENTITY\b|\bAUTO_INCREMENT\b|\bAUTOINCREMENT\b",
                 body, re.IGNORECASE):
        table.generated.add(name)
    if re.search(r"\bPRIMARY\s+KEY\b", body, re.IGNORECASE):
        table.primary_key = (name,)
    if re.search(r"\bUNIQUE\b", body, re.IGNORECASE):
        _append_unique(table, (name,))
    ref = re.search(
        rf"\bREFERENCES\s+(?P<table>{_QUALIFIED_IDENT})\s*\((?P<cols>[^)]*)\)",
        body, re.IGNORECASE | re.DOTALL,
    )
    if ref:
        _append_fk(table, ForeignKey(
            columns=(name,),
            referenced_table=_table_key(ref.group("table")),
            referenced_columns=_column_list(ref.group("cols")),
        ))


def _parse_create_table(statement: str, source: str, schema: Schema) -> bool:
    match = _CREATE_TABLE_RE.match(statement)
    if not match:
        return False
    opening = match.end() - 1
    closing = _matching_paren(statement, opening)
    if closing is None:
        return True
    key = _table_key(match.group("table"))
    table = TableConstraints(name=key)
    for clause in _split_top_level(statement[opening + 1:closing]):
        if not _parse_table_constraint(table, clause):
            _parse_column(table, clause)
    table.sources.append(source)
    schema[key] = table
    return True


def _parse_alter_table(statement: str, source: str, schema: Schema) -> bool:
    match = _ALTER_TABLE_RE.match(statement)
    if not match:
        return False
    key = _table_key(match.group("table"))
    table = schema.setdefault(key, TableConstraints(name=key))
    if source not in table.sources:
        table.sources.append(source)
    body = match.group("body").strip()
    body = re.sub(r"^ADD\s+", "", body, count=1, flags=re.IGNORECASE)
    if re.match(r"^COLUMN\b", body, re.IGNORECASE):
        body = re.sub(r"^COLUMN\s+", "", body, count=1, flags=re.IGNORECASE)
        _parse_column(table, body)
    elif not _parse_table_constraint(table, body):
        set_not_null = re.match(
            rf"^ALTER\s+COLUMN\s+(?P<col>{_IDENT})\s+SET\s+NOT\s+NULL\b",
            body, re.IGNORECASE,
        )
        if set_not_null:
            table.not_null.add(_unquote_identifier(set_not_null.group("col")).lower())
    return True


def parse_migration_sql(sql: str, *, source: str = "<memory>",
                        schema: Schema | None = None) -> Schema:
    """Parse supported CREATE/ALTER/UNIQUE INDEX DDL into table constraints."""
    result = schema if schema is not None else {}
    try:
        for statement in _iter_statements(sql):
            if _parse_create_table(statement, source, result):
                continue
            if _parse_alter_table(statement, source, result):
                continue
            index = _UNIQUE_INDEX_RE.match(statement)
            if index:
                key = _table_key(index.group("table"))
                table = result.setdefault(key, TableConstraints(name=key))
                _append_unique(table, _column_list(index.group("cols")))
                if source not in table.sources:
                    table.sources.append(source)
    except Exception:
        pass
    return result


def discover_migration_files(codebase_root: str) -> list[str]:
    """Return migration SQL paths in stable lexical order."""
    if not codebase_root or not os.path.isdir(codebase_root):
        return []
    found: set[str] = set()
    for dirname in ("migrations", "migration", "migrate"):
        pattern = os.path.join(os.path.abspath(codebase_root), "**", dirname, "**", "*.sql")
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                found.add(os.path.normpath(path))
    return sorted(found, key=lambda path: path.replace("\\", "/").lower())


def load_migration_schema(codebase_root: str) -> Schema:
    """Read and merge all discovered migration DDL. Unreadable files are skipped."""
    schema: Schema = {}
    root = os.path.abspath(codebase_root) if codebase_root else ""
    for path in discover_migration_files(codebase_root):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                sql = handle.read()
        except (OSError, UnicodeError):
            continue
        source = os.path.relpath(path, root).replace("\\", "/") if root else path
        parse_migration_sql(sql, source=source, schema=schema)
    return schema


def touched_tables(text: str, schema: Schema) -> list[str]:
    """Schema tables named by the honey/seed text, in deterministic order."""
    if not text or not schema:
        return []
    touched: list[str] = []
    for key in sorted(schema):
        if re.search(rf"(?<![A-Za-z0-9_$]){re.escape(key)}(?![A-Za-z0-9_$])",
                     text, re.IGNORECASE):
            touched.append(key)
    return touched


def render_schema_grounding(schema: Schema, tables: Iterable[str]) -> str:
    """Render migration-backed constraints for the author prompt."""
    selected = [schema[key.lower()] for key in tables if key.lower() in schema]
    if not selected:
        return ""
    lines = [
        "## Migration schema constraints (CURRENT deterministic DDL grounding)",
        "",
        "Generated DB tests must keep every INSERT consistent with these constraints. "
        "Do not duplicate UNIQUE/PRIMARY KEY tuples, omit required NOT NULL columns, or "
        "reference parent keys that the isolated fixture does not seed.",
    ]
    for table in sorted(selected, key=lambda item: item.name):
        lines.extend(["", f"- table `{table.name}`"])
        if table.primary_key:
            lines.append(f"  - PRIMARY KEY ({', '.join(table.primary_key)})")
        for columns in sorted(table.unique):
            lines.append(f"  - UNIQUE ({', '.join(columns)})")
        if table.not_null:
            lines.append("  - NOT NULL: " + ", ".join(sorted(table.not_null)))
        omittable = (table.defaults | table.generated)
        if omittable:
            lines.append(
                "  - DEFAULT/GENERATED (may be omitted): "
                + ", ".join(sorted(omittable)))
        for fk in sorted(
                table.foreign_keys,
                key=lambda item: (item.columns, item.referenced_table,
                                  item.referenced_columns)):
            lines.append(
                f"  - FOREIGN KEY ({', '.join(fk.columns)}) REFERENCES "
                f"{fk.referenced_table} ({', '.join(fk.referenced_columns)})")
        if table.sources:
            lines.append("  - migration: " + ", ".join(sorted(table.sources)))
    return "\n".join(lines)


def _literal_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return _UNKNOWN
    if value.upper() == "NULL":
        return None
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return ("text", value[1:-1].replace("''", "'"))
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return ("number", value.lstrip("+"))
    if value.upper() in {"TRUE", "FALSE"}:
        return ("boolean", value.upper())
    return _UNKNOWN


def _rows_after_values(text: str, start: int) -> list[list[str]]:
    rows: list[list[str]] = []
    pos = start
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "(":
            break
        closing = _matching_paren(text, pos)
        if closing is None:
            break
        rows.append(_split_top_level(text[pos + 1:closing]))
        pos = closing + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ",":
            break
        pos += 1
    return rows


def _test_scope(text: str, offset: int, edit_id: str) -> str:
    """Nearest Python/JS test declaration, or the edit id when no scope is visible."""
    prefix = text[:offset]
    matches = list(re.finditer(
        r"^[ \t]*(?:async\s+)?def\s+(?P<py>test_[A-Za-z0-9_]+)\s*\("
        r"|^[ \t]*(?P<js>(?:it|test))\s*\(\s*['\"][^'\"]+['\"]",
        prefix, re.MULTILINE,
    ))
    if not matches:
        return edit_id
    match = matches[-1]
    return f"{match.start()}:{match.group('py') or match.group('js')}"


def _extract_insert_rows(edit: dict[str, Any], schema: Schema) -> list[_InsertRow]:
    rel = str(edit.get("file", "")).replace("\\", "/")
    if not _TEST_PATH_RE.search(rel):
        return []
    text = (edit.get("content") if edit.get("kind") == "create_file"
            else edit.get("replacement_new")) or ""
    text = str(text)
    edit_id = str(edit.get("id", "?"))
    rows: list[_InsertRow] = []
    for match in _INSERT_RE.finditer(text):
        table = _table_key(match.group("table"))
        if table not in schema:
            continue
        columns = _column_list(match.group("cols"))
        if not columns or len(columns) != len(set(columns)):
            continue
        for raw_values in _rows_after_values(text, match.end()):
            if len(raw_values) != len(columns):
                continue
            rows.append(_InsertRow(
                edit_id=edit_id,
                file=rel,
                scope=_test_scope(text, match.start(), edit_id),
                table=table,
                values={column: _literal_value(raw)
                        for column, raw in zip(columns, raw_values)},
            ))
    return rows


def validate_test_inserts(edits: Iterable[dict[str, Any]], schema: Schema) -> dict[str, str]:
    """Return edit-id findings for provable migration-constraint violations."""
    if not schema:
        return {}
    rows: list[_InsertRow] = []
    for edit in edits:
        if isinstance(edit, dict):
            rows.extend(_extract_insert_rows(edit, schema))
    if not rows:
        return {}

    reasons: dict[str, list[str]] = {}

    def flag(edit_id: str, reason: str) -> None:
        bucket = reasons.setdefault(edit_id, [])
        if reason not in bucket:
            bucket.append(reason)

    for row in rows:
        table = schema[row.table]
        required = table.not_null - table.defaults - table.generated
        missing = sorted(column for column in required if column not in row.values)
        explicit_null = sorted(
            column for column in table.not_null
            if column in row.values and row.values[column] is None
        )
        if missing:
            flag(row.edit_id, f"{row.table} INSERT omits NOT NULL column(s): "
                 + ", ".join(missing))
        if explicit_null:
            flag(row.edit_id, f"{row.table} INSERT sets NOT NULL column(s) to NULL: "
                 + ", ".join(explicit_null))

    scopes = sorted({(row.file, row.edit_id, row.scope) for row in rows})
    for file, edit_id, scope in scopes:
        file_rows = [
            row for row in rows
            if (row.file, row.edit_id, row.scope) == (file, edit_id, scope)
        ]
        for table_name in sorted({row.table for row in file_rows}):
            table = schema[table_name]
            table_rows = [row for row in file_rows if row.table == table_name]
            unique_sets = list(table.unique)
            if table.primary_key:
                unique_sets.append(table.primary_key)
            for columns in unique_sets:
                seen: dict[tuple[Any, ...], _InsertRow] = {}
                for row in table_rows:
                    values = tuple(row.values.get(column, _UNKNOWN) for column in columns)
                    if any(value is _UNKNOWN or value is None for value in values):
                        continue
                    previous = seen.get(values)
                    if previous:
                        label = ", ".join(columns)
                        reason = (f"{table_name} INSERT duplicates UNIQUE ({label}) "
                                  f"tuple {values!r}")
                        flag(previous.edit_id, reason)
                        flag(row.edit_id, reason)
                    else:
                        seen[values] = row

        seeded_tables = {row.table for row in file_rows}
        for row in file_rows:
            table = schema[row.table]
            for fk in table.foreign_keys:
                if fk.referenced_table not in seeded_tables:
                    continue
                child = tuple(row.values.get(column, _UNKNOWN) for column in fk.columns)
                if any(value is _UNKNOWN or value is None for value in child):
                    continue
                parents = [
                    tuple(parent.values.get(column, _UNKNOWN)
                          for column in fk.referenced_columns)
                    for parent in file_rows
                    if parent.table == fk.referenced_table
                ]
                all_parents_known = parents and all(
                    not any(value is _UNKNOWN or value is None for value in parent)
                    for parent in parents
                )
                if all_parents_known and child not in set(parents):
                    flag(
                        row.edit_id,
                        f"{row.table} INSERT FOREIGN KEY ({', '.join(fk.columns)}) "
                        f"value {child!r} has no matching seeded "
                        f"{fk.referenced_table} ({', '.join(fk.referenced_columns)})",
                    )

    return {edit_id: "; ".join(items) for edit_id, items in sorted(reasons.items())}
