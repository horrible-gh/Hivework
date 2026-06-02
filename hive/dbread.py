"""DBREAD — the generic, read-only DB lens for the converge data-state read.

Why this exists (the converge undecidable branch):
  ``hive.converge`` is tool-OFF and static. When its cause→symptom check comes back
  ``undecidable`` — the outcome depends on a STORED row value static evidence cannot
  determine (which row has result_doc_id set, what doc_review_status it carries, …) —
  the one fact that breaks the tie lives in the live DB, not the code. This module is
  the deterministic GLUE that fetches that one fact so the re-pass can rule on FACT
  instead of an assumed data state. The 120b model NEVER calls this and NEVER writes
  SQL: the converger emits a STRUCTURED need (table + key columns + read columns), and
  the glue here turns it into one mechanical, parameterised SELECT.

Design invariants:
  - NEUTRAL: no FlowGate (or any caller) schema knowledge. It reads whatever
    ``hive.config.DbConnection`` it is handed — like the retriever reads any codebase.
  - READ-ONLY BY CONSTRUCTION: the only statement this module ever builds is SELECT.
    sqlite is opened ``mode=ro`` so the file is never mutated; for networked DBs a
    SELECT-only account is recommended defence-in-depth but not required for safety.
  - SAFE IDENTIFIERS: table/column names originate from model output, so every
    identifier is validated against ``_IDENT`` and rejected otherwise — values are
    always passed as bound parameters, never interpolated.
  - NEVER mutates, and surfaces every failure as ``DbReadError`` for the caller to
    catch — the converge glue treats any failure as "data unavailable" and falls back
    to its static path / a ``needs_data`` termination, exactly like a missing link.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from hive.config import DbConnection

logger = logging.getLogger("hive.dbread")

# A safe SQL identifier: a leading letter/underscore then word chars. Anything else
# (spaces, quotes, dots, semicolons, parens) is rejected — these names come from the
# converger's JSON, so this is the anti-injection gate for the only un-parameterised
# part of the query (table / column identifiers).
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# mysql/mariadb share a wire protocol and the same PyMySQL driver; "mariadb" is an
# accepted alias of "mysql". Postgres is its own driver. sqlite is stdlib.
_MYSQL_KINDS = {"mysql", "mariadb"}
_PG_KINDS = {"postgres", "postgresql", "pg"}


class DbReadError(Exception):
    """Any failure reading the DB (bad identifier, missing driver, connect/query error).

    The converge glue catches this and degrades to "data unavailable" — it is never
    allowed to crash an investigation.
    """


@dataclass
class ReadResult:
    """Rows for one structured read, plus the exact SELECT that produced them.

    ``sql`` (with ``params`` rendered for display) is kept so the honey / needs_data
    block can show the operator EXACTLY what was run — the read is auditable, not magic.
    """
    rows: list[dict[str, Any]]
    sql: str
    params: list[Any]

    @property
    def found(self) -> bool:
        return bool(self.rows)


def _check_ident(name: str, what: str) -> str:
    name = (name or "").strip()
    if not _IDENT.match(name):
        raise DbReadError(f"unsafe {what} identifier: {name!r}")
    return name


def _build_select(table: str, columns: list[str] | None,
                  where: dict[str, Any] | None, limit: int,
                  placeholder: str) -> tuple[str, list[Any]]:
    """Build a parameterised single-table SELECT from validated identifiers.

    ``placeholder`` is the driver's bound-param marker ("?" for sqlite, "%s" for
    pymysql/psycopg). Values go through it; identifiers are validated, never interpolated.
    """
    tbl = _check_ident(table, "table")
    cols = "*"
    if columns:
        cols = ", ".join(_check_ident(c, "column") for c in columns)
    sql = f"SELECT {cols} FROM {tbl}"
    params: list[Any] = []
    if where:
        clauses = []
        for col, val in where.items():
            cc = _check_ident(col, "where-column")
            if val is None:
                clauses.append(f"{cc} IS NULL")
            else:
                clauses.append(f"{cc} = {placeholder}")
                params.append(val)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
    sql += f" LIMIT {int(limit)}"
    return sql, params


def _render(sql: str, params: list[Any], placeholder: str) -> str:
    """Render a human-readable SQL string with params inlined — for display/audit only."""
    out = sql
    for p in params:
        out = out.replace(placeholder, repr(p), 1)
    return out


def _read_sqlite(conn: DbConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    if not conn.path:
        raise DbReadError("sqlite connection has no 'path' (point it at the .db FILE)")
    # Read-only URI: the file is opened mode=ro so nothing can mutate it. immutable=0
    # so a concurrently-written dev DB is still read correctly.
    uri = f"file:{conn.path}?mode=ro"
    try:
        c = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error as e:
        raise DbReadError(f"sqlite connect failed for {conn.path}: {e}") from e
    try:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        raise DbReadError(f"sqlite query failed: {e}") from e
    finally:
        c.close()


def _read_mysql(conn: DbConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    try:
        import pymysql  # optional dependency — only needed for mysql/mariadb targets
        from pymysql.cursors import DictCursor
    except ImportError as e:
        raise DbReadError(
            "mysql/mariadb target needs PyMySQL installed (pip install pymysql)") from e
    try:
        c = pymysql.connect(host=conn.host or "localhost", port=conn.port or 3306,
                            user=conn.user, password=conn.secret(), database=conn.dbname,
                            connect_timeout=5, read_default_file=None,
                            cursorclass=DictCursor)
    except Exception as e:  # pymysql.err.* — surface as DbReadError
        raise DbReadError(f"mysql connect failed for {conn.host}/{conn.dbname}: {e}") from e
    try:
        with c.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        raise DbReadError(f"mysql query failed: {e}") from e
    finally:
        c.close()


def _read_postgres(conn: DbConnection, sql: str, params: list[Any]) -> list[dict[str, Any]]:
    try:
        import psycopg  # psycopg 3
        from psycopg.rows import dict_row
    except ImportError as e:
        raise DbReadError(
            "postgres target needs psycopg installed (pip install psycopg[binary])") from e
    try:
        c = psycopg.connect(host=conn.host or "localhost", port=conn.port or 5432,
                            user=conn.user, password=conn.secret(), dbname=conn.dbname,
                            connect_timeout=5, autocommit=True, row_factory=dict_row)
    except Exception as e:
        raise DbReadError(f"postgres connect failed for {conn.host}/{conn.dbname}: {e}") from e
    try:
        with c.cursor() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        raise DbReadError(f"postgres query failed: {e}") from e
    finally:
        c.close()


def read_rows(conn: DbConnection, table: str, *, columns: list[str] | None = None,
              where: dict[str, Any] | None = None, limit: int = 20) -> ReadResult:
    """Run ONE read-only single-table SELECT and return the rows as dicts.

    This is the primitive the converge data-read glue calls. ``table``/``columns``/
    ``where`` keys are validated identifiers; ``where`` VALUES are bound parameters.
    Raises ``DbReadError`` on any problem (the caller degrades to data-unavailable).
    """
    kind = (conn.kind or "sqlite").strip().lower()
    placeholder = "?" if kind == "sqlite" else "%s"
    sql, params = _build_select(table, columns, where, limit, placeholder)
    rendered = _render(sql, params, placeholder)
    logger.info("dbread: %s on %s", rendered, kind)

    if kind == "sqlite":
        rows = _read_sqlite(conn, sql, params)
    elif kind in _MYSQL_KINDS:
        rows = _read_mysql(conn, sql, params)
    elif kind in _PG_KINDS:
        rows = _read_postgres(conn, sql, params)
    else:
        raise DbReadError(f"unknown db kind: {conn.kind!r} (use sqlite|mysql|mariadb|postgres)")

    return ReadResult(rows=rows, sql=rendered, params=params)


def probe(conn: DbConnection) -> bool:
    """Cheap connectivity check — connect and run a trivial read. Raises DbReadError on failure.

    Used to confirm a configured connection actually works before relying on it (e.g.
    a first-run sanity check), without needing to know any table names.
    """
    kind = (conn.kind or "sqlite").strip().lower()
    if kind == "sqlite":
        _read_sqlite(conn, "SELECT 1", [])
    elif kind in _MYSQL_KINDS:
        _read_mysql(conn, "SELECT 1", [])
    elif kind in _PG_KINDS:
        _read_postgres(conn, "SELECT 1", [])
    else:
        raise DbReadError(f"unknown db kind: {conn.kind!r}")
    return True
