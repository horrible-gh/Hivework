"""Unit tests for hive.dbread + the db_connections config layer.

Covers the converge data-state read plumbing WITHOUT any paid call: a temp sqlite DB
is built in-process and read back through the generic module, plus the config map's
codebase matching, read-only enforcement, and the anti-injection identifier gate.
"""
import json, os, sqlite3, sys, tempfile, unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.config import load_config, DbConnection
from hive.dbread import read_rows, probe, list_schema, DbReadError


def _make_sqlite(path: str) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE documents (doc_id TEXT, doc_review_status TEXT, result_doc_id INTEGER)")
    c.execute("INSERT INTO documents VALUES ('R1', '', 42)")
    c.execute("INSERT INTO documents VALUES ('M1', NULL, NULL)")
    c.execute("INSERT INTO documents VALUES ('A1', 'approved', 7)")
    c.commit()
    c.close()


class TestSqliteRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        _make_sqlite(self.db)
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def test_read_all_columns(self):
        res = read_rows(self.conn, "documents", where={"doc_id": "R1"})
        self.assertTrue(res.found)
        self.assertEqual(res.rows[0]["doc_review_status"], "")
        self.assertEqual(res.rows[0]["result_doc_id"], 42)

    def test_read_selected_columns(self):
        res = read_rows(self.conn, "documents", columns=["doc_review_status"],
                        where={"doc_id": "A1"})
        self.assertEqual(res.rows[0], {"doc_review_status": "approved"})

    def test_where_is_null(self):
        # value None must render IS NULL, not "= NULL" (which never matches)
        res = read_rows(self.conn, "documents", where={"result_doc_id": None})
        self.assertEqual(len(res.rows), 1)
        self.assertEqual(res.rows[0]["doc_id"], "M1")

    def test_no_match_returns_empty_not_error(self):
        res = read_rows(self.conn, "documents", where={"doc_id": "ZZZ"})
        self.assertFalse(res.found)
        self.assertEqual(res.rows, [])

    def test_rendered_sql_audit(self):
        res = read_rows(self.conn, "documents", columns=["doc_id"], where={"doc_id": "R1"})
        # identifiers are delimited (sqlite/pg double-quote) so reserved words work
        self.assertIn('SELECT "doc_id" FROM "documents"', res.sql)
        self.assertIn("'R1'", res.sql)

    def test_probe_ok(self):
        self.assertTrue(probe(self.conn))

    def test_where_list_renders_in_clause(self):
        # a multi-value selector → col IN (?, ?) — used by chained reads that resolve
        # several upstream keys at once
        res = read_rows(self.conn, "documents", columns=["doc_id"],
                        where={"doc_id": ["R1", "A1"]})
        self.assertIn("IN (", res.sql)
        got = sorted(r["doc_id"] for r in res.rows)
        self.assertEqual(got, ["A1", "R1"])

    def test_where_empty_list_matches_nothing(self):
        # an empty upstream (chained read whose source returned no rows) must match
        # nothing deterministically, not raise
        res = read_rows(self.conn, "documents", where={"doc_id": []})
        self.assertEqual(res.rows, [])

    def test_reserved_word_identifier(self):
        # NR174 regression: a column legitimately named like a SQL keyword (``from``)
        # passes the bare-name gate but breaks an un-delimited SELECT. Delimiting it
        # must make the read succeed instead of raising a syntax error.
        c = sqlite3.connect(self.db)
        c.execute('CREATE TABLE links ("from" TEXT, "to" TEXT)')
        c.execute('INSERT INTO links VALUES (\'memo_item\', \'doc\')')
        c.commit()
        c.close()
        res = read_rows(self.conn, "links", columns=["to"], where={"from": "memo_item"})
        self.assertTrue(res.found)
        self.assertEqual(res.rows[0]["to"], "doc")


class TestListSchema(unittest.TestCase):
    """Schema introspection: the authoritative table/column list the converger uses
    so it names real objects instead of guessing (NR174 'items' vs the real table)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        _make_sqlite(self.db)
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def test_lists_tables_and_columns(self):
        schema = list_schema(self.conn)
        self.assertIn("documents", schema)
        self.assertEqual(schema["documents"],
                         ["doc_id", "doc_review_status", "result_doc_id"])

    def test_skips_sqlite_internal_tables(self):
        # add an index so sqlite_autoindex / sqlite_* internals could appear
        c = sqlite3.connect(self.db)
        c.execute("CREATE TABLE more (id INTEGER PRIMARY KEY AUTOINCREMENT, x TEXT)")
        c.execute("INSERT INTO more (x) VALUES ('a')")
        c.commit()
        c.close()
        schema = list_schema(self.conn)
        self.assertIn("more", schema)
        self.assertFalse(any(t.startswith("sqlite_") for t in schema))

    def test_reserved_word_table_introspects(self):
        c = sqlite3.connect(self.db)
        c.execute('CREATE TABLE links ("from" TEXT, "to" TEXT)')
        c.commit()
        c.close()
        schema = list_schema(self.conn)
        self.assertEqual(schema["links"], ["from", "to"])

    def test_missing_db_is_dbreaderror(self):
        with self.assertRaises(DbReadError):
            list_schema(DbConnection(kind="sqlite", path="/no/such/file.db"))


class TestReadOnly(unittest.TestCase):
    """mode=ro must reject any attempt to mutate, and read_rows only ever builds SELECT."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        _make_sqlite(self.db)
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def test_ro_blocks_write_at_driver(self):
        # Directly confirm the mode=ro URI the module uses forbids writes.
        c = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        with self.assertRaises(sqlite3.OperationalError):
            c.execute("DELETE FROM documents")
        c.close()


class TestIdentifierGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        _make_sqlite(self.db)
        self.conn = DbConnection(kind="sqlite", path=self.db)

    def test_bad_table_rejected(self):
        with self.assertRaises(DbReadError):
            read_rows(self.conn, "documents; DROP TABLE documents")

    def test_bad_column_rejected(self):
        with self.assertRaises(DbReadError):
            read_rows(self.conn, "documents", columns=["doc_id, (SELECT 1)"])

    def test_bad_where_column_rejected(self):
        with self.assertRaises(DbReadError):
            read_rows(self.conn, "documents", where={"doc_id = 1 OR 1=1": "x"})


class TestMissingSqlitePath(unittest.TestCase):
    def test_missing_file_is_dbreaderror(self):
        conn = DbConnection(kind="sqlite", path="/no/such/file.db")
        with self.assertRaises(DbReadError):
            read_rows(conn, "documents")

    def test_empty_path_is_dbreaderror(self):
        conn = DbConnection(kind="sqlite", path="")
        with self.assertRaises(DbReadError):
            read_rows(conn, "documents")


class TestUnknownKind(unittest.TestCase):
    def test_unknown_kind_rejected(self):
        conn = DbConnection(kind="oracle", host="x")
        with self.assertRaises(DbReadError):
            read_rows(conn, "t")


class TestConfigDbConnections(unittest.TestCase):
    def _cfg(self, obj):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return load_config(path=path)

    def test_parsed_sqlite_entry(self):
        cfg = self._cfg({"db_connections": {
            "flowgate": {"kind": "sqlite", "path": "X/flowgate.db"}}})
        conn = cfg.db_connections["flowgate"]
        self.assertEqual(conn.kind, "sqlite")
        self.assertEqual(conn.path, "X/flowgate.db")

    def test_parsed_mysql_entry(self):
        cfg = self._cfg({"db_connections": {"tw": {
            "kind": "mysql", "host": "localhost", "port": 3306,
            "dbname": "tw", "user": "u", "password": "p"}}})
        conn = cfg.db_connections["tw"]
        self.assertEqual(conn.kind, "mysql")
        self.assertEqual(conn.port, 3306)
        self.assertEqual(conn.secret(), "p")

    def test_default_no_db_connections(self):
        cfg = load_config(path="/nonexistent/hive.config.json")
        self.assertEqual(cfg.db_connections, {})

    def test_match_by_leaf_name_case_insensitive(self):
        cfg = self._cfg({"db_connections": {
            "flowgate": {"kind": "sqlite", "path": "a.db"}}})
        got = cfg.db_for_codebase("C:/workspace/projects/FlowGate")
        self.assertIsNotNone(got)
        self.assertEqual(got.path, "a.db")

    def test_match_by_explicit_codebase(self):
        cfg = self._cfg({"db_connections": {
            "primary": {"kind": "sqlite", "path": "a.db",
                        "codebase": "C:/work/SomeApp"}}})
        got = cfg.db_for_codebase("C:/work/SomeApp")
        self.assertIsNotNone(got)
        self.assertEqual(got.path, "a.db")

    def test_no_match_returns_none(self):
        cfg = self._cfg({"db_connections": {
            "flowgate": {"kind": "sqlite", "path": "a.db"}}})
        self.assertIsNone(cfg.db_for_codebase("C:/work/Unrelated"))

    def test_no_codebase_returns_none(self):
        cfg = self._cfg({"db_connections": {
            "flowgate": {"kind": "sqlite", "path": "a.db"}}})
        self.assertIsNone(cfg.db_for_codebase(None))

    def test_password_env_wins(self):
        os.environ["TEST_DB_PW_X"] = "fromenv"
        try:
            cfg = self._cfg({"db_connections": {"tw": {
                "kind": "mysql", "password": "raw", "password_env": "TEST_DB_PW_X"}}})
            self.assertEqual(cfg.db_connections["tw"].secret(), "fromenv")
        finally:
            del os.environ["TEST_DB_PW_X"]


if __name__ == "__main__":
    unittest.main()
