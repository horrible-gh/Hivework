"""Migration schema grounding and deterministic generated-test INSERT gate."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import schema_ground, specify
from hive.providers import WorkerResult


MIGRATION_028 = """
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL
);

CREATE TABLE project_modules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_project_modules_project_name UNIQUE (project_id, name),
    CONSTRAINT fk_project_modules_project
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
);
"""


def _edit(content: str, edit_id: str = "T1") -> dict:
    return {
        "id": edit_id,
        "kind": "create_file",
        "file": "tests/test_project_modules.py",
        "content": content,
        "rationale": "add isolated DB regression coverage",
        "evidence": ["028_project_modules.sql"],
        "confidence": "high",
        "anchor_status": "verified",
    }


class TestMigrationSchemaParsing(unittest.TestCase):
    def test_028_constraints_are_extracted(self):
        schema = schema_ground.parse_migration_sql(
            MIGRATION_028, source="server/migrations/028_project_modules.sql")

        modules = schema["project_modules"]
        self.assertEqual(modules.primary_key, ("id",))
        self.assertIn(("project_id", "name"), modules.unique)
        self.assertEqual(
            modules.not_null,
            {"project_id", "name", "title", "created_at"},
        )
        self.assertIn("created_at", modules.defaults)
        self.assertIn("id", modules.generated)
        self.assertEqual(
            modules.foreign_keys,
            [schema_ground.ForeignKey(
                columns=("project_id",),
                referenced_table="projects",
                referenced_columns=("project_id",),
            )],
        )

    def test_alter_table_and_unique_index_are_merged(self):
        schema = schema_ground.parse_migration_sql("""
            CREATE TABLE widgets (id TEXT, account_id TEXT);
            ALTER TABLE widgets ADD COLUMN label TEXT NOT NULL;
            ALTER TABLE widgets ADD CONSTRAINT fk_widget_account
                FOREIGN KEY (account_id) REFERENCES accounts(id);
            CREATE UNIQUE INDEX uq_widget_label ON widgets(account_id, label);
        """)
        widgets = schema["widgets"]
        self.assertIn("label", widgets.not_null)
        self.assertIn(("account_id", "label"), widgets.unique)
        self.assertEqual(widgets.foreign_keys[0].referenced_table, "accounts")

    def test_discovery_is_sorted_and_migration_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            paths = [
                os.path.join(root, "server", "migrations", "028_b.sql"),
                os.path.join(root, "db", "migration", "001_a.sql"),
                os.path.join(root, "schema.sql"),
            ]
            for path in paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("CREATE TABLE x (id TEXT);")

            found = schema_ground.discover_migration_files(root)
            self.assertEqual(found, sorted(paths[:2], key=lambda p: p.lower()))

    def test_grounding_only_renders_tables_named_by_honey(self):
        schema = schema_ground.parse_migration_sql(
            MIGRATION_028 + "\nCREATE TABLE audit_log (id TEXT PRIMARY KEY);")
        tables = schema_ground.touched_tables(
            "Seed project_modules with two rows for the regression.", schema)
        block = schema_ground.render_schema_grounding(schema, tables)

        self.assertIn("- table `project_modules`", block)
        self.assertIn("UNIQUE (project_id, name)", block)
        self.assertIn("FOREIGN KEY (project_id) REFERENCES projects", block)
        self.assertNotIn("- table `audit_log`", block)


class TestGeneratedInsertValidation(unittest.TestCase):
    def setUp(self):
        self.schema = schema_ground.parse_migration_sql(MIGRATION_028)

    def test_t908_duplicate_project_name_is_flagged(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "INSERT INTO project_modules (project_id, name, title) "
            "VALUES ('p1', 'alpha', 'Alpha');\n"
            "INSERT INTO project_modules (project_id, name, title) "
            "VALUES ('p1', 'alpha', 'Duplicate');"
        )], self.schema)

        self.assertIn("T1", findings)
        self.assertIn("duplicates UNIQUE (project_id, name)", findings["T1"])

    def test_not_null_omission_and_explicit_null_are_flagged(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "INSERT INTO project_modules (project_id, name) VALUES ('p1', NULL);"
        )], self.schema)

        self.assertIn("omits NOT NULL column(s): title", findings["T1"])
        self.assertIn("sets NOT NULL column(s) to NULL: name", findings["T1"])
        self.assertNotIn("created_at", findings["T1"])

    def test_fk_mismatch_is_flagged_when_parent_fixture_is_explicit(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "INSERT INTO projects (project_id, project_name) VALUES ('p1', 'One');\n"
            "INSERT INTO project_modules (project_id, name, title) "
            "VALUES ('p2', 'alpha', 'Alpha');"
        )], self.schema)

        self.assertIn("FOREIGN KEY (project_id)", findings["T1"])
        self.assertIn("has no matching seeded projects", findings["T1"])

    def test_unknown_values_and_unseeded_parent_fail_open(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "INSERT INTO project_modules (project_id, name, title) VALUES (?, ?, ?);\n"
            "INSERT INTO project_modules (project_id, name, title) VALUES (?, ?, ?);"
        )], self.schema)
        self.assertEqual(findings, {})

    def test_unknown_parent_key_makes_fk_check_fail_open(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "INSERT INTO projects (project_id, project_name) VALUES (?, 'Dynamic');\n"
            "INSERT INTO projects (project_id, project_name) VALUES ('p1', 'One');\n"
            "INSERT INTO project_modules (project_id, name, title) "
            "VALUES ('p2', 'alpha', 'Alpha');"
        )], self.schema)
        self.assertEqual(findings, {})

    def test_same_tuple_in_separate_test_functions_is_allowed(self):
        findings = schema_ground.validate_test_inserts([_edit(
            "def test_one(test_db):\n"
            "    test_db.execute(\"INSERT INTO project_modules "
            "(project_id, name, title) VALUES ('p1', 'alpha', 'Alpha')\")\n\n"
            "def test_two(test_db):\n"
            "    test_db.execute(\"INSERT INTO project_modules "
            "(project_id, name, title) VALUES ('p1', 'alpha', 'Alpha')\")\n"
        )], self.schema)
        self.assertEqual(findings, {})

    def test_non_test_edit_is_ignored(self):
        edit = _edit(
            "INSERT INTO project_modules (project_id, name, title) "
            "VALUES ('p1', 'alpha', 'Alpha'), ('p1', 'alpha', 'Again');")
        edit["file"] = "server/bootstrap.py"
        self.assertEqual(
            schema_ground.validate_test_inserts([edit], self.schema), {})


class TestSpecifySchemaWiring(unittest.TestCase):
    def test_prompt_is_grounded_and_duplicate_insert_downgrades(self):
        with tempfile.TemporaryDirectory() as root:
            migrations = os.path.join(root, "server", "migrations")
            os.makedirs(migrations)
            with open(os.path.join(migrations, "028_project_modules.sql"),
                      "w", encoding="utf-8") as handle:
                handle.write(MIGRATION_028)

            honey = os.path.join(root, "honey.md")
            contract = os.path.join(root, "contract.md")
            output = os.path.join(root, "edit_spec.json")
            with open(honey, "w", encoding="utf-8") as handle:
                handle.write(
                    "Add an isolated test that seeds project_modules for T908.")
            with open(contract, "w", encoding="utf-8") as handle:
                handle.write("Return the edit-spec JSON.")

            authored = {
                "edits": [_edit(
                    "INSERT INTO project_modules (project_id, name, title) "
                    "VALUES ('p1', 'alpha', 'Alpha'), "
                    "('p1', 'alpha', 'Duplicate');"
                )],
                "deferred": [],
                "gate": {"commands": ["pytest"], "apply": False},
                "termination": "ready_to_apply",
                "notes": "",
            }
            worker_results = [
                WorkerResult(
                    stdout=json.dumps(authored), stderr="", exit_code=0, latency_s=0.01),
                WorkerResult(
                    stdout=json.dumps({"reviews": [{
                        "id": "T1",
                        "effective": True,
                        "coherent": True,
                        "in_scope": True,
                        "reason": "valid red test",
                    }]}),
                    stderr="", exit_code=0, latency_s=0.01,
                ),
            ]

            with mock.patch("hive.specify.call_worker",
                            side_effect=worker_results) as worker:
                result = specify.run_specify(
                    honey, root, output, contract_path=contract)

            author_prompt = worker.call_args_list[0].args[2]
            self.assertIn("## Migration schema constraints", author_prompt)
            self.assertIn("UNIQUE (project_id, name)", author_prompt)
            self.assertEqual(result["termination"], "needs_reinvestigation")
            self.assertEqual(
                result["reinvestigation"]["gate"], "schema_constraints")
            self.assertIn("T1", result["effectiveness"]["ineffective_ids"])
            self.assertFalse(result["edits"][0]["effectiveness"]["ok"])


if __name__ == "__main__":
    unittest.main()
