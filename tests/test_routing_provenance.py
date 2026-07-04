"""Routing provenance + drift guard (group 0081).

The 07-04 misrouting (judge silently flipped openai→copilot in the gitignored
local profile) was only reconstructable from a leftover startup-banner log line.
These tests pin the two defenses added against a recurrence:

1. Ledger provenance — start_run persists the resolved per-stage routing plus the
   config file path/sha256 (``runs.routing_json``), so a post-hoc audit never has
   to trust the file on disk again.
2. Drift guard — check_routing_drift compares a loaded config against the
   committed certified manifest and names every mismatching key.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.config import check_routing_drift, load_config
from hive.ledger import Ledger, NullLedger

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CERTIFIED = os.path.join(REPO_ROOT, "config", "hive.routing.certified.json")


def _write_profile(tmpdir: str, judge_provider: str, judge_model: str,
                   jury_size: int = 3, max_axes: int = 10) -> str:
    """A minimal schema_v2 profile with a parameterized judge block."""
    path = os.path.join(tmpdir, "profile.json")
    cfg = {
        "schema_version": 2,
        "pipeline": {
            "fanout": {"provider": "openai",
                       "model": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
            "judge": {"provider": judge_provider, "model": judge_model,
                      "jury_size": jury_size, "max_axes": max_axes},
            "converge": {"provider": "copilot", "model": "gpt-5-mini"},
            "designer": {"provider": "copilot", "model": "claude-sonnet-4.5"},
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


class TestConfigProvenance(unittest.TestCase):
    """load_config stamps WHERE the routing came from; routing_snapshot carries it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = _write_profile(self.tmpdir, "openai", "openai/gpt-oss-120b")
        self.cfg = load_config(path=self.path)

    def test_source_path_and_sha_stamped(self):
        self.assertEqual(self.cfg.source_path, self.path)
        self.assertEqual(len(self.cfg.source_sha256), 64)

    def test_snapshot_carries_resolved_judge_routing(self):
        snap = self.cfg.routing_snapshot()
        self.assertEqual(snap["roles"]["judge"],
                         {"provider": "openai", "model": "openai/gpt-oss-120b"})
        self.assertEqual(snap["judge_caps"],
                         {"votes_per_axis": 3, "max_axes": 10})
        self.assertEqual(snap["config_path"], self.path)
        self.assertEqual(snap["config_sha256"], self.cfg.source_sha256)


class TestRoutingDrift(unittest.TestCase):
    """check_routing_drift vs the committed certified manifest."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_certified_manifest_is_tracked_and_parses(self):
        self.assertTrue(os.path.exists(CERTIFIED),
                        "certified routing manifest must ship with the repo")
        with open(CERTIFIED, encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["roles"]["judge"]["provider"], "openai")
        self.assertEqual(manifest["judge_caps"]["votes_per_axis"], 3)

    def test_matching_config_reports_no_drift(self):
        cfg = load_config(path=_write_profile(self.tmpdir, "openai",
                                              "openai/gpt-oss-120b"))
        self.assertEqual(check_routing_drift(cfg), [])

    def test_the_0081_misrouting_is_named(self):
        # The exact 07-04 drift: judge flipped to copilot/gpt-5-mini.
        cfg = load_config(path=_write_profile(self.tmpdir, "copilot", "gpt-5-mini"))
        drifts = check_routing_drift(cfg)
        joined = "\n".join(drifts)
        self.assertIn("judge.provider", joined)
        self.assertIn("copilot", joined)
        self.assertIn("judge.model", joined)

    def test_cap_drift_is_named(self):
        # The cap half of the same drift event: jury 6 / max_axes 12.
        cfg = load_config(path=_write_profile(self.tmpdir, "openai",
                                              "openai/gpt-oss-120b",
                                              jury_size=6, max_axes=12))
        joined = "\n".join(check_routing_drift(cfg))
        self.assertIn("judge_caps.votes_per_axis", joined)
        self.assertIn("judge_caps.max_axes", joined)

    def test_missing_manifest_is_silent(self):
        cfg = load_config(path=_write_profile(self.tmpdir, "copilot", "gpt-5-mini"))
        self.assertEqual(
            check_routing_drift(cfg, manifest_path=os.path.join(self.tmpdir, "nope.json")),
            [])


class TestLedgerRoutingColumn(unittest.TestCase):
    """runs.routing_json is persisted (and migrated onto pre-0081 DBs)."""

    def test_routing_json_persisted(self):
        db = os.path.join(tempfile.mkdtemp(), "ledger.db")
        ldg = Ledger(db)
        snap = {"config_path": "x.json", "config_sha256": "ab" * 32,
                "roles": {"judge": {"provider": "openai",
                                    "model": "openai/gpt-oss-120b"}}}
        ldg.start_run(seed="s", codebase="/r", model_queen="m", model_fanout="m",
                      routing_json=json.dumps(snap))
        ldg.close()
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT routing_json FROM runs").fetchone()
        conn.close()
        self.assertEqual(json.loads(row[0])["roles"]["judge"]["provider"], "openai")

    def test_legacy_runs_table_gains_column(self):
        db = os.path.join(tempfile.mkdtemp(), "legacy.db")
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, ts TEXT, seed TEXT,"
            " work_type TEXT, codebase TEXT, model_queen TEXT, model_fanout TEXT,"
            " axes_n INTEGER, rounds INTEGER, conflicts_n INTEGER,"
            " remaining_n INTEGER, parse_errs INTEGER, total_in_chars INTEGER,"
            " total_out_chars INTEGER, total_est_tokens INTEGER,"
            " total_real_tokens INTEGER, elapsed_s REAL, honey_path TEXT, status TEXT)")
        conn.commit()
        conn.close()
        ldg = Ledger(db)
        ldg.start_run(seed="s", codebase="/r", model_queen="m", model_fanout="m",
                      routing_json="{}")
        ldg.close()
        conn = sqlite3.connect(db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
        conn.close()
        self.assertIn("routing_json", cols)

    def test_start_run_without_routing_still_works(self):
        db = os.path.join(tempfile.mkdtemp(), "ledger.db")
        ldg = Ledger(db)
        ldg.start_run(seed="s", codebase="/r", model_queen="m", model_fanout="m")
        self.assertIsNotNone(ldg.run_id)
        ldg.close()


class TestCostSummary(unittest.TestCase):
    """cost_summary aggregates the CURRENT run's calls per provider/model."""

    def test_aggregates_by_provider_model(self):
        db = os.path.join(tempfile.mkdtemp(), "ledger.db")
        ldg = Ledger(db)
        ldg.start_run(seed="s", codebase="/r", model_queen="m", model_fanout="m")
        for _ in range(3):
            ldg.record_call("judge", "A", "copilot", "gpt-5-mini",
                            prompt="p" * 40, output="o" * 20, latency_s=1.0)
        ldg.record_call("lens", "A", "openai", "Qwen", prompt="p" * 40,
                        output="o" * 20, latency_s=1.0, real_tokens=100)
        summary = ldg.cost_summary()
        ldg.close()
        by_key = {(r["provider"], r["model"]): r for r in summary}
        self.assertEqual(by_key[("copilot", "gpt-5-mini")]["calls"], 3)
        # copilot never reports real tokens — the aggregate must say so, not 0.
        self.assertIsNone(by_key[("copilot", "gpt-5-mini")]["real_tokens"])
        self.assertEqual(by_key[("openai", "Qwen")]["calls"], 1)
        self.assertEqual(by_key[("openai", "Qwen")]["real_tokens"], 100)

    def test_null_ledger_returns_empty(self):
        self.assertEqual(NullLedger().cost_summary(), [])


if __name__ == "__main__":
    unittest.main()
