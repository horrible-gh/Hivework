"""Unit tests for hive.parse — comb stdout parser.

Tests against real comb output files from smoke/loop/combs/:
  - comb_D2.txt: Has ● tool-trace lines, one complete JSON, trailing content
  - comb_G.txt:  Extensive ● traces, complete JSON, trailing blank lines
  - comb_RECONCILE2.txt: ● traces + ✗ failed searches, complete JSON

Verifies:
  ① Leading ● tool-trace lines are skipped
  ② First complete top-level {...} is extracted
  ③ Trailing broken/duplicate JSON is discarded
  ④ Key fields (axis_id, termination, root_cause_signal) parse correctly
"""

import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.parse import extract_first_json, parse_comb_file

# Paths to real comb files
COMBS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "smoke", "loop", "combs"
)


class TestParseCombD2(unittest.TestCase):
    """Tests for comb_D2.txt parsing."""

    def setUp(self):
        self.path = os.path.join(COMBS_DIR, "comb_D2.txt")
        self.assertTrue(os.path.exists(self.path),
                        f"comb_D2.txt not found at {self.path}")
        self.parsed = parse_comb_file(self.path)

    def test_is_dict(self):
        """Parser returns a dict."""
        self.assertIsInstance(self.parsed, dict)

    def test_axis_id(self):
        """axis_id is 'D2'."""
        self.assertEqual(self.parsed["axis_id"], "D2")

    def test_termination(self):
        """D2 terminates as 'needs_pm'."""
        self.assertEqual(self.parsed["termination"], "needs_pm")

    def test_root_cause_signal_present(self):
        """D2 has a non-null root_cause_signal mentioning inbox_routes and queries.json."""
        sig = self.parsed["root_cause_signal"]
        self.assertIsNotNone(sig)
        self.assertIn("inbox_routes.py", sig)
        self.assertIn("queries.json:128", sig)

    def test_findings_count(self):
        """D2 has 4 findings."""
        self.assertEqual(len(self.parsed["findings"]), 4)

    def test_findings_have_reachable(self):
        """Each finding has a 'reachable' field."""
        for f in self.parsed["findings"]:
            self.assertIn("reachable", f)

    def test_conditional_reachable_exists(self):
        """At least one finding has reachable containing 'conditional'."""
        reachables = [str(f.get("reachable", "")).lower()
                      for f in self.parsed["findings"]]
        self.assertTrue(any("conditional" in r for r in reachables),
                        f"No 'conditional' reachable found: {reachables}")

    def test_tool_traces_skipped(self):
        """The raw file starts with tool-trace lines, but parsed result is valid JSON."""
        with open(self.path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Raw file should start with non-JSON content
        first_line = raw.strip().split('\n')[0]
        self.assertFalse(first_line.strip().startswith('{'),
                         "Expected tool-trace preamble before JSON")
        # But parse succeeds
        result = extract_first_json(raw)
        self.assertEqual(result["axis_id"], "D2")


class TestParseCombG(unittest.TestCase):
    """Tests for comb_G.txt parsing."""

    def setUp(self):
        self.path = os.path.join(COMBS_DIR, "comb_G.txt")
        self.assertTrue(os.path.exists(self.path),
                        f"comb_G.txt not found at {self.path}")
        self.parsed = parse_comb_file(self.path)

    def test_axis_id(self):
        """axis_id is 'G'."""
        self.assertEqual(self.parsed["axis_id"], "G")

    def test_termination(self):
        """G terminates as 'needs_runtime'."""
        self.assertEqual(self.parsed["termination"], "needs_runtime")

    def test_root_cause_signal_present(self):
        """G has a root_cause_signal mentioning queries.json."""
        sig = self.parsed["root_cause_signal"]
        self.assertIsNotNone(sig)
        self.assertIn("queries.json", sig)

    def test_findings_count(self):
        """G has 4 findings."""
        self.assertEqual(len(self.parsed["findings"]), 4)

    def test_regression_commit(self):
        """G has a regression commit mentioning 0572ded4."""
        reg = self.parsed.get("regression", {})
        self.assertIsNotNone(reg.get("commit"))
        self.assertIn("0572ded4", reg["commit"])

    def test_extensive_tool_traces_skipped(self):
        """comb_G.txt has many ● trace lines (200+) — all skipped."""
        with open(self.path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Count ● lines
        trace_lines = [l for l in raw.split('\n')
                       if l.strip().startswith('●') or l.strip().startswith('✗')]
        self.assertGreater(len(trace_lines), 20,
                           "Expected many tool-trace lines in comb_G.txt")
        # Parse still succeeds
        result = extract_first_json(raw)
        self.assertEqual(result["axis_id"], "G")


class TestParseCombReconcile2(unittest.TestCase):
    """Tests for comb_RECONCILE2.txt parsing."""

    def setUp(self):
        self.path = os.path.join(COMBS_DIR, "comb_RECONCILE2.txt")
        self.assertTrue(os.path.exists(self.path),
                        f"comb_RECONCILE2.txt not found at {self.path}")
        self.parsed = parse_comb_file(self.path)

    def test_axis_id(self):
        """axis_id is 'RECONCILE2'."""
        self.assertEqual(self.parsed["axis_id"], "RECONCILE2")

    def test_termination(self):
        """RECONCILE2 terminates as 'resolved'."""
        self.assertEqual(self.parsed["termination"], "resolved")

    def test_root_cause_signal_present(self):
        """RECONCILE2 has root_cause_signal mentioning inbox_routes and queries.json:128."""
        sig = self.parsed["root_cause_signal"]
        self.assertIsNotNone(sig)
        self.assertIn("inbox_routes.py", sig)
        self.assertIn("queries.json:128", sig)

    def test_findings_count(self):
        """RECONCILE2 has 6 findings."""
        self.assertEqual(len(self.parsed["findings"]), 6)

    def test_has_failed_search_traces(self):
        """comb_RECONCILE2.txt contains ✗ (failed search) traces — still parsed."""
        with open(self.path, 'r', encoding='utf-8') as f:
            raw = f.read()
        failed = [l for l in raw.split('\n') if l.strip().startswith('✗')]
        self.assertGreater(len(failed), 0,
                           "Expected ✗ failed search traces in comb_RECONCILE2.txt")
        # Parse still succeeds
        result = extract_first_json(raw)
        self.assertEqual(result["axis_id"], "RECONCILE2")

    def test_trailing_content_discarded(self):
        """After the JSON, any trailing content is discarded."""
        with open(self.path, 'r', encoding='utf-8') as f:
            raw = f.read()
        result = extract_first_json(raw)
        # The result should only be the first complete JSON
        self.assertIsInstance(result, dict)
        # Verify it's not contaminated with content from duplicates
        self.assertEqual(result["axis_id"], "RECONCILE2")


class TestExtractFirstJsonEdgeCases(unittest.TestCase):
    """Edge-case tests for extract_first_json."""

    def test_no_json_raises(self):
        """Raises ValueError when no JSON found."""
        with self.assertRaises(ValueError):
            extract_first_json("no json here at all")

    def test_only_traces(self):
        """Raises ValueError when only tool traces, no JSON."""
        raw = "● Read foo.py\n  │ some file\n  └ 10 lines\n"
        with self.assertRaises(ValueError):
            extract_first_json(raw)

    def test_simple_json(self):
        """Parses a simple JSON object."""
        raw = '{"key": "value", "num": 42}'
        result = extract_first_json(raw)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["num"], 42)

    def test_json_with_leading_traces(self):
        """Parses JSON after leading ● traces."""
        raw = '● Something\n  └ done\n{"axis_id": "X"}\n'
        result = extract_first_json(raw)
        self.assertEqual(result["axis_id"], "X")

    def test_json_with_trailing_garbage(self):
        """Extracts first JSON, ignores trailing broken JSON."""
        raw = '{"first": true}\n{"broken": tru'
        result = extract_first_json(raw)
        self.assertTrue(result["first"])

    def test_nested_braces_in_strings(self):
        """Handles braces inside JSON string values."""
        raw = '{"msg": "use {braces} and } here", "ok": true}'
        result = extract_first_json(raw)
        self.assertTrue(result["ok"])
        self.assertIn("{braces}", result["msg"])

    def test_brace_fragment_in_trace_before_json(self):
        """A shell snippet with braces in the tool-trace must not be mistaken
        for the comb JSON (regression: digest decompose run grabbed
        ``ForEach-Object { $_.Name }`` and failed to parse it)."""
        raw = (
            "● List files (shell)\n"
            "  │ Get-ChildItem | ForEach-Object { $_.Name }\n"
            "  └ 40 lines...\n\n"
            '{"fanout_decision": "fanout", "tasks": [{"id": "G1"}]}\n'
        )
        result = extract_first_json(raw)
        self.assertEqual(result["fanout_decision"], "fanout")
        self.assertEqual(result["tasks"][0]["id"], "G1")

    def test_small_valid_json_noise_before_comb(self):
        """A small valid JSON object echoed in a trace must not win over the
        larger real comb object that follows."""
        raw = '● echo {"x": 1}\n{"axis_id": "D2", "findings": [1, 2, 3]}'
        result = extract_first_json(raw)
        self.assertEqual(result["axis_id"], "D2")


class TestRepairStrayEscapes(unittest.TestCase):
    """Recovery of a recurring worker defect: an array string element whose
    delimiter quotes were backslash-escaped (T890 decompose failure)."""

    def test_over_escaped_array_element_recovered(self):
        # The exact shape that failed: a value containing single quotes was
        # emitted as ``\"mode='next'\"`` — invalid JSON the strict scan rejects.
        raw = (
            '{\n'
            '  "tasks": [\n'
            '    {"id": "T4", "search_plan": {"keywords": [\n'
            '      "0a0dd2d",\n'
            "      \\\"mode='next'\\\",\n"
            "      \\\"mode='info'\\\",\n"
            '      "in_progress"\n'
            '    ]}}\n'
            '  ]\n'
            '}\n'
        )
        result = extract_first_json(raw)
        kws = result["tasks"][0]["search_plan"]["keywords"]
        self.assertIn("mode='next'", kws)
        self.assertIn("mode='info'", kws)
        self.assertIn("0a0dd2d", kws)
        self.assertIn("in_progress", kws)

    def test_windows_path_globs_untouched_by_repair(self):
        # A line that starts with a real ``"`` (e.g. an escaped Windows path)
        # must not be mangled by the repair — and valid input never triggers it.
        raw = (
            '{"file_globs": [\n'
            '  "C:\\\\workspace\\\\projects\\\\FlowGate\\\\**\\\\*.py"\n'
            ']}\n'
        )
        result = extract_first_json(raw)
        self.assertEqual(result["file_globs"][0],
                         "C:\\workspace\\projects\\FlowGate\\**\\*.py")

    def test_inline_over_escaped_elements_recovered(self):
        # T890 follow-up: queen emitted the whole keyword array on ONE line, with
        # bad ``\"..\"`` elements mixed in with well-formed ``"..."`` ones. The
        # whole-line rule never fires here; the inline boundary rule must.
        raw = (
            '{\n'
            '  "tasks": [\n'
            '    {"id": "T6", "search_plan": {"keywords": '
            '["action-bar", "ActionBar", \\"mode=\'info\'\\", '
            '\\"mode=\'next\'\\", "R tab", "disabled"]}}\n'
            '  ]\n'
            '}\n'
        )
        result = extract_first_json(raw)
        kws = result["tasks"][0]["search_plan"]["keywords"]
        self.assertEqual(
            kws,
            ["action-bar", "ActionBar", "mode='info'", "mode='next'",
             "R tab", "disabled"],
        )

    def test_genuine_in_value_escaped_quotes_preserved(self):
        # A genuinely escaped quote inside a string value (preceded by ``=``, not
        # at an array boundary) is valid JSON and must survive the repair pass
        # untouched — even when the brief sits alongside a recoverable defect.
        raw = (
            '{\n'
            '  "tasks": [\n'
            '    {"id": "T2",\n'
            '     "brief": "snippet showing mode assignment '
            '(e.g. \'mode=\\"info\\"\') and disabled logic.",\n'
            '     "search_plan": {"keywords": '
            '[\\"mode=\'next\'\\", "disabled"]}}\n'
            '  ]\n'
            '}\n'
        )
        result = extract_first_json(raw)
        task = result["tasks"][0]
        self.assertIn('mode="info"', task["brief"])
        self.assertEqual(
            task["search_plan"]["keywords"], ["mode='next'", "disabled"]
        )

    def test_unrecoverable_still_raises(self):
        with self.assertRaises(ValueError):
            extract_first_json('{"k": [\\"a\\", garbage notjson ]}')


if __name__ == "__main__":
    unittest.main()
