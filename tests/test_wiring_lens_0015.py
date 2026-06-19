"""R0015 — regression tests for the decoy-blind-spot fix (TR0007).

Background: the live Lv1 cycle (run462) missed a sort regression because the
FIND drone read the intact ``_file_tree_sort_key`` helper + its unit tests and
declared the backend correct, then misattributed the disorder to the frontend —
never verifying the helper is actually CALLED at the sort site. NR0005 pinned
three root causes:
  RC-1  drone infers wiring from a symbol's existence instead of citing the call site
  RC-2  drone cites tests as correctness proof without distinguishing unit-of-helper
        (green, irrelevant) from e2e (would be red) and without running them
  RC-3  converge lenses refute only the WINNING attribution, never the exclusionary
        "component X is correct, so the cause is Y" premise that drops the true locus

This locks in the fix: a new ``wiring`` converge lens (RC-3) wired into the code
defaults, the loader default, and the default profile; plus the comb-contract
wiring/test-proof clauses (RC-1/RC-2).
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import converge as C
from hive.config import ConvergeLensConfig, load_config

_ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestWiringLensDefinition(unittest.TestCase):
    """RC-3: the wiring lens exists and targets the exclusion premise."""

    def test_wiring_lens_registered(self):
        self.assertIn("wiring", C._LENS_DEFINITIONS)

    def test_wiring_lens_targets_callsite_not_definition(self):
        desc = C._LENS_DEFINITIONS["wiring"].lower()
        # Must talk about the CALL SITE / invocation, not just the definition.
        self.assertIn("call site", desc)
        self.assertIn("invoked", desc)
        # Must name the "definition present, not wired" failure class and that
        # unit-isolation tests do not prove wiring.
        self.assertIn("definition", desc)
        self.assertIn("unit", desc)
        self.assertIn("refute", desc)


class TestWiringLensDefaults(unittest.TestCase):
    """RC-3: wiring is on by default everywhere a lens panel reads its set."""

    def test_dataclass_default_includes_wiring(self):
        self.assertIn("wiring", ConvergeLensConfig().lenses)

    def test_loader_default_includes_wiring_when_unspecified(self):
        # A config with a lens block but NO explicit lenses list → loader default.
        cfg = self._load({"pipeline": {"converge": {"provider": "copilot",
                          "model": "m", "lens": {"enabled": True}}}})
        self.assertIn("wiring", cfg.converge_lens.lenses)

    def test_explicit_lenses_still_respected(self):
        # An explicit list is NOT silently augmented — operator intent wins.
        cfg = self._load({"pipeline": {"converge": {"provider": "copilot",
                          "model": "m", "lens": {"enabled": True,
                          "lenses": ["omission"]}}}})
        self.assertEqual(cfg.converge_lens.lenses, ["omission"])

    def test_default_profile_enables_wiring(self):
        cfg = load_config(path=os.path.join(_ROOT, "config", "hive.config.default.json"))
        self.assertTrue(cfg.converge_lens.enabled)
        self.assertIn("wiring", cfg.converge_lens.lenses)

    def _load(self, raw):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "hive.config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        return load_config(path=path)


class TestCombContractWiringClause(unittest.TestCase):
    """RC-1/RC-2: the FIND contract demands call-site proof for exclusions."""

    def _contract(self):
        with open(os.path.join(_ROOT, "recipes", "comb_contract_v2.md"),
                  encoding="utf-8") as f:
            return f.read().lower()

    def test_requires_callsite_for_exclusion(self):
        c = self._contract()
        self.assertIn("call site", c)
        self.assertIn("definition present, not wired", c)

    def test_tests_are_not_wiring_proof(self):
        c = self._contract()
        # Must explicitly say unit/isolation tests don't prove end-to-end wiring.
        self.assertIn("unit", c)
        self.assertIn("end-to-end", c)


if __name__ == "__main__":
    unittest.main()
