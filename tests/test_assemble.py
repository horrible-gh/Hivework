"""Unit tests for hive.assemble — recipe-driven assembler system override.

The default assembler (`ASSEMBLE_SYSTEM`) renders an *investigation* honey. A
recipe may instead supply its own assembler system prompt via an
`ASSEMBLE SYSTEM OVERRIDE` section (a fenced block), so the same pipeline can
produce a different output shape (e.g. a digest) without touching the engine.

Verifies:
  ① An investigation recipe (no override section) → _extract_assemble_override None
  ② A recipe with the override section → its fenced text is returned
  ③ build_assemble_prompt uses the override when provided
  ④ build_assemble_prompt falls back to the default ASSEMBLE_SYSTEM when None
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive.assemble import (
    ASSEMBLE_SYSTEM,
    build_assemble_prompt,
    _extract_assemble_override,
)

RECIPES_DIR = os.path.join(os.path.dirname(__file__), "..", "recipes")


class TestAssembleOverrideExtraction(unittest.TestCase):
    """Tests for _extract_assemble_override."""

    def test_investigation_recipe_has_no_override(self):
        """recipe_code_bug.md is an investigation recipe — no override section."""
        path = os.path.join(RECIPES_DIR, "recipe_code_bug.md")
        self.assertTrue(os.path.exists(path), f"missing {path}")
        self.assertIsNone(_extract_assemble_override(path))

    def test_extracts_override_from_recipe(self):
        """A recipe with an ASSEMBLE SYSTEM OVERRIDE fenced block returns its text."""
        recipe = (
            "# Recipe — Demo\n\n"
            "## ASSEMBLE SYSTEM OVERRIDE\n\n"
            "```text\n"
            "# ROLE: DIGEST ASSEMBLER\n"
            "Compress, do not investigate.\n"
            "```\n\n"
            "## ① Entrance\nbody\n"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(recipe)
            path = f.name
        try:
            override = _extract_assemble_override(path)
            self.assertIsNotNone(override)
            self.assertIn("DIGEST ASSEMBLER", override)
            self.assertIn("Compress, do not investigate.", override)
            # The fence markers themselves must not be included.
            self.assertNotIn("```", override)
        finally:
            os.unlink(path)

    def test_missing_fence_returns_none(self):
        """Override heading without a following fenced block → None (malformed)."""
        recipe = "## ASSEMBLE SYSTEM OVERRIDE\n\n## Next heading\nbody\n"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(recipe)
            path = f.name
        try:
            self.assertIsNone(_extract_assemble_override(path))
        finally:
            os.unlink(path)


class TestBuildAssemblePrompt(unittest.TestCase):
    """Tests for the assemble_system parameter of build_assemble_prompt."""

    def _build(self, assemble_system):
        return build_assemble_prompt(
            combs=[{"axis_id": "A", "finding": "x"}],
            conflicts=[],
            recipe_section3="§3 rules here",
            seed_text="seed",
            rounds_used=0,
            assemble_system=assemble_system,
        )

    def test_default_used_when_none(self):
        """No override → the default investigation ASSEMBLE_SYSTEM is used."""
        prompt = self._build(None)
        self.assertIn("ROLE: ASSEMBLER", prompt)
        self.assertIn("§3 rules here", prompt)

    def test_override_replaces_default(self):
        """An override replaces the default system prompt entirely."""
        prompt = self._build("# ROLE: DIGEST ASSEMBLER\nCompress only.")
        self.assertIn("ROLE: DIGEST ASSEMBLER", prompt)
        self.assertNotIn("ROLE: ASSEMBLER (honey builder)", prompt)
        # Recipe §3 and combs are still appended.
        self.assertIn("§3 rules here", prompt)
        self.assertIn("axis_id", prompt)


if __name__ == "__main__":
    unittest.main()
