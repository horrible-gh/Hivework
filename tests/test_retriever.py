"""Tests for hive.retriever glob normalization.

Regression guard for the investigate_e2e run-1 silent-empty bug: the queen emits
ABSOLUTE-path globs, but ``rg -g`` matches its pattern against paths RELATIVE to
the search root, so an absolute glob excluded everything → 0 hits → the JUDGE
ruled on an empty bundle. ``_norm_glob`` relativizes (or basename-degrades) each
glob so ``rg -g`` can actually match. These were invisible to the stub suite
because stubs never exercised real ``rg`` glob semantics on absolute paths.
"""
from hive.retriever import _norm_glob

ROOT = "C:/workspace/projects/Documents/projects/FlowGate"


def test_absolute_glob_under_root_becomes_relative():
    g = "C:/workspace/projects/Documents/projects/FlowGate/210_design/D031_*.md"
    assert _norm_glob(g, ROOT) == "210_design/D031_*.md"


def test_absolute_glob_with_backslashes_normalizes_to_relative():
    g = r"C:\workspace\projects\Documents\projects\FlowGate\420_task_reports\TR880_*.md"
    assert _norm_glob(g, ROOT) == "420_task_reports/TR880_*.md"


def test_case_insensitive_drive_prefix_match():
    # Windows paths: queen may differ in drive-letter / separator case.
    g = "c:/WORKSPACE/projects/Documents/projects/FlowGate/210_design/**/*.md"
    assert _norm_glob(g, ROOT) == "210_design/**/*.md"


def test_absolute_glob_outside_root_degrades_to_basename():
    # Code repo lives elsewhere; an out-of-root absolute glob can't be relativized,
    # so it degrades to a basename pattern rg matches at any depth (better than the
    # old behaviour of matching nothing).
    g = "C:/workspace/projects/FlowGate/client/src/**/*.vue"
    assert _norm_glob(g, ROOT) == "*.vue"


def test_relative_glob_passes_through():
    assert _norm_glob("210_design/**/*.md", ROOT) == "210_design/**/*.md"


def test_basename_only_glob_passes_through():
    assert _norm_glob("*.md", ROOT) == "*.md"


def test_root_with_trailing_slash_is_handled():
    g = "C:/workspace/projects/Documents/projects/FlowGate/210_design/D031_*.md"
    assert _norm_glob(g, ROOT + "/") == "210_design/D031_*.md"
