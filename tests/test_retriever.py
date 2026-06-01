"""Tests for hive.retriever glob normalization.

Regression guard for the investigate_e2e run-1 silent-empty bug: the queen emits
ABSOLUTE-path globs, but ``rg -g`` matches its pattern against paths RELATIVE to
the search root, so an absolute glob excluded everything → 0 hits → the JUDGE
ruled on an empty bundle. ``_norm_glob`` relativizes (or basename-degrades) each
glob so ``rg -g`` can actually match. These were invisible to the stub suite
because stubs never exercised real ``rg`` glob semantics on absolute paths.
"""
import os

from hive.retriever import _norm_glob, _validate_globs

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


# ── _validate_globs: existence-probe defence against garbage / over-broad globs.
#    These run real ``rg --files`` against a temp tree, the only thing that can
#    distinguish a structurally-valid non-path (``message/author/date``) from a
#    real one (``server/sql``) — exactly the gap a text filter cannot close.

def _make_tree(tmp_path):
    """A small repo: server/sql/q.json, server/mod/a.py..., plus many docs."""
    (tmp_path / "server" / "sql").mkdir(parents=True)
    (tmp_path / "server" / "sql" / "q.json").write_text("{}", encoding="utf-8")
    (tmp_path / "server" / "mod").mkdir(parents=True)
    for i in range(5):
        (tmp_path / "server" / "mod" / f"a{i}.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    for i in range(40):
        (tmp_path / "docs" / f"d{i}.md").write_text("# doc\n", encoding="utf-8")
    return str(tmp_path)


def test_garbage_glob_zero_match_is_dropped(tmp_path):
    # git-log fields the queen mistook for a path → 0 files → dropped.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(
        ["server/sql/*.json", "message/author/date/**/*"], root)
    assert kept == ["server/sql/*.json"]
    assert "message/author/date/**/*" in diag["dropped_empty"]


def test_overbroad_glob_dropped_when_narrower_exists(tmp_path):
    # **/* matches the whole tree; a narrow usable glob is present → drop the broad one.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(
        ["server/sql/*.json", "**/*"], root, overbroad_files=10)
    assert kept == ["server/sql/*.json"]
    assert "**/*" in diag["dropped_overbroad"]


def test_overbroad_kept_as_last_resort(tmp_path):
    # No narrower scope survives → keep the broad glob rather than search nothing.
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(["**/*"], root, overbroad_files=10)
    assert kept == ["**/*"]
    assert diag["dropped_overbroad"] == []  # not reported as dropped when kept


def test_all_garbage_falls_back_to_whole_tree(tmp_path):
    # Every glob is garbage → kept=[] so rg searches the whole tree (a grounded
    # hit anywhere beats a guaranteed-empty bundle on nonexistent dirs).
    root = _make_tree(tmp_path)
    kept, diag = _validate_globs(["no/such/dir/**/*", "also/missing/**/*"], root)
    assert kept == []
    assert len(diag["dropped_empty"]) == 2


def test_usable_globs_pass_through(tmp_path):
    root = _make_tree(tmp_path)
    kept, _ = _validate_globs(["server/sql/*.json", "server/mod/**/*"], root)
    assert kept == ["server/sql/*.json", "server/mod/**/*"]
