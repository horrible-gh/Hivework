"""Regression guard for B0001 (NR hivework.default.0049.0003) — commit-path
import isolation.

The commit launcher runs the live working-tree ``hive.py`` as its CLI entry. Before
the fix, ``hive.py`` eager-imported the whole pipeline (decompose, fanout, specify,
converge, investigate, …) at module top. Because this CLI is self-hosting — it
commits Hivework's own changes while those very files are mid-edit — a half-saved
WIP module (e.g. a ``try`` block momentarily missing its ``except`` →
``SyntaxError`` at import time) made ``python hive.py commit-plan`` die at parse
time, blocking the commit precisely when it was most needed.

The fix defers every WIP-prone *stage* import into the command handler that uses
it, so the commit / commit-plan path only touches light, stable modules
(config, ledger, commit→parse, providers, backup, secrets).

These tests run in a subprocess so loading ``hive.py`` (and any stage modules it
might wrongly pull) can't pollute the parent interpreter's ``sys.modules``.
"""

import os
import subprocess
import sys
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stage modules that get actively edited during development and therefore carry
# the WIP-SyntaxError risk. None of these may sit on the commit path.
WIP_STAGE_MODULES = [
    "hive.decompose",
    "hive.fanout",
    "hive.conflict_scan",
    "hive.reconcile",
    "hive.assemble",
    "hive.specify",
    "hive.reinvestigate",
    "hive.apply",
    "hive.converge",
    "hive.coordinator",
    "hive.investigate",
    "hive.be_root",
]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_loading_cli_entry_does_not_eager_import_stage_modules():
    """Importing hive.py's top level must not transitively load any stage module."""
    code = """
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location('hive_cli_entry', 'hive.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        WIP = %r
        leaked = [m for m in WIP if m in sys.modules]
        print('LEAKED=' + ','.join(leaked))
        # The commit path's own light deps SHOULD be importable.
        for need in ('hive.commit', 'hive.config', 'hive.ledger'):
            assert need in sys.modules, 'commit-path dep missing: ' + need
        sys.exit(1 if leaked else 0)
    """ % (WIP_STAGE_MODULES,)
    r = _run(code)
    assert r.returncode == 0, (
        "hive.py eager-imported WIP-prone stage modules — commit path is no "
        "longer isolated:\n" + r.stdout + r.stderr
    )


def test_commit_plan_help_survives_a_broken_stage_module():
    """A SyntaxError in a stage module must NOT break the commit-plan entry point.

    This reproduces the exact B0001 failure mode: a half-saved ``try`` (no
    ``except``/``finally``) in ``hive/specify.py``. We inject it into a private
    copy of the repo's ``hive`` package on a temp sys.path so the real source is
    never touched, then confirm ``hive.py commit-plan --help`` still parses/loads.
    """
    code = """
        import importlib.util, sys, os, tempfile, shutil

        # Break a stage module IN MEMORY by pre-seeding sys.modules with a stub
        # that raises on attribute import is not enough — the bug is at *import*
        # time. Instead, install a finder that makes `import hive.specify` raise
        # SyntaxError, mimicking a half-saved WIP file.
        import importlib.abc, importlib.machinery

        class BrokenSpecifyLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise SyntaxError("expected 'except' or 'finally' block")

        class BrokenSpecifyFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == 'hive.specify':
                    return importlib.machinery.ModuleSpec(fullname, BrokenSpecifyLoader())
                return None

        sys.meta_path.insert(0, BrokenSpecifyFinder())

        # Sanity: importing the stage module now blows up exactly like a WIP file.
        try:
            import hive.specify  # noqa
            print('UNEXPECTED: hive.specify imported cleanly'); sys.exit(2)
        except SyntaxError:
            pass

        # The fix's promise: the CLI entry still loads despite specify being broken.
        spec = importlib.util.spec_from_file_location('hive_cli_entry', 'hive.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # must NOT raise

        # And the commit-plan argparse path is reachable (no stage import on it).
        sys.argv = ['hive.py', 'commit-plan', '--help']
        try:
            mod.main()
        except SystemExit as e:
            assert e.code == 0, 'commit-plan --help exited ' + str(e.code)
        print('OK')
        sys.exit(0)
    """
    r = _run(code)
    assert r.returncode == 0, (
        "commit-plan entry did not survive a broken stage module:\n"
        + r.stdout + r.stderr
    )
