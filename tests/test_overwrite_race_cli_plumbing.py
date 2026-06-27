"""Regression test for NR0008 (hivework.default.0058): the CLI→config→run_specify boundary
that armed lever L3.

NR0008 root cause: the L3 overwrite-race lever (``hive.overwrite_race_synth`` lowering ① +
oracle ②) was fully built AND wired into ``run_specify()``, but EVERY CLI specify entry point
starved it of the two inputs it cannot derive on its own — the racing ``source_file`` (so
lowering fires on a multi-file honey, where the lever's own auto-resolve declines) and the
vitest harness ``setup_block`` (so the oracle is runnable). Unlike its siblings ⑦/L2, L3 had
no config schema and no ``*_specify_kwargs`` plumbing. So the paid full run (run598) fail-
opened to the model's weak Option-B and ran no verify — yet the existing unit tests stayed
green because they hand-fed ``source_file`` + ``setup_block`` DIRECTLY, bypassing exactly the
CLI/config boundary that was missing.

These tests lock that boundary: a real ``targets.<name>.overwrite_race`` config block flows
through ``load_config`` → ``overwrite_race_for_codebase`` → ``overwrite_race_specify_kwargs``,
and feeding the resolved kwargs into the same post-processors ``run_specify`` calls turns the
run598 multi-file-honey case from Option-B/no-verify into Option-A/verify-wired. Deterministic,
zero model cost.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive import overwrite_race_synth as orc  # noqa: F401  (kept for parity / future asserts)
from hive.config import load_config, OverwriteRaceConfig


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_hive_cli():
    """Load the top-level ``hive.py`` CLI module to reach ``overwrite_race_specify_kwargs``.

    Loaded by file path (not ``import hive``) exactly like test_commit_launcher_isolation —
    hive.py's top level is import-cheap (no stage modules) by design (B0001)."""
    path = os.path.join(_REPO, "hive.py")
    spec = importlib.util.spec_from_file_location("hive_cli_entry_nr0008", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A faithful 0062-shape race component: one ref written by an async clobber-writer (assigns
# from the awaited GET) AND a live event handler — the appear-then-vanish race.
_RACE_COMPONENT = """import { ref } from 'vue'
import { getRequest } from '../api'

const mentionCopy = ref<{ kind: string; copiedAt: string } | null>(null)

async function fetchMentionCopy(id: string): Promise<void> {
  try {
    const res = await getRequest<any>(`/api/v1/mention-copy?doc_id=${id}`)
    const d = (res.data as any) ?? {}
    mentionCopy.value = d.copied ? { kind: d.mention_kind, copiedAt: d.copied_at } : null
  } catch {
  }
}

function onMentionCopied(e: any): void {
  const detail = e.detail
  if (!detail) return
  mentionCopy.value = { kind: detail.kind, copiedAt: detail.copiedAt }
}
"""

# A second, unrelated client file the honey ALSO names — this is what makes the run598 honey
# multi-file, so the lever's own ``_resolve_component_file`` declines (several candidates).
_OTHER_TS = """export function useFlowGateSse(): void {
  // unrelated composable, present so the honey names >1 file
}
"""

# Inline harness (what oracle ② drives). The real one constructs the two writers over the ref.
_HARNESS = "import { makeRaceHarness } from './race_harness'\n"

# A run598-shaped honey: it NAMES BOTH client files (the multi-file case).
_MULTI_FILE_HONEY = (
    "The mentionCopy badge in client/src/components/DocHeader.vue appears then vanishes "
    "after a focus refresh; see also client/src/composables/useFlowGateSse.ts for the SSE "
    "path that triggers the stale refetch."
)

# The weak fix the model author lowers on its own (Option B): delete the reset assignment.
_OPTION_B_EDIT = {
    "id": "AUTHOR_OPTION_B",
    "file": "client/src/components/DocHeader.vue",
    "anchor_old": "    mentionCopy.value = null",
    "replacement_new": "",
    "rationale": "Prevent non-silent fetches from clearing the local mentionCopy badge.",
    "confidence": "medium",
}

_SRC_REL = "client/src/components/DocHeader.vue"
_OTHER_REL = "client/src/composables/useFlowGateSse.ts"


def _make_codebase() -> str:
    """Write a 0062-shape codebase (race component + a second named file) and return its root."""
    root = tempfile.mkdtemp()
    comp = os.path.join(root, *_SRC_REL.split("/"))
    other = os.path.join(root, *_OTHER_REL.split("/"))
    os.makedirs(os.path.dirname(comp), exist_ok=True)
    os.makedirs(os.path.dirname(other), exist_ok=True)
    with open(comp, "w", encoding="utf-8") as fh:
        fh.write(_RACE_COMPONENT)
    with open(other, "w", encoding="utf-8") as fh:
        fh.write(_OTHER_TS)
    return root


def _write_config(root: str, *, with_binding: bool, with_harness: bool = True) -> str:
    """Write a config JSON binding (or not) an ``overwrite_race`` block to ``root``."""
    targets: dict = {"flowgate": {}}
    if with_binding:
        block: dict = {
            "source_file": _SRC_REL,
            "codebase": root,
            "test_dir": "client/src/test",
        }
        if with_harness:
            block["setup_block"] = _HARNESS
        targets["flowgate"]["overwrite_race"] = block
    cfg_path = os.path.join(root, "hive.config.test.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


def _run_specify_tail(spec: dict, honey: str, codebase: str, kwargs: dict) -> dict:
    """Re-run the L3 post-processors run_specify calls at its end, with the CLI-resolved kwargs.

    Deterministic stand-in for the tail of ``run_specify`` (specify.py: lower → gate → oracle)
    so the boundary is exercised with NO model author/credit."""
    spec = specify._lower_overwrite_race(
        spec, honey, codebase, source_file=kwargs.get("overwrite_race_source_file"))
    spec = specify._apply_guard_consistency_gate(spec)
    spec = specify._synthesize_overwrite_race_red_test(
        spec, honey, codebase,
        setup_block=kwargs.get("overwrite_race_setup_block"),
        fixture_call=kwargs.get("overwrite_race_fixture_call", "makeRaceHarness()"),
        source_file=kwargs.get("overwrite_race_source_file"),
        test_dir=kwargs.get("overwrite_race_test_dir", "client/src/test"))
    return spec


class TestConfigSchema(unittest.TestCase):
    """load_config parses targets.<name>.overwrite_race and resolves it per-codebase."""

    def test_parses_and_resolves_binding(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True))
        orc_cfg = cfg.overwrite_race_for_codebase(root)
        self.assertIsInstance(orc_cfg, OverwriteRaceConfig)
        self.assertEqual(orc_cfg.source_file, _SRC_REL)
        self.assertEqual(orc_cfg.resolve_setup_block(root), _HARNESS)
        self.assertEqual(orc_cfg.fixture_call, "makeRaceHarness()")  # default applied
        self.assertEqual(orc_cfg.test_dir, "client/src/test")

    def test_no_binding_resolves_none(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=False))
        self.assertIsNone(cfg.overwrite_race_for_codebase(root))


class TestSpecifyKwargsHelper(unittest.TestCase):
    """hive.overwrite_race_specify_kwargs — the CLI→run_specify plumbing NR0008 added."""

    def setUp(self):
        self.cli = _load_hive_cli()

    def test_forwards_source_file_and_harness(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True))
        kw = self.cli.overwrite_race_specify_kwargs(cfg, root)
        self.assertEqual(kw.get("overwrite_race_source_file"), _SRC_REL)
        self.assertEqual(kw.get("overwrite_race_setup_block"), _HARNESS)
        self.assertEqual(kw.get("overwrite_race_fixture_call"), "makeRaceHarness()")
        self.assertEqual(kw.get("overwrite_race_test_dir"), "client/src/test")

    def test_noop_without_binding(self):
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=False))
        self.assertEqual(self.cli.overwrite_race_specify_kwargs(cfg, root), {})

    def test_source_file_only_when_no_harness(self):
        # Lowering needs only source_file; the oracle needs the harness. A binding with no
        # harness still arms lowering (Option-A) while the oracle stays a safe no-op.
        root = _make_codebase()
        cfg = load_config(path=_write_config(root, with_binding=True, with_harness=False))
        kw = self.cli.overwrite_race_specify_kwargs(cfg, root)
        self.assertEqual(kw.get("overwrite_race_source_file"), _SRC_REL)
        self.assertNotIn("overwrite_race_setup_block", kw)


class TestEndToEndBoundary(unittest.TestCase):
    """The run598 gap and its closure, exercised through the real config + helper."""

    def setUp(self):
        self.cli = _load_hive_cli()
        self.root = _make_codebase()

    def test_autonomous_resolution_lowers_option_a_without_plumbing(self):
        # T0009 (the crutch removed): with NO overwrite_race config plumbing (source_file=None,
        # setup_block=None) on the MULTI-FILE honey, the lever now DRIVES its per-file
        # recogniser across every honey-named candidate and auto-resolves the UNIQUE racing
        # file ITSELF — so lowering fires Option-A (4 guard roles) on the right file with no
        # hand-fed answer in the preset. This is the autonomous closure of the run598 gap that
        # previously required a config source_file binding (the "목발").
        spec = {"edits": [dict(_OPTION_B_EDIT)], "termination": "ready_to_apply", "verify": {}}
        spec = _run_specify_tail(spec, _MULTI_FILE_HONEY, self.root, kwargs={})  # no plumbing
        guard_roles = {e.get("guard_role") for e in spec["edits"] if e.get("guard_role")}
        self.assertEqual(
            guard_roles, {"token_decl", "live_advance", "async_capture", "async_reject"},
            "autonomous resolution must lower the Option-A guard with NO config source_file")
        # It picked the racing file out of the multi-file candidate set, not the inert sibling.
        guard_edits = [e for e in spec["edits"] if e.get("guard_role")]
        self.assertTrue(all(e.get("file") == _SRC_REL for e in guard_edits),
                        "the auto-resolved source_file must be the racing component, not the "
                        "inert useFlowGateSse.ts the honey also names")
        # The ORACLE still declines without a harness (목발 ② — the harness cannot be grounded
        # from the codebase, so it remains the genuinely hard remaining piece). Honest: lowering
        # walks on its own, but verify's RED→GREEN proof still needs makeRaceHarness realised.
        self.assertFalse((spec.get("verify") or {}).get("red_test_node"),
                         "oracle must still decline without a harness")

    def test_plumbing_closes_gap_to_option_a_and_verify(self):
        # AFTER NR0008: the config binding flows load_config → helper → kwargs, and feeding
        # those to the same post-processors lowers the Option-A guard (4 roled edits) AND
        # wires the last-write-wins oracle — the three-pieces end-to-end that run598 missed.
        cfg = load_config(path=_write_config(self.root, with_binding=True))
        kw = self.cli.overwrite_race_specify_kwargs(cfg, self.root)
        spec = {"edits": [dict(_OPTION_B_EDIT)], "termination": "ready_to_apply", "verify": {}}
        spec = _run_specify_tail(spec, _MULTI_FILE_HONEY, self.root, kwargs=kw)
        guard_roles = {e.get("guard_role") for e in spec["edits"] if e.get("guard_role")}
        self.assertEqual(
            guard_roles, {"token_decl", "live_advance", "async_capture", "async_reject"},
            "the Option-A generation guard's four symbols must all be lowered")
        self.assertTrue((spec.get("verify") or {}).get("red_test_node"),
                        "the last-write-wins oracle must be wired into verify")
        # The guard edits target the bound source file.
        guard_edits = [e for e in spec["edits"] if e.get("guard_role")]
        self.assertTrue(all(e.get("file") == _SRC_REL for e in guard_edits))


class TestAllSpecifyEntrypointsArmL3(unittest.TestCase):
    """Static guard: no specify entry point may forward write_sink kwargs but forget L3."""

    def test_every_write_sink_call_is_paired_with_overwrite_race(self):
        with open(os.path.join(_REPO, "hive.py"), encoding="utf-8") as fh:
            src = fh.read()
        ws = src.count("specify_kwargs.update(write_sink_specify_kwargs(")
        orc_n = src.count("specify_kwargs.update(overwrite_race_specify_kwargs(")
        self.assertGreaterEqual(ws, 4, "expected the four known specify entry points")
        self.assertEqual(
            orc_n, ws,
            "every specify entry point that arms lever L2 must also arm lever L3 (NR0008) — "
            "a write_sink_specify_kwargs call without a sibling overwrite_race_specify_kwargs "
            "re-opens the run598 starvation gap")


if __name__ == "__main__":
    unittest.main()
