"""Unit tests for hive.overwrite_race_synth + specify lever-L3 wiring.

Lever L3 (hivework.default.0057.0004-T): the fix-synthesis backend for the 0062 client
write-write overwrite-race — a reactive ref written by a LIVE event handler AND an ASYNC
refetch, where the late stale async write clobbers the live value (appear-then-vanish). It
closes NR0003's three holes:
  ① LOWERING — recognise the symptom from the live file and inject the deterministic
     guarded-write edits (generation token + helper, live-writer advance, async capture +
     stale-reject), the template specify never had.
  ② ORACLE — synthesise a last-write-wins red test (RED while unguarded, GREEN once the
     guard discards the stale response), the behaviour oracle the race class never had.
  ③ CONSISTENCY GATE — require the guard's three symbols to land together (a token advanced
     but never compared is inert), the sibling-consistency gate FK-only L1 never had.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive import overwrite_race_synth as orc


# A minimal but faithful Vue <script setup> component carrying the 0062 race: one ref written
# by an async clobber-writer (assigns from the awaited GET) and a live event handler.
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

# A vitest harness exposing the two writers over the ref (what the oracle drives). The real
# recipe wires the component; this is the shape the synthesiser binds to.
_VITEST_HARNESS = """import { makeRaceHarness } from './race_harness'
"""

_HONEY = "The mentionCopy badge in DocHeader.vue appears then vanishes after a focus refresh."


def _write_component(text: str = _RACE_COMPONENT, name: str = "DocHeader.vue"):
    """Write ``text`` to a temp file and return (codebase_root, abs_path, rel_path)."""
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "client", "src", "main")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return d, path, os.path.relpath(path, d).replace("\\", "/")


def _guard_spec_from(symptom) -> dict:
    """A ready spec whose source edits are the lever-L3 guarded-write lowering."""
    edits = orc.synthesize_guarded_write_edits(symptom)
    for e in edits:
        e["file"] = symptom.source_file
    return {"edits": edits, "termination": "ready_to_apply", "notes": ""}


class TestDetectSymptom(unittest.TestCase):
    """overwrite_race_synth.detect_overwrite_race_symptom — recognition + decline."""

    def test_recognises_race_symptom(self):
        root, path, _ = _write_component()
        sym = orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path)
        self.assertIsNotNone(sym)
        self.assertEqual(sym.ref, "mentionCopy")
        self.assertEqual(sym.async_writer.name, "fetchMentionCopy")
        self.assertEqual([w.name for w in sym.live_writers], ["onMentionCopied"])
        self.assertEqual(sym.gen_token, "mentionCopyGeneration")
        self.assertEqual(sym.guard_fn, "_shouldReplaceMentionCopy")

    def test_resolves_file_from_honey_path(self):
        # No explicit source_file: the component is resolved from the honey + codebase root.
        root, path, rel = _write_component()
        honey = f"{_HONEY} See {rel}."
        sym = orc.detect_overwrite_race_symptom(honey, root)
        self.assertIsNotNone(sym)
        self.assertEqual(sym.source_file, rel)

    def test_declines_already_guarded_async_writer(self):
        guarded = _RACE_COMPONENT.replace(
            "    const d = (res.data as any) ?? {}",
            "    const d = (res.data as any) ?? {}\n"
            "    if (fetchGeneration !== mentionCopyGeneration) return")
        guarded = guarded.replace(
            "async function fetchMentionCopy(id: string): Promise<void> {",
            "async function fetchMentionCopy(id: string): Promise<void> {\n"
            "  const fetchGeneration = ++mentionCopyGeneration")
        root, path, _ = _write_component(guarded)
        self.assertIsNone(orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path))

    def test_declines_without_live_writer(self):
        # Drop the live event handler → no writer can advance the generation → guard inert.
        only_async = _RACE_COMPONENT.split("function onMentionCopied")[0]
        root, path, _ = _write_component(only_async)
        self.assertIsNone(orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path))

    def test_declines_two_async_clobber_writers(self):
        two = _RACE_COMPONENT + """
async function refetchMentionCopy(id: string): Promise<void> {
  const res2 = await getRequest<any>(`/api/v1/mention-copy?doc_id=${id}`)
  mentionCopy.value = res2.data
}
"""
        root, path, _ = _write_component(two)
        self.assertIsNone(orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path))

    def test_honey_scopes_to_named_ref(self):
        # A second racing ref exists, but the honey names only mentionCopy → unique pick.
        two_ref = _RACE_COMPONENT + """
const otherBadge = ref<string | null>(null)
async function fetchOther(id: string): Promise<void> {
  const r = await getRequest<any>(`/o?id=${id}`)
  otherBadge.value = r.data
}
function onOther(e: any): void { otherBadge.value = e.v }
"""
        root, path, _ = _write_component(two_ref)
        # honey names mentionCopy only → fires for it
        sym = orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path)
        self.assertIsNotNone(sym)
        self.assertEqual(sym.ref, "mentionCopy")
        # honey naming neither + two structural candidates → decline (don't guess)
        self.assertIsNone(orc.detect_overwrite_race_symptom(
            "something unrelated", root, source_file=path))


class TestAutonomousMultiFileResolution(unittest.TestCase):
    """detect_overwrite_race_symptom — T0009 multi-candidate driver (no config crutch).

    The 0062 honey is MULTI-file, so the single-file ``_resolve_component_file`` declines and,
    BEFORE T0009, the lever needed the answer hand-fed via ``targets.<name>.overwrite_race.
    source_file`` (the preset "목발"). These tests assert the hive now DERIVES source_file
    itself by running the per-file recogniser across every honey-named candidate, with explicit
    smoking-gun signals on 0 / several matches."""

    def _write_multi(self, files: dict[str, str]) -> tuple[str, dict[str, str]]:
        """Write several files under one codebase; return (root, {name: rel_path})."""
        root = tempfile.mkdtemp()
        rels: dict[str, str] = {}
        for name, text in files.items():
            p = os.path.join(root, "client", "src", "components", name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            rels[name] = os.path.relpath(p, root).replace("\\", "/")
        return root, rels

    # An inert sibling the honey also names — present so the honey is multi-file.
    _INERT_TS = "export function useFlowGateSse(): void {\n  // no race here\n}\n"

    def test_auto_resolves_unique_racing_file_among_candidates(self):
        root, rels = self._write_multi(
            {"DocHeader.vue": _RACE_COMPONENT, "useFlowGateSse.ts": self._INERT_TS})
        honey = (f"mentionCopy in {rels['DocHeader.vue']} appears then vanishes; see also "
                 f"{rels['useFlowGateSse.ts']} for the SSE refetch.")
        # NO source_file supplied — the driver must pick the racing file itself.
        sym = orc.detect_overwrite_race_symptom(honey, root)
        self.assertIsNotNone(sym, "the unique racing file must be auto-resolved")
        self.assertEqual(sym.source_file, rels["DocHeader.vue"])
        self.assertEqual(sym.ref, "mentionCopy")

    def test_zero_race_candidates_declines_quietly_with_signal(self):
        # Two named files, neither racing → honest "nothing to fix" (None), not a guess.
        inert2 = "export function helper(): number {\n  return 1\n}\n"
        root, rels = self._write_multi(
            {"useFlowGateSse.ts": self._INERT_TS, "helper.ts": inert2})
        honey = f"check {rels['useFlowGateSse.ts']} and {rels['helper.ts']} for the badge."
        self.assertIsNone(orc.detect_overwrite_race_symptom(honey, root))

    def test_several_racing_files_decline_as_ambiguous(self):
        # Two DISTINCT racing files both named → ambiguous, never guess which (failure mode #2).
        other_race = _RACE_COMPONENT.replace("DocHeader", "DocFooter")
        root, rels = self._write_multi(
            {"DocHeader.vue": _RACE_COMPONENT, "DocFooter.vue": other_race})
        honey = (f"mentionCopy races in {rels['DocHeader.vue']} and {rels['DocFooter.vue']} "
                 "both.")
        self.assertIsNone(orc.detect_overwrite_race_symptom(honey, root),
                          "two racing candidates must decline, not silently pick one")

    def test_candidate_enumerator_dedups_and_resolves(self):
        root, rels = self._write_multi(
            {"DocHeader.vue": _RACE_COMPONENT, "useFlowGateSse.ts": self._INERT_TS})
        honey = (f"{rels['DocHeader.vue']} {rels['DocHeader.vue']} "  # named twice
                 f"{rels['useFlowGateSse.ts']}")
        cands = orc._candidate_component_files(honey, root)
        self.assertEqual(len(cands), 2, "duplicates must collapse, both existing files kept")


class TestLowering(unittest.TestCase):
    """overwrite_race_synth.synthesize_guarded_write_edits — the ① guarded-write template."""

    def setUp(self):
        self.root, self.path, self.rel = _write_component()
        self.sym = orc.detect_overwrite_race_symptom(_HONEY, self.root, source_file=self.path)
        self.text = open(self.path, encoding="utf-8").read()

    def test_emits_four_guard_edits_with_roles(self):
        edits = orc.synthesize_guarded_write_edits(self.sym)
        roles = [e["guard_role"] for e in edits]
        self.assertEqual(roles, ["token_decl", "live_advance", "async_capture", "async_reject"])

    def test_anchors_are_uniquely_applyable_and_produce_guarded_form(self):
        out = self.text
        for e in orc.synthesize_guarded_write_edits(self.sym):
            self.assertEqual(out.count(e["anchor_old"]), 1, e["id"])
            out = out.replace(e["anchor_old"], e["replacement_new"], 1)
        # the three guard symbols are now threaded through both writers
        self.assertIn("let mentionCopyGeneration = 0", out)
        self.assertIn("function _shouldReplaceMentionCopy(generation: number)", out)
        self.assertIn("mentionCopyGeneration += 1", out)               # live advance
        self.assertIn("const fetchGeneration = ++mentionCopyGeneration", out)  # async capture
        self.assertIn("if (!_shouldReplaceMentionCopy(fetchGeneration)) return", out)  # reject
        # the stale-reject sits BEFORE the async commit (so it actually gates it)
        self.assertLess(out.index("if (!_shouldReplaceMentionCopy(fetchGeneration)) return"),
                        out.index("mentionCopy.value = d.copied"))

    def test_failopen_on_empty_symptom(self):
        self.assertEqual(orc.synthesize_guarded_write_edits(None), [])


class TestOracle(unittest.TestCase):
    """overwrite_race_synth.synthesize_overwrite_race_red_test — the ② last-write-wins oracle."""

    def setUp(self):
        self.root, self.path, self.rel = _write_component()
        self.sym = orc.detect_overwrite_race_symptom(_HONEY, self.root, source_file=self.path)
        self.spec = _guard_spec_from(self.sym)

    def test_synthesises_for_guard_repair_with_harness(self):
        out = orc.synthesize_overwrite_race_red_test(
            self.spec, _HONEY, self.root, symptom=self.sym, setup_block=_VITEST_HARNESS)
        self.assertIsNotNone(out)
        self.assertEqual(out["edit"]["kind"], "create_file")
        self.assertTrue(out["edit"]["file"].endswith(".test.ts"))
        body = out["edit"]["content"]
        self.assertIn("makeRaceHarness()", body)
        self.assertIn("startStaleAsyncWrite()", body)
        self.assertIn("ctx.liveWrite()", body)
        self.assertIn("toEqual(live)", body)
        self.assertEqual(out["node"].split("::")[0], out["edit"]["file"])

    def test_declines_without_guard_repair(self):
        plain = {"edits": [{"id": "E1", "file": "x.vue",
                            "anchor_old": "a", "replacement_new": "b"}],
                 "termination": "ready_to_apply"}
        self.assertIsNone(orc.synthesize_overwrite_race_red_test(
            plain, _HONEY, self.root, symptom=self.sym, setup_block=_VITEST_HARNESS))

    def test_declines_without_harness(self):
        self.assertIsNone(orc.synthesize_overwrite_race_red_test(
            self.spec, _HONEY, self.root, symptom=self.sym, setup_block=None))


class TestSpecifyLowerWrapper(unittest.TestCase):
    """specify._lower_overwrite_race — injects guard edits, no-clobber, kill-switch."""

    def setUp(self):
        self.root, self.path, self.rel = _write_component()

    def test_injects_guard_edits_when_author_lowered_none(self):
        spec = {"edits": [], "termination": "needs_reinvestigation", "notes": ""}
        out = specify._lower_overwrite_race(spec, _HONEY, self.root, source_file=self.path)
        ids = {e["id"] for e in out["edits"]}
        self.assertEqual(ids, {"RACE_GUARD_TOKEN", "RACE_GUARD_LIVE_0",
                               "RACE_GUARD_ASYNC_CAPTURE", "RACE_GUARD_ASYNC_REJECT"})
        # the lowered guard is a concrete fix → applyable, loop-back cleared
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", out)
        # every injected edit is stamped with the symptom's source file
        for e in out["edits"]:
            self.assertEqual(e["file"], self.rel)
        self.assertIn("overwrite-race guard lowered (lever L3)", out["notes"])

    def test_noop_when_author_already_has_guard(self):
        spec = {"edits": [{"id": "E1", "file": self.rel,
                           "anchor_old": "x", "replacement_new": "let xGeneration = 0"}],
                "termination": "ready_to_apply"}
        out = specify._lower_overwrite_race(spec, _HONEY, self.root, source_file=self.path)
        self.assertEqual(len(out["edits"]), 1)  # untouched

    def test_noop_when_symptom_absent(self):
        root, path, _ = _write_component("const x = 1\n")
        spec = {"edits": [], "termination": "needs_reinvestigation"}
        out = specify._lower_overwrite_race(spec, "no race here", root, source_file=path)
        self.assertEqual(out["edits"], [])

    def test_kill_switch_is_noop(self):
        spec = {"edits": [], "termination": "needs_reinvestigation"}
        with mock.patch.dict(os.environ, {specify._OVERWRITE_RACE_LOWER_ENV_OFF: "1"}):
            out = specify._lower_overwrite_race(spec, _HONEY, self.root, source_file=self.path)
        self.assertEqual(out["edits"], [])


class TestGuardConsistencyGate(unittest.TestCase):
    """specify._apply_guard_consistency_gate — the ③ three-symbols-together gate."""

    def _complete_spec(self):
        root, path, rel = _write_component()
        sym = orc.detect_overwrite_race_symptom(_HONEY, root, source_file=path)
        return _guard_spec_from(sym)

    def test_complete_guard_passes(self):
        out = specify._apply_guard_consistency_gate(self._complete_spec())
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertTrue(out["guard_consistency"]["complete"])

    def test_noop_on_non_guard_spec(self):
        spec = {"edits": [{"id": "E1", "file": "x.py",
                           "anchor_old": "a", "replacement_new": "b"}],
                "termination": "ready_to_apply"}
        out = specify._apply_guard_consistency_gate(spec)
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("guard_consistency", out)

    def test_inert_guard_missing_compare_downgrades(self):
        # token advanced/captured but the async writer never COMPARES → the stale write still
        # clobbers → inert → downgrade.
        spec = self._complete_spec()
        spec["edits"] = [e for e in spec["edits"] if e["guard_role"] != "async_reject"]
        out = specify._apply_guard_consistency_gate(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_GUARD_INERT)
        self.assertIn("compare", out["guard_consistency"]["missing"])

    def test_inert_guard_missing_advance_downgrades(self):
        # a compare with nothing advancing the token → it always passes → inert.
        spec = self._complete_spec()
        spec["edits"] = [e for e in spec["edits"]
                         if e["guard_role"] not in ("live_advance", "async_capture")]
        out = specify._apply_guard_consistency_gate(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertEqual(out["reinvestigation"]["reason_code"], specify.RI_GUARD_INERT)
        self.assertIn("advance", out["guard_consistency"]["missing"])

    def test_inert_guard_missing_decl_downgrades(self):
        spec = self._complete_spec()
        spec["edits"] = [e for e in spec["edits"] if e["guard_role"] != "token_decl"]
        out = specify._apply_guard_consistency_gate(spec)
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertIn("decl", out["guard_consistency"]["missing"])

    def test_kill_switch_is_noop(self):
        spec = self._complete_spec()
        spec["edits"] = [e for e in spec["edits"] if e["guard_role"] != "async_reject"]
        with mock.patch.dict(os.environ, {specify._GUARD_CONSISTENCY_ENV_OFF: "1"}):
            out = specify._apply_guard_consistency_gate(spec)
        self.assertEqual(out["termination"], "ready_to_apply")  # unchecked


class TestSpecifyOracleWrapper(unittest.TestCase):
    """specify._synthesize_overwrite_race_red_test — verify wiring, no-clobber, kill-switch."""

    def setUp(self):
        self.root, self.path, self.rel = _write_component()
        self.sym = orc.detect_overwrite_race_symptom(_HONEY, self.root, source_file=self.path)

    def test_wires_verify(self):
        spec = specify._synthesize_overwrite_race_red_test(
            _guard_spec_from(self.sym), _HONEY, self.root,
            setup_block=_VITEST_HARNESS, source_file=self.path)
        self.assertTrue(spec["verify"]["red_test_node"].endswith("_over_stale_async"))
        self.assertEqual(spec["verify"]["test_edit_ids"], ["RACE_ORACLE_RED"])
        self.assertTrue(any(e.get("id") == "RACE_ORACLE_RED" for e in spec["edits"]))
        self.assertIn("overwrite-race oracle synthesised (lever L3)", spec["notes"])

    def test_does_not_clobber_existing_red_test(self):
        spec = _guard_spec_from(self.sym)
        spec["verify"] = {"red_test_node": "x.test.ts::test_x", "test_edit_ids": ["A"]}
        out = specify._synthesize_overwrite_race_red_test(
            spec, _HONEY, self.root, setup_block=_VITEST_HARNESS, source_file=self.path)
        self.assertEqual(out["verify"]["red_test_node"], "x.test.ts::test_x")
        self.assertFalse(any(e.get("id") == "RACE_ORACLE_RED" for e in out["edits"]))

    def test_failopen_without_harness(self):
        spec = specify._synthesize_overwrite_race_red_test(
            _guard_spec_from(self.sym), _HONEY, self.root,
            setup_block=None, source_file=self.path)
        self.assertNotIn("verify", spec)

    def test_kill_switch_is_noop(self):
        with mock.patch.dict(os.environ, {specify._OVERWRITE_RACE_ORACLE_ENV_OFF: "1"}):
            spec = specify._synthesize_overwrite_race_red_test(
                _guard_spec_from(self.sym), _HONEY, self.root,
                setup_block=_VITEST_HARNESS, source_file=self.path)
        self.assertNotIn("verify", spec)


class TestL3PipelineClosesLoop(unittest.TestCase):
    """The L3 triplet on the real 0062-shape file, in run_specify order: lower → gate →
    oracle. From an author who recognised the race but lowered NO guard, the pipeline injects
    the guard, certifies it is whole, and attaches the independent last-write-wins oracle."""

    def test_lower_then_gate_then_oracle(self):
        root, path, rel = _write_component()
        spec = {"edits": [], "termination": "needs_reinvestigation", "notes": ""}

        spec = specify._lower_overwrite_race(spec, _HONEY, root, source_file=path)
        self.assertEqual(spec["termination"], "ready_to_apply")

        spec = specify._apply_guard_consistency_gate(spec)
        self.assertTrue(spec["guard_consistency"]["complete"])
        self.assertEqual(spec["termination"], "ready_to_apply")

        spec = specify._synthesize_overwrite_race_red_test(
            spec, _HONEY, root, setup_block=_VITEST_HARNESS, source_file=path)
        self.assertTrue(spec["verify"]["red_test_node"].endswith("_over_stale_async"))
        self.assertIn("RACE_ORACLE_RED", spec["verify"]["test_edit_ids"])
        # both writers carry the guard, certified by an INDEPENDENT oracle
        blob = "\n".join(e.get("replacement_new", "") + e.get("content", "")
                         for e in spec["edits"])
        self.assertIn("mentionCopyGeneration += 1", blob)
        self.assertIn("if (!_shouldReplaceMentionCopy(fetchGeneration)) return", blob)


if __name__ == "__main__":
    unittest.main()
