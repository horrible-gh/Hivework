"""Unit tests for hive.write_sink_synth + specify._synthesize_write_sink_red_test.

Lever L2 (hivework.default.0048.0003-NR): the INDEPENDENT write-sink behaviour oracle that
L1's loop-back was missing. When the spec repairs a write-sink (a routing ``insert_*event``
edit — the 0082 FK mis-routing) and the symptom is a mutating endpoint that 500s, synthesise
a TestClient red test that issues the request and asserts NOT 500 — RED on the FK violation,
GREEN once the callee-swap routes the event to the table its FK references. Coverage:
  ① synthesises edit + verify node when a routing repair + a runnable harness both resolve
  ② declines (fail-open) without a routing repair, without a harness, without a TestClient
     fixture, and for an unseeded ``{param}`` request template
  ③ grounds the mutating verb+path from the honey when no explicit request is supplied
  ④ the specify wrapper wires spec.verify, never clobbers an existing red test, and is a
     no-op under the kill-switch HIVE_NO_WRITE_SINK_ORACLE
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hive import specify
from hive import write_sink_synth as ws

# A minimal harness: a pytest TestClient fixture named ``client`` (what the generated test
# binds to). The real recipe seeds a group + mounts the dispose router; this is the shape.
_HARNESS = """import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    app = _build_seeded_app()
    yield TestClient(app)
"""


def _write_sink_spec(**over):
    """A ready spec whose source edit repairs the 0082 write-sink (callee-swap)."""
    spec = {
        "edits": [{
            "id": "E1", "file": "server/modules/flow_gate/process_service.py",
            "anchor_old": '        db.insert_event(group_id, "group_disposed", note=note)',
            "replacement_new": '        db.insert_group_event(group_id, "group_disposed", note=note)',
            "confidence": "high", "anchor_status": "verified",
        }],
        "termination": "ready_to_apply",
        "notes": "",
    }
    spec.update(over)
    return spec


_REQUEST = {"verb": "post", "path": "/api/v1/groups/grp_ws/dispose",
            "json": {"reason_option": "OTHER", "reason_detail": "t"}}


class TestSynthesizeWriteSinkRedTest(unittest.TestCase):
    """write_sink_synth.synthesize_write_sink_red_test — recognition + construction."""

    def test_synthesises_for_routing_repair_with_harness(self):
        out = ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertIsNotNone(out)
        self.assertEqual(out["edit"]["kind"], "create_file")
        self.assertTrue(out["edit"]["file"].endswith(".py"))
        self.assertEqual(out["node"].split("::")[0], out["edit"]["file"])
        body = out["edit"]["content"]
        # binds to the harness fixture and asserts NOT 500 on the mutating request
        self.assertIn("def " + out["node"].split("::")[1] + "(client):", body)
        self.assertIn("client.post('/api/v1/groups/grp_ws/dispose'", body)
        self.assertIn("response.status_code != 500", body)
        # the harness itself is prepended so the test is runnable
        self.assertIn("TestClient", body)

    def test_no_routing_repair_declines(self):
        # A source edit that is not an insert_*event routing repair → not a write-sink fix.
        spec = _write_sink_spec()
        spec["edits"][0]["replacement_new"] = "    return compute_total(items)"
        spec["edits"][0]["anchor_old"] = "    return compute(items)"
        self.assertIsNone(ws.synthesize_write_sink_red_test(
            spec, "", "", setup_block=_HARNESS, request=_REQUEST))

    def test_no_harness_declines(self):
        self.assertIsNone(ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=None, request=_REQUEST))

    def test_harness_without_testclient_fixture_declines(self):
        bad = "import pytest\n\n@pytest.fixture\ndef thing():\n    return 1\n"
        self.assertIsNone(ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=bad, request=_REQUEST))

    def test_unseeded_path_template_declines(self):
        # A request still carrying a {param} cannot reach the write → decline, don't guess.
        req = dict(_REQUEST, path="/api/v1/groups/{group_id}/dispose")
        self.assertIsNone(ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=_HARNESS, request=req))

    def test_non_mutating_request_declines(self):
        req = dict(_REQUEST, verb="get")
        self.assertIsNone(ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=_HARNESS, request=req))

    def test_grounds_route_from_honey_when_no_request(self):
        # With no explicit request, the mutating verb+path are grounded from the honey.
        binding = [{"verb": "post", "full_path": "/api/v1/groups/g1/dispose"}]
        with mock.patch.object(ws, "_resolve_http_bindings", return_value=binding):
            out = ws.synthesize_write_sink_red_test(
                _write_sink_spec(), "POST /api/v1/groups/g1/dispose 500s", "/code",
                setup_block=_HARNESS, request=None)
        self.assertIsNotNone(out)
        self.assertIn("client.post('/api/v1/groups/g1/dispose')", out["edit"]["content"])

    def test_grounded_template_route_declines(self):
        # A grounded route that still carries a path param cannot be seeded → decline.
        binding = [{"verb": "post", "full_path": "/api/v1/groups/{group_id}/dispose"}]
        with mock.patch.object(ws, "_resolve_http_bindings", return_value=binding):
            self.assertIsNone(ws.synthesize_write_sink_red_test(
                _write_sink_spec(), "honey", "/code", setup_block=_HARNESS, request=None))

    def test_request_dataclass_accepted(self):
        req = ws.WriteSinkRequest(verb="post", path="/api/v1/groups/g/dispose", json=None)
        out = ws.synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=_HARNESS, request=req)
        self.assertIsNotNone(out)
        # no body → no json= kwarg in the generated call
        self.assertIn("client.post('/api/v1/groups/g/dispose')", out["edit"]["content"])


class TestSpecifyWriteSinkWrapper(unittest.TestCase):
    """specify._synthesize_write_sink_red_test — verify wiring, no-clobber, kill-switch."""

    def test_wrapper_wires_verify(self):
        spec = specify._synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertIn("verify", spec)
        self.assertTrue(spec["verify"]["red_test_node"].endswith("_not_500"))
        self.assertEqual(spec["verify"]["test_edit_ids"], ["WRITE_SINK_RED"])
        self.assertTrue(any(e.get("id") == "WRITE_SINK_RED" for e in spec["edits"]))
        self.assertIn("write-sink oracle synthesised (lever L2)", spec["notes"])

    def test_wrapper_does_not_clobber_existing_red_test(self):
        spec = _write_sink_spec()
        spec["verify"] = {"red_test_node": "tests/test_author.py::test_x",
                          "test_edit_ids": ["AUTHORED"]}
        out = specify._synthesize_write_sink_red_test(
            spec, "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertEqual(out["verify"]["red_test_node"], "tests/test_author.py::test_x")
        self.assertFalse(any(e.get("id") == "WRITE_SINK_RED" for e in out["edits"]))

    def test_wrapper_overrides_decoy_pinning_test_on_divergence_loopback(self):
        # L1 looped back (RI_SAME_FACET_DIVERGENCE) because the author's own red test pins
        # the decoy it normalised away. L2 supersedes that node with its independent oracle
        # and lifts the loop-back so apply can certify by execution (NR0003 RC1 — why L2 exists).
        spec = _write_sink_spec()
        spec["termination"] = "needs_reinvestigation"
        spec["reinvestigation"] = {"reason_code": specify.RI_SAME_FACET_DIVERGENCE,
                                   "gate": "same_facet_consistency"}
        spec["verify"] = {"red_test_node": "server/tests/test_decoy.py::test_pins_insert_event",
                          "test_edit_ids": ["DECOY"]}
        out = specify._synthesize_write_sink_red_test(
            spec, "", "", setup_block=_HARNESS, request=_REQUEST)
        # the independent oracle is now authoritative
        self.assertTrue(out["verify"]["red_test_node"].endswith("_not_500"))
        self.assertIn("WRITE_SINK_RED", out["verify"]["test_edit_ids"])
        # the loop-back is lifted: apply can proceed to red→green
        self.assertEqual(out["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", out)
        self.assertIn("lifted the same-facet-divergence", out["notes"])

    def test_wrapper_does_not_override_unrelated_loopback(self):
        # A loop-back from a DIFFERENT gate is not L2's to lift; the existing node stands.
        spec = _write_sink_spec()
        spec["termination"] = "needs_reinvestigation"
        spec["reinvestigation"] = {"reason_code": "deferred_root_cause"}
        spec["verify"] = {"red_test_node": "tests/test_author.py::test_x",
                          "test_edit_ids": ["AUTHORED"]}
        out = specify._synthesize_write_sink_red_test(
            spec, "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertEqual(out["verify"]["red_test_node"], "tests/test_author.py::test_x")
        self.assertEqual(out["termination"], "needs_reinvestigation")
        self.assertFalse(any(e.get("id") == "WRITE_SINK_RED" for e in out["edits"]))

    def test_wrapper_failopen_without_harness(self):
        spec = specify._synthesize_write_sink_red_test(
            _write_sink_spec(), "", "", setup_block=None, request=_REQUEST)
        self.assertNotIn("verify", spec)
        self.assertFalse(any(e.get("id") == "WRITE_SINK_RED" for e in spec["edits"]))

    def test_wrapper_kill_switch_is_noop(self):
        with mock.patch.dict(os.environ, {specify._WRITE_SINK_ENV_OFF: "1"}):
            spec = specify._synthesize_write_sink_red_test(
                _write_sink_spec(), "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertNotIn("verify", spec)
        self.assertFalse(any(e.get("id") == "WRITE_SINK_RED" for e in spec["edits"]))


class TestL1L2PairClosesLoop(unittest.TestCase):
    """The L1+L2 PAIR on the real 0082 divergence, through the actual gate functions in
    run_specify order: L1 normalises the decoy + loops back on the author's decoy-pinning
    self-test, then L2 supersedes it with the independent oracle and lifts the loop-back."""

    def _divergent_spec_with_decoy_test(self):
        # dispose site = DECOY arg-swap (insert_event(doc_id)), close site = CANONICAL
        # callee-swap (insert_group_event(group_id)) — the exact NR0003 RC2 divergence.
        return {
            "edits": [
                {"id": "E1", "file": "server/modules/flow_gate/process_service.py",
                 "anchor_old": '        db.insert_event(group_id, "group_disposed", note=note)',
                 "replacement_new": '        db.insert_event(doc_id, "group_disposed", note=note)',
                 "confidence": "high", "anchor_status": "verified"},
                {"id": "E2", "file": "server/modules/flow_gate/process_service.py",
                 "anchor_old": "    db.insert_event(\n        group_id,\n        event_type,\n        note=note,\n    )",
                 "replacement_new": "    db.insert_group_event(\n        group_id,\n        event_type,\n        note=note,\n    )",
                 "confidence": "high", "anchor_status": "verified"},
                {"id": "E3", "kind": "create_file",
                 "file": "server/tests/test_dispose_event.py",
                 # the author's self-test PINS the decoy (asserts insert_event), not the canonical sink
                 "content": ('with patch("...process_service.db.insert_event") as m:\n'
                             '    assert m.call_args.args[0] == doc_id\n'),
                 "confidence": "high"},
            ],
            "termination": "ready_to_apply",
            "verify": {"red_test_node": "server/tests/test_dispose_event.py::test_x",
                       "test_edit_ids": ["E3"]},
            "notes": "",
        }

    def test_pair_normalises_then_certifies_independently(self):
        spec = self._divergent_spec_with_decoy_test()

        # L1: normalises the decoy E1 to the canonical callee-swap, and — because the author's
        # red test pins the decoy it removed — loops back rather than certify on it.
        spec = specify._apply_same_facet_consistency_gate(spec)
        e1 = next(e for e in spec["edits"] if e["id"] == "E1")
        self.assertIn("insert_group_event(group_id,", e1["replacement_new"])  # normalised
        self.assertEqual(spec["termination"], "needs_reinvestigation")
        self.assertEqual(spec["reinvestigation"]["reason_code"],
                         specify.RI_SAME_FACET_DIVERGENCE)

        # L2: supersedes the decoy-pinning node with the independent not-500 oracle and lifts
        # the loop-back, so apply can now certify the normalised routing by execution.
        spec = specify._synthesize_write_sink_red_test(
            spec, "", "", setup_block=_HARNESS, request=_REQUEST)
        self.assertEqual(spec["termination"], "ready_to_apply")
        self.assertNotIn("reinvestigation", spec)
        self.assertTrue(spec["verify"]["red_test_node"].endswith("_not_500"))
        self.assertIn("WRITE_SINK_RED", spec["verify"]["test_edit_ids"])
        # both source sites now route to the canonical sink, certified by an INDEPENDENT test
        for e in spec["edits"]:
            if e.get("id") in ("E1", "E2"):
                self.assertIn("insert_group_event", e["replacement_new"])


if __name__ == "__main__":
    unittest.main()
