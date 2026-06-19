"""Tests for the code-map BE-root axis (hive.be_root, Hook A / M028).

The queen name-matches a frontend file and misses the live backend gate one hop
away; be_root_axis traces FE symptom → live handler → service and injects a
CODEMAP_BE_ROOT front axis. These exercise the node-pick method with a fake
code-map (no model, no network), the mechanism the run/investigate paths wire in.
"""
import json

from hive.be_root import be_root_axis


class _FakeCM:
    """Minimal code-map: one FE file → one live endpoint → one handler → service."""

    def route_table(self, root):
        return [{"path": "/api/v1/widgets", "handler_file": "be/widgets.py",
                 "handler_func": "list_widgets", "live": True, "register_index": 0}]

    def endpoints_in_file(self, root, f):
        return ["/api/v1/widgets"]

    def handler_callees(self, root, hf, hfunc):
        return [{"module_file": "be/widget_service.py"}]

    def field_producers(self, root, fld):
        return []

    def find_symbol(self, root, tok):
        return []


_LEAVES = [{"id": "T1",
            "search_plan": {"file_globs": ["client/src/Widgets.vue"]}}]


def _pick_fn(prompt):
    return json.dumps({"file": "client/src/Widgets.vue",
                       "endpoint": "/api/v1/widgets", "field": ""})


def test_be_root_axis_off_without_call_fn(tmp_path):
    # Strict opt-in: no call_fn → true no-op (the broad trace was a regression).
    assert be_root_axis("symptom", _LEAVES, str(tmp_path), cm=_FakeCM()) is None


def test_be_root_axis_traces_backend_root(tmp_path):
    axis = be_root_axis("widgets list is empty", _LEAVES, str(tmp_path),
                        cm=_FakeCM(), call_fn=_pick_fn)
    assert axis is not None
    assert axis["id"] == "CODEMAP_BE_ROOT"
    globs = axis["search_plan"]["file_globs"]
    # Both the live handler and the service it reaches are surfaced as the root.
    assert "be/widgets.py" in globs
    assert "be/widget_service.py" in globs


def test_be_root_axis_noop_when_pick_grounds_nothing(tmp_path):
    # A pick that resolves to no backend path yields no axis (safe no-op).
    axis = be_root_axis("symptom", _LEAVES, str(tmp_path), cm=_FakeCM(),
                        call_fn=lambda p: json.dumps({"file": "x", "endpoint": "",
                                                      "field": ""}))
    assert axis is None


def test_be_root_axis_noop_without_codemap(tmp_path):
    # No code-map available (e.g. before the module lands) → no-op.
    assert be_root_axis("s", _LEAVES, str(tmp_path), cm=None, call_fn=_pick_fn) is None
