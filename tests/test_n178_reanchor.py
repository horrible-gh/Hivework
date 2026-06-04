"""N178 — whitespace-drift re-anchor (specify) + import↔usage all-or-nothing (apply).

Self-contained file (does not touch the shared test_specify/test_apply modules) so it
can land alongside concurrent work on those files.

Two fixes are covered:
  1. specify._reanchor_drifted — an anchor that drifted by insignificant whitespace only
     is re-synced to the exact live bytes (instead of dead-ending in needs_reinvestigation
     that the reactive bridge can only terminate).
  2. apply.build_proposal — on a partial split, an import edit whose paired usage edit is
     held is held too, so the import and its use ship all-or-nothing (no dangling import).
"""
import os

from hive import specify
from hive.apply import build_proposal


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return p


# ── Fix 1: whitespace-drift re-anchor ──────────────────────────────────────────

# Live uses 4-space indents; the author transcribed the anchor with 2-space indents.
_LIVE_VUE = (
    "function submit() {\n"
    "    try {\n"
    "        await save()\n"
    "    } catch (e) {\n"
    "        showToast(e.message)\n"
    "    }\n"
    "}\n"
)


def _drifted_spec():
    return {
        "termination": "ready_to_apply",
        "edits": [{
            "id": "E7",
            "file": "src/NewRequirementModal.vue",
            # 2-space indent, NOT byte-equal to the 4-space live text → drift
            "anchor_old": "  } catch (e) {\n"
                          "    showToast(e.message)\n"
                          "  }\n",
            "replacement_new": "  } catch (e) {\n"
                               "    showToast({ message: e.message, type: 'error' })\n"
                               "  }\n",
            "anchor_status": "verified",
            "confidence": "high",
        }],
    }


def test_reanchor_recovers_whitespace_drift(tmp_path):
    root = str(tmp_path)
    _write(root, "src/NewRequirementModal.vue", _LIVE_VUE)
    spec = _drifted_spec()

    # verify marks it not_found (the 2-space anchor is absent from the 4-space live file)
    specify._verify_anchors_live(spec, root)
    assert spec["edits"][0]["anchor_status"] == "not_found"

    # re-anchor recovers it: exact live bytes re-lifted, status back to verified
    specify._reanchor_drifted(spec, root)
    e = spec["edits"][0]
    assert e["anchor_status"] == "verified"
    assert "reanchored" in e
    # the new anchor is the exact live block (4-space indents), present once in live
    assert e["anchor_old"] == (
        "    } catch (e) {\n"
        "        showToast(e.message)\n"
        "    }\n")
    assert _LIVE_VUE.count(e["anchor_old"]) == 1
    # the change is preserved; context lines now carry live's 4/8-space indentation
    assert "showToast({ message: e.message, type: 'error' })" in e["replacement_new"]
    assert e["replacement_new"].startswith("    } catch (e) {\n")
    # and normalize then does NOT downgrade it
    specify._normalize_spec(spec)
    assert spec["termination"] == "ready_to_apply"


def test_reanchor_refuses_when_ambiguous(tmp_path):
    # The same block appears twice → >1 normalized match → NOT re-anchored (honest NR).
    root = str(tmp_path)
    _write(root, "src/Dup.vue", _LIVE_VUE + "\n" + _LIVE_VUE)
    spec = _drifted_spec()
    spec["edits"][0]["file"] = "src/Dup.vue"
    specify._verify_anchors_live(spec, root)
    specify._reanchor_drifted(spec, root)
    assert spec["edits"][0]["anchor_status"] == "not_found"
    assert "reanchored" not in spec["edits"][0]


def test_reanchor_refuses_when_gone(tmp_path):
    # Nothing in live resembles the anchor → stays not_found (no fabrication).
    root = str(tmp_path)
    _write(root, "src/Gone.vue", "function noop() {\n  return 1\n}\n")
    spec = _drifted_spec()
    spec["edits"][0]["file"] = "src/Gone.vue"
    specify._verify_anchors_live(spec, root)
    specify._reanchor_drifted(spec, root)
    assert spec["edits"][0]["anchor_status"] == "not_found"
    assert "reanchored" not in spec["edits"][0]


def test_reanchor_skips_genuine_stale_ambiguity(tmp_path):
    # An anchor present >1x is 'stale' (not 'not_found') — re-anchor leaves it for
    # _disambiguate_anchors, never fuzzy-relifts an ambiguous one.
    root = str(tmp_path)
    _write(root, "src/Stale.vue", "foo()\nfoo()\n")
    spec = {
        "termination": "ready_to_apply",
        "edits": [{"id": "E1", "file": "src/Stale.vue",
                   "anchor_old": "foo()", "replacement_new": "bar()",
                   "anchor_status": "verified"}],
    }
    specify._verify_anchors_live(spec, root)
    assert spec["edits"][0]["anchor_status"] == "stale"
    specify._reanchor_drifted(spec, root)
    assert spec["edits"][0]["anchor_status"] == "stale"  # untouched
    assert "reanchored" not in spec["edits"][0]


# ── Fix 2: import↔usage all-or-nothing on a partial split ───────────────────────

_LIVE_SCRIPT = (
    "<script setup>\n"
    "import { ref } from 'vue'\n"
    "\n"
    "function submit() {\n"
    "  doStuff()\n"
    "}\n"
    "</script>\n"
)


def test_import_edit_held_when_paired_usage_held(tmp_path):
    root = str(tmp_path)
    _write(root, "src/Foo.vue", _LIVE_SCRIPT)
    spec = {
        "termination": "ready_to_apply",
        "edits": [
            {  # E6 — adds the import (applicable on its own)
                "id": "E6", "file": "src/Foo.vue",
                "anchor_old": "import { ref } from 'vue'",
                "replacement_new": "import { ref } from 'vue'\n"
                                   "import { useToast } from './toast'",
            },
            {  # E7 — the paired USAGE, anchor drifted/missing → not applicable, held
                "id": "E7", "file": "src/Foo.vue",
                "anchor_old": "  doStuffMISSING()",
                "replacement_new": "  const { showToast } = useToast()\n  doStuff()",
            },
        ],
    }
    prop = build_proposal(spec, root)
    by_id = {str(r["id"]): r for r in prop["edits"]}
    assert by_id["E7"]["applicable"] is False
    # the import edit is held too — its binding is unused without E7
    assert by_id["E6"]["writable"] is False
    assert "import edit held" in by_id["E6"].get("held_reason", "")
    assert prop["writable_ids"] == []


def test_import_edit_not_held_when_usage_writable(tmp_path):
    root = str(tmp_path)
    _write(root, "src/Bar.vue", _LIVE_SCRIPT)
    spec = {
        "termination": "ready_to_apply",
        "edits": [
            {"id": "E6", "file": "src/Bar.vue",
             "anchor_old": "import { ref } from 'vue'",
             "replacement_new": "import { ref } from 'vue'\n"
                                "import { useToast } from './toast'"},
            {"id": "E7", "file": "src/Bar.vue",
             "anchor_old": "  doStuff()",
             "replacement_new": "  const { showToast } = useToast()\n  showToast('hi')"},
        ],
    }
    prop = build_proposal(spec, root)
    by_id = {str(r["id"]): r for r in prop["edits"]}
    # both apply, binding is used → nothing held, fully ready
    assert by_id["E6"]["writable"] is True
    assert by_id["E7"]["writable"] is True
    assert prop["ready"] is True
