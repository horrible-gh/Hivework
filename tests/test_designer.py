"""Tests for the designer (설계자) stage (group 0079, logic SSOT 0079.0005-L).

The author model call is injected (``author_fn``) so these tests exercise the pure
decision logic — grounded-gate, precheck-against-the-keymaster, re-author-on-decline,
demote-on-partial, strict no_go, recipe parsing — against a real tiny codebase the
keymaster grounds oracles on. No model, no network."""

from hive import designer


# ── a tiny live codebase the keymaster can ground an http_read oracle against ──
def _codebase(tmp_path):
    root = tmp_path / "code"
    (root / "app").mkdir(parents=True)
    (root / "app" / "routes.py").write_text(
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/api/v1")\n\n'
        '@router.get("/projects")\n'
        "def list_projects():\n"
        '    return {"modules": _rows()}\n', encoding="utf-8")
    return str(root)


# A grounded oracle criterion that resolves against _codebase (route + field both real).
_GOOD_DESIGN = (
    "# Design\n\n"
    "recipe: code_bug\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: GET /api/v1/projects returns a non-empty modules list\n"
    "  oracle:\n"
    "    kind: http_read\n"
    "    verb: get\n"
    "    route: /api/v1/projects\n"
    "    json_path: modules\n"
    "    must: non_empty\n")

# An oracle naming a route that does NOT exist in the codebase → keymaster declines.
_BAD_DESIGN = (
    "# Design\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    "  prose: GET /api/v1/ghost returns something\n"
    "  oracle:\n"
    "    kind: http_read\n"
    "    verb: get\n"
    "    route: /api/v1/ghost\n"
    "    json_path: nope\n"
    "    must: exists\n")


def _verdicts(located=True):
    return [{"axis_id": "A1", "title": "t",
             "verdict": {"located": located, "file": "app/routes.py",
                         "lines": "3-5", "reason": "route here"}}]


# ── grounded gate (L §2.1) ─────────────────────────────────────────────────────
def test_no_grounded_verdict_needs_reinvestigation(tmp_path):
    root = _codebase(tmp_path)
    out = designer.run_designer(
        "seed", _verdicts(located=False), "", root, role=None,
        author_fn=lambda p: _GOOD_DESIGN)
    assert out["decision"] == "needs_reinvestigation"
    assert out["reason"] == "no_grounded_verdict"
    assert out["attempts"] == 0


def test_empty_verdicts_needs_reinvestigation(tmp_path):
    root = _codebase(tmp_path)
    out = designer.run_designer("seed", [], "", root, role=None,
                                author_fn=lambda p: _GOOD_DESIGN)
    assert out["decision"] == "needs_reinvestigation"


# ── proceed (L §2.1 happy path) ────────────────────────────────────────────────
def test_grounded_resolving_criterion_proceeds(tmp_path):
    root = _codebase(tmp_path)
    out = designer.run_designer("seed", _verdicts(), "honey", root, role=None,
                                author_fn=lambda p: _GOOD_DESIGN)
    assert out["decision"] == "proceed"
    assert out["valid_ids"] == ["AC1"]
    assert out["recipe"] == "code_bug"
    assert out["attempts"] == 1
    assert "## 수용기준" in out["design_text"]


def test_prompt_carries_grounding_evidence(tmp_path):
    root = _codebase(tmp_path)
    seen = {}

    def _author(prompt):
        seen["p"] = prompt
        return _GOOD_DESIGN

    designer.run_designer("seed text", _verdicts(), "honeytext", root, role=None,
                          author_fn=_author)
    assert "app/routes.py:3-5" in seen["p"]      # grounded location injected
    assert "seed text" in seen["p"]
    assert "honeytext" in seen["p"]


# ── no_go: all criteria decline, strict stop (L §4 invariant) ──────────────────
def test_all_declined_no_go_after_retries(tmp_path):
    root = _codebase(tmp_path)
    calls = []

    def _author(prompt):
        calls.append(prompt)
        return _BAD_DESIGN

    out = designer.run_designer("seed", _verdicts(), "", root, role=None,
                                author_fn=_author)
    assert out["decision"] == "no_go"
    assert out["reason"] == "all_criteria_declined"
    assert out["attempts"] == designer.MAX_AUTHORING_ATTEMPTS
    assert len(calls) == designer.MAX_AUTHORING_ATTEMPTS      # re-authored once
    # the second prompt carries the decline feedback so the author can fix it
    assert "DECLINED" in calls[1]


# ── re-author recovers on attempt 2 ────────────────────────────────────────────
def test_reauthor_recovers_on_second_attempt(tmp_path):
    root = _codebase(tmp_path)
    seq = iter([_BAD_DESIGN, _GOOD_DESIGN])
    out = designer.run_designer("seed", _verdicts(), "", root, role=None,
                                author_fn=lambda p: next(seq))
    assert out["decision"] == "proceed"
    assert out["attempts"] == 2
    assert out["valid_ids"] == ["AC1"]


# ── partial decline → proceed with demotion (L §2.1 demote_declined) ───────────
def test_partial_decline_demotes_and_proceeds(tmp_path):
    root = _codebase(tmp_path)
    mixed = (
        "# Design\n\n"
        "## 수용기준\n"
        "- id: AC1\n"
        "  prose: good one\n"
        "  oracle: {kind: http_read, verb: get, route: /api/v1/projects, "
        "json_path: modules, must: non_empty}\n"
        "- id: AC2\n"
        "  prose: ghost one\n"
        "  oracle: {kind: http_read, verb: get, route: /api/v1/ghost, "
        "json_path: nope, must: exists}\n")
    out = designer.run_designer("seed", _verdicts(), "", root, role=None,
                                author_fn=lambda p: mixed)
    assert out["decision"] == "proceed"
    assert out["valid_ids"] == ["AC1"]
    assert [d["id"] for d in out["declined"]] == ["AC2"]
    # AC2 is demoted out of the keymaster input but preserved as reference prose
    assert "강등된 수용기준" in out["design_text"]
    assert "ghost one" in out["design_text"]
    # re-reading the demoted design yields only the valid criterion for the keymaster
    from hive.acceptance_synth import read_acceptance_criteria
    ids = [c["id"] for c in read_acceptance_criteria(out["design_text"])]
    assert ids == ["AC1"]


# ── model failure → no_go (never raises) ───────────────────────────────────────
def test_author_exception_is_no_go(tmp_path):
    root = _codebase(tmp_path)

    def _boom(prompt):
        raise RuntimeError("provider down")

    out = designer.run_designer("seed", _verdicts(), "", root, role=None,
                                author_fn=_boom)
    assert out["decision"] == "no_go"
    assert out["reason"] == "model_error"
    assert out["attempts"] == 1


# ── max_criteria cap (L §5 no silent truncation) ──────────────────────────────
def test_criteria_cap_truncates_and_flags(tmp_path):
    root = _codebase(tmp_path)
    items = "\n".join(
        f"- id: AC{i}\n"
        f"  prose: p{i}\n"
        f"  oracle: {{kind: http_read, verb: get, route: /api/v1/projects, "
        f"json_path: modules, must: non_empty}}"
        for i in range(designer.MAX_CRITERIA + 3))
    design = "## 수용기준\n" + items + "\n"
    out = designer.run_designer("seed", _verdicts(), "", root, role=None,
                                author_fn=lambda p: design, max_criteria=designer.MAX_CRITERIA)
    assert out["decision"] == "proceed"
    assert out["truncated"] is True
    assert len(out["valid_ids"]) == designer.MAX_CRITERIA


# ── recipe parsing (L §2.4) ────────────────────────────────────────────────────
def test_parse_recipe():
    assert designer.parse_recipe("blah\nrecipe: code_feature\nmore") == "code_feature"
    assert designer.parse_recipe("recipe = `code_bug`") == "code_bug"
    assert designer.parse_recipe("no judgement here") is None
