"""Coordinator (R0001, group 0003) — W0+W1 unit tests.

Covers each engine piece against its design invariants and the work-instruction
acceptance criteria (T0002 §5): L-01 anti-fabrication, L-02 code-axis guard,
L-03 strict gate + loose/hard-cap ready, L-06 best-effort termination + need_
eradication, P-03 header sync, and the non-interactive 1-shot wiring. All LLM
calls are mocked — no live workers.
"""
import io
import tokenize
from pathlib import Path
from unittest import mock

import pytest

from hive.providers import WorkerResult
from hive.coordinator import expected as L01
from hive.coordinator import decompose_fork as L02
from hive.coordinator import gate as L03
from hive.coordinator import finalize as L06
from hive.coordinator import engine
from hive.coordinator.model import Gap, GapState, MAX_SLOTS, MAX_AXES


def _wr(stdout: str) -> WorkerResult:
    return WorkerResult(stdout=stdout, stderr="", exit_code=0, latency_s=0.01)


def _patch_llm(stdout: str):
    """Patch the shared LLM harness so every structured_call returns `stdout`."""
    return mock.patch("hive.coordinator._llm.call_worker", return_value=_wr(stdout))


# ───────────────────────────── L-01 expected ─────────────────────────────────
def test_expected_blank_returns_empty_without_calling_llm():
    with mock.patch("hive.coordinator._llm.call_worker") as cw:
        assert L01.extract_expected("   ") == []
        cw.assert_not_called()                     # E1: no spend on blank input


def test_expected_rung1_states_target():
    out = ('{"axes":[{"axis_label":"highlight","observed_phrase":"진행중",'
           '"intent":{"states_target":true,"target_state":"현재(파랑)"}}]}')
    with _patch_llm(out):
        got = L01.extract_expected("메모가 진행중으로 떠 이상해")
    assert len(got) == 1
    assert got[0]["expected"] == "현재(파랑)"
    assert got[0]["refutable"] is True


def test_expected_rung3_negation_when_no_target_stated():
    out = ('{"axes":[{"axis_label":"highlight","observed_phrase":"진행중",'
           '"intent":{"states_target":false,"target_state":null}}]}')
    with _patch_llm(out):
        got = L01.extract_expected("메모가 진행중으로 떠 이상해")
    assert got and got[0]["expected"].startswith("not: ")   # always-refutable fallback


def test_expected_grounding_zero_is_dropped():
    # observed_phrase not present in the symptom → grounding 0 → below TAU_EMIT.
    out = ('{"axes":[{"axis_label":"x","observed_phrase":"foobar_hallucination",'
           '"intent":{"states_target":false,"target_state":null}}]}')
    with _patch_llm(out):
        assert L01.extract_expected("메모가 진행중으로 떠 이상해") == []


def test_expected_schema_invalid_returns_empty():
    with _patch_llm("this is not json at all"):
        assert L01.extract_expected("something is wrong") == []   # E6: never fabricate


def test_expected_capped_at_max_axes():
    axes = ",".join(
        '{"axis_label":"a%d","observed_phrase":"진행중%d",'
        '"intent":{"states_target":true,"target_state":"현재%d"}}' % (i, i, i)
        for i in range(MAX_AXES + 3))
    sym = " ".join("진행중%d" % i for i in range(MAX_AXES + 3))
    with _patch_llm('{"axes":[' + axes + ']}'):
        got = L01.extract_expected(sym)
    assert len(got) == MAX_AXES


# ───────────────────────────── L-02 decompose ────────────────────────────────
def test_decompose_blank_returns_empty():
    with mock.patch("hive.coordinator._llm.call_worker") as cw:
        assert L02.decompose("") == []
        cw.assert_not_called()


def test_decompose_keeps_grounded_context_slot():
    out = ('{"slots":[{"slot":"which module","kind":"context",'
           '"what_is_unsaid":"user did not say which module",'
           '"changes_outcome_score":0.8,"targets_symbol_or_file":false,'
           '"is_single_token_lexical":false,"proposes_investigation":false}]}')
    with _patch_llm(out):
        got = L02.decompose("the module selector is missing on the screen")
    assert len(got) == 1
    assert got[0]["expected_carveout"] is False
    assert got[0]["status"] == "open"


def test_decompose_drops_code_axis_directive():
    out = ('{"slots":[{"slot":"read function build_decompose_prompt","kind":"scope",'
           '"what_is_unsaid":"x","changes_outcome_score":0.9,'
           '"targets_symbol_or_file":true,"is_single_token_lexical":false,'
           '"proposes_investigation":false}]}')
    with _patch_llm(out):
        assert L02.decompose("the module selector is missing") == []  # CON-2 guard


def test_decompose_drops_ungrounded_hallucination():
    out = ('{"slots":[{"slot":"zzz qqq","kind":"context","what_is_unsaid":"",'
           '"changes_outcome_score":0.9,"targets_symbol_or_file":false,'
           '"is_single_token_lexical":false,"proposes_investigation":false}]}')
    with _patch_llm(out):
        assert L02.decompose("the module selector is missing") == []  # E3


def test_decompose_expected_kind_sets_carveout():
    out = ('{"slots":[{"slot":"expected module behaviour","kind":"expected",'
           '"what_is_unsaid":"x","changes_outcome_score":0.8,'
           '"targets_symbol_or_file":false,"is_single_token_lexical":false,'
           '"proposes_investigation":false}]}')
    with _patch_llm(out):
        got = L02.decompose("the module selector behaviour is wrong")
    assert got and got[0]["expected_carveout"] is True


def test_decompose_truncates_non_silently():
    slots = ",".join(
        '{"slot":"module aspect %d","kind":"context",'
        '"what_is_unsaid":"x","changes_outcome_score":0.9,'
        '"targets_symbol_or_file":false,"is_single_token_lexical":false,'
        '"proposes_investigation":false}' % i for i in range(MAX_SLOTS + 2))
    msg = "module " + " ".join("aspect" for _ in range(MAX_SLOTS + 2))
    with _patch_llm('{"slots":[' + slots + ']}'):
        got = L02.decompose(msg)
    assert len(got) == MAX_SLOTS
    assert got[-1]["provenance"].get("truncated") == 2   # no silent drop (E7)


# ───────────────────────────── L-03 gate / ready ─────────────────────────────
def _cand(slot, hint, carveout=False):
    return {"id": slot, "slot": slot, "kind": "context",
            "expected_carveout": carveout, "load_bearing_hint": hint,
            "salience": 0.9, "provenance": {}}


def test_gate_expected_carveout_promoted_unconditionally():
    gaps = L03.gate_load_bearing([_cand("exp", 0.0, carveout=True)], {})
    assert len(gaps) == 1 and gaps[0].expected_carveout is True   # FR-4


def test_gate_threshold_keeps_high_drops_low():
    gaps = L03.gate_load_bearing(
        [_cand("hi", 0.9), _cand("lo", 0.1)], {})
    slots = {g.slot for g in gaps}
    assert slots == {"hi"}                                        # FR-3 minimal-query


def test_gate_already_known_slot_dropped():
    draft = {"caller_supplied_context": "the module is users"}
    gaps = L03.gate_load_bearing([_cand("module", 0.9)], draft)
    assert gaps == []


def test_ready_hardcap_seals_open_gaps():
    gs = GapState(uuid="u", symptom_raw="x")
    gs.gaps = [Gap(id="a", slot="a", status="open")]
    gs.caps["rounds_left"] = 0
    assert L03.ready(gs) == "sealed"
    assert gs.gaps[0].status == "skipped"
    assert gs.gaps[0].provenance.get("reason") == "cap_reached"


def test_ready_loose_pass_when_all_resolved():
    gs = GapState(uuid="u", symptom_raw="x")
    gs.gaps = [Gap(id="a", slot="a", status="answered", answer="yes")]
    gs.caps["rounds_left"] = 4
    assert L03.ready(gs) == "ready"


def test_ready_empty_gaps_immediate_ready():
    gs = GapState(uuid="u", symptom_raw="x")
    gs.caps["rounds_left"] = 4
    assert L03.ready(gs) == "ready"                               # AC-3 no-op pass


def test_ready_collecting_when_open_and_no_cap():
    gs = GapState(uuid="u", symptom_raw="x")
    gs.gaps = [Gap(id="a", slot="a", status="open")]
    gs.caps["rounds_left"] = 4
    assert L03.ready(gs) == "collecting"


# ───────────────────────────── L-06 finalize ─────────────────────────────────
def test_finalize_almost_empty_still_hands_off():
    gs = GapState(uuid="u", symptom_raw="메모가 이상해", status="sealed")
    seed = L06.finalize(gs)
    assert L06.CALLER_CONTEXT_SECTION in seed
    assert "symptom: 메모가 이상해" in seed
    assert "expected:" not in seed                               # E2 omit, not blank


def test_finalize_expected_present():
    gs = GapState(uuid="u", symptom_raw="x", status="sealed",
                  expected=[{"expected": "현재(파랑)"}])
    seed = L06.finalize(gs)
    assert "- expected: 현재(파랑)" in seed


def test_finalize_skipped_to_provenance_answered_to_context():
    gs = GapState(uuid="u", symptom_raw="x", status="sealed")
    gs.gaps = [
        Gap(id="a", slot="which_module", status="answered", answer="users"),
        Gap(id="b", slot="which_screen", status="skipped"),
    ]
    seed = L06.finalize(gs)
    assert "which_module: users" in seed                         # answered → context
    assert "skipped_slots: [which_screen]" in seed               # skipped → provenance
    assert "which_screen: " not in seed


def test_finalize_rejects_banned_status_vocab():
    gs = GapState(uuid="u", symptom_raw="x", status="sealed")
    gs.gaps = [Gap(id="a", slot="a", status="needs_answer")]     # banned vocab
    with pytest.raises(L06.CoordinatorVocabError):
        L06.finalize(gs)


def test_assert_single_terminal_is_handoff():
    assert L06.assert_single_terminal() == "HANDOFF"


def test_header_stays_in_sync_with_pipeline():
    from hive.investigate import CALLER_CONTEXT_SECTION as PIPE_HEADER
    assert L06.CALLER_CONTEXT_SECTION == PIPE_HEADER             # P-03 seam sync guard


# ───────────────────────────── engine (1-shot wiring) ────────────────────────
def _fake_dispatch(provider, model, prompt, **kw):
    if "USER SYMPTOM" in prompt:        # L-01 expected
        return _wr('{"axes":[{"axis_label":"h","observed_phrase":"진행중",'
                   '"intent":{"states_target":true,"target_state":"현재"}}]}')
    if "USER MESSAGE" in prompt:        # L-02 decompose
        return _wr('{"slots":[{"slot":"which module","kind":"context",'
                   '"what_is_unsaid":"x","changes_outcome_score":0.3,'
                   '"targets_symbol_or_file":false,"is_single_token_lexical":false,'
                   '"proposes_investigation":false}]}')
    return _wr("{}")


def test_run_coordinator_noninteractive_seals_and_enriches():
    seed = "메모가 진행중으로 떠 이상해. module 선택이 안돼."
    with mock.patch("hive.coordinator._llm.call_worker", side_effect=_fake_dispatch):
        res = engine.run_coordinator(seed, model="claude-sonnet-4.5")
    assert res["status"] == "sealed"                             # W1 best-effort seal
    assert L06.CALLER_CONTEXT_SECTION in res["enriched_seed"]
    assert seed in res["enriched_seed"]                          # original preserved
    assert res["expected"] and res["expected"][0]["expected"] == "현재"


def test_run_coordinator_blank_seed_still_hands_off():
    with mock.patch("hive.coordinator._llm.call_worker") as cw:
        res = engine.run_coordinator("")
        cw.assert_not_called()
    assert isinstance(res["enriched_seed"], str)                 # AC-C: never blocks
    assert res["status"] == "sealed"


# ───────────────────────────── need_ ban lint (CI guard) ─────────────────────
def test_no_banned_vocab_in_coordinator_source():
    """L-06 §2.3: no need_/needs_ prefix or awaiting/blocked/punt/abstain
    work-around may appear as an IDENTIFIER anywhere in the coordinator package
    (strings/comments are excluded — the policy is described there in prose)."""
    pkg = Path(engine.__file__).parent
    banned_prefix = ("need_", "needs_")
    banned_semantic = ("awaiting_", "blocked_", "punt", "abstain")
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.NAME:
                continue
            name = tok.string.lower()
            if name.startswith(banned_prefix) or any(p in name for p in banned_semantic):
                offenders.append(f"{path.name}:{tok.start[0]} {tok.string}")
    assert not offenders, "banned vocab identifiers: " + "; ".join(offenders)
