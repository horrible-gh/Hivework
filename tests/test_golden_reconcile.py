"""Tests for hive.py `_reconcile_golden_scores` — orphaned golden-score splicing.

T0011 follow-up ("왜 채점은 하지 않는가"): golden scoring is a post-run, hand-
authored verdict written to ``golden*.scored.json`` in the run's workdir. The
reconcile hook carries that orphaned file into runs.jsonl (last-wins) so the
report shows the real recall, not 미계측 — without ever fabricating a score for a
run that has none.
"""
import importlib.util
import json
import logging
import os

_HIVE_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hive.py")


def _load_hive():
    spec = importlib.util.spec_from_file_location("_hive_script_under_test", _HIVE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_repo(tmp_path, runs, scored_by_dir):
    """Lay out a fake repo: perf/metrics/runs.jsonl + smoke/<dir>/golden*.scored.json."""
    metrics = tmp_path / "perf" / "metrics"
    metrics.mkdir(parents=True)
    with open(metrics / "runs.jsonl", "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for seed_dir, golden in scored_by_dir.items():
        d = tmp_path / seed_dir
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "golden_0082.scored.json", "w", encoding="utf-8") as fh:
            json.dump(golden, fh, ensure_ascii=False)
    return str(tmp_path), str(metrics / "runs.jsonl")


def _read_latest(jsonl_path):
    latest = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                latest[rec["run_id"]] = rec
    return latest


SCORE = {"seeded": 1, "recalled": 1, "false_positives": 0, "verified_fixed": 0,
         "per_bug": [{"id": "0082", "level": 3, "found": True, "fixed": False}]}


def test_splices_orphaned_score_for_unscored_run(tmp_path):
    """A run whose seed-dir holds a scored golden but whose jsonl line lacks a
    golden block gets a new scored line appended (last-wins supersedes the stub)."""
    hive = _load_hive()
    repo, jsonl = _make_repo(
        tmp_path,
        runs=[{"run_id": "run511", "ts": "2026-06-21T13:51:09+00:00",
               "seed": "smoke/0039_tsr0009/seed.md"}],
        scored_by_dir={"smoke/0039_tsr0009": SCORE})
    n = hive._reconcile_golden_scores(logging.getLogger("t"), repo=repo)
    assert n == 1
    latest = _read_latest(jsonl)
    assert latest["run511"]["golden"]["recalled"] == 1
    assert latest["run511"]["golden"]["per_bug"][0]["found"] is True


def test_idempotent_no_double_splice(tmp_path):
    """Re-running never appends a second scored line for an already-scored run."""
    hive = _load_hive()
    repo, jsonl = _make_repo(
        tmp_path,
        runs=[{"run_id": "run511", "ts": "2026-06-21T13:51:09+00:00",
               "seed": "smoke/0039_tsr0009/seed.md"}],
        scored_by_dir={"smoke/0039_tsr0009": SCORE})
    assert hive._reconcile_golden_scores(logging.getLogger("t"), repo=repo) == 1
    # second pass: the run now carries golden → skipped
    assert hive._reconcile_golden_scores(logging.getLogger("t"), repo=repo) == 0
    lines = [l for l in open(jsonl, encoding="utf-8").read().splitlines() if l.strip()]
    assert sum(1 for l in lines if '"golden"' in l) == 1


def test_unscored_run_stays_unscored(tmp_path):
    """No scored file in the seed-dir → no golden block fabricated (미계측 ≠ 0%)."""
    hive = _load_hive()
    repo, jsonl = _make_repo(
        tmp_path,
        runs=[{"run_id": "run999", "ts": "2026-06-21T00:00:00+00:00",
               "seed": "smoke/9999_none/seed.md"}],
        scored_by_dir={})  # no scored file anywhere
    # create the seed dir but with no scored json
    (tmp_path / "smoke" / "9999_none").mkdir(parents=True)
    assert hive._reconcile_golden_scores(logging.getLogger("t"), repo=repo) == 0
    assert "golden" not in _read_latest(jsonl)["run999"]


def test_already_scored_run_is_left_alone(tmp_path):
    """A run that already has a golden block is never re-touched even if a scored
    file is present (the existing verdict wins)."""
    hive = _load_hive()
    pre = {"run_id": "run511", "ts": "2026-06-21T13:51:09+00:00",
           "seed": "smoke/0039_tsr0009/seed.md",
           "golden": {"seeded": 1, "recalled": 0, "false_positives": 0,
                      "verified_fixed": 0,
                      "per_bug": [{"id": "0082", "level": 3, "found": False, "fixed": False}]}}
    repo, jsonl = _make_repo(tmp_path, runs=[pre],
                             scored_by_dir={"smoke/0039_tsr0009": SCORE})
    assert hive._reconcile_golden_scores(logging.getLogger("t"), repo=repo) == 0
    assert _read_latest(jsonl)["run511"]["golden"]["recalled"] == 0
