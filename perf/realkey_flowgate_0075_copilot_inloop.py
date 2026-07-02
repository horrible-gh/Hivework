"""Real-key trial (group 0075) — 조각B: REAL copilot-in-loop authoring on the REAL FlowGate.

R0001 (0075) "실 과금이던 뭐던 남김없이 다 하라": 0073/0074 cut the first real keys but the
AUTHORING stage was always hand-written (``_spec_with_fix`` in those harnesses) — box-0 only
ever ran zero-model. NR0003 (0075) named the sole remaining piece 조각B = flip that no-model
authoring to a REAL copilot in-the-loop call (real billing). This harness does exactly that.

It reuses 0073's proven, non-destructive path (real workshop tree, real pytest, master-snapshot
restore) but REPLACES the hand-authored spec with ``specify.run_specify`` on provider=copilot:

    honey (feature: widen TITLE_MAX_LEN) + design (## 수용기준 unit_value 120)
      → hive.specify.run_specify(provider="copilot", ...)      # REAL copilot authors the fix
          → ground_anchors lifts live TITLE_MAX_LEN=100         # local, free
          → copilot writes edit {TITLE_MAX_LEN 100 -> 120}      # BILLED author call
          → _synthesize_acceptance_red_test attaches box-0 RED  # zero-model
      → verify.verify_red_green(spec, root, runner)             # REAL pytest red->green, restore

The proof (GO) = copilot (not the harness) authored the widening edit AND box-0's gate certified
it red->green on the real tree AND the tree was restored byte-identical. Cost is read back from
the ledger + the HIVE_CALL_LOG credit footer.

Run:  set HIVE_CALL_LOG=<path> ; PYTHONPATH=<Hivework> PYTHONIOENCODING=utf-8 \
      python perf/realkey_flowgate_0075_copilot_inloop.py
Exits non-zero on any miss (author produced no widening edit / not grounded / not red->green /
not restored). Engine (hive/*) unchanged; workshop non-destructive.
"""
import json
import os
import re
import sys
import tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from hive.config import load_config  # noqa: E402
from hive import specify  # noqa: E402
from hive import verify as verifymod  # noqa: E402
import importlib.util as _ilu  # noqa: E402


def _load_hive_cli():
    path = os.path.join(_REPO, "hive.py")
    spec = _ilu.spec_from_file_location("hive_cli_entry_realkey_0075", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FLOWGATE = os.environ.get(
    "REALKEY_FLOWGATE_ROOT",
    r"C:\workspace\projects\Hivework-test\FlowGate\server",
)
_TARGET_REL = "modules/flow_gate/linter.py"
_TARGET = f"{_TARGET_REL}::TITLE_MAX_LEN"
_EXPECTED = 120  # the widened cap the copilot author must land

# The acceptance design (box-0 seed). read_acceptance_criteria needs BOTH id and prose.
_CRITERIA = (
    "# Feature design — raise the document title cap\n\n"
    "## 수용기준\n"
    "- id: AC1\n"
    f"  prose: {_TARGET} must equal {_EXPECTED} (the widened title length cap)\n"
    "  oracle:\n"
    "    kind: unit_value\n"
    f"    target: {_TARGET}\n"
    "    must: equals\n"
    f"    expected: {_EXPECTED}\n"
)

# The honey the copilot author lowers into an edit-spec. It names the exact symbol/file so the
# local anchor-grounding pre-flight lifts the CURRENT value (100) and the author writes a real
# 100->120 change. This is a feature honey (not a bug), authored by hand as the design input —
# the AUTHORING (honey -> edit) is what copilot does, and that is the billed 조각B step.
_HONEY = (
    "# Feature: widen the document title length cap\n\n"
    "## Goal\n"
    f"The document title length cap `TITLE_MAX_LEN` in `{_TARGET_REL}` is currently 100.\n"
    f"Widen it to {_EXPECTED} so longer document titles are accepted.\n\n"
    "## Edit target\n"
    f"- File: `{_TARGET_REL}`\n"
    f"- Symbol: `TITLE_MAX_LEN` (a module-level constant, currently `= 100`)\n"
    f"- Required change: set `TITLE_MAX_LEN = {_EXPECTED}`\n"
)


def _write_config(root: str) -> str:
    """0073's config: bind the real FlowGate root to a pytest runner + acceptance doc."""
    targets = {
        "flowgate": {
            "tests": {
                "command": ["python", "-m", "pytest", "-q"],
                "codebase": root,
                "timeout_sec": 300,
                "env": {"PYTHONDONTWRITEBYTECODE": "1"},  # same-byte-width flip .pyc staleness
            },
            "acceptance": {
                "codebase": root,
                "criteria_text": _CRITERIA,
                "test_dir": "tests",
            },
        }
    }
    fd, cfg_path = tempfile.mkstemp(prefix="realkey_0075_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"targets": targets}, fh)
    return cfg_path


def _purge_pyc(tgt_abs: str) -> None:
    pyc_dir = os.path.join(os.path.dirname(tgt_abs), "__pycache__")
    stem = os.path.splitext(os.path.basename(tgt_abs))[0]
    if os.path.isdir(pyc_dir):
        for f in os.listdir(pyc_dir):
            if f.startswith(stem + ".") and f.endswith(".pyc"):
                try:
                    os.remove(os.path.join(pyc_dir, f))
                except OSError:
                    pass


def _read_credits(call_log: str | None) -> float | None:
    """Sum the 'AI Credits N.NN' lines the copilot CLI writes to stderr (captured by
    providers._tee_call into HIVE_CALL_LOG). None when no log / no credit line."""
    if not call_log or not os.path.exists(call_log):
        return None
    total = 0.0
    seen = False
    with open(call_log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"AI Credits\s+([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                total += float(m.group(1))
                seen = True
    return total if seen else None


def main() -> int:
    cli = _load_hive_cli()
    root = _FLOWGATE
    out: dict = {"codebase_root": root, "target": _TARGET, "piece": "B (copilot-in-loop)"}

    if not os.path.isdir(root):
        out["error"] = f"workshop FlowGate root not found: {root}"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 4

    call_log = os.environ.get("HIVE_CALL_LOG")
    tgt_abs = os.path.join(root, _TARGET_REL)
    before_bytes = open(tgt_abs, "rb").read()
    _purge_pyc(tgt_abs)

    cfg = load_config(path=_write_config(root))
    kw = cli.acceptance_specify_kwargs(cfg, root)
    provider_kwargs = cli.build_provider_kwargs(cfg)
    role = cfg.role("specify")
    review_role = cfg.role("review")
    out["author"] = {"provider": role.provider, "model": role.model}

    # honey + design -> scratch files for run_specify.
    scratch = tempfile.mkdtemp(prefix="realkey_0075_")
    honey_path = os.path.join(scratch, "feature.honey.md")
    with open(honey_path, "w", encoding="utf-8") as fh:
        fh.write(_HONEY)
    spec_out = os.path.join(scratch, "feature.edit_spec.json")

    ldg = cli.open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=honey_path, codebase=root,
                  model_queen=role.model, model_fanout=role.model)
    author_err = None
    spec = None
    try:
        # THE billed 조각B call: copilot authors the fix (+ box-0 red test is attached inside).
        spec = specify.run_specify(
            honey_path=honey_path,
            codebase_root=root,
            output_path=spec_out,
            model=role.model,
            provider=role.provider,
            review=True,
            review_model=review_role.model,
            review_provider=review_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
            author_retries=role.retries,
            acceptance_criteria_text=kw.get("acceptance_criteria_text"),
            acceptance_setup_block=kw.get("acceptance_setup_block"),
            acceptance_app_fixture=kw.get("acceptance_app_fixture"),
            acceptance_test_dir=kw.get("acceptance_test_dir", "tests"),
        )
        ldg.finish_run(honey_path=spec_out,
                       axes_n=len(spec.get("edits") or []), status="done")
    except Exception as e:  # copilot down / no auth / parse fail — report, don't crash silently
        author_err = f"{type(e).__name__}: {e}"
        ldg.finish_run(status="failed")
    finally:
        ldg.close()

    if spec is None:
        out["author_error"] = author_err
        out["credits"] = _read_credits(call_log)
        out["GO"] = False
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    edits = spec.get("edits") or []
    out["authored_edits"] = edits
    out["termination"] = spec.get("termination")
    # Did copilot author the widening edit on the real symbol?
    widening = next(
        (e for e in edits
         if _TARGET_REL.replace("\\", "/") in (e.get("file", "").replace("\\", "/"))
         and str(_EXPECTED) in str(e.get("replacement_new", ""))
         and "TITLE_MAX_LEN" in str(e.get("replacement_new", "")) + str(e.get("anchor_old", ""))),
        None)
    out["copilot_authored_widening_edit"] = bool(widening)

    vblock = spec.get("verify") or {}
    node = vblock.get("red_test_node")
    grounded = bool(node)
    out["box0_grounded"] = grounded
    out["synth_node"] = node

    verdict = {"transition": "SKIPPED", "verified": False}
    if grounded and widening:
        runner = cfg.test_runner_for_codebase(root)
        backup_root = os.path.join(tempfile.gettempdir(), "realkey_0075_backups")
        verdict = verifymod.verify_red_green(spec, root, runner, backup_root, ttl_hours=1)
    out["transition"] = verdict.get("transition")
    out["verified"] = bool(verdict.get("verified"))
    out["red_status"] = (verdict.get("red") or {}).get("status")
    out["green_status"] = (verdict.get("green") or {}).get("status")

    after_bytes = open(tgt_abs, "rb").read()
    out["source_restored"] = (before_bytes == after_bytes)
    out["credits"] = _read_credits(call_log)

    out["checks"] = {
        "copilot_authored_the_fix": out["copilot_authored_widening_edit"],
        "box0_grounded_red_test": grounded,
        "red_to_green_verified": out["verified"] and out["transition"] == "red_to_green",
        "source_restored": out["source_restored"],
    }
    out["GO"] = all(out["checks"].values())
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
