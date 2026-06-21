#!/usr/bin/env python3
"""Hivework CLI — full-loop orchestrator entry point.

Runs the 6-stage pipeline:
  ① decompose → ② fan-out → ③ parse → ④ conflict-scan → ⑤ reconcile → ⑥ assemble

Usage:
  python hive.py run --seed <seed.md> --recipe <recipe.md> \
      --codebase <root> --out <honey.md> [--workdir <dir>] [--round-cap 2]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

from hive.config import load_config
from hive.ledger import open_ledger
from hive.decompose import run_decompose, independent_axes
from hive.be_root import be_root_axis
from hive.providers import call_worker
from hive.fanout import run_fanout, load_comb_contract
from hive.parse import partition_combs
from hive.conflict_scan import scan_conflicts
from hive.reconcile import run_reconcile_loop
from hive.assemble import run_assemble
from hive.specify import run_specify
from hive.reinvestigate import run_reinvestigation_loop
from hive.apply import run_apply
from hive.commit import run_propose, run_commit, render_commit_summary_lines
from hive.converge import run_converge
from hive.coordinator import run_coordinator
from hive.coordinator.gapstate import open_store as open_gapstate_store
from hive.investigate import (
    _converge_fragments,
    _rebuild_bundles,
    format_caller_context,
    render_local_honey,
    rerun_reinvestigation,
    run_investigate,
    seed_edit_targets,
)
from hive import backup as backup_store
from hive import secrets as hive_secrets


def _force_utf8_io() -> None:
    """Make our own stdout/stderr UTF-8 so Korean logs don't crash on Windows.

    On Windows the console defaults to cp932; logging Korean (seed paths,
    conflict details, copilot output snippets) raises UnicodeEncodeError and
    kills the first launch. Reconfiguring here is self-protecting regardless
    of how hive.py is invoked (no reliance on the caller's environment).
    PYTHONIOENCODING is also set as a cheap defence for any child Python.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _load_secrets() -> str | None:
    """Load Hivework's out-of-repo secrets file into the environment (env-vars WIN, file
    fills gaps). Thin back-compat wrapper around the canonical ``hive.secrets.load_secrets``
    so the CLI and the test suite (``tests/conftest.py``) share ONE loader — a token set
    once by ``hive_setup`` (``~/.hivework/.env``) reaches every entry point without a
    launcher batch / manual ``set``. See ``hive/secrets.py`` for resolution + precedence.
    """
    return hive_secrets.load_secrets()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def build_provider_kwargs(cfg) -> dict:
    """One shared kwargs dict handed to every ``call_worker`` in a command.

    Carries BOTH providers' connection settings: the copilot CLI (``exe`` /
    ``allow_flag``) and the OpenAI-compatible HTTP endpoint (``base_url`` /
    ``api_key_env``, from the config ``openai`` block — what makes that provider
    vendor-neutral). Each handler ignores the keys meant for the other (copilot via
    ``**_ignored``, the HTTP handler likewise), so one dict safely serves a run that
    mixes providers across roles.
    """
    kwargs: dict[str, str] = {}
    if cfg.copilot.exe:
        kwargs["exe"] = cfg.copilot.exe
    if cfg.copilot.allow:
        kwargs["allow_flag"] = cfg.copilot.allow
    # Capability enforcement: read-only copilot workers (--deny-tool=write,shell).
    # One semantic flag set here, translated once per provider in the handler;
    # codex/http ignore it (already read-only). See CopilotConfig.read_only.
    kwargs["read_only"] = cfg.copilot.read_only
    # Pin the copilot billing account: resolve the configured token (raw ``token``
    # wins, else read the env var named by ``token_env``) and hand it to the
    # copilot handler, which injects it as COPILOT_GITHUB_TOKEN so the run bills
    # THIS account regardless of the ambient shell/login. Left unset → handler
    # warns and falls back to the CLI's stored login (prior behavior).
    copilot_token = cfg.copilot.token
    if not copilot_token and cfg.copilot.token_env:
        copilot_token = os.environ.get(cfg.copilot.token_env)
    if copilot_token:
        kwargs["copilot_token"] = copilot_token
    # Codex CLI provider settings (ignored by the other handlers via **_ignored):
    # ``codex_exe`` pins the binary, ``codex_lock_timeout_sec`` is the cross-process
    # serialization mutex's wait bound — an explicit config number, always passed.
    if cfg.codex.exe:
        kwargs["codex_exe"] = cfg.codex.exe
    kwargs["codex_lock_timeout_sec"] = cfg.codex.lock_timeout_sec
    if cfg.openai.base_url:
        kwargs["base_url"] = cfg.openai.base_url
    if cfg.openai.api_key_env:
        kwargs["api_key_env"] = cfg.openai.api_key_env
    return kwargs


def http_shape_specify_kwargs(cfg, codebase_root: str | None) -> dict:
    """run_specify kwargs that bind lever ⑦'s red test to the target's TestClient.

    Resolves the per-codebase ``http_shape`` harness (config ``targets.<name>.http_shape``)
    and forwards its setup block / fixture name / test dir into ``run_specify``. Returns
    an empty dict when the codebase has no entry — synthesis then stays a no-op (or falls
    back to auto-discovery), so behaviour is unchanged until a harness is configured.
    """
    hs = cfg.http_shape_for_codebase(codebase_root)
    if not hs:
        return {}
    out: dict = {"http_shape_test_dir": hs.test_dir or "tests"}
    setup_block = hs.resolve_setup_block(codebase_root)
    if setup_block:
        out["http_shape_setup_block"] = setup_block
    if hs.app_fixture:
        out["http_shape_app_fixture"] = hs.app_fixture
    return out


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full 6-stage pipeline."""
    start_time = time.time()
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    cfg.apply_cli_model(args.model)

    # ── Cost guard-rail (safety.allow_swarm): refuse the open-ended swarm before
    #    spending a single credit. The swarm fan-out (one agentic drone per axis,
    #    billed per internal turn) is the run path's blow-up risk; when it is
    #    locked off in config, route the operator to the cheap investigate path
    #    rather than silently launching drones.
    if not cfg.safety.allow_swarm:
        logger.error("=" * 60)
        logger.error("Swarm `run` path is DISABLED by config (safety.allow_swarm=false).")
        logger.error("This blocks fan-out + reconcile drones (the cost/​hang risk).")
        logger.error("Use the cheap path instead:")
        logger.error("  python hive.py investigate --specify --seed <s> --codebase <r> --out <o>")
        logger.error("To deliberately allow the swarm, set safety.allow_swarm=true in hive.config.json.")
        logger.error("=" * 60)
        raise SystemExit(2)

    provider_kwargs = build_provider_kwargs(cfg)

    queen_role = cfg.queen
    fanout_role = cfg.role("fanout")
    assemble_role = cfg.role("assemble")
    specify_role = cfg.role("specify")

    # Setup workdir
    workdir = args.workdir or os.path.join(os.path.dirname(args.out), "hive_workdir")
    os.makedirs(workdir, exist_ok=True)
    combs_dir = os.path.join(workdir, "combs")
    os.makedirs(combs_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Hivework full-loop orchestrator")
    logger.info("  seed:     %s", args.seed)
    logger.info("  recipe:   %s", args.recipe)
    logger.info("  codebase: %s", args.codebase)
    logger.info("  output:   %s", args.out)
    logger.info("  workdir:  %s", workdir)
    logger.info("  round-cap: %d", args.round_cap)
    logger.info("  queen:    %s/%s", queen_role.provider, queen_role.model)
    logger.info("  fanout:   %s/%s", fanout_role.provider, fanout_role.model)
    logger.info("  assemble: %s/%s", assemble_role.provider, assemble_role.model)
    if args.specify:
        logger.info("  specify:  %s/%s (chained)", specify_role.provider, specify_role.model)
    logger.info("=" * 60)

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.seed, codebase=args.codebase,
                  model_queen=queen_role.model, model_fanout=fanout_role.model)

    honey_path = ""
    final_combs: list[dict] = []
    conflicts: list[dict] = []
    remaining_conflicts: list[dict] = []
    rounds_used = 0
    parse_errors: list[str] = []

    try:
        # Load seed text (+ opt-in requester comments folded in as a labelled section)
        with open(args.seed, 'r', encoding='utf-8') as f:
            seed_text = f.read()
        caller_context = format_caller_context(getattr(args, "comment", None))
        if caller_context:
            seed_text += caller_context
            logger.info("Caller-supplied context: %d comment(s) folded into seed",
                        len(args.comment))
        logger.info("Seed loaded: %d chars", len(seed_text))

        # ────────────────────────────────────────────────────────────
        # STAGE L-01 coordinator (opt-in, pre-decompose seed enrichment)
        # ────────────────────────────────────────────────────────────
        # The queen's front-end interpreter (R0001): extracts the true `expected`
        # and structures the symptom into the Caller-supplied context section the
        # decompose stage already reads. Opt-in via --coordinator; when absent the
        # seed flows to decompose byte-for-byte unchanged (CON / D-01 §4).
        if getattr(args, "coordinator", False):
            coord_role = cfg.role("coordinator")
            logger.info("─" * 60)
            logger.info("STAGE coordinator (%s/%s)", coord_role.provider, coord_role.model)
            logger.info("─" * 60)
            store = open_gapstate_store(cfg.ledger.db_path, cfg.apply.backup_ttl_hours,
                                        enabled=cfg.ledger.enabled)
            coord_result = run_coordinator(
                seed_text=seed_text,
                codebase_root=args.codebase,
                recipe_path=args.recipe,
                model=coord_role.model,
                provider=coord_role.provider,
                ledger=ldg,
                provider_kwargs=provider_kwargs,
                timeout=coord_role.worker_timeout(),
                store=store,
            )
            if store is not None:
                store.close()
            seed_text = coord_result["enriched_seed"]
            coord_path = os.path.join(workdir, "coordinator_result.json")
            with open(coord_path, "w", encoding="utf-8") as f:
                json.dump(coord_result, f, indent=2, ensure_ascii=False)
            logger.info("Coordinator: status=%s, expected=%d axes, skipped=%d slots "
                        "→ %s", coord_result["status"], len(coord_result["expected"]),
                        len(coord_result["provenance"]["skipped_slots"]), coord_path)

        # ────────────────────────────────────────────────────────────
        # STAGE ① decompose
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ① decompose")
        logger.info("─" * 60)

        decompose_result = run_decompose(
            seed_text=seed_text,
            recipe_path=args.recipe,
            codebase_root=args.codebase,
            model=queen_role.model,
            provider=queen_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
            timeout=queen_role.worker_timeout(),
        )

        # Save decompose result
        decompose_path = os.path.join(workdir, "decompose_result.json")
        with open(decompose_path, 'w', encoding='utf-8') as f:
            json.dump(decompose_result, f, indent=2, ensure_ascii=False)
        logger.info("Decompose result saved to %s", decompose_path)

        # RC-B (NR hivework.default.0008.0009): the swarm fans out INDEPENDENT leaf
        # axes only. A task that declares depends_on — a synthesis / comparison /
        # integration step — cannot be a blind parallel drone: it needs the OTHER
        # axes' combs, which an independent swarm worker never sees, so it can only
        # return findings:[] ("No prior investigation evidence available; cannot
        # synthesize…" was the literal run-451 output). decompose's own rule #3
        # already places such tasks in a LATER `steps` entry; this stage used to
        # FLATTEN every task into the swarm and ignore that, drowning the run in a
        # guaranteed-empty synthesis axis. Honour the dependency: dependent tasks are
        # dropped from the fan-out (synthesis belongs to the assemble/queen stage).
        all_tasks = decompose_result.get("tasks", [])
        axes = independent_axes(all_tasks)
        dropped_axes = [t.get("id", "?") for t in all_tasks if t.get("depends_on")]
        if dropped_axes:
            logger.info("RC-B: %d dependent axis(es) excluded from swarm fan-out "
                        "(synthesis/integration is an assemble-stage job, not a blind "
                        "drone): %s", len(dropped_axes), ", ".join(dropped_axes))
        logger.info("Decompose produced %d axes (%d independent → fan-out)",
                    len(all_tasks), len(axes))

        # Hook A (M028 / NR hivework.default.0004.0003 §3): the be-root mitigation
        # was only wired into `investigate`, so the swarm `run` path drove the queen's
        # known call-graph blindness with NO mitigation (run 418's decompose carried
        # no CODEMAP_BE_ROOT axis). Wire it here too: the code-map traces FE symptom →
        # live handler → service/producer and injects it as a front swarm axis. Strict
        # opt-in (--be-root): it adds one node-pick model call and is a true no-op when
        # the code-map is absent or grounds nothing.
        if getattr(args, "be_root", False):
            def _be_pick(prompt: str) -> str:
                wr = call_worker(queen_role.provider, queen_role.model, prompt,
                                 cwd=None, timeout=queen_role.worker_timeout(),
                                 **provider_kwargs)
                return wr.stdout if wr.exit_code == 0 else ""
            be_axis = be_root_axis(seed_text, axes, args.codebase, call_fn=_be_pick)
            if be_axis is not None:
                axes = [be_axis] + [a for a in axes
                                    if a.get("id") != "CODEMAP_BE_ROOT"]
                logger.info("Hook A: injected CODEMAP_BE_ROOT swarm axis → %s",
                            be_axis["search_plan"]["file_globs"])
            else:
                logger.info("Hook A: code-map grounded no BE-root axis (no-op)")

        # ────────────────────────────────────────────────────────────
        # STAGE ② fan-out
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ② fan-out")
        logger.info("─" * 60)

        contract_path = os.path.join(
            os.path.dirname(args.recipe), "comb_contract_v2.md"
        )
        if not os.path.exists(contract_path):
            contract_path = None

        comb_files = run_fanout(
            axes=axes,
            seed_text=seed_text,
            codebase_root=args.codebase,
            workdir=workdir,
            contract_path=contract_path,
            model=fanout_role.model,
            provider=fanout_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
            max_workers=cfg.fanout.parallel,
            respecify_retries=cfg.fanout.retries,
            max_calls=cfg.fanout.max_calls,
        )
        logger.info("Fan-out complete: %d comb files", len(comb_files))

        # ────────────────────────────────────────────────────────────
        # STAGE ③ parse
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ③ parse")
        logger.info("─" * 60)

        # G1 (NR hivework.default.0005.0003 RC-2): partition_combs applies the SAME
        # comb-shape gate the ledger uses to the pipeline INPUT. A tool-argument
        # object ({"path":..,"pattern":..,"glob":..}) decodes as valid JSON, so it
        # used to parse cleanly and flow through conflict-scan → reconcile →
        # assemble as if it were evidence (fake "honey"). Now a parse with no
        # `findings` list is excluded from the evidence set — recorded in
        # parse_errors so the dropped axis stays visible in telemetry — so noise
        # can never masquerade as honey evidence.
        combs, excluded_notes, parse_fail_notes = partition_combs(comb_files)
        for note in parse_fail_notes:
            logger.error("  PARSE FAIL %s", note)
        for note in excluded_notes:
            logger.warning("  NON-COMB excluded %s", note)
        parse_errors.extend(parse_fail_notes)
        parse_errors.extend(excluded_notes)
        for parsed in combs:
            logger.info("  Parsed axis %s: termination=%s",
                        parsed.get("axis_id", "?"),
                        parsed.get("termination", "?"))

        # Save parsed combs
        parsed_path = os.path.join(workdir, "parsed_combs.json")
        with open(parsed_path, 'w', encoding='utf-8') as f:
            json.dump(combs, f, indent=2, ensure_ascii=False)
        logger.info("Parsed %d combs (errors: %d), saved to %s",
                    len(combs), len(parse_errors), parsed_path)

        if parse_errors:
            errors_path = os.path.join(workdir, "parse_errors.json")
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(parse_errors, f, indent=2, ensure_ascii=False)

        # ────────────────────────────────────────────────────────────
        # STAGE ④ conflict-scan
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ④ conflict-scan")
        logger.info("─" * 60)

        conflicts = scan_conflicts(combs)
        logger.info("Conflict scan: %d conflicts detected", len(conflicts))
        for c in conflicts:
            logger.info("  [%s] %s vs %s: %s",
                        c["type"], c["axis_a"], c.get("axis_b", "-"), c["detail"])

        # Save conflicts
        conflicts_path = os.path.join(workdir, "conflicts.json")
        with open(conflicts_path, 'w', encoding='utf-8') as f:
            json.dump(conflicts, f, indent=2, ensure_ascii=False)

        # ────────────────────────────────────────────────────────────
        # STAGE ⑤ reconcile loop
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ⑤ reconcile")
        logger.info("─" * 60)

        comb_contract = load_comb_contract(contract_path, args.codebase)

        if conflicts:
            final_combs, remaining_conflicts, rounds_used = run_reconcile_loop(
                combs=combs,
                comb_files=comb_files,
                seed_text=seed_text,
                codebase_root=args.codebase,
                workdir=workdir,
                comb_contract=comb_contract,
                model=queen_role.model,
                round_cap=args.round_cap,
                provider=queen_role.provider,
                ledger=ldg,
                provider_kwargs=provider_kwargs,
            )
        else:
            final_combs = combs
            remaining_conflicts = []
            rounds_used = 0
            logger.info("No conflicts — reconcile skipped")

        # Save final state
        final_path = os.path.join(workdir, "final_combs.json")
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_combs, f, indent=2, ensure_ascii=False)
        remaining_path = os.path.join(workdir, "remaining_conflicts.json")
        with open(remaining_path, 'w', encoding='utf-8') as f:
            json.dump(remaining_conflicts, f, indent=2, ensure_ascii=False)
        logger.info("Reconcile: %d rounds, %d remaining conflicts",
                    rounds_used, len(remaining_conflicts))

        # ────────────────────────────────────────────────────────────
        # STAGE ⑥ assemble
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ⑥ assemble")
        logger.info("─" * 60)

        honey_path = run_assemble(
            combs=final_combs,
            conflicts=remaining_conflicts,
            seed_text=seed_text,
            recipe_path=args.recipe,
            codebase_root=args.codebase,
            output_path=args.out,
            model=assemble_role.model,
            rounds_used=rounds_used,
            provider=assemble_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
        )

        # ────────────────────────────────────────────────────────────
        # STAGE ⑦ specify (optional, chained via --specify)
        # ────────────────────────────────────────────────────────────
        if args.specify and (
            not honey_path or not os.path.exists(honey_path)
            or os.path.getsize(honey_path) == 0
        ):
            logger.warning("Skipping chained specify: no usable honey at %r "
                           "(assemble produced nothing)", honey_path)
        elif args.specify:
            logger.info("─" * 60)
            logger.info("STAGE ⑦ specify (chained — honey → edit-spec, propose only)")
            logger.info("─" * 60)
            spec_out = args.spec_out or (os.path.splitext(args.out)[0] + ".edit_spec.json")
            review_role = cfg.role("review")
            specify_kwargs = dict(author_retries=specify_role.retries)
            if specify_role.timeout_sec is not None:
                specify_kwargs["author_timeout"] = specify_role.timeout_sec
            specify_kwargs.update(http_shape_specify_kwargs(cfg, args.codebase))
            try:
                spec = run_specify(
                    honey_path=honey_path,
                    codebase_root=args.codebase,
                    output_path=spec_out,
                    contract_path=args.contract,
                    model=specify_role.model,
                    provider=specify_role.provider,
                    review_model=review_role.model,
                    review_provider=review_role.provider,
                    ledger=ldg,
                    provider_kwargs=provider_kwargs,
                    db_conn=cfg.db_for_codebase(args.codebase),
                    **specify_kwargs,
                )
                logger.info("Edit-spec: %s (%d edits, %d deferred, termination=%s)",
                            spec_out, len(spec.get("edits") or []),
                            len(spec.get("deferred") or []), spec.get("termination", "?"))
            except Exception as e:
                # The honey is the valuable artifact and is already on disk; a
                # specify hiccup must not lose it. Persist a resume marker (partial-
                # save) so specify can re-run from it without re-investigating.
                resume_path = _write_specify_resume(
                    honey_path, spec_out, specify_role, review_role, args, str(e))
                logger.error("Chained specify failed (honey is intact at %s): %s",
                             honey_path, e)
                logger.error("Resume specify WITHOUT re-investigating: %s", resume_path)

        ldg.finish_run(
            honey_path=honey_path,
            axes_n=len(final_combs),
            rounds=rounds_used,
            conflicts_n=len(conflicts),
            remaining_n=len(remaining_conflicts),
            parse_errs=len(parse_errors),
            status="done",
        )
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

    # ────────────────────────────────────────────────────────────
    # Done
    # ────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("  Honey: %s", honey_path)
    logger.info("  Axes parsed: %d", len(final_combs))
    logger.info("  Reconcile rounds: %d", rounds_used)
    logger.info("  Remaining conflicts: %d", len(remaining_conflicts))
    logger.info("  Parse errors: %d", len(parse_errors))
    logger.info("=" * 60)

    # Best-effort: project this finished cycle into perf/metrics/runs.jsonl so the
    # performance report has fresh input without a hand-authored line (closes
    # NR0005 §2 — the hive is the producer, report.py the consumer).
    _emit_runs_jsonl(ldg.run_id, workdir, cfg.ledger.db_path, logger)


def _emit_runs_jsonl(run_id, workdir, db_path, logger) -> None:
    """Append this run's telemetry to ``perf/metrics/runs.jsonl`` (best-effort).

    Inverse of ``perf/metrics/report.py``: ledger → jsonl. Before this hook the
    report's input had to be authored by hand each cycle (NR0005 §2). Any failure
    is swallowed — a telemetry write must never break a completed run. comb_shaped
    is derived from the workdir's ``final_combs.json`` (no ledger column for it);
    the golden block is omitted (it needs the external NR0004 scorer).
    """
    if run_id is None:
        return
    try:
        import importlib.util
        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "perf", "metrics")
        emit_path = os.path.join(metrics_dir, "emit.py")
        out_path = os.path.join(metrics_dir, "runs.jsonl")
        spec = importlib.util.spec_from_file_location("_hive_runs_emit", emit_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if mod.emit(db_path, run_id, out_path, workdir=workdir):
            logger.info("  Telemetry: appended run%s → %s", run_id, out_path)
    except Exception as e:  # best-effort: never fatal
        logger.warning("Telemetry emit skipped (%s)", e)


def _write_specify_resume(honey_path, spec_out, specify_role, review_role, args,
                          error: str) -> str:
    """Persist a resume marker so a failed chained specify can be re-run from the
    SAVED honey, skipping decompose→retrieve→judge entirely.

    The honey IS the investigate stage's full output and is already on disk; this
    marker just records the exact standalone-``specify`` invocation that reproduces
    the chained author/review against it. Best-effort: a write failure must not mask
    the original specify error, so any OSError is swallowed.
    """
    resume_path = os.path.splitext(spec_out)[0] + ".specify_resume.json"
    marker = {
        "stage": "specify",
        "reason": "chained specify failed — investigate output (honey) is intact",
        "error": error,
        "honey_path": os.path.abspath(honey_path),
        "spec_out": os.path.abspath(spec_out),
        "codebase": os.path.abspath(args.codebase),
        "docs": os.path.abspath(args.docs) if getattr(args, "docs", None) else None,
        "specify": {"provider": specify_role.provider, "model": specify_role.model,
                    "timeout_sec": specify_role.timeout_sec, "retries": specify_role.retries},
        "review": {"provider": review_role.provider, "model": review_role.model},
        "resume_cmd": (
            f"{sys.executable} {os.path.abspath(__file__)} specify "
            f"--honey {os.path.abspath(honey_path)} "
            f"--codebase {os.path.abspath(args.codebase)} "
            f"--out {os.path.abspath(spec_out)}"
            + (f" --docs {os.path.abspath(args.docs)}" if getattr(args, "docs", None) else "")
        ),
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(resume_path)), exist_ok=True)
        with open(resume_path, "w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return resume_path


def run_investigate_command(args: argparse.Namespace) -> None:
    """Execute the cheap (M004) investigation path: decompose → bridge → retrieve → judge.

    Replaces the open-ended swarm fan-out with one queen decomposition, zero-cost
    local retrieval per axis, and a budgeted JUDGE verdict. The only spend is the
    1 decompose call plus ≤ ``judge.max_calls_per_axis`` per judged axis over
    ≤ ``judge.max_axes`` axes (all from hive.config.json).
    """
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    cfg.apply_cli_model(args.model)

    provider_kwargs = build_provider_kwargs(cfg)

    queen = cfg.queen
    judge_role = cfg.role("judge")
    default_globs = args.globs.split(",") if args.globs else None

    logger.info("=" * 60)
    logger.info("Hivework investigate — decompose → retrieve(local) → judge")
    logger.info("  seed:     %s", args.seed)
    logger.info("  codebase: %s", args.codebase)
    logger.info("  docs:     %s", args.docs or "(none)")
    logger.info("  output:   %s", args.out)
    logger.info("  queen:    %s/%s", queen.provider, queen.model)
    logger.info("  judge:    %s/%s (≤%d calls/axis, ≤%d axes)",
                judge_role.provider, judge_role.model,
                cfg.judge.max_calls_per_axis, cfg.judge.max_axes)
    logger.info("=" * 60)

    with open(args.seed, "r", encoding="utf-8") as f:
        seed_text = f.read()
    caller_context = format_caller_context(getattr(args, "comment", None))
    if caller_context:
        seed_text += caller_context
        logger.info("Caller-supplied context: %d comment(s) folded into seed",
                    len(args.comment))

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.seed, codebase=args.codebase,
                  model_queen=queen.model, model_fanout=judge_role.model)
    run_id = ldg.run_id
    result: dict = {}
    investigate_ok = False
    try:
        result = run_investigate(
            seed_text=seed_text, recipe_path=args.recipe, code_root=args.codebase,
            docs_root=args.docs, output_path=args.out, cfg=cfg, ledger=ldg,
            provider_kwargs=provider_kwargs, default_globs=default_globs,
        )
        ldg.finish_run(honey_path=args.out, axes_n=result.get("axes_judged", 0),
                       status="done")
        investigate_ok = True
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

    # Best-effort telemetry: project this finished investigate cycle into
    # perf/metrics/runs.jsonl, mirroring the swarm run path (see _emit_runs_jsonl
    # call above). Before this hook the investigate path wrote the ledger but never
    # the report's input, so every investigate run was invisible in runs.html until
    # a manual emit.py (hivework.0035.0014-T "실행 레포트가 안보인다"). The golden
    # block is omitted here (it needs the external scorer); an operator splices it
    # later via `emit.py --golden-json`, which report.load_runs dedups last-wins.
    if investigate_ok:
        _emit_runs_jsonl(run_id, os.path.dirname(os.path.abspath(args.out)),
                         cfg.ledger.db_path, logger)

    located = sum(1 for v in result.get("verdicts", []) if v["verdict"]["located"])
    logger.info("=" * 60)
    logger.info("Investigate complete: %d/%d axes located",
                located, result.get("axes_judged", 0))
    logger.info("  Report: %s (+ .md)", args.out)
    logger.info("=" * 60)

    # ── Optional chained specify: cheap-path verdicts → LOCAL honey → edit-spec.
    #    No assemble model call — the honey is templated from the verdicts (free),
    #    then the single specify author lowers it against live code. This is what
    #    lets a create/edit task get its edit-spec WITHOUT swarm fan-out.
    if getattr(args, "specify", False):
        # The seed may name explicit edit targets even when no axis located one
        # (Defect 2): those are still authorable from their grounded live text, so
        # specify must run if EITHER a verdict located something OR the seed pinned
        # a concrete file target.
        seed_targets = seed_edit_targets(seed_text, args.codebase, args.docs)
        if located == 0 and not seed_targets:
            logger.warning("Skipping chained specify: no located verdict and no "
                           "seed-named target to author an edit from.")
            return
        honey_path = os.path.splitext(args.out)[0] + ".honey.md"
        with open(honey_path, "w", encoding="utf-8") as f:
            f.write(render_local_honey(result, seed_text, args.codebase, args.docs))
        logger.info("Local honey rendered (no assemble call): %s", honey_path)

        specify_role = cfg.role("specify")
        spec_out = args.spec_out or (os.path.splitext(args.out)[0] + ".edit_spec.json")
        logger.info("─" * 60)
        logger.info("specify (chained — local honey → edit-spec, propose only) %s/%s",
                    specify_role.provider, specify_role.model)
        logger.info("─" * 60)
        ldg2 = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
        # Without start_run the ledger's _run_id stays None, so begin_call no-ops
        # and the chained specify worker calls leave NO rows. Start a run here so
        # the post-converge worker shows up in the configured ledger DB.
        ldg2.start_run(seed=args.seed, codebase=args.codebase,
                       model_queen=specify_role.model, model_fanout=specify_role.model)
        try:
            review_role = cfg.role("review")
            specify_kwargs = dict(author_retries=specify_role.retries)
            if specify_role.timeout_sec is not None:
                specify_kwargs["author_timeout"] = specify_role.timeout_sec
            specify_kwargs.update(http_shape_specify_kwargs(cfg, args.codebase))

            def _respecify():
                spec = run_specify(
                    honey_path=honey_path, codebase_root=args.codebase,
                    docs_root=args.docs, output_path=spec_out,
                    contract_path=args.contract, model=specify_role.model,
                    provider=specify_role.provider, review_model=review_role.model,
                    review_provider=review_role.provider, ledger=ldg2,
                    provider_kwargs=provider_kwargs,
                    db_conn=cfg.db_for_codebase(args.codebase), **specify_kwargs)
                logger.info("Edit-spec: %s (%d edits, %d deferred, termination=%s)",
                            spec_out, len(spec.get("edits") or []),
                            len(spec.get("deferred") or []),
                            spec.get("termination", "?"))
                return spec

            def _read_honey():
                try:
                    with open(honey_path, encoding="utf-8") as f:
                        return f.read()
                except OSError:
                    return ""

            spec = _respecify()
            # Reaction #3: route an NR to the cheapest re-entry instead of dead-ending.
            # The plan is always logged for free; the live re-run loop (gated by
            # cfg.reinvestigation.live, capped by max_rounds, with an honest no-change
            # early stop) re-grounds → re-specifies up to the cap.
            spec, result = run_reinvestigation_loop(
                spec, result, cfg=cfg,
                rerun=lambda pl, res: rerun_reinvestigation(
                    pl, res, seed_text=seed_text, code_root=args.codebase,
                    docs_root=args.docs, cfg=cfg, ledger=ldg2,
                    provider_kwargs=provider_kwargs, honey_out=honey_path),
                respecify=_respecify, read_honey=_read_honey)
        except Exception as e:
            # Partial-save: the honey (the whole investigate stage's output) is
            # already on disk, so specify can be RESUMED from it without re-running
            # decompose→retrieve→judge. Drop a resume marker next to the honey that
            # records exactly how, so a single slow/failed author call does not force
            # a full re-investigate (T892).
            resume_path = _write_specify_resume(
                honey_path, spec_out, specify_role, review_role, args, str(e))
            logger.error("Chained specify failed (verdicts + honey intact at %s): %s",
                         honey_path, e)
            logger.error("Resume specify WITHOUT re-investigating: %s", resume_path)
            ldg2.finish_run(honey_path=honey_path, status="failed")
        else:
            ldg2.finish_run(honey_path=honey_path, status="done")
        finally:
            ldg2.close()


def run_reconverge_command(args: argparse.Namespace) -> None:
    """Re-run ONLY the converge stage on a saved investigate ``verdict.json``.

    Rebuilds each axis's evidence bundle from its stored ``search_plan`` via free LOCAL
    retrieve (no decompose / judge re-spend, and deterministic — the SAME evidence each
    run), then runs converge ONCE with the configured converge role and the codebase's
    read-only DB. Lets a converge change — a guard, a prompt tweak, or a model swap — be
    tested for ~1 call instead of an 8-minute full pipeline, and with NO decompose
    non-determinism between runs. Propose-only; writes nothing to the target codebase.
    """
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    cfg.apply_cli_model(args.model)
    provider_kwargs = build_provider_kwargs(cfg)

    with open(args.verdict, "r", encoding="utf-8") as f:
        result = json.load(f)
    with open(args.seed, "r", encoding="utf-8") as f:
        seed_text = f.read()

    verdicts = list(result.get("verdicts") or [])
    located = [v for v in verdicts if (v.get("verdict") or {}).get("located")]
    conv_role = cfg.role("converge")
    db_conn = cfg.db_for_codebase(args.codebase)

    logger.info("=" * 60)
    logger.info("Hivework reconverge — rebuild bundles (local, free) → converge only")
    logger.info("  verdict:  %s (%d verdict(s), %d located)",
                args.verdict, len(verdicts), len(located))
    logger.info("  codebase: %s", args.codebase)
    logger.info("  converge: %s/%s%s", conv_role.provider, conv_role.model,
                f" (DB data-state read: {db_conn.kind})" if db_conn else "")
    logger.info("=" * 60)

    if len(located) < 2:
        logger.info("reconverge: <2 located verdict(s) — nothing to stitch (free skip)")
        return

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.seed, codebase=args.codebase,
                  model_queen=conv_role.model, model_fanout=conv_role.model)
    try:
        bundles = _rebuild_bundles(verdicts, args.codebase, args.docs)
        split_cfg = cfg.converge_split
        lens_cfg = cfg.converge_lens
        fanout_role = cfg.role("fanout")
        cres = run_converge(
            seed_text=seed_text, verdicts=_converge_fragments(verdicts), bundles=bundles,
            provider=conv_role.provider, model=conv_role.model, code_root=args.codebase,
            ledger=ldg, provider_kwargs=provider_kwargs, k=6, max_hops=2, db_conn=db_conn,
            split_enabled=split_cfg.enabled, split_max_loci=split_cfg.max_loci,
            split_provider=split_cfg.provider, split_model=split_cfg.model,
            lens_lenses=(lens_cfg.lenses if lens_cfg.enabled else []),
            lens_provider=(lens_cfg.provider or fanout_role.provider),
            lens_model=(lens_cfg.model or fanout_role.model),
            lens_min_refute=lens_cfg.min_refute)
        ldg.finish_run(status="done")
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

    cv = cres.as_dict()
    result["converge"] = cv
    cc = cv.get("causal_check") or {}
    ad = cv.get("attributed_defect") or {}
    logger.info("=" * 60)
    logger.info("reconverge result:")
    logger.info("  converged:         %s", cv.get("converged"))
    logger.info("  attributed:        %s:%s", ad.get("file"), ad.get("lines"))
    logger.info("  causal verdict:    %s (data_dependent=%s, data_premise_refuted=%s)",
                cc.get("verdict"), cc.get("data_dependent"),
                cc.get("data_premise_refuted"))
    logger.info("  data_state_backed: %s", cv.get("data_state_backed"))
    logger.info("  summary:           %s", cv.get("summary"))
    logger.info("=" * 60)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("reconverge: updated verdict written → %s", args.out)
        honey_out = os.path.splitext(args.out)[0] + ".honey.md"
        with open(honey_out, "w", encoding="utf-8") as f:
            f.write(render_local_honey(result, seed_text, args.codebase, args.docs))
        logger.info("reconverge: re-rendered honey → %s", honey_out)


def run_specify_command(args: argparse.Namespace) -> None:
    """Execute the standalone specify stage: honey + live code → edit-spec JSON.

    This runs after a honey exists (from `hive run`, or an existing NR honey).
    It is a single-author stage and Stage-1 propose-only: nothing is written to
    the target codebase.
    """
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    cfg.apply_cli_model(args.model)

    provider_kwargs = build_provider_kwargs(cfg)

    role = cfg.role("specify")
    review_role = cfg.role("review")

    logger.info("=" * 60)
    logger.info("Hivework specify — honey → edit-spec (Stage-1: propose only)")
    logger.info("  honey:    %s", args.honey)
    logger.info("  codebase: %s", args.codebase)
    logger.info("  docs:     %s", args.docs or "(none)")
    logger.info("  output:   %s", args.out)
    logger.info("  contract: %s", args.contract or "(default) recipes/edit_spec_contract_v1.md")
    logger.info("  author:   %s/%s", role.provider, role.model)
    logger.info("  reviewer: %s/%s", review_role.provider, review_role.model)
    logger.info("=" * 60)

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.honey, codebase=args.codebase,
                  model_queen=role.model, model_fanout=role.model)
    specify_kwargs = dict(author_retries=role.retries)
    if role.timeout_sec is not None:
        specify_kwargs["author_timeout"] = role.timeout_sec
    specify_kwargs.update(http_shape_specify_kwargs(cfg, args.codebase))
    spec: dict = {}
    try:
        spec = run_specify(
            honey_path=args.honey,
            codebase_root=args.codebase,
            docs_root=args.docs,
            output_path=args.out,
            contract_path=args.contract,
            model=role.model,
            provider=role.provider,
            review_model=review_role.model,
            review_provider=review_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
            db_conn=cfg.db_for_codebase(args.codebase),
            **specify_kwargs,
        )
        ldg.finish_run(honey_path=args.out, axes_n=len(spec.get("edits") or []),
                       status="done")
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

    logger.info("=" * 60)
    logger.info("Specify complete: %d edits, %d deferred, termination=%s",
                len(spec.get("edits") or []), len(spec.get("deferred") or []),
                spec.get("termination", "?"))
    logger.info("  Edit-spec (SSOT): %s", args.out)
    logger.info("=" * 60)


def _resolve_repair_honey(spec_path: str, explicit: str | None) -> str | None:
    """Locate the investigate honey a spec was authored from, for the repair loop.

    An explicit ``--honey`` always wins. Otherwise try the siblings specify's naming
    produces: the spec is ``<base>.edit_spec.json`` (so the honey is ``<base>`` or
    ``<base>.honey.md``) or just ``<spec>.honey.md``. Returns the first that exists,
    else None — the caller then falls back to plain verify (still monotonic).
    """
    if explicit:
        return explicit if os.path.exists(explicit) else None
    candidates = []
    if spec_path.endswith(".edit_spec.json"):
        base = spec_path[: -len(".edit_spec.json")]
        candidates += [base, base + ".honey.md", base + ".md"]
    root = os.path.splitext(spec_path)[0]
    candidates += [root, root + ".honey.md", spec_path + ".honey.md"]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            return c
    return None


def _build_repair_regenerator(args: argparse.Namespace, cfg, logger):
    """Build the specify-backed regenerate callback for ``apply --repair`` (or None).

    Mirrors the chained-specify wiring (roles, provider_kwargs, http-shape, db_conn)
    so a re-authored fix is produced by the SAME author the spec came from, only with
    the failing test output appended as evidence. Returns None when the source honey
    cannot be located — the apply path then degrades to single-shot verify, never an
    error, keeping the feature monotonic.
    """
    honey_path = _resolve_repair_honey(args.spec, getattr(args, "honey", None))
    if not honey_path:
        logger.warning("apply: --repair set but the source honey could not be located "
                       "(pass --honey) — falling back to single-shot verify")
        return None
    from hive.repair import make_specify_regenerator
    specify_role = cfg.role("specify")
    review_role = cfg.role("review")
    specify_kwargs = dict(author_retries=specify_role.retries)
    if specify_role.timeout_sec is not None:
        specify_kwargs["author_timeout"] = specify_role.timeout_sec
    specify_kwargs.update(http_shape_specify_kwargs(cfg, args.codebase))
    spec_out = os.path.splitext(args.spec)[0] + ".repair"
    logger.info("apply: --repair will re-author fixes via specify (%s/%s) from honey %s",
                specify_role.provider, specify_role.model, honey_path)
    return make_specify_regenerator(
        honey_path=honey_path,
        codebase_root=args.codebase,
        spec_out=spec_out,
        model=specify_role.model,
        provider=specify_role.provider,
        review_model=review_role.model,
        review_provider=review_role.provider,
        provider_kwargs=build_provider_kwargs(cfg),
        db_conn=cfg.db_for_codebase(args.codebase),
        **specify_kwargs,
    )


def run_apply_command(args: argparse.Namespace) -> None:
    """Execute the standalone apply stage: edit-spec JSON → proposal (+ optional write).

    This runs after specify has produced an edit-spec. apply calls no worker: it
    re-verifies each anchor against the LIVE codebase, renders unified diffs, and
    emits a proposal. By default it is propose-only — nothing is written.

    With ``--write`` and a READY proposal, the edits are applied to the live
    codebase after the originals are snapshotted into a scratch backup bundle
    (all-or-nothing, with rollback). A non-ready proposal is never written.
    """
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    backup_root = cfg.apply.backup_root()
    ttl_hours = cfg.apply.backup_ttl_hours

    # Runtime red→green verify (the closed loop): opt-in (--verify) and only fires when
    # the run's codebase has a configured test_runner AND the spec carries a verify block.
    # --repair implies --verify and adds the self-repair loop on top of it.
    do_repair = getattr(args, "repair", False)
    verify = getattr(args, "verify", False) or do_repair
    runner = cfg.test_runner_for_codebase(args.codebase) if verify else None
    if verify and runner is None:
        logger.warning("apply: --verify set but no test_runner configured for "
                       "codebase %r — runtime verify will be skipped", args.codebase)
    repair = _build_repair_regenerator(args, cfg, logger) if (do_repair and runner) else None

    mode = "WRITE (apply to live code)" if args.write else "propose only"
    logger.info("=" * 60)
    logger.info("Hivework apply — edit-spec → proposal (%s)", mode)
    logger.info("  spec:     %s", args.spec)
    logger.info("  codebase: %s", args.codebase or "(from spec's codebase_root)")
    logger.info("  docs:     %s", args.docs or "(none)")
    logger.info("  output:   %s", args.out or "(none — stdout summary only)")
    if args.write:
        logger.info("  backups:  %s (ttl %dh)", backup_root, ttl_hours)
    logger.info("=" * 60)

    proposal = run_apply(
        spec_path=args.spec,
        codebase_root=args.codebase,
        docs_root=args.docs,
        output_path=args.out,
        write=args.write,
        backup_root=backup_root if (args.write or verify) else None,
        ttl_hours=ttl_hours,
        partial=getattr(args, "partial", False),
        verify=verify,
        runner=runner,
        repair=repair,
        repair_max_iters=getattr(args, "repair_max_iters", 2),
    )

    logger.info("=" * 60)
    logger.info("Apply complete: %s — %d/%d edits applicable",
                "READY" if proposal["ready"] else "NOT READY",
                proposal["n_applicable"], proposal["n_edits"])
    if args.out:
        logger.info("  Proposal: %s", args.out)
    logger.info("=" * 60)

    # A successful --partial write is a success even though the spec is not globally
    # ready: the individually-ready edits shipped. Exit 0 so the caller sees it.
    if proposal.get("applied_partial"):
        w = proposal.get("write") or {}
        logger.info("Apply: PARTIAL write applied %d edit(s) %s; %d item(s) still "
                    "unresolved", len(w.get("written", [])), w.get("applied_ids"),
                    len(proposal["not_ready_reasons"]))
        return

    # Non-zero exit when not ready so a caller (or chained pipeline) can branch.
    if not proposal["ready"]:
        if proposal.get("partial_ready"):
            logger.info("  %d edit(s) %s are individually ready — re-run with "
                        "--partial to apply just those",
                        len(proposal["writable_ids"]), proposal["writable_ids"])
        sys.exit(2)

    # Ready but the write itself failed (e.g. anchor collided at write time and
    # everything was rolled back) — distinct exit so callers don't treat it as done.
    write = proposal.get("write")
    if args.write and write and not write.get("ok"):
        sys.exit(3)


def run_commit_plan_command(args: argparse.Namespace) -> None:
    """Execute the commit-plan stage: live git working tree → commit-plan JSON.

    A single author worker reads the live ``git status`` plus the commit-plan
    contract and groups the changes into atomic conventional commits. Propose-only:
    nothing is committed. The output plan JSON is the SSOT consumed by ``commit``.
    """
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    cfg.apply_cli_model(args.model)

    provider_kwargs = build_provider_kwargs(cfg)

    role = cfg.role("commit")

    logger.info("=" * 60)
    logger.info("Hivework commit-plan — git working tree → commit-plan (propose only)")
    logger.info("  repo:     %s", args.repo)
    logger.info("  output:   %s", args.out)
    logger.info("  contract: %s", args.contract or "(default) recipes/commit_plan_contract_v1.md")
    logger.info("  author:   %s/%s", role.provider, role.model)
    if args.feedback or args.prev_plan:
        logger.info("  revising: prev_plan=%s feedback=%s",
                    args.prev_plan or "(none)", "yes" if args.feedback else "no")
    logger.info("=" * 60)

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.repo, codebase=args.repo,
                  model_queen=role.model, model_fanout=role.model)
    plan: dict = {}
    try:
        plan = run_propose(
            repo_root=args.repo,
            output_path=args.out,
            contract_path=args.contract,
            model=role.model,
            provider=role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
            prev_plan_path=args.prev_plan,
            feedback=args.feedback,
            filename_only_threshold=cfg.commit_stage.filename_only_threshold,
        )
        ldg.finish_run(honey_path=args.out,
                       axes_n=len(plan.get("commits") or []), status="done")
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

    logger.info("=" * 60)
    logger.info("Commit-plan complete: %d commits, termination=%s",
                len(plan.get("commits") or []), plan.get("termination", "?"))
    logger.info("  Commit-plan (SSOT): %s", args.out)
    logger.info("=" * 60)


def run_commit_command(args: argparse.Namespace) -> None:
    """Execute the commit stage: commit-plan JSON → proposal (+ optional write).

    This calls no worker: it re-verifies each planned commit against the LIVE git
    state, renders a proposal table, and (by default) does nothing else. With
    ``--write`` and a READY plan, the commits are created scoped to their pathspecs,
    all-or-nothing (a mid-sequence failure soft-resets HEAD back). A non-ready plan
    is never committed.
    """
    logger = logging.getLogger("hive")

    mode = "WRITE (create commits)" if args.write else "dry run"
    logger.info("=" * 60)
    logger.info("Hivework commit — commit-plan → proposal (%s)", mode)
    logger.info("  plan:     %s", args.plan)
    logger.info("  repo:     %s", args.repo or "(from plan's repo_root)")
    logger.info("  output:   %s", args.out or "(none — stdout summary only)")
    logger.info("=" * 60)

    proposal = run_commit(
        plan_path=args.plan,
        repo_root=args.repo,
        output_path=args.out,
        write=args.write,
    )

    logger.info("=" * 60)
    logger.info("Commit complete: %s — %d/%d commits committable",
                "READY" if proposal["ready"] else "NOT READY",
                proposal["n_committable"], proposal["n_commits"])
    for line in render_commit_summary_lines(proposal):
        logger.info("  %s", line)
    if args.out:
        logger.info("  Proposal: %s", args.out)
    logger.info("=" * 60)

    # Non-zero exit when not ready so a caller (or chained pipeline) can branch.
    if not proposal["ready"]:
        sys.exit(2)

    # Ready but the commit itself failed and was rolled back — distinct exit.
    write = proposal.get("write")
    if args.write and write and not write.get("ok"):
        sys.exit(3)


def run_restore_command(args: argparse.Namespace) -> None:
    """Restore a backup bundle, undoing an earlier ``apply --write``."""
    logger = logging.getLogger("hive")
    cfg = load_config(profile=getattr(args, "profile", None))
    backup_root = cfg.apply.backup_root()

    bundle = args.bundle
    if not bundle and args.latest:
        bundle = backup_store.latest_bundle(backup_root)
        if not bundle:
            logger.error("No backup bundles found under %s", backup_root)
            sys.exit(1)
    if not bundle:
        logger.error("Provide --bundle <dir> or --latest")
        sys.exit(1)

    manifest = backup_store.load_manifest(bundle)
    logger.info("Restoring bundle %s", bundle)
    logger.info("  codebase_root: %s", manifest.get("codebase_root"))
    logger.info("  files:         %d", len(manifest.get("files", [])))
    restored = backup_store.restore_bundle(bundle)
    for path in restored:
        logger.info("  restored: %s", path)
    logger.info("Restore complete: %d file(s)", len(restored))


def main() -> None:
    """CLI entry point."""
    _force_utf8_io()
    _load_secrets()  # pull ~/.hivework/.env (or $HIVE_ENV_FILE) into env; real env wins

    parser = argparse.ArgumentParser(
        prog="hive",
        description="Hivework full-loop orchestrator — automated investigation pipeline",
    )
    # Global profile selector — picks config/hive.config.<profile>.json (default:
    # 'default'). Goes before the sub-command: `hive --profile small investigate ...`.
    parser.add_argument(
        "--profile", default=None,
        help="Config profile under config/ (default: 'default').",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 'run' sub-command
    run_parser = subparsers.add_parser("run", help="Run the full 6-stage pipeline")
    run_parser.add_argument(
        "--seed", required=True,
        help="Path to seed markdown file (investigation instruction)",
    )
    run_parser.add_argument(
        "--recipe", required=True,
        help="Path to recipe card markdown (e.g., smoke/loop/recipe_code_bug.md)",
    )
    run_parser.add_argument(
        "--codebase", required=True,
        help="Root path of the target codebase to investigate",
    )
    run_parser.add_argument(
        "--out", required=True,
        help="Output path for the honey markdown",
    )
    run_parser.add_argument(
        "--workdir", default=None,
        help="Working directory for intermediate outputs (default: alongside --out)",
    )
    run_parser.add_argument(
        "--round-cap", type=int, default=2,
        help="Maximum reconcile rounds (default: 2, per recipe)",
    )
    run_parser.add_argument(
        "--model", default=None,
        help="Model for copilot workers (default: per-role config, gpt-5-mini)",
    )
    run_parser.add_argument(
        "--comment", action="append", metavar="TEXT", default=None,
        help="Requester's direct input/hint, folded into the seed as authoritative "
             "intent (locations still verified). Repeatable; optional.",
    )
    run_parser.add_argument(
        "--coordinator", action="store_true",
        help="Enable the coordinator front-end (R0001): a Sonnet-tier pre-decompose "
             "stage that extracts the true 'expected' and structures the symptom into "
             "the Caller-supplied context section. Opt-in; absent, the pipeline is "
             "unchanged.",
    )
    run_parser.add_argument(
        "--be-root", dest="be_root", action="store_true",
        help="Hook A (M028): wire the code-map BE-root axis into the swarm path. The "
             "queen name-matches a frontend file and misses the live backend gate/"
             "producer one hop away; the code-map traces it and injects a "
             "CODEMAP_BE_ROOT axis the swarm reads. Opt-in (one node-pick model call); "
             "no-op when the code-map is absent or grounds nothing.",
    )
    run_parser.add_argument(
        "--specify", action="store_true",
        help="Chain the specify stage after assemble: honey → edit-spec (propose only)",
    )
    run_parser.add_argument(
        "--spec-out", default=None,
        help="Output path for the chained edit-spec (default: <out>.edit_spec.json)",
    )
    run_parser.add_argument(
        "--contract", default=None,
        help="Path to the edit-spec contract for --specify (default: recipes/edit_spec_contract_v1.md)",
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'investigate' sub-command — the cheap path: decompose → retrieve(local) → judge
    inv_parser = subparsers.add_parser(
        "investigate",
        help="Cheap investigation: decompose → local retrieve → judge verdicts "
             "(replaces swarm fan-out; spend = 1 decompose + judge budget)",
    )
    inv_parser.add_argument(
        "--seed", required=True,
        help="Path to seed markdown file (investigation instruction)",
    )
    inv_parser.add_argument(
        "--codebase", required=True,
        help="Root path of the target codebase to investigate (local FIND scope)",
    )
    inv_parser.add_argument(
        "--out", required=True,
        help="Output path for the verdict report JSON (a sibling .md is also written)",
    )
    inv_parser.add_argument(
        "--recipe", default=None,
        help="Path to recipe card markdown (for decompose §1 fixed axes)",
    )
    inv_parser.add_argument(
        "--docs", default=None,
        help="Root of design docs (for design-excerpt retrieval); optional",
    )
    inv_parser.add_argument(
        "--globs", default=None,
        help="Comma-separated fallback file globs when an axis names no path "
             "(e.g. 'server/**/*.py,client/**/*.vue')",
    )
    inv_parser.add_argument(
        "--model", default=None,
        help="Model override for all roles (default: per-role config)",
    )
    inv_parser.add_argument(
        "--comment", action="append", metavar="TEXT", default=None,
        help="Requester's direct input/hint, folded into the seed as authoritative "
             "intent (locations still verified). Repeatable; optional.",
    )
    inv_parser.add_argument(
        "--specify", action="store_true",
        help="Chain specify after the verdicts: render a LOCAL honey from the "
             "verdicts (no assemble call) → edit-spec JSON (propose only)",
    )
    inv_parser.add_argument(
        "--spec-out", default=None,
        help="Output path for the chained --specify edit-spec "
             "(default: <out>.edit_spec.json)",
    )
    inv_parser.add_argument(
        "--contract", default=None,
        help="Path to the edit-spec contract for --specify "
             "(default: recipes/edit_spec_contract_v1.md)",
    )
    inv_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'specify' sub-command — lower a honey into an edit-spec (Stage-1: propose only)
    spec_parser = subparsers.add_parser(
        "specify",
        help="Lower an assembled honey into an applicable edit-spec JSON (propose only)",
    )
    spec_parser.add_argument(
        "--honey", required=True,
        help="Path to the assembled honey markdown (investigation report)",
    )
    spec_parser.add_argument(
        "--codebase", required=True,
        help="Root of the LIVE codebase — anchors are lifted from here, not the honey",
    )
    spec_parser.add_argument(
        "--docs", default=None,
        help="Root of the design docs, when they live in a separate tree from the "
             "code (mirrors investigate's --docs). A document-update direction is "
             "lowered against this tree instead of the nearest source file.",
    )
    spec_parser.add_argument(
        "--out", required=True,
        help="Output path for the edit-spec JSON (the SSOT)",
    )
    spec_parser.add_argument(
        "--contract", default=None,
        help="Path to the edit-spec contract (default: recipes/edit_spec_contract_v1.md)",
    )
    spec_parser.add_argument(
        "--model", default=None,
        help="Model override for the specify author (default: per-role config)",
    )
    spec_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'apply' sub-command — render an edit-spec into a proposal (Stage-1: propose only)
    apply_parser = subparsers.add_parser(
        "apply",
        help="Render an edit-spec into a unified-diff proposal against live code "
             "(propose only — never writes)",
    )
    apply_parser.add_argument(
        "--spec", required=True,
        help="Path to the edit-spec JSON produced by specify (the SSOT)",
    )
    apply_parser.add_argument(
        "--codebase", default=None,
        help="Root of the LIVE codebase (default: the spec's codebase_root). "
             "Anchors are re-verified here, not trusted from the spec.",
    )
    apply_parser.add_argument(
        "--docs", default=None,
        help="Root of the design docs, when they live in a separate tree from the "
             "code (mirrors investigate's --docs). Used as an additional base to "
             "resolve an edit's file path when it is not found under --codebase.",
    )
    apply_parser.add_argument(
        "--out", default=None,
        help="Output path for the proposal markdown (default: none, summary to log)",
    )
    apply_parser.add_argument(
        "--write", action="store_true",
        help="Apply a READY proposal's edits to the live codebase (default: propose "
             "only). Originals are backed up to a scratch bundle first; not-ready "
             "proposals are never written.",
    )
    apply_parser.add_argument(
        "--partial", action="store_true",
        help="With --write on a NOT-READY proposal, apply just the individually "
             "applicable + effective edits instead of nothing — so a verified fix "
             "is not blocked by a deferred sibling. Unresolved items are reported.",
    )
    apply_parser.add_argument(
        "--verify", action="store_true",
        help="Run the spec's red test red→green against the configured test_runner "
             "before declaring READY: apply the test edit only (must fail), then the "
             "source fix (must pass), then restore. A fix unconfirmed by execution is "
             "held NOT READY. No-op when the codebase has no test_runner configured.",
    )
    apply_parser.add_argument(
        "--repair", action="store_true",
        help="Self-repair loop (implies --verify): when the red test stays RED after "
             "the fix, feed its failing output back to specify and re-author the fix, "
             "up to --repair-max-iters times. Monotonic — it can only turn a still_red "
             "into a verified green, never make a fix worse. Needs the source honey "
             "(--honey, else derived from the spec path) to re-author.",
    )
    apply_parser.add_argument(
        "--repair-max-iters", type=int, default=2,
        help="Max repair iterations (regenerate→verify rounds, excludes the initial "
             "verify). Default 2.",
    )
    apply_parser.add_argument(
        "--honey", default=None,
        help="Path to the investigate honey the spec was authored from. Used by "
             "--repair to re-author the fix. Defaults to a sibling of --spec "
             "(<spec-without-.edit_spec.json> or <spec>.honey.md).",
    )
    apply_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'commit-plan' sub-command — author a commit-plan from the live git tree
    commit_plan_parser = subparsers.add_parser(
        "commit-plan",
        help="Group the repo's uncommitted changes into a commit-plan JSON "
             "(propose only — never commits)",
    )
    commit_plan_parser.add_argument(
        "--repo", required=True,
        help="Root of the git work tree whose changes to group into commits",
    )
    commit_plan_parser.add_argument(
        "--out", required=True,
        help="Output path for the commit-plan JSON (the SSOT)",
    )
    commit_plan_parser.add_argument(
        "--contract", default=None,
        help="Path to the commit-plan contract (default: recipes/commit_plan_contract_v1.md)",
    )
    commit_plan_parser.add_argument(
        "--feedback", default=None,
        help="PM feedback to revise a rejected plan (use with --prev-plan)",
    )
    commit_plan_parser.add_argument(
        "--prev-plan", default=None,
        help="Path to a previously rejected commit-plan JSON to revise",
    )
    commit_plan_parser.add_argument(
        "--model", default=None,
        help="Model override for the commit author (default: per-role config, haiku)",
    )
    commit_plan_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'commit' sub-command — execute a commit-plan against live git (dry run / --write)
    commit_parser = subparsers.add_parser(
        "commit",
        help="Re-verify a commit-plan against live git and render a proposal "
             "(dry run; --write creates the commits)",
    )
    commit_parser.add_argument(
        "--plan", required=True,
        help="Path to the commit-plan JSON produced by commit-plan (the SSOT)",
    )
    commit_parser.add_argument(
        "--repo", default=None,
        help="Root of the git work tree (default: the plan's repo_root). "
             "Commits are re-verified here, not trusted from the plan.",
    )
    commit_parser.add_argument(
        "--out", default=None,
        help="Output path for the proposal markdown (default: none, summary to log)",
    )
    commit_parser.add_argument(
        "--write", action="store_true",
        help="Create the commits for a READY plan (default: dry run). Each commit "
             "is scoped to its pathspecs; not-ready plans are never committed.",
    )
    commit_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'restore' sub-command — undo an apply --write from its backup bundle
    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore files from an apply --write backup bundle (undo a write)",
    )
    restore_parser.add_argument(
        "--bundle", default=None,
        help="Path to the backup bundle directory to restore from",
    )
    restore_parser.add_argument(
        "--latest", action="store_true",
        help="Restore the most recent backup bundle in the store",
    )
    restore_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    # 'reconverge' — re-run ONLY converge on a saved verdict.json (cheap converge iteration)
    reconv_parser = subparsers.add_parser(
        "reconverge",
        help="Re-run ONLY converge on a saved verdict.json (rebuild bundles locally; "
             "~1 call, deterministic — test a converge guard/prompt/model without a full run)",
    )
    reconv_parser.add_argument(
        "--verdict", required=True,
        help="Path to a saved investigate verdict.json (carries verdicts + search_plans)",
    )
    reconv_parser.add_argument(
        "--seed", required=True,
        help="Path to the original seed markdown (converge re-reads the symptom)",
    )
    reconv_parser.add_argument(
        "--codebase", required=True,
        help="Root path of the target codebase (local re-retrieve + live-code grounding)",
    )
    reconv_parser.add_argument(
        "--docs", default=None,
        help="Root of design docs (optional; mirrors the original run)",
    )
    reconv_parser.add_argument(
        "--out", default=None,
        help="Write the updated verdict (+ a sibling .honey.md) here (optional)",
    )
    reconv_parser.add_argument(
        "--model", default=None,
        help="Model override for all roles (default: per-role config)",
    )
    reconv_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, 'verbose', False))

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "investigate":
        run_investigate_command(args)
    elif args.command == "specify":
        run_specify_command(args)
    elif args.command == "reconverge":
        run_reconverge_command(args)
    elif args.command == "apply":
        run_apply_command(args)
    elif args.command == "commit-plan":
        run_commit_plan_command(args)
    elif args.command == "commit":
        run_commit_command(args)
    elif args.command == "restore":
        run_restore_command(args)


if __name__ == "__main__":
    main()
