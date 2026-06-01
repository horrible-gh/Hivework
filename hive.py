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
from hive.decompose import run_decompose
from hive.fanout import run_fanout, load_comb_contract
from hive.parse import parse_comb_file
from hive.conflict_scan import scan_conflicts
from hive.reconcile import run_reconcile_loop
from hive.assemble import run_assemble
from hive.specify import run_specify
from hive.apply import run_apply
from hive.commit import run_propose, run_commit
from hive.investigate import render_local_honey, run_investigate
from hive import backup as backup_store


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


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the orchestrator."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full 6-stage pipeline."""
    start_time = time.time()
    logger = logging.getLogger("hive")
    cfg = load_config()
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

    provider_kwargs: dict[str, str] = {}
    if cfg.copilot.exe:
        provider_kwargs["exe"] = cfg.copilot.exe
    if cfg.copilot.allow:
        provider_kwargs["allow_flag"] = cfg.copilot.allow

    queen_role = cfg.queen
    swarm_role = cfg.swarm
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
    logger.info("  swarm:    %s/%s", swarm_role.provider, swarm_role.model)
    logger.info("  assemble: %s/%s", assemble_role.provider, assemble_role.model)
    if args.specify:
        logger.info("  specify:  %s/%s (chained)", specify_role.provider, specify_role.model)
    logger.info("=" * 60)

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.seed, codebase=args.codebase,
                  model_queen=queen_role.model, model_swarm=swarm_role.model)

    honey_path = ""
    final_combs: list[dict] = []
    conflicts: list[dict] = []
    remaining_conflicts: list[dict] = []
    rounds_used = 0
    parse_errors: list[str] = []

    try:
        # Load seed text
        with open(args.seed, 'r', encoding='utf-8') as f:
            seed_text = f.read()
        logger.info("Seed loaded: %d chars", len(seed_text))

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
        )

        # Save decompose result
        decompose_path = os.path.join(workdir, "decompose_result.json")
        with open(decompose_path, 'w', encoding='utf-8') as f:
            json.dump(decompose_result, f, indent=2, ensure_ascii=False)
        logger.info("Decompose result saved to %s", decompose_path)

        axes = decompose_result.get("tasks", [])
        logger.info("Decompose produced %d axes", len(axes))

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
            model=swarm_role.model,
            provider=swarm_role.provider,
            ledger=ldg,
            provider_kwargs=provider_kwargs,
        )
        logger.info("Fan-out complete: %d comb files", len(comb_files))

        # ────────────────────────────────────────────────────────────
        # STAGE ③ parse
        # ────────────────────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info("STAGE ③ parse")
        logger.info("─" * 60)

        combs: list[dict] = []
        for axis_id, comb_path in sorted(comb_files.items()):
            try:
                parsed = parse_comb_file(comb_path)
                combs.append(parsed)
                logger.info("  Parsed axis %s: termination=%s",
                            parsed.get("axis_id", axis_id),
                            parsed.get("termination", "?"))
            except (ValueError, FileNotFoundError) as e:
                parse_errors.append(f"{axis_id}: {e}")
                logger.error("  PARSE FAIL axis %s: %s", axis_id, e)

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
            try:
                spec = run_specify(
                    honey_path=honey_path,
                    codebase_root=args.codebase,
                    output_path=spec_out,
                    contract_path=args.contract,
                    model=specify_role.model,
                    provider=specify_role.provider,
                    ledger=ldg,
                    provider_kwargs=provider_kwargs,
                )
                logger.info("Edit-spec: %s (%d edits, %d deferred, termination=%s)",
                            spec_out, len(spec.get("edits") or []),
                            len(spec.get("deferred") or []), spec.get("termination", "?"))
            except Exception as e:
                # The honey is the valuable artifact and is already on disk; a
                # specify hiccup must not lose it. Surface, but don't fail the run.
                logger.error("Chained specify failed (honey is intact at %s): %s",
                             honey_path, e)

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


def run_investigate_command(args: argparse.Namespace) -> None:
    """Execute the cheap (M004) investigation path: decompose → bridge → retrieve → judge.

    Replaces the open-ended swarm fan-out with one queen decomposition, zero-cost
    local retrieval per axis, and a budgeted JUDGE verdict. The only spend is the
    1 decompose call plus ≤ ``judge.max_calls_per_axis`` per judged axis over
    ≤ ``judge.max_axes`` axes (all from hive.config.json).
    """
    logger = logging.getLogger("hive")
    cfg = load_config()
    cfg.apply_cli_model(args.model)

    provider_kwargs: dict[str, str] = {}
    if cfg.copilot.exe:
        provider_kwargs["exe"] = cfg.copilot.exe
    if cfg.copilot.allow:
        provider_kwargs["allow_flag"] = cfg.copilot.allow

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

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.seed, codebase=args.codebase,
                  model_queen=queen.model, model_swarm=judge_role.model)
    result: dict = {}
    try:
        result = run_investigate(
            seed_text=seed_text, recipe_path=args.recipe, code_root=args.codebase,
            docs_root=args.docs, output_path=args.out, cfg=cfg, ledger=ldg,
            provider_kwargs=provider_kwargs, default_globs=default_globs,
        )
        ldg.finish_run(honey_path=args.out, axes_n=result.get("axes_judged", 0),
                       status="done")
    except Exception:
        ldg.finish_run(status="failed")
        raise
    finally:
        ldg.close()

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
        if located == 0:
            logger.warning("Skipping chained specify: no located verdict to author "
                           "an edit from (specify would have nothing to lower).")
            return
        honey_path = os.path.splitext(args.out)[0] + ".honey.md"
        with open(honey_path, "w", encoding="utf-8") as f:
            f.write(render_local_honey(result, seed_text))
        logger.info("Local honey rendered (no assemble call): %s", honey_path)

        specify_role = cfg.role("specify")
        spec_out = args.spec_out or (os.path.splitext(args.out)[0] + ".edit_spec.json")
        logger.info("─" * 60)
        logger.info("specify (chained — local honey → edit-spec, propose only) %s/%s",
                    specify_role.provider, specify_role.model)
        logger.info("─" * 60)
        ldg2 = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
        try:
            review_role = cfg.role("review")
            spec = run_specify(
                honey_path=honey_path, codebase_root=args.codebase,
                docs_root=args.docs,
                output_path=spec_out, contract_path=args.contract,
                model=specify_role.model, provider=specify_role.provider,
                review_model=review_role.model, review_provider=review_role.provider,
                ledger=ldg2, provider_kwargs=provider_kwargs,
            )
            logger.info("Edit-spec: %s (%d edits, %d deferred, termination=%s)",
                        spec_out, len(spec.get("edits") or []),
                        len(spec.get("deferred") or []), spec.get("termination", "?"))
        except Exception as e:
            logger.error("Chained specify failed (verdicts + honey intact at %s): %s",
                         honey_path, e)
        finally:
            ldg2.close()


def run_specify_command(args: argparse.Namespace) -> None:
    """Execute the standalone specify stage: honey + live code → edit-spec JSON.

    This runs after a honey exists (from `hive run`, or an existing NR honey).
    It is a single-author stage and Stage-1 propose-only: nothing is written to
    the target codebase.
    """
    logger = logging.getLogger("hive")
    cfg = load_config()
    cfg.apply_cli_model(args.model)

    provider_kwargs: dict[str, str] = {}
    if cfg.copilot.exe:
        provider_kwargs["exe"] = cfg.copilot.exe
    if cfg.copilot.allow:
        provider_kwargs["allow_flag"] = cfg.copilot.allow

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
                  model_queen=role.model, model_swarm=role.model)
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
    cfg = load_config()
    backup_root = cfg.apply.backup_root()
    ttl_hours = cfg.apply.backup_ttl_hours

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
        backup_root=backup_root if args.write else None,
        ttl_hours=ttl_hours,
    )

    logger.info("=" * 60)
    logger.info("Apply complete: %s — %d/%d edits applicable",
                "READY" if proposal["ready"] else "NOT READY",
                proposal["n_applicable"], proposal["n_edits"])
    if args.out:
        logger.info("  Proposal: %s", args.out)
    logger.info("=" * 60)

    # Non-zero exit when not ready so a caller (or chained pipeline) can branch.
    if not proposal["ready"]:
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
    cfg = load_config()
    cfg.apply_cli_model(args.model)

    provider_kwargs: dict[str, str] = {}
    if cfg.copilot.exe:
        provider_kwargs["exe"] = cfg.copilot.exe
    if cfg.copilot.allow:
        provider_kwargs["allow_flag"] = cfg.copilot.allow

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
                  model_queen=role.model, model_swarm=role.model)
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
    cfg = load_config()
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

    parser = argparse.ArgumentParser(
        prog="hive",
        description="Hivework full-loop orchestrator — automated investigation pipeline",
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
