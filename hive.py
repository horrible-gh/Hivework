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

    logger.info("=" * 60)
    logger.info("Hivework specify — honey → edit-spec (Stage-1: propose only)")
    logger.info("  honey:    %s", args.honey)
    logger.info("  codebase: %s", args.codebase)
    logger.info("  output:   %s", args.out)
    logger.info("  contract: %s", args.contract or "(default) recipes/edit_spec_contract_v1.md")
    logger.info("  author:   %s/%s", role.provider, role.model)
    logger.info("=" * 60)

    ldg = open_ledger(cfg.ledger.enabled, cfg.ledger.db_path)
    ldg.start_run(seed=args.honey, codebase=args.codebase,
                  model_queen=role.model, model_swarm=role.model)
    spec: dict = {}
    try:
        spec = run_specify(
            honey_path=args.honey,
            codebase_root=args.codebase,
            output_path=args.out,
            contract_path=args.contract,
            model=role.model,
            provider=role.provider,
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(getattr(args, 'verbose', False))

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "specify":
        run_specify_command(args)


if __name__ == "__main__":
    main()
