# Hivework

A tool that orchestrates codebase investigation and fixes as a **multi-stage
pipeline** rather than a single model call. It investigates broadly with a swarm of
cheap workers, converges conflicts through re-investigation, and spends expensive
models only where they're needed.

It doesn't stop at investigation — it goes one box further each time:
**investigate → specify → apply**. The actual mutation is the thin, mechanical last
step. By default `apply` only **proposes a diff and never writes**; pass `--write`
to actually apply a READY proposal to the code (originals are backed up first so the
write is undoable — see below).

## Pipeline

```
① decompose → ② fan-out → ③ parse → ④ conflict-scan → ⑤ reconcile → ⑥ assemble
   (investigate = `run`, produces the honey)
        │
        ▼
   ⑦ specify   honey + LIVE code → edit-spec (JSON, SSOT)      ← single author, calls a model
        │
        ▼
   ⑧ apply     edit-spec → unified-diff proposal (propose only) ← no worker call, pure local
```

Three user-facing stages: **run (investigate) → specify (lower to an edit-spec) →
apply (render the proposal)**.

## Prerequisites

- Python 3.11+
- Worker provider: the `copilot` CLI (per-role models live in `hive.config.json`)
- stdout/stderr are forced to UTF-8 so Korean logs don't crash the Windows console

## 1. run — investigate (6 stages, produces a honey)

```powershell
python hive.py run `
  --seed <seed.md> `
  --recipe recipes\recipe_code_bug.md `
  --codebase <project-root> `
  --out <honey.md> `
  [--workdir <dir>] [--round-cap 2] [--model <model>] [-v]
```

- Add `--specify` to chain the specify stage right after assemble and produce
  `<out>.edit_spec.json` in one go (use `--spec-out` / `--contract` to override the
  output / contract paths).

## 2. specify — lower to an edit-spec (honey → edit-spec)

Lowers the honey's *prose* fix directions into precisely **applicable**
`anchor_old → replacement_new` edits. This is a **single-author** stage, not fan-out
(code edits must be internally coherent).

```powershell
python hive.py specify `
  --honey <honey.md> `
  --codebase <LIVE code root> `
  --out <edit_spec.json> `
  [--contract recipes\edit_spec_contract_v1.md] [--model <model>] [-v]
```

Key rules (contract: `recipes/edit_spec_contract_v1.md`):

- Anchors are lifted from **LIVE code, not the honey**, byte-for-byte (the honey may
  be stale). The result is fed back as `anchor_status: verified | stale | not_found`.
- A direction that can't be expressed as before→after (needs runtime / policy /
  multi-file design) goes to `deferred[]`, not `edits[]`.
- If the **honey's premise is false** (the code is already correct), no edit is made:
  it lands in `deferred` with `termination: needs_reinvestigation`.
- `gate.apply` is always false (Stage-1). specify never writes to the code.
- The output is a single JSON edit-spec (the SSOT). The diff is a derived view
  rendered by apply.

## 3. apply — render the proposal, and optionally write (edit-spec → diff [→ code])

Reads the edit-spec (SSOT) produced by specify, **re-verifies each anchor against
LIVE code**, and renders unified diffs. This is a **pure, deterministic local stage
with no worker call** (no model invoked).

```powershell
python hive.py apply `
  --spec <edit_spec.json> `
  [--codebase <LIVE code root>] `   # defaults to the spec's codebase_root
  [--out <proposal.md>] `           # omit for a log summary only
  [--write] `                       # actually apply a READY proposal to the code
  [-v]
```

Behaviour:

- **Anchor re-verification**: even if specify marked an edit `verified`, apply
  re-locates it. If live code has changed, it's caught as **drift and refused**.
- **applicable = the anchor occurs exactly once** in live code. Zero matches
  (missing / already-applied) or multiple matches (ambiguous) are not applicable.
- **ready verdict** = `termination == ready_to_apply` AND every edit is applicable.
- **Default is propose-only**: without `--write`, the target code is never written.
  `gate.apply` in the spec is model-authored and is never trusted to drive a write —
  the write switch is the human-held `--write` flag.
- Exit codes: **2** when not ready (so a caller can branch), **0** when ready (and,
  with `--write`, the write succeeded), **3** when ready but the write failed and was
  rolled back.
- The proposal contains the verdict + a per-edit unified diff + the gate commands to
  run after you apply (and, after a `--write`, the files written + the backup bundle).

### `--write` — applying for real (with an undo window)

`--write` only fires on a **READY** proposal; a not-ready proposal is never written.
The write is **all-or-nothing**: before touching anything, every target file's
original is snapshotted into a scratch **backup bundle** outside the codebase, each
edit's anchor is re-verified unique at the instant of writing, and if any anchor is no
longer unique the whole change is **rolled back** from the snapshot.

Backups are a time-boxed undo window (config `apply.backup_dir` / `apply.backup_ttl_hours`,
default `.apply_backups/` and 168h). Each `--write` run first purges expired bundles.
To undo a write while its bundle is still alive:

```powershell
python hive.py restore --bundle <bundle-dir>   # path is printed in the proposal/log
python hive.py restore --latest                # most recent bundle in the store
```

## Tests

```powershell
python -m pytest -q
```

## Layout

```
hive/
  decompose.py  fanout.py  parse.py  conflict_scan.py  reconcile.py  assemble.py
  specify.py    apply.py    backup.py
  config.py     ledger.py   providers.py
recipes/
  edit_spec_contract_v1.md   # specify's output contract (also the author's role prompt)
tests/
hive.py                      # CLI entry point (run / specify / apply / restore)
hive.config.json             # per-role provider/model config
```
