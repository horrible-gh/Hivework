# Hivework

> ⚠️ **Status: WIP / experimental** — actively developed, interfaces may change.

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

Lowers the honey's *prose* fix directions into precisely **applicable** edits — either
an `anchor_old → replacement_new` change to an existing file, or a `create_file` edit
(`kind:"create_file"` + `content`, no anchor) when the honey calls for a brand-new file.
This is a **single-author** stage, not fan-out (code edits must be internally coherent).

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
- **Edit kinds**: an existing-file change is an anchor edit (`anchor_old → replacement_new`);
  a brand-new file is a `create_file` edit carrying `content` and no anchor (valid only when
  the target does not yet exist). See the contract's `[Edit kinds — anchor edit vs create_file]`.
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
- **applicable** — an anchor edit is applicable when its anchor occurs **exactly once**
  in live code (zero matches = missing/already-applied, multiple = ambiguous, neither
  applicable); a `create_file` edit is applicable when its target path **does not yet
  exist** (creating over an existing file is refused).
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
edit's anchor is re-verified unique at the instant of writing (a `create_file` target is
re-verified still absent), and if any check fails the whole change is **rolled back** —
modified files restored from the snapshot, newly created files deleted.

Backups are a time-boxed undo window (config `apply.backup_dir` / `apply.backup_ttl_hours`,
default `.apply_backups/` and 168h). Each `--write` run first purges expired bundles.
To undo a write while its bundle is still alive:

```powershell
python hive.py restore --bundle <bundle-dir>   # path is printed in the proposal/log
python hive.py restore --latest                # most recent bundle in the store
```

## 4. commit — group changes into atomic commits, and optionally commit

A second, independent two-stage pair that mirrors `specify → apply`, applied to git.
It turns a messy working tree into a sequence of clean, atomic
[Conventional Commits](https://www.conventionalcommits.org/) and then — only on a
human-held switch — creates them.

```
commit-plan   git working tree + contract → commit-plan (JSON, SSOT)  ← single author, calls a model
     │
     ▼
commit        commit-plan → re-verify vs live git → proposal [→ git commit]  ← no worker call, deterministic
```

Hivework owns its own commit policy
(`recipes/commit_plan_contract_v1.md`) — it does **not** depend on any external rule
document at runtime.

### 4.1 commit-plan — author the plan (propose only)

A single author worker reads the live `git status` and the contract, then groups the
changes into atomic commits. Nothing is committed.

```powershell
python hive.py commit-plan `
  --repo <git work tree> `
  --out <commit_plan.json> `
  [--feedback "consolidate the docs into one commit"] [--prev-plan <old.json>] `
  [--contract recipes\commit_plan_contract_v1.md] [--model <model>] [-v]
```

- The commit author defaults to **haiku** (`hive.config.json` role `commit`), a tier
  above the gpt-5-mini swarm, since grouping wants more judgement.
- Rejected a plan? Re-run with `--prev-plan <old.json> --feedback "..."` to revise
  rather than start blind (e.g. "too many doc commits — merge them").
- `gate.commit` in the plan is model-authored and is always forced `false`. The
  author never authorizes the write.

### 4.2 commit — re-verify and (optionally) commit

Reads the plan (SSOT), **re-verifies every commit against the live git state**, and
renders a proposal table. Calls no worker. Default is a dry run.

```powershell
python hive.py commit `
  --plan <commit_plan.json> `
  [--repo <git work tree>] `   # defaults to the plan's repo_root
  [--out <proposal.md>] `
  [--write] `                  # actually create the commits
  [-v]
```

Behaviour:

- **A commit is committable** only when its message is `type(scope): description`
  (English, allowed type), every file is currently changed in git, and no file is
  assigned to more than one commit (no hunk-splitting). `leftover` (changed files in
  no commit) is computed and surfaced.
- **ready verdict** = `termination == ready_to_commit` AND every commit committable.
- **`--write` only fires on a READY plan.** Each commit is created scoped to its
  pathspecs (`git add -- <files>` then `git commit -- <files> -m …`) so out-of-scope
  changes never leak in. The sequence is **all-or-nothing**: files are re-checked as
  still-changed at write time, and a mid-sequence failure soft-resets HEAD back to
  where it started (working-tree changes preserved — git is the undo, no backup
  bundle needed).
- Exit codes: **2** when not ready, **0** when ready (and, with `--write`, the commit
  succeeded), **3** when ready but the commit failed and was rolled back.

Approval is just re-running with `--write`: review the dry-run table, then commit.

## 5. digest — summarise a corpus into one document (reuses `run`)

`digest` is not a separate verb — it is `run` pointed at a **digest recipe** instead of an
investigation recipe. The same 6-stage pipeline (decompose → fan-out → … → assemble) then performs
**faithful, lossy compression** of a corpus of M items into one structured digest, rather than a
root-cause investigation. What flips the behaviour is the recipe: a digest recipe carries an
`ASSEMBLE SYSTEM OVERRIDE` section (a fenced block) whose text replaces the default investigation
assembler with a digest assembler. Investigation recipes omit that section and are unaffected.

```powershell
# 1. Stage the corpus: put each item (one markdown file per item) in a folder.
#    digest has no --corpus flag yet, so --codebase points at that folder.
# 2. Run with the digest recipe:
python hive.py run `
  --seed     <seed.md> `              # say "this is a DIGEST run; each file in the root is one item"
  --recipe   recipes\recipe_digest.md `
  --codebase <corpus_folder> `        # the staged folder of items
  --out      <digest.md> `
  --round-cap 1
```

The digest recipe embeds one **profile** (the scenario test-run profile) that fixes the four domain
seams — grouping key, sub-digest schema, merge granularity, output shape. To digest a different
kind of corpus, swap the `ACTIVE PROFILE` section; the engine never changes.

Output is a digest document: per-domain roll-up tables, an overall coverage map (every item → group,
proving zero orphans), an open-questions/contradictions block, and a metadata line. No fixes, no
recommendations — only faithful compression.

## Tests

```powershell
python -m pytest -q
```

## Layout

```
hive/
  decompose.py  fanout.py  parse.py  conflict_scan.py  reconcile.py  assemble.py
  specify.py    apply.py    backup.py
  commit.py
  config.py     ledger.py   providers.py
recipes/
  recipe_code_bug.md           # investigation recipe card (find a root cause)
  recipe_code_feature.md       # creation recipe card (build a new feature → create_file edits)
  recipe_digest.md             # digest recipe card (faithful compression of a corpus) + ASSEMBLE SYSTEM OVERRIDE
  edit_spec_contract_v1.md     # specify's output contract (also the author's role prompt)
  commit_plan_contract_v1.md   # commit-plan's output contract (also the author's role prompt)
tests/
hive.py                        # CLI entry point (run / specify / apply / commit-plan / commit / restore)
hive.config.json               # per-role provider/model config
```
