# Recipe — Digest Engine Card (general engine · v1)

> **Identity:** `digest` is Hivework's domain-agnostic faithful-compression mode — a peer to `run`
> (investigate) and `specify`/`apply` (fix), not a helper bolted onto any external workflow.
> The ENGINE below is fixed and general; everything domain-specific lives in the ACTIVE PROFILE
> section (embedded here because digest currently reuses `run`, which has no `--profile` flag yet).
> "scenario digest" is one profile of this engine, never its definition — swap the ACTIVE PROFILE
> to digest a different kind of corpus.

> The pipeline shape is the same as every other recipe — decompose → fan-out → conflict →
> reconcile → assemble — but the semantics of each stage are **faithful, lossy compression**, not
> root-cause investigation. This is a DIGEST run: compress the corpus, do not investigate code.

---

## ASSEMBLE SYSTEM OVERRIDE

```text
# ROLE: DIGEST ASSEMBLER — automated pipeline final stage (faithful compression, NOT investigation)

You are the ASSEMBLER in a Hivework DIGEST run. You receive:
1. All sub-digest comb results (per-group worker outputs) as JSON
2. Reconcile round results (if any)
3. The recipe output-shape rules (provided after this prompt)

Your ONLY job: synthesise the sub-digests into ONE faithful DIGEST document — a lossy compression
of the corpus, NOT a root-cause investigation.

## Output — follow the recipe output-shape rules EXACTLY. Produce:
- A digest header (corpus name, date, items M, groups K, termination) — NOT an investigation header.
- Per-domain roll-up sections (one per group/domain), each a roll-up TABLE of items with their
  result + a direct evidence pointer (source item id : the line that establishes it).
- An overall COVERAGE MAP table: every source item id -> its domain/group -> included? — this proves
  zero orphans and is mandatory.
- An open-questions / gaps / contradictions block.
- A single digest metadata line: date - groups - items covered M/total - open items.

## Hard rules (non-negotiable):
- FAITHFUL LOSSY COMPRESSION ONLY. Do NOT analyse, diagnose, hypothesise, or recommend.
- FORBIDDEN (these belong to investigation, not digest): a root-cause "key conclusion"; call-chain
  arrow (->) traces; "Gap analysis" framed as investigation; "Design SSOT comparison"; "Repro
  hypothesis"; "Fix options A/B/C"; any recommendation or fix direction; investigation header fields
  (investigation-id / investigator / status: investigation-only).
- NO new facts. Every claim must cite a source item id present in the combs.
- Surface gaps and contradictions explicitly; never smooth them away.
- Output pure markdown. No JSON wrapping.
```

---

## ① Entrance — cut (partition the corpus into groups)

Partition the **M corpus items** into **K groups** using the ACTIVE PROFILE's grouping key. Each
group is assigned to one worker, which produces a **sub-digest** covering exactly its items.

**Coverage invariant (non-negotiable):** every input item lands in **exactly one group** — no
orphan item, no item in two groups. Emit a numbered coverage manifest (every item ID → its group)
in the decompose output before any worker runs.

| Axis | What | Why |
|---|---|---|
| **Grouping key** | Partition by the ACTIVE PROFILE key. When silent, default to 5–7 items per group by natural locality. | The faithful unit of compression is domain knowledge a generic cut cannot know. |
| **Coverage manifest** | List every input item ID with its assigned group before workers run. | Audit trail for the coverage invariant; makes gaps detectable at a glance. |
| **Sub-digest schema** | Each worker compresses its group per the ACTIVE PROFILE schema. | A fixed schema is the contract between decompose and assemble; without it sub-digests are incompatible. |

---

## ② Loop — reconcile (structural conflict, gap-fill, contradiction surfacing)

Digest has **no root-cause signal**. Conflict triggers are structural/coverage-based:

- **Round cap: 1–2.** If not closed in two rounds, hard boundary → surface to PM.
- **Triggers:** (a) two sub-digests make contradictory claims about the same item; (b) duplicate
  coverage (item in two groups); (c) coverage gap (item digested by no group).
- **Reconcile:** dedup (assign the item to exactly one group, re-digest the other), gap-fill
  (assign orphan to nearest group), contradiction-surfacing (record the disagreement explicitly —
  never silently pick a winner).
- **Termination classes:** `converged` / `needs_source` (item unreadable/ambiguous). There is no
  "ask a human" class — a scope judgment is surfaced by converging on the best-supported call.

---

## ③ Exit — assemble (the digest document)

**Hard rules (non-negotiable):**

| Rule | Statement |
|---|---|
| Faithful lossy compression | The digest reduces the corpus; it does not extend it. |
| No new facts | Nothing may appear that is not traceable to a source item. |
| Every claim traces to a source item ID | Each statement cites the source item(s), e.g. `[#TR852]`. |
| Gaps & contradictions surfaced | Open questions, contradictions, `needs_source` appear in a dedicated end block — never smoothed away. |

The output shape is supplied by the ACTIVE PROFILE.

---

# ACTIVE PROFILE — Scenario Test-Run Digest

> One profile of the engine above. It specialises the four engine seams for a batch of
> test-scenario / task reports. Each corpus file (one report per file, e.g. `TR####_*.md`) is one
> item; its item ID is its report number. To digest a different corpus, replace this whole ACTIVE
> PROFILE section; the engine above never changes.

## (1) Grouping key — ① Entrance

| Axis | What | Why |
|---|---|---|
| **Grouping key** | Group items **by component / subsystem under test**; 5–7 items per group; merge the smallest residual into the nearest functional neighbour to avoid orphans. | Component is the domain unit for a test batch; a count-only cut mixes unrelated subsystems. |
| **Ambiguous tag** | If an item's component is absent/ambiguous, assign to the group whose scope best matches its subject; append `(reassigned)` in the group header. | Silent mis-assignment violates the coverage invariant; annotation keeps it auditable. |

Coverage invariant: every item id lands in exactly one group. No orphans.

## (2) Sub-digest schema — ② Fan-out (each worker emits one sub-digest, exactly three sections)

### 2-A Pass / Fail table
| Item ID | Result | One-line evidence |
|---|---|---|
| TR### | pass / fail / blocked | `<the line in the report that establishes the result>` |

- Every item ID in the group must appear; no silent omissions.
- `blocked` = could not determine result from the report; pair with a reason.
- Evidence is a direct pointer/quote, not a paraphrase.

### 2-B Regression points
Bullet list of previously-passing behaviours now at risk. Each: behaviour at risk → the item that
revealed it → component boundary crossed. If none: `(none identified in this group)`.

### 2-C Remaining items
Bullet list with a disposition tag: `[untested]` / `[blocked]` / `[needs-follow-up]`.

## (3) Merge granularity — ③ Assemble
Synthesise sub-digests into **scenario-domain units** (one section per component cluster; isolated
groups keep their own unit). Engine hard rules apply in full. Every pass/fail row retains its item
ID — no anonymous aggregation. Surface gaps: `[gap: <component area> — no coverage]`.

## (4) Output shape — final digest, four blocks in order
1. **Per-domain roll-up** — one `### Domain:` section per unit, with a roll-up table (Item ID /
   Component / Result / Evidence pointer), plus aggregated regression points and remaining items.
2. **Overall coverage map** — table of every item id → domain unit → result; confirms zero orphans.
3. **Open questions / Regressions block** — union of all regression points + remaining items,
   de-duplicated, sorted by severity (fail > blocked > ambiguous).
4. **Digest metadata line** —
   `Digest date: <date> · Groups: <N> · Items covered: <M>/<total> · Open items: <K>`

## Success criteria
- Coverage map has exactly as many rows as items in scope; no id repeated or missing.
- Every group has 5–7 items (±1 only for the remainder group, documented).
- Regression block non-empty if any item failed.
- No unresolved `[contradiction]` left in domain units.
