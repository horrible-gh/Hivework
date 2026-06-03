# Recipe — Code-Feature Creation Card (v0 prototype)

> Per M003 §6: "A recipe = one page of domain build-time knowledge." A single card holds the
> entrance (how to cut), the loop (how long to re-investigate), and the exit (how to assemble).
> At runtime, code + free workers execute exactly per this card.
> Sibling to recipe_code_bug.md. That card drives INVESTIGATION (find a root cause). This card
> drives CREATION (build a small new artifact). The pipeline shape is identical — decompose →
> fan-out → conflict → reconcile → assemble — but the semantics of each cut axis and each
> conflict differ.

## ① Entrance — fixed cut axes for BUILDING a new feature

The **first task in `steps[0]` is always the contract axis** — a single worker, one session,
whose sole output is the shared interface: function/class signatures, exported types, naming
conventions, and any cross-cutting constants the feature introduces (no implementation). All
implementation axes list this task in `depends_on` and run in `steps[1]` (parallel). This
ordering is non-negotiable: it is what stops independent implementation workers from diverging
on naming and call signatures before the work is reconciled.

Always **add/reinforce the following** on top of the raw decompose output (A–G):

| Axis | Step | What | Why (not inferable from the raw seed) |
|---|---|---|---|
| **Contract / interface definition** | `steps[0]` — single worker, no parallelism | Define the public interface: exported names, parameter names and types, return types, error conventions, shared constants. Produce a ~1-page interface spec, no implementation. | Without a fixed contract written first, every parallel implementation worker invents its own names and signatures; integration collisions become unavoidable rather than exceptional. |
| **Wiring / registration** | `steps[1]` alongside impl axes | Identify every anchor point where the new file must be wired in: imports in existing files, route registrations, DI entries, index re-exports, migration references. | The registration pattern (routes, adapters, DI) is a domain convention. A worker who sees only one impl axis will not know where to hook the new file into the existing graph. |
| **Design SSOT alignment** | `steps[1]` alongside impl axes | Grep `Documents/projects/<proj>/` (R/D/L/P/DB/M) for the feature's intended spec; confirm axis scope matches recorded intent and surface open design questions before implementation. | The feature may be partially specified in an existing D/R/M doc. Catching spec-vs-impl drift at decompose time is cheaper than reconciling it post-assembly. |

The remaining A–G axes (per domain) are assigned to `steps[1]`, each with
`depends_on: ["<contract-axis-id>"]`.

## ② Loop — integration-conflict re-investigation cap + termination classification

- Round cap: **2**. New-feature work is structurally predictable; if collisions are still
  unresolved after 2 reconcile rounds, classify as a hard boundary and surface to PM.
- **Conflict-detection trigger:** an **integration mismatch** between two or more combs —
  signature disagreement, naming collision, duplicate definition, or import-path collision for
  the new file. Distinct from a bug-card conflict: there is no root-cause signal to compare; the
  trigger is a structural clash in the assembled code.
- **Required fields in the re-integration brief:** the exact `file:line` of each conflicting
  definition or call site, the axis that produced each side, and a single narrowed question of
  the form "which signature / name / path should win, and what must change in the other axis to
  conform." Reference the contract-axis output as the tiebreaker — if a conflict contradicts the
  contract, the implementation axis changes, not the contract.
- Termination classes: converged / needs_design (ambiguity in the original spec needs a D/R doc
  update). There is NO "ask a human" class: a business judgment on scope or naming is surfaced by
  RESOLVING with the best-supported proposal, not by punting to a person. A hard boundary is
  emitted together with the evidence for why it cannot be closed in-loop.

## ③ Exit — honey output shape (code feature = **edit-spec-centric**)

The assembled honey for a creation task lowers into an **edit-spec** ready to feed
`specify → apply`. The predominant operation is `create_file`; anchor edits (`kind: "edit"`, or
`kind` absent) are present but secondary (wiring the new file in). Edit-kind shapes are defined
in `edit_spec_contract_v1.md` — use `"create_file"` for new files, `"edit"`/absent for anchored
modifications. Never use `"edit_file"` (not a recognised value).

Fixed exit skeleton:
1. **Header block** — feature ID / date / author / status / creation-only flag.
2. **Numbered sections = cut axes, 1:1** — what each axis worker found or produced.
3. **Contract summary block** — the final agreed interface (names, signatures, types, error
   conventions). The single source of truth for all downstream edits; reproduced verbatim here.
4. **Edit-spec table** — ordered file operations. `create_file` entries come first (no prior
   content to diff); anchor entries follow, each with a minimal context snippet locating the
   insertion point.
5. **Integration check list** — one row per conflict reconciled in-loop, with the winning
   resolution noted.
6. **Design SSOT delta** — if the work revealed a feature-spec vs D/R/M gap, list the doc IDs
   needing updates (out of scope for apply; flagged for PM/author).
7. **Dependency note** — any migration or dependency install not captured in the edit-spec,
   called out explicitly so `apply` does not silently omit it.
