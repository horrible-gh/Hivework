# Recipe — Code-Bug Investigation Card (FlowGate domain · v0 prototype)

> Per M003 §6: "A recipe = one page of domain build-time knowledge." A single card holds the entrance (how to cut), the loop (how long to re-investigate), and the exit (how to assemble). At runtime, code + free workers execute exactly per this card.
> Origin: in full-loop smoke #001 the honey was thinner than NR150 (Opus). Of the three causes, this card reinforces ② axis coverage and ③ per-axis depth.

## ① Entrance — fixed cut axes (including domain-convention axes that cannot be inferred from the raw seed)

Always **add/reinforce the following** on top of the raw decompose output (A–G):

| Axis | What | Why (not inferable from the raw seed) |
|---|---|---|
| **Design SSOT grep** | Grep the design-basis docs under `Documents/projects/<proj>/` (M0xx memos / D0xx designs / DB0xx / R0xx requirements / NR0xx prior investigations) for "the intended spec of this behavior." Not "the code does X so it's a bug" but "the design intended X, the code does Y" — a contrast. | FlowGate's doc-numbering system (R/D/L/P/DB/M/N/T) is a domain convention. This is the very axis found missing in M002. |
| **Head-resolver / SQL gate** | The branches, loops, early-returns, and **SQL WHERE conditions** that decide "is this code block actually reached." Especially the head/lookup query conditions in `server/sql/queries/*.json`. | Chicken-and-egg gates (e.g. `result_doc_id IS NOT NULL` blocking the first insert) are invisible if you only check that code "exists." This was the identity of NR150 Gap A. |
| **Regression blame** | Use `git log` / `git blame` to pin the recent change / introducing commit for the lines tied to the symptom. "Since when did it break / is it already fixed." | From current code alone you cannot tell "regression" from "always was this way." |

The remaining A–G (FE badge · submit UI · BE endpoint · async pipeline · DB schema · global search · logs) are used as-is from the raw decompose output.

## ② Loop — conflict re-investigation cap + termination classification

- Round cap: **2**. (Code bugs usually converge in 1 round; if still unclosed after 2, classify as a hard boundary.)
- **Conflict-detection trigger:** `root_cause_signal` mismatch between combs OR `termination` divergence (one resolved, another unresolved, etc.).
- **Required fields in the re-investigation brief:** cite the file:line evidence of both conflicting conclusions, plus a narrowed question covering "only what is needed to resolve this contradiction." **Always ask about execution reachability** ("does that code actually run") — not merely "does the code exist."
- Termination classes: converged / needs_runtime (execution logs required) / needs_external (external authority). There is NO "ask a human" class: a business/scope judgment is surfaced by RESOLVING with the best-supported proposal, not by punting to a person. A hard boundary is emitted "together with the evidence for why it cannot be closed."

## ③ Exit — honey output shape (code bug = **trace-centric**)

Fixed cursor-NR skeleton + code-domain switches:
1. Header block (ID / date / investigator / status / investigation-only).
2. Numbered sections = cut axes, 1:1.
3. **Call-chain code-block traces** (`file:line` → `→` arrows). ← required for the code domain; absent in the legal domain.
4. **Gap analysis table** (each Gap = file:line + issue + regression commit).
5. **Design SSOT comparison table** (design ID §section → intent vs actual code).
6. Source table (conclusion row → axis / drone / file:line) + conflict-convergence results.
7. Repro hypothesis (minimal reproduction sequence) + Fix-direction options A/B/C + recommendation.
