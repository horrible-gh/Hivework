"""Coordinator — the queen's front-end interpreter (R0001, group 0003).

Attaches BEFORE decompose: turns a user's casual prose into a refined seed by
extracting the true ``expected`` (L-01) and structuring the symptom/context into
the ``## Caller-supplied context (requester's direct input)`` section the
pipeline already reads (P-03 handoff). Opt-in via ``hive run --coordinator``;
absent, the pipeline is byte-for-byte unchanged.

Scope = M0002 W0 + W1 (the minimal, NON-interactive 1-shot skeleton):
  L-01 expected extraction · L-02 context decompose (dedicated fork) ·
  L-03 load-bearing gate + ready() · L-06 best-effort termination · DB-01
  gap-state persistence · P-03 handoff section.

OUT (W2, follow-up T): the interactive ping-pong / questions[] round-trip
(P-01/P-02), MC forms + manifest (D-02/L-05), adaptive expertise (L-04). Because
P-01 is W2, the W1 coordinator never asks the user back — it best-effort seals on
a single pass (M0002 §6 W1 definition).

Design bundle: hivework.default.0001 (0018-AC final-approved).
"""
from hive.coordinator.engine import run_coordinator

__all__ = ["run_coordinator"]
