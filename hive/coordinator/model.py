"""Coordinator data model + tuning constants (single source of truth).

Every numeric threshold here is a DEFERRED placeholder per the matching L-doc §1
(L-01/L-02/L-03): provisional, to be calibrated against the M035/M036/M037
ground-truth corpus once the core 3 cases are stable. Keep them in THIS module so
there is exactly one place to tune (each L-doc names this its numeric authority).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# ── DEFERRED numeric thresholds (provisional; corpus-calibrate later) ──────────
TAU_EMIT = 0.5         # L-01 §1: min confidence to emit an expected axis
MAX_AXES = 3           # L-01 §1: max expected axes per symptom
TAU_SALIENCE = 0.4     # L-02 §1: min salience to keep a slot candidate
MAX_SLOTS = 7          # L-02 §1: max slot candidates per message
TAU_LB = 0.5           # L-03 §1: min outcome-change to promote to load-bearing
ROUNDS_CAP = 4         # L-03 §1: caps.rounds_left initial (interactive loop = W2)
RETRY_MAX = 1          # L-01/L-02 §1: structured-output retry budget

# score weights (L-01 §2.3 / L-02 §2.3) — grounding-dominant so a hallucinated
# observed/slot (grounding==0) cannot clear its threshold.
W_G, W_S, W_P = 0.6, 0.25, 0.15        # expected: grounding / specificity / support
WD_G, WD_O, WD_U = 0.45, 0.30, 0.25    # decompose: grounding / outcome / unsaid

# ── L-06 §1: need_ eradication policy + terminal-set invariant ────────────────
BANNED_PREFIXES = ("need_", "needs_")
BANNED_SEMANTICS = ("awaiting_", "blocked_", "punt", "abstain")
ALLOWED_STATUS_VOCAB = ("open", "answered", "skipped")
ALLOWED_PROVENANCE_KEYS = ("interactive", "sealed", "skipped_slots")
TERMINAL_SET = ("HANDOFF",)

# L-02 §1: dedicated reasoning fork id (shares plumbing with the queen, NOT the
# reasoning prompt/profile — CON-2).
FORK_PROFILE = "coordinator_decompose"


@dataclass
class Gap:
    """A load-bearing information gap (P-01 §2.2). In a gap_state, every Gap is
    load_bearing (P-01 §8); non-load-bearing candidates are dropped at the L-03
    gate before promotion."""
    id: str
    slot: str
    load_bearing: bool = True
    expected_carveout: bool = False     # expected => always free-form (FR-4)
    kind: str = "context"               # expected|context|scope|constraint|env
    question: str = ""                  # surface question (filled in W2 / D-02)
    format: str = "free"                # "mc"|"free"; W1 is always "free"
    options: list[str] = field(default_factory=list)
    escape: bool = True                 # always an explicit "don't know" (FR-4)
    status: str = "open"                # open|answered|skipped
    answer: str | None = None
    salience: float = 0.0
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Gap":
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Turn:
    seq: int
    role: str              # coordinator|user|caller
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GapState:
    """Runtime shape of a coordinator session (P-01 §2.1). Persistence/TTL is
    DB-01 (gapstate.py); this is the in-memory form."""
    uuid: str
    track: str = "api"                  # interactive|api
    interactive_flag: bool = False
    symptom_raw: str = ""
    expected: list[dict] = field(default_factory=list)   # L-01 output
    transcript: list[Turn] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    seed_base: str = ""                 # the original seed the section attaches to
    seed_draft: dict = field(default_factory=lambda: {"caller_supplied_context": ""})
    # budget_left None = unbounded here; a real cost cap is W2/D-03 tuning.
    caps: dict = field(default_factory=lambda: {"rounds_left": ROUNDS_CAP,
                                                "budget_left": None})
    status: str = "collecting"          # collecting|ready|sealed
    expertise: float = 0.0              # L-04 E position (W2; carried fwd-compat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "track": self.track,
            "interactive_flag": self.interactive_flag,
            "symptom_raw": self.symptom_raw,
            "expected": self.expected,
            "transcript": [t.to_dict() for t in self.transcript],
            "gaps": [g.to_dict() for g in self.gaps],
            "seed_base": self.seed_base,
            "seed_draft": self.seed_draft,
            "caps": self.caps,
            "status": self.status,
            "expertise": self.expertise,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GapState":
        gs = cls(uuid=d["uuid"])
        gs.track = d.get("track", "api")
        gs.interactive_flag = bool(d.get("interactive_flag", False))
        gs.symptom_raw = d.get("symptom_raw", "")
        gs.expected = list(d.get("expected", []))
        gs.transcript = [Turn(**t) for t in d.get("transcript", [])]
        gs.gaps = [Gap.from_dict(g) for g in d.get("gaps", [])]
        gs.seed_base = d.get("seed_base", "")
        gs.seed_draft = d.get("seed_draft", {"caller_supplied_context": ""})
        gs.caps = d.get("caps", {"rounds_left": ROUNDS_CAP, "budget_left": None})
        gs.status = d.get("status", "collecting")
        gs.expertise = float(d.get("expertise", 0.0))
        return gs
