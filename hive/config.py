"""Config layer — per-role provider/model configuration for Hivework.

Loads hive.config.json from repo root (or a specified path). Falls back to
built-in defaults when the file is absent or a key is missing. Defaults
reproduce today's behavior exactly: all roles use provider=copilot, model=gpt-5-mini.

Precedence: explicit CLI --model overrides all role models.
"""
import json, logging, os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hive.config")

_DEFAULTS: dict[str, Any] = {
    "roles": {
        "queen":    {"provider": "copilot", "model": "gpt-5-mini"},
        "swarm":    {"provider": "copilot", "model": "gpt-5-mini"},
        "assemble": {"provider": "copilot", "model": "gpt-5-mini"},
        "specify":  {"provider": "copilot", "model": "gpt-5-mini"},
        # Effectiveness reviewer (specify's second pass). A tool-OFF single-shot
        # judgement — the edit diff (anchor_old→replacement_new) and, since the
        # anchor-grounding pre-flight, the current values are all in the prompt —
        # so it is a prime candidate to move OFF the per-internal-turn-billed
        # copilot onto deepinfra (mirrors judge). Default mirrors specify (copilot)
        # so behaviour is unchanged until hive.config.json opts into deepinfra.
        "review":   {"provider": "copilot", "model": "gpt-5-mini"},
        # Commit author defaults to haiku (a tier up): grouping changes into clean
        # atomic commits wants more judgement than the gpt-5-mini swarm default.
        "commit":   {"provider": "copilot", "model": "claude-haiku-4.5"},
        # The JUDGE re-search reviewer (M004 §4). Copilot's ceiling is sonnet
        # (opus unavailable), so the quality-critical judge sits at sonnet.
        "judge":    {"provider": "copilot", "model": "claude-sonnet-4.5"},
    },
    "copilot": {"exe": None, "allow": "--allow-all", "timeout_sec": 300},
    "ledger":  {"enabled": True, "db_path": "hive_ledger.db"},
    "apply":   {"backup_dir": ".apply_backups", "backup_ttl_hours": 168},
    # Cost caps for the JUDGE-directed follow-up loop. The retriever is built
    # for ≤1 re-search per axis (≤2 model calls: 1 judge + ≤1 re-judge); these
    # knobs surface that budget in config so spend is controllable, not hidden.
    # max_axes is a RUNAWAY-CEILING, not an aggressive cap: with a cheap judge
    # provider, judging all leaf axes costs cents, so we judge every leaf up to
    # this ceiling. A low cap (was 3) silently dropped decisive grep-once axes
    # past position N (N164: css_rules cut → specify ran blind → self-reversal).
    "judge":   {"max_calls_per_axis": 2, "max_axes": 12, "max_parallel": 2},
    # Cost guard-rails. allow_swarm=false makes `hive.py run` refuse to launch the
    # open-ended swarm (fan-out + reconcile) and point at the cheap investigate path.
    "safety":  {"allow_swarm": True},
}

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "hive.config.json")
# Hivework repo root — relative config paths (e.g. apply.backup_dir) resolve here.
_REPO_ROOT = os.path.dirname(os.path.abspath(_DEFAULT_CONFIG_PATH))


@dataclass
class RoleConfig:
    provider: str = "copilot"
    model: str = "gpt-5-mini"
    # Optional per-role worker overrides. ``timeout_sec=None`` means the caller's
    # built-in default applies (e.g. run_specify's 600s author cap). ``retries`` is
    # how many EXTRA attempts a transient worker failure (timeout / non-zero exit /
    # empty output) gets before the stage gives up — 0 preserves single-shot
    # behavior. Surfaced per-role because a slow agentic CLI (codex specify author)
    # needs a different cap than a fast tool-OFF API call (T892 timeout).
    timeout_sec: int | None = None
    retries: int = 0


@dataclass
class CopilotConfig:
    exe: str | None = None
    allow: str = "--allow-all"
    timeout_sec: int = 300


@dataclass
class LedgerConfig:
    enabled: bool = True
    db_path: str = "hive_ledger.db"


@dataclass
class ApplyConfig:
    """Settings for ``apply --write`` — the scratch backup / undo window."""
    backup_dir: str = ".apply_backups"
    backup_ttl_hours: int = 168

    def backup_root(self) -> str:
        """Absolute backup root; a relative backup_dir resolves under the repo root."""
        if os.path.isabs(self.backup_dir):
            return self.backup_dir
        return os.path.join(_REPO_ROOT, self.backup_dir)


@dataclass
class SafetyConfig:
    """Cost guard-rails enforced by the CLI before any spend.

    ``allow_swarm`` gates the open-ended swarm ``run`` path (fan-out drones +
    reconcile re-investigation). When false, ``hive.py run`` refuses to start and
    points the operator at the cheap ``investigate`` path — a config kill-switch
    against accidentally launching the multi-worker, per-internal-turn-billed
    swarm. Default true preserves today's behavior; set false in
    ``hive.config.json`` to lock the swarm off unless deliberately re-enabled.
    """
    allow_swarm: bool = True


@dataclass
class JudgeConfig:
    """Cost caps for the JUDGE-directed follow-up loop (M004 §4 budget).

    The retriever's follow-up mechanism is built for ≤1 re-search per axis
    (``max_calls_per_axis`` model calls: 1 judge + ≤1 re-judge). ``max_axes``
    is a runaway-ceiling on judged leaf axes (not an aggressive cap): a cheap
    judge provider makes judging every leaf cost cents, so the ceiling is high
    enough to cover a normal decompose (5–10 leaves) and only guards against a
    pathological axis explosion. ``max_parallel`` caps concurrent judge calls —
    surfacing the whole budget here so spend is controllable rather than hidden.
    """
    max_calls_per_axis: int = 2
    max_axes: int = 12
    max_parallel: int = 2


@dataclass
class Config:
    queen: RoleConfig = field(default_factory=RoleConfig)
    swarm: RoleConfig = field(default_factory=RoleConfig)
    assemble_role: RoleConfig = field(default_factory=RoleConfig)
    specify: RoleConfig = field(default_factory=RoleConfig)
    review: RoleConfig = field(default_factory=RoleConfig)
    commit: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-haiku-4.5"))
    judge_role: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def role(self, name: str) -> RoleConfig:
        """Return the RoleConfig for a role ('queen', 'swarm', 'assemble', 'specify', 'review', 'commit', 'judge')."""
        if name == "assemble":
            return self.assemble_role
        if name == "judge":
            return self.judge_role
        return getattr(self, name, RoleConfig())

    def apply_cli_model(self, model: str | None) -> None:
        """Apply a CLI --model override to all roles (preserves --model semantics)."""
        if model is None:
            return
        for role in (self.queen, self.swarm, self.assemble_role, self.specify,
                     self.review, self.commit, self.judge_role):
            role.model = model


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | None = None) -> Config:
    """Load config from JSON file, falling back to built-in defaults for missing keys."""
    resolved_path = path or _DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            logger.debug("Loaded config from %s", resolved_path)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load config from %s: %s — using defaults", resolved_path, e)
            raw = {}
    else:
        logger.debug("Config file not found at %s — using defaults", resolved_path)

    merged = _deep_merge(_DEFAULTS, raw)
    roles = merged.get("roles", {})
    copilot_raw = merged.get("copilot", {})
    ledger_raw = merged.get("ledger", {})
    apply_raw = merged.get("apply", {})
    judge_raw = merged.get("judge", {})
    safety_raw = merged.get("safety", {})

    def _role(name: str, default_model: str = "gpt-5-mini") -> RoleConfig:
        r = roles.get(name, {})
        timeout = r.get("timeout_sec")
        return RoleConfig(provider=r.get("provider", "copilot"),
                          model=r.get("model", default_model),
                          timeout_sec=int(timeout) if timeout is not None else None,
                          retries=int(r.get("retries", 0)))

    return Config(
        queen=_role("queen"),
        swarm=_role("swarm"),
        assemble_role=_role("assemble"),
        specify=_role("specify"),
        review=_role("review"),
        commit=_role("commit", default_model="claude-haiku-4.5"),
        judge_role=_role("judge", default_model="claude-sonnet-4.5"),
        copilot=CopilotConfig(
            exe=copilot_raw.get("exe"),
            allow=copilot_raw.get("allow", "--allow-all"),
            timeout_sec=int(copilot_raw.get("timeout_sec", 300)),
        ),
        ledger=LedgerConfig(
            enabled=bool(ledger_raw.get("enabled", True)),
            db_path=ledger_raw.get("db_path", "hive_ledger.db"),
        ),
        apply=ApplyConfig(
            backup_dir=apply_raw.get("backup_dir", ".apply_backups"),
            backup_ttl_hours=int(apply_raw.get("backup_ttl_hours", 168)),
        ),
        judge=JudgeConfig(
            max_calls_per_axis=int(judge_raw.get("max_calls_per_axis", 2)),
            max_axes=int(judge_raw.get("max_axes", 12)),
            max_parallel=int(judge_raw.get("max_parallel", 2)),
        ),
        safety=SafetyConfig(
            allow_swarm=bool(safety_raw.get("allow_swarm", True)),
        ),
    )
