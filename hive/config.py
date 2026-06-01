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
    "judge":   {"max_calls_per_axis": 2, "max_axes": 3, "max_parallel": 2},
}

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "hive.config.json")
# Hivework repo root — relative config paths (e.g. apply.backup_dir) resolve here.
_REPO_ROOT = os.path.dirname(os.path.abspath(_DEFAULT_CONFIG_PATH))


@dataclass
class RoleConfig:
    provider: str = "copilot"
    model: str = "gpt-5-mini"


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
class JudgeConfig:
    """Cost caps for the JUDGE-directed follow-up loop (M004 §4 budget).

    The retriever's follow-up mechanism is built for ≤1 re-search per axis
    (``max_calls_per_axis`` model calls: 1 judge + ≤1 re-judge). ``max_axes``
    bounds how many axes a single run will judge, and ``max_parallel`` caps
    concurrent judge calls — surfacing the whole budget here so spend is
    controllable rather than hidden in a counter.
    """
    max_calls_per_axis: int = 2
    max_axes: int = 3
    max_parallel: int = 2


@dataclass
class Config:
    queen: RoleConfig = field(default_factory=RoleConfig)
    swarm: RoleConfig = field(default_factory=RoleConfig)
    assemble_role: RoleConfig = field(default_factory=RoleConfig)
    specify: RoleConfig = field(default_factory=RoleConfig)
    commit: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-haiku-4.5"))
    judge_role: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def role(self, name: str) -> RoleConfig:
        """Return the RoleConfig for a role ('queen', 'swarm', 'assemble', 'specify', 'commit', 'judge')."""
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
                     self.commit, self.judge_role):
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

    def _role(name: str, default_model: str = "gpt-5-mini") -> RoleConfig:
        r = roles.get(name, {})
        return RoleConfig(provider=r.get("provider", "copilot"),
                          model=r.get("model", default_model))

    return Config(
        queen=_role("queen"),
        swarm=_role("swarm"),
        assemble_role=_role("assemble"),
        specify=_role("specify"),
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
            max_axes=int(judge_raw.get("max_axes", 3)),
            max_parallel=int(judge_raw.get("max_parallel", 2)),
        ),
    )
