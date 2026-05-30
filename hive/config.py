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
    },
    "copilot": {"exe": None, "allow": "--allow-all", "timeout_sec": 300},
    "ledger":  {"enabled": True, "db_path": "hive_ledger.db"},
}

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "hive.config.json")


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
class Config:
    queen: RoleConfig = field(default_factory=RoleConfig)
    swarm: RoleConfig = field(default_factory=RoleConfig)
    assemble_role: RoleConfig = field(default_factory=RoleConfig)
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)

    def role(self, name: str) -> RoleConfig:
        """Return the RoleConfig for a given role name ('queen', 'swarm', 'assemble')."""
        if name == "assemble":
            return self.assemble_role
        return getattr(self, name, RoleConfig())

    def apply_cli_model(self, model: str | None) -> None:
        """Apply a CLI --model override to all roles (preserves --model semantics)."""
        if model is None:
            return
        for role in (self.queen, self.swarm, self.assemble_role):
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

    def _role(name: str) -> RoleConfig:
        r = roles.get(name, {})
        return RoleConfig(provider=r.get("provider", "copilot"), model=r.get("model", "gpt-5-mini"))

    return Config(
        queen=_role("queen"),
        swarm=_role("swarm"),
        assemble_role=_role("assemble"),
        copilot=CopilotConfig(
            exe=copilot_raw.get("exe"),
            allow=copilot_raw.get("allow", "--allow-all"),
            timeout_sec=int(copilot_raw.get("timeout_sec", 300)),
        ),
        ledger=LedgerConfig(
            enabled=bool(ledger_raw.get("enabled", True)),
            db_path=ledger_raw.get("db_path", "hive_ledger.db"),
        ),
    )
