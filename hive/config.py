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
        # The CONVERGER (hive.converge): a single tool-OFF call that stitches the
        # per-axis verdicts into one executed call path and attributes the defect
        # to one node. Same cost class / shape as judge, so it defaults to the same
        # provider/model; hive.config.json routes it to deepinfra like judge.
        "converge": {"provider": "copilot", "model": "claude-sonnet-4.5"},
    },
    "copilot": {"exe": None, "allow": "--allow-all", "timeout_sec": 300},
    # OpenAI-compatible HTTP endpoint for the 'openai'/'deepinfra' provider. These
    # are the SAME backend; the defaults below are the DeepInfra preset (kept for
    # back-compat). Point at any other vendor (OpenAI proper, a self-hosted vLLM, …)
    # by overriding base_url + api_key_env in hive.config.json — no code change.
    "openai":  {"base_url": "https://api.deepinfra.com/v1/openai",
                "api_key_env": "DEEPINFRA_TOKEN"},
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
class OpenAiConfig:
    """OpenAI-compatible HTTP endpoint for the ``openai`` / ``deepinfra`` provider.

    Both provider names share one backend (hive/providers.py); this block is what
    makes it vendor-neutral. ``base_url`` is the chat-completions root and
    ``api_key_env`` is the NAME of the env var holding the key (the key itself lives
    in the out-of-repo secrets file, never here). Defaults are the DeepInfra preset
    so existing configs that only say ``"provider": "deepinfra"`` keep working; the
    setup wizard writes a chosen preset (DeepInfra, OpenAI, …) or a custom endpoint.
    """
    base_url: str = "https://api.deepinfra.com/v1/openai"
    api_key_env: str = "DEEPINFRA_TOKEN"


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
class CommitConfig:
    """Settings for the commit-plan author stage.

    ``filename_only_threshold`` caps credit spend on huge change sets (commonly
    documentation dumps of hundreds/thousands of files). When the number of changed
    paths EXCEEDS this value, the author is told to group by file PATH/NAME only and
    NOT to open file contents — grouping a thousand docs by reading each is the
    expensive case. At or below the threshold the author may open files to judge
    grouping. Set to 0 to disable filename-only mode entirely (always allow opening).
    """
    filename_only_threshold: int = 50


@dataclass
class DbConnection:
    """One target codebase's read-only DB connection (for the converge data-state read).

    NEUTRAL by design — no FlowGate (or any caller) semantics. ``kind`` dispatches
    the driver; the rest are standard connection coordinates. The converge data-read
    glue only ever issues SELECTs, so a connection is read-only BY CONSTRUCTION
    regardless of the account's grants; a SELECT-only DB account is recommended
    defence-in-depth, not a correctness requirement.

    - sqlite: only ``path`` (the ``.db`` FILE, not its folder) is used; opened with
      ``mode=ro`` so the file is never mutated. ``host``/``user``/``password`` ignored.
    - mysql / mariadb (kind ``"mysql"`` or ``"mariadb"`` — same wire driver) and
      postgres (kind ``"postgres"``): ``host``/``port``/``dbname``/``user`` plus a
      secret. The secret is ``password`` (raw — fine for a dev SELECT-only account)
      OR ``password_env`` (name of an env var holding it — keeps the secret out of a
      committed/open config). ``password_env`` wins when both are set.

    ``codebase`` optionally binds this entry to an explicit codebase path; when
    absent the entry is matched to a run by its key vs the ``--codebase`` leaf name
    (case-insensitive), so ``"flowgate"`` matches ``…/FlowGate``.
    """
    kind: str = "sqlite"
    path: str = ""
    host: str = ""
    port: int | None = None
    dbname: str = ""
    user: str = ""
    password: str = ""
    password_env: str = ""
    codebase: str = ""

    def secret(self) -> str:
        """Resolve the password: ``password_env`` (env lookup) wins, else raw ``password``."""
        if self.password_env:
            return os.environ.get(self.password_env, "")
        return self.password


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
    converge_role: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    openai: OpenAiConfig = field(default_factory=OpenAiConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    commit_stage: CommitConfig = field(default_factory=CommitConfig)
    # Per-codebase read-only DB connections, keyed by a short name (e.g. "flowgate").
    # Empty by default — the converge data-state read is SKIPPED when a run's codebase
    # has no entry (graceful: converge falls back to its static path / needs_data).
    db_connections: dict[str, DbConnection] = field(default_factory=dict)

    def db_for_codebase(self, codebase_root: str | None) -> DbConnection | None:
        """Resolve the DB connection for a run's ``--codebase`` path, or None.

        Match order: (1) an entry whose explicit ``codebase`` path-aligns with the
        run's codebase, then (2) an entry whose KEY equals the codebase's leaf folder
        name (case-insensitive) — so ``"flowgate"`` matches ``…/FlowGate``. Returns
        None when nothing matches (the common no-DB case), never raises.
        """
        if not codebase_root or not self.db_connections:
            return None
        norm = codebase_root.replace("\\", "/").rstrip("/").lower()
        leaf = norm.rsplit("/", 1)[-1]
        # (1) explicit codebase binding wins (handles key≠folder-name)
        for conn in self.db_connections.values():
            cb = (conn.codebase or "").replace("\\", "/").rstrip("/").lower()
            if cb and (cb == norm or norm.endswith("/" + cb) or cb.endswith("/" + norm)):
                return conn
        # (2) key vs codebase leaf name
        for key, conn in self.db_connections.items():
            if key.strip().lower() == leaf:
                return conn
        return None

    def role(self, name: str) -> RoleConfig:
        """Return the RoleConfig for a role ('queen', 'swarm', 'assemble', 'specify', 'review', 'commit', 'judge')."""
        if name == "assemble":
            return self.assemble_role
        if name == "judge":
            return self.judge_role
        if name == "converge":
            return self.converge_role
        return getattr(self, name, RoleConfig())

    def apply_cli_model(self, model: str | None) -> None:
        """Apply a CLI --model override to all roles (preserves --model semantics)."""
        if model is None:
            return
        for role in (self.queen, self.swarm, self.assemble_role, self.specify,
                     self.review, self.commit, self.judge_role, self.converge_role):
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
    openai_raw = merged.get("openai", {})
    ledger_raw = merged.get("ledger", {})
    apply_raw = merged.get("apply", {})
    judge_raw = merged.get("judge", {})
    safety_raw = merged.get("safety", {})
    commit_raw = merged.get("commit_stage", {})
    db_raw = merged.get("db_connections", {})

    def _db_conn(d: dict) -> DbConnection:
        port = d.get("port")
        return DbConnection(
            kind=str(d.get("kind", "sqlite")).strip().lower(),
            path=str(d.get("path", "")),
            host=str(d.get("host", "")),
            port=int(port) if port is not None else None,
            dbname=str(d.get("dbname", "")),
            user=str(d.get("user", "")),
            password=str(d.get("password", "")),
            password_env=str(d.get("password_env", "")),
            codebase=str(d.get("codebase", "")),
        )

    db_connections = {
        str(k): _db_conn(v) for k, v in db_raw.items() if isinstance(v, dict)
    }

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
        converge_role=_role("converge", default_model="claude-sonnet-4.5"),
        copilot=CopilotConfig(
            exe=copilot_raw.get("exe"),
            allow=copilot_raw.get("allow", "--allow-all"),
            timeout_sec=int(copilot_raw.get("timeout_sec", 300)),
        ),
        openai=OpenAiConfig(
            base_url=str(openai_raw.get("base_url",
                                        "https://api.deepinfra.com/v1/openai")),
            api_key_env=str(openai_raw.get("api_key_env", "DEEPINFRA_TOKEN")),
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
        commit_stage=CommitConfig(
            filename_only_threshold=int(
                commit_raw.get("filename_only_threshold", 50)),
        ),
        db_connections=db_connections,
    )
