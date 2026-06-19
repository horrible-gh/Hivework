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
        # The COORDINATOR (hive.coordinator, R0001): the queen's front-end
        # interpreter. Sonnet-tier by policy (D-03 §1) — it is the beginner's
        # entry gate, so a low tier is forbidden; absolute cost is small (≈1 call
        # per run, D-03 §3). Opt-in via `hive run --coordinator`.
        "coordinator": {"provider": "copilot", "model": "claude-sonnet-4.5"},
    },
    "copilot": {"exe": None, "allow": "--allow-all", "timeout_sec": 300, "read_only": True},
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
    "judge":   {"max_calls_per_axis": 2, "max_axes": 12, "max_parallel": 2,
                "votes_per_axis": 1, "max_total_calls": 0},
    # Cost guard-rails. allow_swarm=false makes `hive.py run` refuse to launch the
    # open-ended swarm (fan-out + reconcile) and point at the cheap investigate path.
    "safety":  {"allow_swarm": True},
    # Spend knobs for the fan-out swarm, previously hard-coded in run_fanout. Defaults
    # reproduce today's behavior: 4 parallel drones, one respecify retry, no call cap.
    "fanout":  {"parallel": 4, "retries": 1, "max_calls": 0},
    # (M013 B3) Capped per-axis REINFORCEMENT by a few quality SCOUT agents (roles.scout)
    # — NOT a swarm (the swarm pattern lives in judge's best-of-N vote). Distinct from the
    # legacy open-ended swarm above. Fires ONLY on an axis the queen flagged (coverage_risk)
    # AND whose local FIND came back empty (needs_reinforcement). Gated by its OWN switch so
    # enabling targeted reinforcement does NOT re-open the full-swarm `run` path. Default OFF
    # (no spend until deliberately enabled). Caps mirror the judge block: max_workers
    # (scouts per axis — keep small; more just re-find the same files) + max_total_calls.
    "reinforce": {"enabled": False, "max_workers": 2, "max_total_calls": 4},
    # (M013 reaction #3) Live one-shot re-investigation of a specify
    # needs_reinvestigation, routed by reason_code + coverage. Default OFF: the plan
    # is always computed & logged for free, but the paid re-run (re_retrieve/
    # re_converge → re-honey → re-specify) only fires when live=true. max_rounds is the
    # hard ceiling on cheap re-runs; the loop also stops early when a re-run changes
    # nothing (honey unchanged) — so it never busy-loops up to the cap for free.
    "reinvestigation": {"live": False, "max_rounds": 2},
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

    def worker_timeout(self, base_default: int = 300) -> int:
        """Effective subprocess timeout (seconds) for this role's worker call.

        An explicit ``timeout_sec`` always wins. Otherwise the default is
        provider-aware: ``codex`` is a slow agentic CLI whose queen/specify
        explorations routinely run 150–290s and tip over a flat 300s wall once
        parallel batch load starves the single serialized codex slot
        (ledger-observed: queen avg 158s, max 290s, intermittent 300s
        TimeoutExpired). Give it a roomier default so a legitimately-slow run
        completes instead of being killed and retried; fast providers (copilot,
        the HTTP endpoint) keep the lean ``base_default``.
        """
        if self.timeout_sec is not None:
            return self.timeout_sec
        return 600 if self.provider == "codex" else base_default


@dataclass
class CopilotConfig:
    exe: str | None = None
    allow: str = "--allow-all"
    timeout_sec: int = 300
    # Which GitHub account the copilot CLI bills. The CLI picks its account from
    # COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN (these OVERRIDE the stored CLI
    # login per `copilot help environment`); with none set it SILENTLY falls back
    # to the logged-in account — which can bill the wrong account with no error.
    # Set ``token`` to pin a specific account (e.g. a test account) regardless of
    # the ambient shell/login: Hive injects it as COPILOT_GITHUB_TOKEN into the
    # copilot subprocess. ``token_env`` instead names an env var to read the token
    # from (keeps the secret out of the config file). ``token`` wins if both set.
    token: str | None = None
    token_env: str | None = None
    # Capability enforcement: when True, every copilot worker runs with
    # --deny-tool=write --deny-tool=shell so it can read/grep the target tree but
    # never edit it or shell out (denial wins over --allow-all). All Hive copilot
    # roles are read-only by design (queen/swarm explore, specify is propose-only,
    # apply writes in deterministic Python), so this defaults True. Flip to False
    # only to deliberately let a copilot worker mutate the target codebase.
    read_only: bool = True


@dataclass
class CodexConfig:
    """Codex CLI provider settings (the OpenAI-equivalent agentic worker).

    ``exe`` pins the codex executable (else it is auto-discovered on PATH).
    ``lock_timeout_sec`` is the cross-process serialization mutex's wait bound in
    seconds: how long a queued codex call sits behind others before giving up with
    a TimeoutError (a safety bound so a stuck holder can't hang a batch overnight).
    An explicit, operator-visible number — change it here, not in code.

    Default 3600 (1h), not 1800: codex per-call timeouts are now roomier (queen
    600s via worker_timeout, specify up to ~800s observed), so under a parallel
    batch the SINGLE serialized codex slot can have a legitimate queue whose total
    wait exceeds 30min — a 1800s bound killed honest waiters mid-queue (ledger:
    "lock not acquired within 1800s"). 3600s absorbs a realistic overnight queue
    while still bounding a genuinely stuck holder.
    """
    exe: str | None = None
    lock_timeout_sec: int = 3600


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
class RunnerConfig:
    """One target codebase's test-runner command (for the runtime red→green verify).

    NEUTRAL by design — no FlowGate (or any caller) semantics, mirroring
    :class:`DbConnection`. ``command`` is the base argv to launch the target's test
    suite (e.g. ``["python", "-m", "pytest", "-q"]``); ``verify_red_green`` appends
    the specify-named red-test node id as the final positional argument, which pytest
    / unittest ``-k`` targets / most runners accept. ``cwd`` is where to launch it
    (relative paths resolve under the run's ``--codebase`` root, so ``"server"`` runs
    pytest inside ``…/FlowGate/server`` where its conftest + DB fixtures live).

    ``env`` injects extra environment for the run (e.g. a venv ``PATH`` / a test DB
    URL) on top of the inherited environment. ``codebase`` optionally binds this entry
    to an explicit codebase path; when absent the entry is matched to a run by its key
    vs the ``--codebase`` leaf name (case-insensitive), so ``"flowgate"`` matches
    ``…/FlowGate`` — the exact resolution :class:`DbConnection` uses.
    """
    command: list[str] = field(default_factory=list)
    cwd: str = ""
    timeout_sec: int = 300
    env: dict[str, str] = field(default_factory=dict)
    codebase: str = ""


@dataclass
class HttpShapeConfig:
    """One target codebase's HTTP-shape red-test harness (lever ⑦ enabler).

    NEUTRAL by design — no FlowGate (or any caller) semantics, mirroring
    :class:`RunnerConfig` / :class:`DbConnection`. It tells specify how to bind the
    synthesised ``GET …`` red test to the TARGET's own seeded TestClient so
    ``apply --verify`` can observe red→green. Supplied one of two ways (never invented):

    - ``app_fixture``: the NAME of an existing seeded-``TestClient`` fixture in the
      target's test tree. Overrides the auto-discovery that DECLINES when the target
      defines several client fixtures (the common case — FlowGate has many), which is
      exactly why naming one is needed.
    - ``setup_block`` (inline) or ``setup_block_file`` (a path, relative resolves under
      the codebase root): explicit pytest source — imports + a ``@pytest.fixture`` that
      builds a seeded TestClient — prepended to the generated test. ``setup_block_file``
      content is read at resolve time; inline ``setup_block`` wins when both are set.

    ``test_dir`` is where the synthesised red test file is created (relative to the
    codebase, e.g. ``"server/tests"`` so it lands beside the target's conftest). With
    NEITHER a fixture name NOR a setup block the lever stays a safe no-op (synthesis
    declines rather than emit a test that errors). ``codebase`` optionally binds this
    entry to an explicit codebase path, exactly like the other per-target blocks.
    """
    app_fixture: str = ""
    setup_block: str = ""
    setup_block_file: str = ""
    test_dir: str = "tests"
    codebase: str = ""

    def resolve_setup_block(self, codebase_root: str | None) -> str | None:
        """Return the harness source: inline ``setup_block`` wins, else read the file.

        A relative ``setup_block_file`` resolves under ``codebase_root``. Returns None
        when neither is set or the file cannot be read (synthesis then falls back to
        ``app_fixture`` / auto-discovery). Never raises."""
        if self.setup_block.strip():
            return self.setup_block
        if self.setup_block_file:
            path = self.setup_block_file
            if not os.path.isabs(path) and codebase_root:
                path = os.path.join(codebase_root, path)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                return None
        return None


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
class ReinforceConfig:
    """Caps for the M013 B3 targeted reinforcement by SCOUT agents (roles.scout) —
    distinct from the legacy open-ended swarm gated by ``safety.allow_swarm``, and NOT a
    swarm itself (the swarm pattern lives in judge's best-of-N vote).

    Reinforcement fires ONLY on a ``needs_reinforcement`` axis — one the queen flagged
    (``coverage_risk``) AND whose blind FIND returned nothing (B2). It spends, so it
    has its OWN ``enabled`` switch (default false): turning on targeted reinforcement
    must not silently re-open the full-swarm ``run`` path. ``max_workers`` caps the scouts
    per axis (keep small — extra scouts just re-find the same files; quality of one scout,
    i.e. ``roles.scout``'s model, is the real lever) and ``max_total_calls`` is the hard
    per-run ceiling — the same one-number budget shape as ``JudgeConfig.max_total_calls``.
    """
    enabled: bool = False
    max_workers: int = 2
    max_total_calls: int = 4


@dataclass
class ReinvestigationConfig:
    """Gate for the M013 reaction-#3 live re-run of a specify needs_reinvestigation.

    The routing plan (``hive.reinvestigate.plan_reinvestigation``) is always computed
    and logged for free. ``live`` gates only the PAID execution of that plan
    (re_retrieve / re_converge → re-honey → re-specify). ``max_rounds`` is the hard
    ceiling on cheap re-runs; the loop ALSO stops early when a re-run changes nothing
    (the re-grounded honey is identical), so a stubborn NR can never busy-loop up to the
    cap. Default off; default cap 2 (one re-run + one confirm) — a tuning knob raised
    only with live A/B evidence. Default off.
    """
    live: bool = False
    max_rounds: int = 2


@dataclass
class ConvergeSplitConfig:
    """Gate + cap for the per-locus SPLIT converge pass (M020 follow-up).

    The default converge is ONE holistic call that must, in a single weak single-shot,
    order the whole path AND pick the one guilty node among several competing located
    loci — too much for a cheap model, so it wanders (SQL ↔ FE ↔ peer) run to run. The
    split pass instead asks ONE NARROW question per located locus ("does THIS locus's
    live code produce the reported symptom?"), each low-variance enough for a cheaper
    model, then COMBINES the answers DETERMINISTICALLY by elimination: when exactly one
    locus survives its cause→symptom check, that one is attributed. The combine is free
    code, not a model judgment, so the wobble at the stitch point disappears.

    It is a PRECISION layer, never a new failure mode: it adopts a verdict ONLY on a
    clean elimination (exactly one survivor). Zero, several, or more located loci than
    ``max_loci`` (a partial evaluation cannot soundly claim "only one survives") all fall
    back to the existing holistic converge + its guards. So turning it on can only improve
    a result or no-op — it can never ship an answer the holistic path would not have.

    ``max_loci`` is the scout-style count cap (the analog of ``ReinforceConfig.max_workers``):
    it bounds the per-locus calls AND defines when the set is too big to evaluate soundly
    (over it → fall back, which is also CHEAPER than N calls). Keep it small; tune the
    per-locus quality with ``model`` (a narrow question tolerates a CHEAPER model), not by
    raising the cap. ``provider``/``model`` empty → reuse the ``roles.converge`` role.
    Default OFF — opt-in, A/B-gated, exactly like reinforce / reinvestigation.
    """
    enabled: bool = False
    max_loci: int = 4
    provider: str = ""
    model: str = ""


@dataclass
class ConvergeLensConfig:
    """Gate + lens set for the adversarial LENS refutation pass at converge.

    converge's causal_check is best-of-1: ONE cheap single-shot rules consistent/contradicted
    and wobbles at depth (right chain, wrong node) and is blind to omission/shadow it was
    never shown. judge de-risks its own noise with best-of-N voting; this is the converge
    analog, pointed at REFUTATION. When converge ships an ACTIONABLE ``consistent``
    attribution, one refuter per lens (the cheap swarm tier) tries to break it; a majority
    refutation demotes converged→False so a wobbly attribution routes to reinvestigation
    instead of an edit. Added cost = exactly ``len(lenses)`` swarm calls per actionable
    converge, zero otherwise. Default OFF — opt-in, A/B-gated, exactly like converge_split.

    ``lenses`` is the set of distinct refutation angles (each = ONE swarm call); empty →
    no-op. The three defaults map to the confirmed failure classes (shadow / omission /
    wobble). ``provider``/``model`` empty → reuse the ``roles.swarm`` role (the best-of-N
    tier, e.g. gpt-oss-120b). ``min_refute`` 0 → simple majority of the lenses.
    """
    enabled: bool = False
    lenses: list[str] = field(default_factory=lambda: [
        "datasource-liveness", "omission", "reproduction"])
    provider: str = ""
    model: str = ""
    min_refute: int = 0


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

    ``votes_per_axis`` is best-of-N judge voting (M010 §5): each axis is judged N
    times INDEPENDENTLY and the located loci are UNIONED (not majority) so a
    noisy-but-correct rare hit survives and converge's causal gate — not a vote
    count — decides precision. Default 1 = today's single judgment (opt-in,
    zero extra cost); ~5 is the validated sweet spot on a cheap judge provider.

    ``max_total_calls`` is the ONE-NUMBER budget cap: a hard ceiling on the TOTAL
    judge model calls in a single investigate run, so cost is controllable without
    reasoning about the ``axes × votes × calls`` product. When the worst case would
    exceed it the pipeline first REDUCES votes (keep axis coverage, shrink voting
    depth), then — only if even one vote across all axes overflows — trims axes. It
    is a worst-case ceiling (assumes every vote spends ``max_calls_per_axis``); real
    runs land at or under it since the re-judge does not always fire. 0 = unlimited
    (today's behaviour).
    """
    max_calls_per_axis: int = 2
    max_axes: int = 12
    max_parallel: int = 2
    votes_per_axis: int = 1
    max_total_calls: int = 0


@dataclass
class FanoutConfig:
    """Tuning knobs for the source-mining fan-out swarm (pipeline.fanout / old roles.swarm).

    These were previously HIDDEN — hard-coded in ``hive.fanout.run_fanout`` rather than
    surfaced in the config — which is exactly the "who is this tuning data for?" gap the
    schema_version 2 layout closes. The on/off gate stays in :class:`SafetyConfig`
    (``allow_swarm``, fed by ``pipeline.fanout.enabled``); this block carries the spend
    knobs:

    - ``parallel``: max concurrent drones (was ``run_fanout``'s hard-coded ``max_workers=4``).
    - ``retries``: extra respecify turns a non-comb reply gets before the axis gives up
      (was the single, unconditional respecify pass). 1 preserves today's behaviour.
    - ``max_calls``: a hard ceiling on TOTAL drone calls per run (0 = unlimited). When set,
      the axis list is trimmed pre-launch so worst-case ``axes x (1 + retries) <= max_calls``.
    """
    parallel: int = 4
    retries: int = 1
    max_calls: int = 0


@dataclass
class Config:
    queen: RoleConfig = field(default_factory=RoleConfig)
    # The legacy blanket-fanout worker (one drone per axis, `hive run`) — a real swarm,
    # gated off by safety.allow_swarm. Distinct from `scout` below.
    swarm: RoleConfig = field(default_factory=RoleConfig)
    # The B3 reinforcement worker: a FEW quality agents sent to dig up the evidence a
    # thin axis's blind grep missed — NOT a swarm (the swarm pattern lives in judge's
    # best-of-N voting). Named `scout` so the model is the obvious reinforcement-quality
    # tuning knob. Falls back to `swarm` when unset, so existing configs are unchanged.
    scout: RoleConfig = field(default_factory=RoleConfig)
    assemble_role: RoleConfig = field(default_factory=RoleConfig)
    specify: RoleConfig = field(default_factory=RoleConfig)
    review: RoleConfig = field(default_factory=RoleConfig)
    commit: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-haiku-4.5"))
    judge_role: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    converge_role: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    # The COORDINATOR (R0001): queen front-end. Sonnet-tier by policy (D-03 §1).
    coordinator: RoleConfig = field(
        default_factory=lambda: RoleConfig(model="claude-sonnet-4.5"))
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    openai: OpenAiConfig = field(default_factory=OpenAiConfig)
    ledger: LedgerConfig = field(default_factory=LedgerConfig)
    apply: ApplyConfig = field(default_factory=ApplyConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    fanout: FanoutConfig = field(default_factory=FanoutConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    reinforce: ReinforceConfig = field(default_factory=ReinforceConfig)
    reinvestigation: ReinvestigationConfig = field(
        default_factory=ReinvestigationConfig)
    converge_split: ConvergeSplitConfig = field(
        default_factory=ConvergeSplitConfig)
    converge_lens: ConvergeLensConfig = field(
        default_factory=ConvergeLensConfig)
    commit_stage: CommitConfig = field(default_factory=CommitConfig)
    # Per-codebase read-only DB connections, keyed by a short name (e.g. "flowgate").
    # Empty by default — the converge data-state read is SKIPPED when a run's codebase
    # has no entry (graceful: converge falls back to its static path / needs_data).
    db_connections: dict[str, DbConnection] = field(default_factory=dict)
    # Per-codebase test runners, keyed by a short name (e.g. "flowgate"). Empty by
    # default — the runtime red→green verify (hive.verify) is SKIPPED when a run's
    # codebase has no entry (graceful, exactly like db_connections above).
    test_runners: dict[str, RunnerConfig] = field(default_factory=dict)
    # Per-codebase HTTP-shape red-test harnesses, keyed by a short name. Empty by
    # default — lever ⑦ synthesis stays a no-op when a run's codebase has no entry
    # (auto-discovery declines on an ambiguous test tree), graceful like the two above.
    http_shape_targets: dict[str, HttpShapeConfig] = field(default_factory=dict)

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

    def test_runner_for_codebase(self, codebase_root: str | None) -> "RunnerConfig | None":
        """Resolve the test runner for a run's ``--codebase`` path, or None.

        Same match order as :meth:`db_for_codebase`: an explicit ``codebase`` binding
        wins, else the entry whose KEY equals the codebase's leaf folder name
        (case-insensitive). Returns None when nothing matches (the common no-runner
        case → runtime verify is skipped), never raises.
        """
        if not codebase_root or not self.test_runners:
            return None
        norm = codebase_root.replace("\\", "/").rstrip("/").lower()
        leaf = norm.rsplit("/", 1)[-1]
        for runner in self.test_runners.values():
            cb = (runner.codebase or "").replace("\\", "/").rstrip("/").lower()
            if cb and (cb == norm or norm.endswith("/" + cb) or cb.endswith("/" + norm)):
                return runner
        for key, runner in self.test_runners.items():
            if key.strip().lower() == leaf:
                return runner
        return None

    def http_shape_for_codebase(self, codebase_root: str | None) -> "HttpShapeConfig | None":
        """Resolve the HTTP-shape harness for a run's ``--codebase`` path, or None.

        Same match order as :meth:`db_for_codebase` / :meth:`test_runner_for_codebase`:
        an explicit ``codebase`` binding wins, else the entry whose KEY equals the
        codebase's leaf folder name (case-insensitive). Returns None when nothing matches
        (the common case → lever ⑦ synthesis stays a no-op), never raises.
        """
        if not codebase_root or not self.http_shape_targets:
            return None
        norm = codebase_root.replace("\\", "/").rstrip("/").lower()
        leaf = norm.rsplit("/", 1)[-1]
        for hs in self.http_shape_targets.values():
            cb = (hs.codebase or "").replace("\\", "/").rstrip("/").lower()
            if cb and (cb == norm or norm.endswith("/" + cb) or cb.endswith("/" + norm)):
                return hs
        for key, hs in self.http_shape_targets.items():
            if key.strip().lower() == leaf:
                return hs
        return None

    def role(self, name: str) -> RoleConfig:
        """Return the RoleConfig for a role ('queen', 'swarm', 'scout', 'assemble', 'specify', 'review', 'commit', 'judge')."""
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
        for role in (self.queen, self.swarm, self.scout, self.assemble_role,
                     self.specify, self.review, self.commit, self.judge_role,
                     self.converge_role):
            role.model = model


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# Config files live in ``<repo>/config/`` as one complete file per profile
# (hive.config.<profile>.json). Resolved off the module dir so the location is
# independent of the caller's working directory.
_CONFIG_DIR = os.path.join(_REPO_ROOT, "config")


def _is_default_profile(profile: str | None) -> bool:
    """A None/blank/'default' profile selects the default file (the auto-generate one)."""
    return profile is None or str(profile).strip() in ("", "default")


def _profile_path(profile: str | None) -> str:
    """Absolute path of a profile's config file under ``config/`` (default: 'default')."""
    name = (str(profile).strip() if profile else "") or "default"
    return os.path.join(_CONFIG_DIR, f"hive.config.{name}.json")


# schema_version 2 (R0001): the unified pipeline layout. Each stage carries its own
# provider/model AND its tuning knobs co-located, under one ordered ``pipeline`` block,
# with names that match the run's stages (decompose / fanout / judge / …) instead of the
# scattered roles + stages + ops of the v1 grouped layout. The map below is the single
# source of truth for which pipeline stage backs which internal role.
_PIPELINE_ROLE_STAGES = (
    "decompose", "fanout", "judge", "reinforce", "converge",
    "assemble", "specify", "review", "commit",
)


def _resolve_model_ref(value: Any, pipeline: dict) -> Any:
    """Resolve a ``"@stage"`` model reference to that pipeline stage's model string.

    A few stages reuse another stage's model rather than naming their own (e.g. the
    converge lens reuses ``@fanout``); the config writes that intent literally as
    ``"@fanout"``. Non-reference values pass through unchanged; a dangling reference
    resolves to an empty string (the "reuse the owning role" sentinel downstream).
    """
    if isinstance(value, str) and value.startswith("@"):
        ref = pipeline.get(value[1:])
        return ref.get("model", "") if isinstance(ref, dict) else ""
    return value


def _expand_schema_v2(raw: dict) -> dict:
    """Expand the schema_version 2 ``coordinator`` + ``pipeline`` layout onto the v1
    grouped keys (roles / stages / safety / fanout), so :func:`_normalize` and every
    downstream consumer keep working unchanged. A no-op when no ``pipeline`` block is
    present (legacy grouped/flat configs fall straight through). Returns a new dict.
    """
    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, dict):
        return raw
    out = dict(raw)
    roles = dict(raw.get("roles") or {})
    stages = dict(raw.get("stages") or {})

    def _role_pm(stage: dict, include_retries: bool = True) -> dict:
        """Pull a stage's provider/model (+ optional timeout/retries) into a role dict."""
        d: dict[str, Any] = {}
        if "provider" in stage:
            d["provider"] = stage["provider"]
        if "model" in stage:
            d["model"] = _resolve_model_ref(stage["model"], pipeline)
        if "timeout_sec" in stage:
            d["timeout_sec"] = stage["timeout_sec"]
        # ``retries`` on decompose/specify is the role's transient-failure retry; on
        # fanout/judge/reinforce it is a STAGE knob (respecify / re-judge depth) and must
        # NOT leak into RoleConfig.retries, so those callers pass include_retries=False.
        if include_retries and "retries" in stage:
            d["retries"] = stage["retries"]
        return d

    coord = raw.get("coordinator")
    if isinstance(coord, dict):
        roles["coordinator"] = _role_pm(coord)

    if isinstance(pipeline.get("decompose"), dict):
        roles["queen"] = _role_pm(pipeline["decompose"])

    fo = pipeline.get("fanout")
    if isinstance(fo, dict):
        roles["swarm"] = _role_pm(fo, include_retries=False)
        out["safety"] = {**(raw.get("safety") or {}),
                         "allow_swarm": bool(fo.get("enabled", True))}
        out["fanout"] = {
            "parallel": int(fo.get("parallel", 4)),
            "retries": int(fo.get("retries", 1)),
            "max_calls": int(fo.get("max_calls", 0)),
        }

    ju = pipeline.get("judge")
    if isinstance(ju, dict):
        roles["judge"] = _role_pm(ju, include_retries=False)
        jstage: dict[str, Any] = {}
        if "jury_size" in ju:
            jstage["votes_per_axis"] = ju["jury_size"]
        if "retries" in ju:
            jstage["max_calls_per_axis"] = int(ju["retries"]) + 1
        if "parallel" in ju:
            jstage["max_parallel"] = ju["parallel"]
        if "max_axes" in ju:
            jstage["max_axes"] = ju["max_axes"]
        if "max_calls" in ju:
            jstage["max_total_calls"] = ju["max_calls"]
        stages["judge"] = {**(stages.get("judge") or {}), **jstage}

    re_ = pipeline.get("reinforce")
    if isinstance(re_, dict):
        roles["scout"] = _role_pm(re_, include_retries=False)
        rstage: dict[str, Any] = {}
        if "enabled" in re_:
            rstage["enabled"] = re_["enabled"]
        if "parallel" in re_:
            rstage["max_workers"] = re_["parallel"]
        if "max_calls" in re_:
            rstage["max_total_calls"] = re_["max_calls"]
        stages["reinforce"] = {**(stages.get("reinforce") or {}), **rstage}

    co = pipeline.get("converge")
    if isinstance(co, dict):
        roles["converge"] = _role_pm(co)
        cstage = dict(stages.get("converge") or {})
        if isinstance(co.get("split"), dict):
            cstage["split"] = {k: v for k, v in co["split"].items() if k != "_comment"}
        if isinstance(co.get("lens"), dict):
            lens = {k: v for k, v in co["lens"].items() if k != "_comment"}
            if "model" in lens:
                lens["model"] = _resolve_model_ref(lens["model"], pipeline)
            cstage["lens"] = lens
        stages["converge"] = cstage

    ri = pipeline.get("reinvestigate")
    if isinstance(ri, dict):
        ristage: dict[str, Any] = {}
        if "enabled" in ri:
            ristage["live"] = ri["enabled"]
        if "rounds" in ri:
            ristage["max_rounds"] = ri["rounds"]
        stages["reinvestigation"] = {**(stages.get("reinvestigation") or {}), **ristage}

    for st in ("assemble", "specify", "review"):
        if isinstance(pipeline.get(st), dict):
            roles[st] = _role_pm(pipeline[st])

    cm = pipeline.get("commit")
    if isinstance(cm, dict):
        roles["commit"] = _role_pm(cm)
        if "filename_only_threshold" in cm:
            stages["commit"] = {**(stages.get("commit") or {}),
                                "filename_only_threshold": cm["filename_only_threshold"]}

    out["roles"] = roles
    out["stages"] = stages
    return out


def _normalize(raw: dict) -> dict:
    """Map the grouped layout (roles / stages / targets / ops / providers) onto the
    flat internal keys the extractor below reads, so BOTH the new config files and any
    legacy flat config load identically — no downstream module or test has to change.

    The schema_version 2 ``coordinator`` + ``pipeline`` layout is expanded onto those
    grouped keys first (:func:`_expand_schema_v2`), so all three layouts converge here.

    Grouped keys win when present. ``_comment`` keys ride along harmlessly: the
    extractor reads named fields only, so annotations need no stripping. Returns a new
    dict; the input is not mutated.
    """
    if not isinstance(raw, dict):
        return {}
    raw = _expand_schema_v2(raw)
    out = dict(raw)

    providers = raw.get("providers")
    if isinstance(providers, dict):
        if isinstance(providers.get("copilot"), dict):
            out["copilot"] = providers["copilot"]
        if isinstance(providers.get("codex"), dict):
            out["codex"] = providers["codex"]
        if isinstance(providers.get("openai"), dict):
            out["openai"] = providers["openai"]

    stages = raw.get("stages")
    if isinstance(stages, dict):
        if isinstance(stages.get("judge"), dict):
            out["judge"] = stages["judge"]
        if isinstance(stages.get("converge"), dict):
            out["converge"] = stages["converge"]
        if isinstance(stages.get("reinforce"), dict):
            out["reinforce"] = stages["reinforce"]
        if isinstance(stages.get("reinvestigation"), dict):
            out["reinvestigation"] = stages["reinvestigation"]
        if isinstance(stages.get("commit"), dict):
            out["commit_stage"] = stages["commit"]

    ops = raw.get("ops")
    if isinstance(ops, dict):
        sr = ops.get("swarm_run")
        if isinstance(sr, dict) and "allow" in sr:
            out["safety"] = {**(raw.get("safety") or {}), "allow_swarm": sr["allow"]}
        if isinstance(ops.get("apply"), dict):
            out["apply"] = ops["apply"]
        if isinstance(ops.get("ledger"), dict):
            out["ledger"] = ops["ledger"]

    targets = raw.get("targets")
    if isinstance(targets, dict):
        db_conns = dict(raw.get("db_connections") or {})
        runners = dict(raw.get("test_runners") or {})
        http_shapes = dict(raw.get("http_shape_targets") or {})
        for name, t in targets.items():
            if not isinstance(t, dict):  # skips a targets-level "_comment", etc.
                continue
            if isinstance(t.get("db"), dict):
                db_conns[name] = t["db"]
            if isinstance(t.get("tests"), dict):
                runners[name] = t["tests"]
            if isinstance(t.get("http_shape"), dict):
                http_shapes[name] = t["http_shape"]
        if db_conns:
            out["db_connections"] = db_conns
        if runners:
            out["test_runners"] = runners
        if http_shapes:
            out["http_shape_targets"] = http_shapes

    return out


def _schema_v2_default_dict() -> dict:
    """Build the schema_version 2 (coordinator + pipeline) dict from the built-in NEUTRAL
    defaults — used only to bootstrap a missing default-profile file (a fresh checkout that
    never ran setup). These are code defaults, not the hand-tuned shipped values; on a
    normal checkout ``config/hive.config.default.json`` already exists so this never fires.
    Emitting the v2 layout keeps the bootstrap file in the SAME shape humans edit."""
    d = _DEFAULTS
    roles = d["roles"]
    j = d["judge"]
    return {
        "schema_version": 2,
        "coordinator": dict(roles["coordinator"]),
        "pipeline": {
            "decompose": dict(roles["queen"]),
            "fanout": {**dict(roles["swarm"]),
                       "enabled": d["safety"]["allow_swarm"],
                       "parallel": d["fanout"]["parallel"],
                       "retries": d["fanout"]["retries"],
                       "max_calls": d["fanout"]["max_calls"]},
            "judge": {**dict(roles["judge"]),
                      "jury_size": j["votes_per_axis"],
                      "retries": max(0, j["max_calls_per_axis"] - 1),
                      "parallel": j["max_parallel"],
                      "max_axes": j["max_axes"],
                      "max_calls": j["max_total_calls"]},
            "reinforce": {**dict(roles["swarm"]),
                          "enabled": d["reinforce"]["enabled"],
                          "parallel": d["reinforce"]["max_workers"],
                          "max_calls": d["reinforce"]["max_total_calls"]},
            "converge": {**dict(roles["converge"]),
                         "split": {"enabled": False, "max_loci": 4}},
            "reinvestigate": {"model": "@judge",
                              "enabled": d["reinvestigation"]["live"],
                              "rounds": d["reinvestigation"]["max_rounds"]},
            "assemble": dict(roles["assemble"]),
            "specify": dict(roles["specify"]),
            "review": dict(roles["review"]),
            "commit": {**dict(roles["commit"]), "filename_only_threshold": 50},
        },
        "providers": {
            "copilot": dict(d["copilot"]),
            "openai": dict(d["openai"]),
        },
        "targets": {},
        "ops": {
            "apply": dict(d["apply"]),
            "ledger": dict(d["ledger"]),
        },
    }


def _write_bootstrap_default(path: str) -> None:
    """Write the neutral schema_version 2 default file (bootstrap safety net)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_schema_v2_default_dict(), f, indent=2)
        f.write("\n")
    logger.info("Wrote bootstrap default config to %s", path)


def load_config(path: str | None = None, profile: str | None = None) -> Config:
    """Load config, falling back to built-in defaults for missing keys.

    Resolution:

    - ``path`` given → load that exact file (explicit override; what tests use).
    - ``path`` None → load ``config/hive.config.<profile or 'default'>.json``. If the
      DEFAULT-profile file is absent (a fresh checkout that never ran setup), a neutral
      bootstrap file is written from the built-in defaults, then loaded.

    The file may use the grouped layout (roles / stages / targets / ops / providers) or
    the legacy flat keys; :func:`_normalize` maps the former onto the latter so both
    load identically.
    """
    if path is not None:
        resolved_path = path
    else:
        resolved_path = _profile_path(profile)
        if not os.path.exists(resolved_path) and _is_default_profile(profile):
            try:
                _write_bootstrap_default(resolved_path)
            except OSError as e:
                logger.warning("Could not write bootstrap default config to %s: %s",
                               resolved_path, e)

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

    merged = _deep_merge(_DEFAULTS, _normalize(raw))
    roles = merged.get("roles", {})
    copilot_raw = merged.get("copilot", {})
    codex_raw = merged.get("codex", {})
    openai_raw = merged.get("openai", {})
    ledger_raw = merged.get("ledger", {})
    apply_raw = merged.get("apply", {})
    judge_raw = merged.get("judge", {})
    fanout_raw = merged.get("fanout", {})
    safety_raw = merged.get("safety", {})
    reinforce_raw = merged.get("reinforce", {})
    reinvest_raw = merged.get("reinvestigation", {})
    split_raw = merged.get("converge", {}).get("split", {}) \
        if isinstance(merged.get("converge"), dict) else {}
    lens_raw = merged.get("converge", {}).get("lens", {}) \
        if isinstance(merged.get("converge"), dict) else {}
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

    runners_raw = merged.get("test_runners", {})

    def _runner(d: dict) -> RunnerConfig:
        cmd = d.get("command", [])
        if isinstance(cmd, str):
            cmd = cmd.split()
        env = d.get("env", {})
        return RunnerConfig(
            command=[str(x) for x in cmd] if isinstance(cmd, list) else [],
            cwd=str(d.get("cwd", "")),
            timeout_sec=int(d.get("timeout_sec", 300)),
            env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
            codebase=str(d.get("codebase", "")),
        )

    test_runners = {
        str(k): _runner(v) for k, v in runners_raw.items() if isinstance(v, dict)
    }

    http_shape_raw = merged.get("http_shape_targets", {})

    def _http_shape(d: dict) -> HttpShapeConfig:
        return HttpShapeConfig(
            app_fixture=str(d.get("app_fixture", "")),
            setup_block=str(d.get("setup_block", "")),
            setup_block_file=str(d.get("setup_block_file", "")),
            test_dir=str(d.get("test_dir", "tests")) or "tests",
            codebase=str(d.get("codebase", "")),
        )

    http_shape_targets = {
        str(k): _http_shape(v) for k, v in http_shape_raw.items() if isinstance(v, dict)
    }

    def _role(name: str, default_model: str = "gpt-5-mini") -> RoleConfig:
        r = roles.get(name, {})
        timeout = r.get("timeout_sec")
        return RoleConfig(provider=r.get("provider", "copilot"),
                          model=r.get("model", default_model),
                          timeout_sec=int(timeout) if timeout is not None else None,
                          retries=int(r.get("retries", 0)))

    # `scout` (B3 reinforcement worker) defaults to the `swarm` role's model when the
    # config does not name it, so an existing roles.swarm carries over unchanged.
    swarm_role = _role("swarm")
    scout_role = _role("scout") if "scout" in roles else swarm_role
    return Config(
        queen=_role("queen"),
        swarm=swarm_role,
        scout=scout_role,
        assemble_role=_role("assemble"),
        specify=_role("specify"),
        review=_role("review"),
        commit=_role("commit", default_model="claude-haiku-4.5"),
        judge_role=_role("judge", default_model="claude-sonnet-4.5"),
        converge_role=_role("converge", default_model="claude-sonnet-4.5"),
        coordinator=_role("coordinator", default_model="claude-sonnet-4.5"),
        copilot=CopilotConfig(
            exe=copilot_raw.get("exe"),
            allow=copilot_raw.get("allow", "--allow-all"),
            timeout_sec=int(copilot_raw.get("timeout_sec", 300)),
            token=copilot_raw.get("token"),
            token_env=copilot_raw.get("token_env"),
            read_only=bool(copilot_raw.get("read_only", True)),
        ),
        codex=CodexConfig(
            exe=codex_raw.get("exe"),
            lock_timeout_sec=int(codex_raw.get("lock_timeout_sec", 3600)),
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
            votes_per_axis=int(judge_raw.get("votes_per_axis", 1)),
            max_total_calls=int(judge_raw.get("max_total_calls", 0)),
        ),
        fanout=FanoutConfig(
            parallel=int(fanout_raw.get("parallel", 4)),
            retries=int(fanout_raw.get("retries", 1)),
            max_calls=int(fanout_raw.get("max_calls", 0)),
        ),
        safety=SafetyConfig(
            allow_swarm=bool(safety_raw.get("allow_swarm", True)),
        ),
        reinforce=ReinforceConfig(
            enabled=bool(reinforce_raw.get("enabled", False)),
            max_workers=int(reinforce_raw.get("max_workers", 2)),
            max_total_calls=int(reinforce_raw.get("max_total_calls", 4)),
        ),
        reinvestigation=ReinvestigationConfig(
            live=bool(reinvest_raw.get("live", False)),
            max_rounds=int(reinvest_raw.get("max_rounds", 2)),
        ),
        converge_split=ConvergeSplitConfig(
            enabled=bool(split_raw.get("enabled", False)),
            max_loci=int(split_raw.get("max_loci", 4)),
            provider=str(split_raw.get("provider", "") or ""),
            model=str(split_raw.get("model", "") or ""),
        ),
        converge_lens=ConvergeLensConfig(
            enabled=bool(lens_raw.get("enabled", False)),
            lenses=[str(x) for x in (lens_raw.get("lenses") or [
                "datasource-liveness", "omission", "reproduction"]) if str(x).strip()],
            provider=str(lens_raw.get("provider", "") or ""),
            model=str(lens_raw.get("model", "") or ""),
            min_refute=int(lens_raw.get("min_refute", 0)),
        ),
        commit_stage=CommitConfig(
            filename_only_threshold=int(
                commit_raw.get("filename_only_threshold", 50)),
        ),
        db_connections=db_connections,
        test_runners=test_runners,
        http_shape_targets=http_shape_targets,
    )
