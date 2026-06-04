#!/usr/bin/env python3
"""Hivework interactive setup - a menu-driven, idempotent installer.

Run via setup.bat / setup.sh (which install requirements first), or directly:

    python hive_setup.py

Rather than marching through fixed steps and demanding a token up front, this is a
MENU: you first pick WHAT to manage, then drill in. Nothing is forced and nothing
already present is overwritten.

  1. Config file        - create hive.config.json from the example, or show status.
  2. Models & providers - per role, pick a provider (copilot / codex / openai-
                          compatible) and a model, with suggested presets and a
                          one-touch 'balanced' layout.
  3. API tokens         - choose WHICH secret to set (the OpenAI-compatible
                          endpoint's key, or a DB password), stored in the
                          out-of-repo secrets file so keys never live in the repo.

The provider is NOT hard-wired to one vendor: copilot and codex sign in through
their own CLI and need no key here at all; only an OpenAI-compatible HTTP endpoint
(DeepInfra, OpenAI, a self-hosted vLLM, ...) needs a base_url + API key.

UI is English only (a language layer can be added later if wanted). Uses ``rich``
for clean output when available, and degrades to plain text if not. Console output
is kept ASCII-only so it cannot crash on a legacy Windows code page (cp932/cp949),
the same hazard hive.py guards against.
"""
from __future__ import annotations

import os
import sys


def _force_utf8_io() -> None:
    """Best-effort UTF-8 stdout/stderr so output never dies on a legacy code page."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8_io()

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(REPO, "hive.config.json")
CONFIG_EXAMPLE = os.path.join(REPO, "hive.config.example.json")


class GoBack(Exception):
    """Raised by a text prompt when the user hits Ctrl+C to abort the CURRENT input
    and return to the enclosing menu - instead of an uncaught KeyboardInterrupt that
    kills the whole program (the old behaviour forced a quit + re-launch). Menu loops
    catch this and redisplay, so Ctrl+C means 'back', not 'lose everything'."""

# ── UI shim: prefer rich, fall back to plain stdio so a deps-less direct run still
#    works. All text is ASCII so neither path can hit a code-page encode error.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    _con = Console()

    def banner(text: str) -> None:
        _con.print(Panel(text, expand=False))

    def info(text: str) -> None:
        # markup/highlight OFF: content lines carry literal '[minimum]', '[y/n]',
        # '$VAR', paths - rich must not eat brackets as tags or recolour them.
        _con.print(text, markup=False, highlight=False)

    def ok(text: str) -> None:
        _con.print(f"[green][ok][/green] {text}")

    def warn(text: str) -> None:
        _con.print(f"[yellow][!][/yellow] {text}")

    def ask_secret(prompt: str) -> str:
        try:
            return Prompt.ask(prompt, password=True, default="", show_default=False)
        except EOFError:
            return ""
        except KeyboardInterrupt:
            raise GoBack  # Ctrl+C aborts this input -> back to the menu, not a crash

    def ask_text(prompt: str, default: str = "") -> str:
        try:
            return Prompt.ask(prompt, default=default).strip()
        except EOFError:
            return default
        except KeyboardInterrupt:
            raise GoBack  # Ctrl+C aborts this input -> back to the menu, not a crash

    def ask_choice(prompt: str, choices: list[str], default: str) -> str:
        # EOF (piped/closed stdin) returns the default so menu loops terminate
        # instead of crashing or spinning.
        try:
            return Prompt.ask(prompt, choices=choices, default=default)
        except EOFError:
            return default

except Exception:  # rich not installed (direct run before pip install) - plain stdio
    import getpass

    def banner(text: str) -> None:
        line = "=" * min(len(text) + 4, 72)
        print(f"\n{line}\n  {text}\n{line}")

    def info(text: str) -> None:
        print(text)

    def ok(text: str) -> None:
        print(f"[ok] {text}")

    def warn(text: str) -> None:
        print(f"[!] {text}")

    def ask_secret(prompt: str) -> str:
        try:
            return getpass.getpass(prompt + " ").strip()
        except EOFError:
            return ""
        except KeyboardInterrupt:
            raise GoBack  # Ctrl+C aborts this input -> back to the menu, not a crash

    def ask_text(prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            val = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            return default
        except KeyboardInterrupt:
            raise GoBack  # Ctrl+C aborts this input -> back to the menu, not a crash
        return val or default

    def ask_choice(prompt: str, choices: list[str], default: str) -> str:
        try:
            val = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        return val if val in choices else default


# ── Selection layer: arrow-key menus when possible, text prompts otherwise ─────────
# rich has no interactive menu, so for arrow-key SELECT and space-toggle CHECKBOX we
# use questionary (optional). When it is absent or stdin is not a TTY (piped input,
# automation, tests), every selector falls back to the text ask_choice / numbered
# loop above - so behaviour and the test mocks (which patch ask_choice) are unchanged.
try:
    import questionary as _q
except Exception:
    _q = None


def _can_prompt() -> bool:
    """questionary present AND a real interactive terminal on stdin (else: text)."""
    if _q is None:
        return False
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:
        return False


def select_one(prompt: str, choices: list[tuple[str, str]], default: str) -> str:
    """Arrow-key pick of ONE value. ``choices`` = [(value, label), ...]. Falls back to
    the text ask_choice (over the values) when not interactive, so tests/automation
    that drive ask_choice keep working unchanged."""
    values = [v for v, _l in choices]
    if not _can_prompt():
        return ask_choice(prompt, values, default)
    qchoices = [_q.Choice(title=label, value=value) for value, label in choices]
    try:
        ans = _q.select(prompt, choices=qchoices,
                        default=default if default in values else None,
                        instruction="(up/down, Enter)").ask()
    except Exception:
        return ask_choice(prompt, values, default)
    return ans if ans is not None else default


def _checkbox_text(prompt: str, items: list[tuple[str, str, bool]]) -> set[str]:
    """Numbered [x]/[ ] toggle loop - the non-TTY / no-questionary fallback for
    multi-select (mirrors questionary.checkbox so both paths behave the same)."""
    order = [value for value, _l, _c in items]
    labels = {value: label for value, label, _c in items}
    checked = {value: chk for value, _l, chk in items}
    while True:
        info(f"\n{prompt}")
        for i, value in enumerate(order, 1):
            mark = "[x]" if checked[value] else "[ ]"
            info(f"  {i}) {mark} {labels[value]}")
        info("  Type a NUMBER to toggle; 'a'=all, 'n'=none, 'd'=done.")
        nums = [str(i) for i in range(1, len(order) + 1)]
        choice = ask_choice("Toggle / done", nums + ["a", "n", "d"], "d")
        if choice == "d":
            return {v for v, on in checked.items() if on}
        if choice == "a":
            checked = {v: True for v in order}
        elif choice == "n":
            checked = {v: False for v in order}
        else:
            v = order[int(choice) - 1]
            checked[v] = not checked[v]


def checkbox(prompt: str, items: list[tuple[str, str, bool]]) -> set[str]:
    """Space-toggle multi-select. ``items`` = [(value, label, checked), ...]. Arrow to
    move, Space to tick, Enter to confirm. Falls back to ``_checkbox_text`` when not
    interactive. Pre-ticked items reflect auto-detection."""
    if not _can_prompt():
        return _checkbox_text(prompt, items)
    qchoices = [_q.Choice(title=label, value=value, checked=checked)
                for value, label, checked in items]
    try:
        ans = _q.checkbox(prompt, choices=qchoices,
                          instruction="(Space toggles, Enter confirms)").ask()
    except Exception:
        return _checkbox_text(prompt, items)
    return set(ans or [])


def _secrets_path() -> str:
    """Where the secrets file lives: $HIVE_ENV_FILE override, else ~/.hivework/.env."""
    return os.environ.get("HIVE_ENV_FILE") or os.path.join(
        os.path.expanduser("~"), ".hivework", ".env")


# OpenAI-compatible endpoint presets. "custom" is handled inline (asks for both).
# Each entry: (label, base_url, default api_key_env). copilot/codex are NOT here -
# they sign in via their own CLI and need no key in this file.
_PRESETS: dict[str, tuple[str, str, str]] = {
    "deepinfra": ("DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_TOKEN"),
    "openai":    ("OpenAI",     "https://api.openai.com/v1",          "OPENAI_API_KEY"),
}


def _secrets_skeleton(api_key_env: str, token: str = "") -> str:
    return (
        "# Hivework secrets - keep this file OUTSIDE the repo (it holds real keys).\n"
        "# A value already in your environment wins; this file only fills gaps.\n"
        "# copilot can auth via its own CLI login OR a COPILOT_GITHUB_TOKEN here;\n"
        "# codex auths via its own CLI login.\n"
        f"{api_key_env}={token}\n"
        "# DB passwords referenced by hive.config.json db_connections[*].password_env\n"
        "# (only for mysql/postgres targets; sqlite needs none), e.g.:\n"
        "# DB_PASSWORD=\n"
    )


def bootstrap_config() -> bool:
    """Create hive.config.json from the example. Returns True if it was just created
    (so the caller may patch the chosen endpoint), False if it already existed."""
    if os.path.isfile(CONFIG):
        ok(f"Config already exists - left untouched:\n    {CONFIG}")
        return False
    if not os.path.isfile(CONFIG_EXAMPLE):
        warn("hive.config.example.json not found - skipping config bootstrap.")
        return False
    import shutil
    shutil.copyfile(CONFIG_EXAMPLE, CONFIG)
    ok(f"Created config from the example:\n    {CONFIG}")
    info("    -> Edit it: set each role's provider/model, and your db_connections.")
    return True


def choose_endpoint() -> tuple[str, str] | None:
    """Pick the OpenAI-compatible endpoint (base_url + api_key_env), or None to skip.

    Provider-neutral: presets are conveniences, not a lock-in. Returns the chosen
    (base_url, api_key_env) so the caller can patch the config + name the secret.
    """
    info("\nProvider endpoint (only the OpenAI-compatible HTTP provider needs one).")
    info("  copilot / codex roles sign in via their own CLI - skip this for them.")
    choice = select_one("Which endpoint do the openai/deepinfra roles use?",
                        [("deepinfra", "DeepInfra (default)"),
                         ("openai",    "OpenAI"),
                         ("custom",    "Custom (your own base_url)"),
                         ("skip",      "Skip - none of my roles use HTTP")],
                        "deepinfra")
    if choice == "skip":
        return None
    if choice == "custom":
        base_url = ask_text("Base URL (OpenAI-compatible, e.g. https://host/v1)").strip()
        if not base_url:
            warn("No base URL given - skipping endpoint configuration.")
            return None
        api_key_env = ask_text("Env-var NAME that will hold the key", "OPENAI_API_KEY")
        return base_url, (api_key_env or "OPENAI_API_KEY").strip()
    _label, base_url, api_key_env = _PRESETS[choice]
    return base_url, api_key_env


def patch_config_endpoint(base_url: str, api_key_env: str) -> None:
    """Write the chosen openai endpoint into the freshly-created hive.config.json.

    Only called when bootstrap_config just created the file, so idempotency holds -
    an existing config is never rewritten. Drops the reference-only ``_openai_presets``
    key the example carries to keep the live config clean.
    """
    import json
    try:
        with open(CONFIG, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"Could not patch config endpoint ({e}); set the openai block by hand.")
        return
    data.pop("_openai_presets", None)
    data["openai"] = {"base_url": base_url, "api_key_env": api_key_env}
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    ok(f"Set openai endpoint -> {base_url}  (key from ${api_key_env})")


# ── Catalogue: providers, suggested models, and the roles config drives ──────────
# How each provider authenticates and which role-kind it suits. copilot/codex carry
# no key in the secrets file; only the OpenAI-compatible HTTP provider does.
_PROVIDERS: dict[str, str] = {
    "copilot":   "Copilot CLI - sign in via the 'copilot' CLI; no key here. Ceiling: sonnet.",
    "codex":     "Codex CLI - sign in via the 'codex' CLI; flat-rate sub. Tool-ON roles.",
    "openai":    "OpenAI-compatible HTTP - needs endpoint + API key. Best for tool-OFF.",
    "deepinfra": "Same HTTP backend as 'openai', DeepInfra preset (cheap tool-OFF).",
}

# Suggested models per provider for the manual per-role editor (first = default);
# 'custom' is always offered. copilot/codex entries are VERIFIED with `--model` on
# 2026-06-03 (rejected ones dropped: copilot has no gpt-5.4-nano; a ChatGPT-account
# codex rejects the codex-specific gpt-5.3-codex/gpt-5.1-codex-mini/gpt-5-codex and
# only takes the general gpt-5.x line).
_MODEL_PRESETS: dict[str, list[str]] = {
    "copilot":   ["gpt-5-mini", "gpt-5.4-mini", "claude-haiku-4.5",
                  "claude-sonnet-4.5", "claude-sonnet-4.6", "gpt-5.4"],
    "codex":     ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"],
    "openai":    ["openai/gpt-oss-20b", "openai/gpt-oss-120b",
                  "Qwen/Qwen3-235B-A22B-Instruct-2507"],
    "deepinfra": ["openai/gpt-oss-20b", "openai/gpt-oss-120b",
                  "Qwen/Qwen3-235B-A22B-Instruct-2507"],
}

# The roles hive.config.json drives, with their nature (guides the provider choice).
_ROLES: list[tuple[str, str]] = [
    ("queen",    "tool-ON  : decompose / explore the tree"),
    ("specify",  "tool-ON  : author the edit spec"),
    ("swarm",    "tool-ON  : fan-out drones (swarm path)"),
    ("assemble", "tool-ON  : stitch drone combs"),
    ("commit",   "tool-ON  : group changes into commits"),
    ("judge",    "tool-OFF : single-shot ruling on a bundle"),
    ("converge", "tool-OFF : stitch per-axis verdicts"),
    ("review",   "tool-OFF : effectiveness review of an edit"),
]


# ── Config read/write ────────────────────────────────────────────────────────────
def _load_config_data() -> dict | None:
    import json
    if not os.path.isfile(CONFIG):
        return None
    try:
        with open(CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"Could not read config ({e}).")
        return None


def _load_example_data() -> dict | None:
    """Read hive.config.example.json into memory WITHOUT writing anything to disk, so
    the wizard can build a config in-memory and create the file only at the very end
    (generation is the last step, not a side effect of opening the editor)."""
    import json
    if not os.path.isfile(CONFIG_EXAMPLE):
        return None
    try:
        with open(CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        warn(f"Could not read the example config ({e}).")
        return None
    data.pop("_openai_presets", None)  # reference-only key the example ships
    return data


def _save_config_data(data: dict) -> None:
    import json
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _role_entry(provider: str, model: str, prev: dict) -> dict:
    """A role's {provider, model} dict, preserving any per-role extras already set."""
    entry = {"provider": provider, "model": model}
    for extra in ("timeout_sec", "retries"):
        if extra in prev:
            entry[extra] = prev[extra]
    return entry


# ── Guided preset generation ─────────────────────────────────────────────────────
# Per-provider model IDs by quality/cost tier, from the model shortlist (2026-06-03,
# _scratch/ai_model_shortlist_*). EDIT HERE ONLY to retune models - the generator and
# the per-role editor read everything from this one table. HTTP provider keyed
# 'openai' (vendor-neutral). NB: copilot/codex IDs follow the CLI's lowercase-hyphen
# form (e.g. 'GPT-5.4 mini' -> 'gpt-5.4-mini'); verify against the live CLI if a model
# is rejected.
_MODEL_TIERS: dict[str, dict[str, str]] = {
    # copilot (verified): gpt-5-mini (cheapest that works; nano is rejected) ->
    # gpt-5.4-mini (mid) -> claude-sonnet-4.6 (strong coding card).
    "copilot": {"min": "gpt-5-mini", "mix": "gpt-5.4-mini", "max": "claude-sonnet-4.6"},
    # codex (verified on a ChatGPT account): only the general gpt-5.x line works.
    # Per actual usage + cost-caution: floor is gpt-5.4-mini (min AND mix), and 'max'
    # reaches only gpt-5.4 - the gpt-5.5 flagship is deliberately NOT in any tier.
    "codex":   {"min": "gpt-5.4-mini", "mix": "gpt-5.4-mini", "max": "gpt-5.4"},
    # deepinfra/openai: oss-20b (cheap) -> oss-120b (main worker, JSON+FC) ->
    # Qwen3-235B-Instruct (strong, JSON+FC). oss-120b is proven live; 20b/Qwen are
    # listed on DeepInfra's pricing but not yet round-tripped here.
    "openai":  {"min": "openai/gpt-oss-20b", "mix": "openai/gpt-oss-120b",
                "max": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
}

# tool-ON roles re-open live files: today only an agentic CLI (copilot/codex) runs the
# tool loop, so the tool-OFF HTTP provider can't take them YET (the agent loop is
# unbuilt, not impossible - see providers.py). tool-OFF roles are single-shot rulings.
_TOOL_ON_ROLES = {"queen", "specify", "swarm", "assemble", "commit"}

# Provider preference PER ROLE (first available wins). Distinct per role so that with
# several providers available the layout MIXES across them instead of piling onto one:
# the heavy authors (queen/specify) favour codex (flat-rate agentic coding); the
# lighter agentic roles (swarm/assemble/commit) favour copilot; the tool-OFF rulings
# favour the cheap HTTP provider. Each list is filtered to what's available, so it
# degrades gracefully (codex-only -> all codex). tool-ON lists never include 'openai'.
_ROLE_PROVIDER_PREF: dict[str, list[str]] = {
    "queen":    ["codex", "copilot"],
    "specify":  ["codex", "copilot"],
    "swarm":    ["copilot", "codex"],
    "assemble": ["copilot", "codex"],
    "commit":   ["copilot", "codex"],
    "judge":    ["openai", "copilot", "codex"],
    "converge": ["openai", "copilot", "codex"],
    "review":   ["openai", "copilot", "codex"],
}

# Overall preset tier -> which model tier each role-kind takes. 'mix' keeps tool-OFF
# on the main worker (oss-120b) rather than the cheapest, since rulings parse JSON.
_TIER_PLAN: dict[str, dict[str, str]] = {
    "minimum": {"on": "min", "off": "min"},
    "mix":     {"on": "mix", "off": "mix"},
    "maximum": {"on": "max", "off": "max"},
}


def _pick_provider(role: str, available: set[str]) -> str | None:
    """The best available provider for a role per its preference list, or None if
    none fits (only for a tool-ON role when neither codex nor copilot is available)."""
    for prov in _ROLE_PROVIDER_PREF.get(role, ["openai", "copilot", "codex"]):
        if prov in available:
            return prov
    return None


def build_tier_preset(available: set[str], tier: str) -> dict[str, tuple[str, str]] | None:
    """Compose a role -> (provider, model) layout from the AVAILABLE providers.

    Returns None when a tool-ON role can't be satisfied (no codex/copilot available)
    - the caller surfaces that as 'you need an agentic CLI for the author roles'.
    """
    plan = _TIER_PLAN[tier]
    out: dict[str, tuple[str, str]] = {}
    for name, _desc in _ROLES:
        prov = _pick_provider(name, available)
        if prov is None:
            return None
        model_tier = plan["on" if name in _TOOL_ON_ROLES else "off"]
        out[name] = (prov, _MODEL_TIERS[prov][model_tier])
    return out


def apply_tier_preset(data: dict, roles_map: dict[str, tuple[str, str]]) -> None:
    """Write a generated role layout into the config, preserving per-role extras."""
    roles = data.setdefault("roles", {})
    for name, (provider, model) in roles_map.items():
        roles[name] = _role_entry(provider, model, roles.get(name, {}))


def detect_providers() -> dict[str, tuple[bool, str]]:
    """Best-effort 'what can this user run?' probe -> {provider: (likely, reason)}.

    codex/copilot: is the CLI on PATH? copilot also counts a stored
    COPILOT_GITHUB_TOKEN. openai (HTTP): is the configured api_key_env present in the
    secrets file? The result only seeds the confirm step's defaults - the user has
    the final say (they may be about to add a token)."""
    keys = _read_env()
    out: dict[str, tuple[bool, str]] = {}

    codex_cli = _cli_found(_CLI_PROVIDERS["codex"][0])
    out["codex"] = (bool(codex_cli),
                    "CLI found" if codex_cli else "CLI NOT on PATH")

    cop_cli = _cli_found(_CLI_PROVIDERS["copilot"][0])
    cop_tok = bool(keys.get("COPILOT_GITHUB_TOKEN"))
    if cop_cli:
        reason = "CLI found" + (", token set" if cop_tok else " (CLI login or token)")
    else:
        reason = "CLI NOT on PATH" + (", but token set" if cop_tok else "")
    out["copilot"] = (bool(cop_cli) or cop_tok, reason)

    api_env = ((_load_config_data() or {}).get("openai") or {}).get(
        "api_key_env", "DEEPINFRA_TOKEN")
    http_tok = bool(keys.get(api_env))
    out["openai"] = (http_tok, f"key ${api_env} {'set' if http_tok else 'NOT set'}")
    return out


# ── Secrets file read/write (preserves existing lines & comments) ─────────────────
def _read_env() -> dict[str, str]:
    keys: dict[str, str] = {}
    path = _secrets_path()
    if not os.path.isfile(path):
        return keys
    try:
        lines = open(path, "r", encoding="utf-8").read().splitlines()
    except OSError:
        return keys
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        keys[k.strip()] = v.strip()
    return keys


def _mask(value: str) -> str:
    v = value.strip().strip('"').strip("'")
    if not v:
        return "(empty - set before a paid run)"
    if len(v) <= 4:
        return "****"
    return v[:2] + "****" + v[-2:]


def _upsert_env(key: str, value: str) -> str:
    """Set ``key=value`` in the secrets file, preserving other lines. Creates the
    file (with the skeleton header) if missing. Returns 'created'|'updated'|'appended'."""
    path = _secrets_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(_secrets_skeleton(key, value))
        return "created"
    lines = open(path, "r", encoding="utf-8").read().splitlines(keepends=True)
    out: list[str] = []
    found = False
    for raw in lines:
        s = raw.strip()
        if s and not s.startswith("#") and s.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}\n")
            found = True
        else:
            out.append(raw if raw.endswith("\n") else raw + "\n")
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(f"{key}={value}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    return "updated" if found else "appended"


# CLI providers. They run as a subprocess that inherits our environment, so a token
# placed in the secrets file (loaded by hive.py _load_secrets) reaches them for free.
# Each entry: (exe candidates mirroring hive/providers.py, optional env-token name,
# sign-in hint). copilot ALSO honours COPILOT_GITHUB_TOKEN as an alternative to its
# interactive CLI login; codex auths only via its own CLI login (token=None).
_CLI_PROVIDERS: dict[str, tuple[list[str], str | None, str]] = {
    "copilot": (["copilot.cmd", "copilot"], "COPILOT_GITHUB_TOKEN",
                "run 'copilot' and sign in to GitHub, or set COPILOT_GITHUB_TOKEN"),
    "codex":   (["codex.cmd", "codex"], None,
                "run 'codex' and sign in to ChatGPT"),
}


def _cli_found(candidates: list[str]) -> str | None:
    """First resolvable exe path among the candidates, or None if none on PATH."""
    import shutil
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


# ── Status ───────────────────────────────────────────────────────────────────────
def show_status() -> None:
    banner("Hivework status")
    if os.path.isfile(CONFIG):
        ok(f"config: {CONFIG}")
        data = _load_config_data() or {}
        roles = data.get("roles", {})
        for name, _desc in _ROLES:
            r = roles.get(name, {})
            info(f"    {name:9s} {str(r.get('provider','-')):9s} {r.get('model','-')}")
        oa = data.get("openai", {})
        if oa:
            info(f"    endpoint  {oa.get('base_url','-')}  (key ${oa.get('api_key_env','-')})")
    else:
        warn(f"config: not created yet ({CONFIG})")
    path = _secrets_path()
    if os.path.isfile(path):
        ok(f"secrets: {path}")
        keys = _read_env()
        for k, v in keys.items():
            info(f"    {k} = {_mask(v)}")
        if not keys:
            info("    (no keys set yet)")
    else:
        warn(f"secrets: not created yet ({path})")

    # CLI providers (copilot / codex): report CLI presence + (for copilot) whether a
    # COPILOT_GITHUB_TOKEN is set, so a copilot user understands their auth options.
    info("provider CLIs (copilot / codex run via their own CLI):")
    keys = _read_env()
    used = {str(r.get("provider")) for r in (_load_config_data() or {}).get("roles", {}).values()}
    for name, (candidates, token_env, hint) in _CLI_PROVIDERS.items():
        found = _cli_found(candidates)
        tag = "" if name in used else "  (not used by current config)"
        auth = ""
        if token_env:
            auth = (f" [auth: ${token_env} set]" if keys.get(token_env)
                    else f" [auth: CLI login, or set ${token_env}]")
        if found:
            info(f"    {name:9s} found: {found}{auth}{tag}")
        else:
            warn(f"    {name:9s} NOT on PATH - {hint}{tag}")


# ── Menus ────────────────────────────────────────────────────────────────────────
def menu_config() -> None:
    while True:
        try:
            choice = select_one("Config file",
                                [("1", "Create hive.config.json from the example (if missing)"),
                                 ("2", "Show status"),
                                 ("b", "Back")], "b")
            if choice == "1":
                bootstrap_config()
            elif choice == "2":
                show_status()
            else:
                return
        except GoBack:
            continue  # Ctrl+C in a sub-prompt -> redisplay this menu


def _edit_role(data: dict, name: str, available: set[str] | None = None) -> None:
    roles = data.setdefault("roles", {})
    cur = roles.get(name, {})
    info(f"\nRole '{name}'  (current: {cur.get('provider','-')}/{cur.get('model','-')})")
    # When ``available`` is given (custom-from-preset path), restrict the provider
    # menu to the providers the user actually selected - and drop the HTTP provider
    # for a tool-ON role, which it can't run. Otherwise (raw per-role editor) show all.
    if available:
        provs = [p for p in _PROVIDERS if p in available]
        if name in _TOOL_ON_ROLES:
            provs = [p for p in provs if p not in ("openai", "deepinfra")]
        if not provs:
            warn(f"No selected provider can run '{name}' - left unchanged.")
            return
    else:
        provs = list(_PROVIDERS.keys())
    default_prov = cur.get("provider") if cur.get("provider") in provs else provs[0]
    provider = select_one(f"Provider for '{name}'",
                          [(p, f"{p:9s} {_PROVIDERS[p]}") for p in provs],
                          str(default_prov))
    models = _MODEL_PRESETS.get(provider, [])
    if models:
        default = cur.get("model") if cur.get("model") in models else models[0]
        pick = select_one(f"Model for {provider}",
                          [(m, m) for m in models] + [("custom", "custom (type your own)")],
                          default)
        model = ask_text("Custom model id", "").strip() if pick == "custom" else pick
    else:
        model = ask_text("Model id", str(cur.get("model", ""))).strip()
    if not model:
        warn("No model given - role left unchanged.")
        return
    roles[name] = _role_entry(provider, model, cur)
    ok(f"{name} -> {provider}/{model}")


def select_providers(detected: dict[str, tuple[bool, str]]) -> set[str]:
    """A space-toggle CHECK-LIST of providers, pre-ticked from detection (a present CLI
    or token auto-ticks it). Arrow + Space to toggle, Enter to confirm - no hand-typing.
    Returns the checked set. (Numbered-toggle text fallback when not interactive.)"""
    order = ["codex", "copilot", "openai"]
    labels = {
        "codex":   "codex    - agentic CLI, flat-rate sub (tool-ON roles)",
        "copilot": "copilot  - agentic CLI, login or COPILOT_GITHUB_TOKEN",
        "openai":  "openai   - OpenAI-compatible HTTP (deepinfra/openai/...), tool-OFF",
    }
    items = [(prov, f"{labels[prov]}  ({detected[prov][1]})", detected[prov][0])
             for prov in order]
    return checkbox("Providers you can use (auto-ticked from detected CLIs/tokens):",
                    items)


def _print_preset(tier: str, roles_map: dict[str, tuple[str, str]]) -> None:
    info(f"\n  [{tier}]")
    for name, _desc in _ROLES:
        prov, model = roles_map[name]
        info(f"    {name:9s} {prov:9s} {model}")


def _custom_roles(data: dict, available: set[str]) -> None:
    """Hand-edit every role, with the provider menu limited to the selected providers.
    Loops until 'done', then saves. Used both for full custom and for tweaking a
    just-applied preset."""
    info("\nCustom roles - provider menu is limited to the providers you selected.")
    while True:
        try:
            roles = data.get("roles", {})
            role_choices: list[tuple[str, str]] = []
            for i, (name, desc) in enumerate(_ROLES, 1):
                r = roles.get(name, {})
                role_choices.append(
                    (str(i), f"{name:9s} {str(r.get('provider','-')):9s} "
                             f"{str(r.get('model','-')):22s} {desc}"))
            role_choices.append(("d", "Done (saves the config)"))
            choice = select_one("Edit which role?", role_choices, "d")
            if choice == "d":
                _save_config_data(data)
                ok(f"Saved config -> {CONFIG}")
                return
            _edit_role(data, _ROLES[int(choice) - 1][0], available)
        except GoBack:
            continue  # Ctrl+C while editing a role -> back to the role list


def menu_guided(data: dict) -> None:
    """User-centred role layout: check the providers you can use, then either take a
    ready preset (shown in full, still tweakable before saving) or hand-pick every
    role. Provider+model are never hand-typed unless you choose to."""
    available = select_providers(detect_providers())
    if not available:
        warn("No providers selected - nothing to generate.")
        return
    if not ({"codex", "copilot"} & available):
        warn("The author roles (queen/specify/swarm/assemble/commit) are agentic and")
        warn("need codex or copilot. Add one (or set its token), then retry.")
        return

    # Preset or full custom, asked up front.
    mode = select_one("How do you want to set the roles?",
                      [("preset", "Use a ready preset (recommended)"),
                       ("custom", "Hand-pick every role (custom)")], "preset")
    if mode == "custom":
        _custom_roles(data, available)
        return

    # Build min/mix/max from the selected providers, show them, let the user pick one.
    presets: dict[str, dict[str, tuple[str, str]]] = {}
    for tier in ("minimum", "mix", "maximum"):
        rm = build_tier_preset(available, tier)
        if rm is not None:
            presets[tier] = rm
            _print_preset(tier, rm)
    if not presets:
        return
    info("\n  (presets only use the providers you selected)")
    tier_labels = {"minimum": "minimum - cheapest models",
                   "mix":     "mix     - balanced (recommended)",
                   "maximum": "maximum - strongest models (cost-caution)"}
    tier = select_one("Which preset?",
                      [(t, tier_labels.get(t, t)) for t in presets] + [("cancel", "Cancel")],
                      "mix" if "mix" in presets else next(iter(presets)))
    if tier == "cancel":
        return
    chosen = presets[tier]

    # Show the chosen preset in full, then: apply as-is / tweak it first / cancel.
    info("\nChosen preset:")
    _print_preset(tier, chosen)
    after = select_one("Apply this preset?",
                       [("apply",     "Apply as-is"),
                        ("customize", "Tweak it first (then save)"),
                        ("cancel",    "Cancel")], "apply")
    if after == "cancel":
        return
    apply_tier_preset(data, chosen)              # seed the config with the preset
    if after == "customize":
        _custom_roles(data, available)           # hand-edit on top of it (saves)
        return
    _save_config_data(data)
    ok(f"Applied '{tier}' preset and saved -> {CONFIG}")
    if "openai" in available and not _read_env().get(
            ((data.get("openai") or {}).get("api_key_env", "DEEPINFRA_TOKEN"))):
        info("  Reminder: tool-OFF roles use the HTTP provider - set its key in API tokens.")


def menu_models() -> None:
    """Models & providers = the guided wizard, entered directly. The first thing the
    user sees is the provider CHECK-LIST (check -> mix -> preset or custom -> show ->
    generate), never a raw per-role table. The numbered per-role editor is the
    'custom' branch only - it is not dumped on entry."""
    data = _load_config_data()
    if data is None:
        # No config yet: draft from the example IN MEMORY - the file is written only
        # at the end (preset apply or custom 'done'), never on entry.
        data = _load_example_data()
        if data is None:
            warn("hive.config.example.json not found - cannot start the wizard.")
            return
        info("\nNo config yet - building a draft from the example. Nothing is written")
        info("until you apply a preset or finish a custom layout.")
    menu_guided(data)


def menu_tokens() -> None:
    while True:
      try:
        info("\nAPI tokens (stored OUTSIDE the repo - keys never live in the code):")
        info(f"  file: {_secrets_path()}")
        keys = _read_env()
        if keys:
            for k, v in keys.items():
                info(f"    {k} = {_mask(v)}")
        else:
            info("    (no keys set yet)")
        choice = select_one("Which token do you want to set?",
            [("1", "OpenAI-compatible endpoint + its API key (openai/deepinfra roles)"),
             ("2", "Copilot token (COPILOT_GITHUB_TOKEN) - optional, instead of CLI login"),
             ("3", "Database password (by env-var name)"),
             ("4", "Set/replace any single key by name"),
             ("b", "Back")], "b")
        if choice == "b":
            return
        if choice == "2":
            info("  copilot also works via its own CLI login; this token is an alternative")
            info("  (handy for headless/automation). codex auths only via its CLI login.")
            val = ask_secret("Paste COPILOT_GITHUB_TOKEN (blank = fill in later):")
            ok(f"COPILOT_GITHUB_TOKEN {_upsert_env('COPILOT_GITHUB_TOKEN', val.strip())}.")
        elif choice == "1":
            endpoint = choose_endpoint()
            if endpoint is None:
                continue
            base_url, api_key_env = endpoint
            if os.path.isfile(CONFIG):
                patch_config_endpoint(base_url, api_key_env)
            else:
                info("  (no config yet) set later:  "
                     f'"openai": {{"base_url":"{base_url}","api_key_env":"{api_key_env}"}}')
            token = ask_secret(f"Paste the {api_key_env} value (blank = fill in later):")
            if token.strip():
                state = _upsert_env(api_key_env, token.strip())
                ok(f"{api_key_env} {state}.")
            else:
                if api_key_env not in _read_env():
                    _upsert_env(api_key_env, "")
                info(f"  Left {api_key_env} blank - set it before a paid run.")
        elif choice == "3":
            name = ask_text("DB password env-var NAME (e.g. DB_PASSWORD)", "").strip()
            if not name:
                continue
            val = ask_secret(f"Paste the {name} value (blank = fill in later):")
            ok(f"{name} {_upsert_env(name, val.strip())}.")
        elif choice == "4":
            name = ask_text("Key NAME to set", "").strip()
            if not name:
                continue
            val = ask_secret(f"Value for {name} (blank = empty):")
            ok(f"{name} {_upsert_env(name, val.strip())}.")
      except GoBack:
        continue  # Ctrl+C in a token sub-prompt -> redisplay this menu


def main() -> None:
    banner("Hivework setup")
    info("Menu-driven and idempotent - pick what to manage; nothing is forced.")
    info("Tip: while typing a value, Ctrl+C goes BACK to the menu (it no longer quits).")
    show_status()
    while True:
        try:
            choice = select_one("What do you want to manage?",
                [("1", "Config file        - create / show hive.config.json"),
                 ("2", "Models & providers - guided: check providers, pick a preset or customize"),
                 ("3", "API tokens         - endpoint + keys in the secrets file"),
                 ("4", "Show status"),
                 ("q", "Quit")], "q")
            if choice == "1":
                menu_config()
            elif choice == "2":
                menu_models()
            elif choice == "3":
                menu_tokens()
            elif choice == "4":
                show_status()
            else:
                info("\nDone. Standalone runs read the secrets file automatically;")
                info("via the launcher, the launcher's environment wins.")
                return
        except GoBack:
            continue  # stray Ctrl+C from a deep prompt -> never quits, just redisplay


if __name__ == "__main__":
    main()
