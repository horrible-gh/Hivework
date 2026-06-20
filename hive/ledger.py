"""SQLite telemetry ledger for Hivework runs.

Records per-run and per-worker-call timing and character/token accounting.
This is for cost visibility only — never for pipeline state.

The ledger is non-fatal: if disabled or any sqlite op throws, a warning is
logged and the pipeline continues.

Token accounting tiers:
  in_chars / out_chars  — EXACT: len(prompt) and len(stdout).
  est_tokens            — ESTIMATE: estimate_tokens(prompt) + estimate_tokens(output).
  real_tokens           — Always NULL. The copilot CLI exposes no token counts
                          (investigated: copilot --output-format json emits timing
                           and code metrics but no prompt/completion token fields).
                          Column is kept nullable for future providers.
"""
import logging, os, sqlite3, threading, time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("hive.ledger")

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, ts TEXT, seed TEXT, work_type TEXT, codebase TEXT,
    model_queen TEXT, model_fanout TEXT,
    axes_n INTEGER, rounds INTEGER, conflicts_n INTEGER, remaining_n INTEGER, parse_errs INTEGER,
    total_in_chars INTEGER, total_out_chars INTEGER,
    total_est_tokens INTEGER, total_real_tokens INTEGER,
    elapsed_s REAL, honey_path TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS worker_calls (
    id INTEGER PRIMARY KEY, run_id INTEGER, stage TEXT, axis_id TEXT,
    provider TEXT, model TEXT,
    in_chars INTEGER, out_chars INTEGER,
    est_tokens INTEGER, real_tokens INTEGER,
    latency_s REAL, comb_path TEXT, ok INTEGER, err TEXT,
    status TEXT, started_at TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(id));
"""

# Columns added after the original schema shipped. ``CREATE TABLE IF NOT EXISTS``
# leaves a pre-existing worker_calls untouched, so older ledger DBs miss these.
# Each is ALTERed in (errors ignored when the column already exists) so begin/
# finish_call work against both fresh and historical databases.
_MIGRATIONS = (
    "ALTER TABLE worker_calls ADD COLUMN status TEXT",
    "ALTER TABLE worker_calls ADD COLUMN started_at TEXT",
    # swarm -> fanout rename (B0001): unify the ledger name with the config
    # (pipeline.fanout). On a historical DB this renames the column in place;
    # on a fresh DB created with the new DDL the column is already `model_fanout`
    # so this RENAME raises "no such column: model_swarm" and is swallowed below.
    "ALTER TABLE runs RENAME COLUMN model_swarm TO model_fanout",
)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Heuristic: max(len(text)//4, len(text.split())).
    This is a rough approximation. Korean and code-dense text can differ
    significantly; a real tokenizer (e.g. tiktoken) is a future refinement.
    """
    if not text:
        return 0
    return max(len(text) // 4, len(text.split()))


class Ledger:
    """Wraps a SQLite connection for run + worker_calls telemetry. Non-fatal."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._run_id: int | None = None
        self._calls: list[dict[str, Any]] = []
        self._start_ts = time.time()
        # The connection is opened ``check_same_thread=False`` so the parallel
        # per-axis judge fan-out (hive.investigate) can record from worker threads.
        # SQLite serialises its own writes, but two threads sharing ONE connection
        # can still interleave statements; this lock serialises every connection
        # access so concurrent ``record_call``s are safe and lossless.
        self._lock = threading.Lock()
        # In-flight calls keyed by call_id: holds in_chars + the prompt-side token
        # estimate so finish_call can complete est_tokens and the run aggregate
        # without re-reading the prompt.
        self._pending: dict[int, dict[str, Any]] = {}
        self._connect()

    def _connect(self) -> None:
        try:
            # ``timeout`` is the Python-side busy wait; the WAL journal lets one
            # writer and many readers proceed at once, and ``busy_timeout`` makes
            # a writer that hits the single-writer lock RETRY for 5s instead of
            # failing immediately. This is what keeps CROSS-PROCESS writes lossless
            # when separate batch processes (219 + 220) record to one ledger file —
            # the in-process ``self._lock`` only covers threads of one process.
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False,
                                         timeout=5.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_DDL)
            for stmt in _MIGRATIONS:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists
            self._conn.commit()
        except Exception as e:
            logger.warning("Ledger: failed to connect/init %s: %s", self._db_path, e)
            self._conn = None

    @property
    def run_id(self) -> int | None:
        """The current run's ledger id (set by start_run), or None. Lets a caller
        (e.g. the runs.jsonl emit hook) reference the row after finish_run."""
        return self._run_id

    def start_run(self, seed: str, codebase: str, model_queen: str, model_fanout: str,
                  ts: str | None = None) -> None:
        """Insert a runs row with status='running'."""
        if self._conn is None:
            return
        ts = ts or datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO runs (ts, seed, work_type, codebase, model_queen, model_fanout, status)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (ts, seed, "investigate", codebase, model_queen, model_fanout, "running"))
                self._conn.commit()
                self._run_id = cur.lastrowid
        except Exception as e:
            logger.warning("Ledger: start_run failed: %s", e)

    def begin_call(self, stage: str, axis_id: str, provider: str, model: str,
                   prompt: str, comb_path: str = "") -> int | None:
        """Insert a worker_calls row at call start with status='wait'.

        Returns the new row id (pass it to ``finish_call``) or None when the
        ledger is unavailable. Recording the row BEFORE the (possibly multi-minute)
        worker call is what makes an in-flight run visible — and guarantees a row
        survives even if the call later times out (see finish_call in except paths).
        ``in_chars`` and ``started_at`` are fixed here; out_chars/latency/ok land
        at finish.

        Status starts at 'wait', NOT 'running': between begin_call and the worker
        actually executing there can be a real blocking gap — a provider may queue
        behind a serialization lock (the codex cross-process mutex). Marking it
        'running' here would label every queued codex call 'running' when only one
        is truly executing and the rest are parked on the lock. The handler calls
        ``mark_running`` the instant it owns the slot, so 'wait' vs 'running'
        reflects reality.
        """
        if self._conn is None or self._run_id is None:
            return None
        in_chars = len(prompt)
        est_prompt = estimate_tokens(prompt)
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO worker_calls"
                    " (run_id, stage, axis_id, provider, model,"
                    "  in_chars, comb_path, status, started_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (self._run_id, stage, axis_id, provider, model,
                     in_chars, comb_path, "wait", started_at))
                self._conn.commit()
                call_id = cur.lastrowid
                self._pending[call_id] = {"in_chars": in_chars, "est_prompt": est_prompt}
                return call_id
        except Exception as e:
            logger.warning("Ledger: begin_call failed: %s", e)
            return None

    def mark_running(self, call_id: int | None) -> None:
        """Flip a 'wait' row to 'running' once the worker truly starts executing.

        The provider handler calls this the instant it owns its slot and begins
        the subprocess/HTTP call — for codex, AFTER acquiring the cross-process
        serialization lock — so the ledger separates "queued behind the lock"
        ('wait') from "actually running" ('running'). Only a still-'wait' row is
        flipped (never clobbers a finished/failed row, and a lost wakeup can't
        resurrect a done row); a None call_id no-ops. Non-fatal.
        """
        if self._conn is None or call_id is None:
            return
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE worker_calls SET status='running'"
                    " WHERE id=? AND status='wait'", (call_id,))
                self._conn.commit()
        except Exception as e:
            logger.warning("Ledger: mark_running failed: %s", e)

    def finish_call(self, call_id: int | None, output: str, latency_s: float,
                    ok: bool = True, err: str = "",
                    real_tokens: int | None = None) -> None:
        """Update a worker_calls row started by ``begin_call`` with the outcome.

        Sets out_chars, est_tokens, latency, ok/err, real_tokens and status
        ('done' on success, 'failed' otherwise). A None call_id (ledger
        unavailable, or begin failed) silently no-ops. ``real_tokens`` is EXACT
        when the provider reports usage (e.g. deepinfra), else None — NULL.
        """
        if self._conn is None or call_id is None:
            return
        out_chars = len(output)
        # A worker call that returns ZERO output is not a clean success, even when
        # the HTTP round-trip exited 0 (NR hivework.default.0004.0003 §2/§5.2: run
        # 418 logged 9 empty combs as ok=1, so the swarm's 0% yield read as
        # "healthy" and the bottleneck was mis-attributed). Demote empty output to
        # ok=0 here so the ledger distinguishes a real answer from a blank one.
        # Observability only — ok is never read to drive pipeline control flow.
        if ok and out_chars == 0 and not err:
            ok = False
            err = "empty output (out_chars=0)"
        try:
            with self._lock:
                pending = self._pending.pop(call_id, None)
                est_prompt = pending["est_prompt"] if pending else 0
                in_chars = pending["in_chars"] if pending else 0
                est_tokens = est_prompt + estimate_tokens(output)
                status = "done" if ok else "failed"
                self._conn.execute(
                    "UPDATE worker_calls SET out_chars=?, est_tokens=?, real_tokens=?,"
                    " latency_s=?, ok=?, err=?, status=? WHERE id=?",
                    (out_chars, est_tokens, real_tokens,
                     latency_s, int(ok), err, status, call_id))
                self._conn.commit()
                self._calls.append({"in_chars": in_chars, "out_chars": out_chars,
                                    "est": est_tokens, "real": real_tokens})
        except Exception as e:
            logger.warning("Ledger: finish_call failed: %s", e)

    def record_call(self, stage: str, axis_id: str, provider: str, model: str,
                    prompt: str, output: str, latency_s: float,
                    comb_path: str = "", ok: bool = True, err: str = "",
                    real_tokens: int | None = None) -> None:
        """Record a completed call in one shot (begin_call + finish_call).

        Convenience for paths that don't need in-flight visibility and for back-
        compat. Live worker call sites should prefer begin_call/finish_call so the
        row appears while the call is running.
        """
        call_id = self.begin_call(stage, axis_id, provider, model, prompt, comb_path)
        self.finish_call(call_id, output, latency_s, ok=ok, err=err,
                         real_tokens=real_tokens)

    def record_local(self, stage: str, axis_id: str, mechanism: str = "",
                     detail: str = "", in_chars: int = 0, out_chars: int = 0,
                     latency_s: float = 0.0) -> None:
        """Insert one worker_calls row for a LOCAL (free, deterministic) step.

        These are the engine's own zero-cost operations — local ripgrep retrieve,
        a live-DB data read, etc. — that never hit a model. They get a row so the
        ledger DB shows the FULL run trace, not just billed model calls, but are
        recorded as provider='local' with est/real tokens left at zero/NULL and
        are NOT folded into ``self._calls`` — so the run's token + char cost
        aggregate stays MODEL-spend only and local rows never inflate it.
        ``mechanism`` (e.g. 'ripgrep', 'sqlite') lands in the model column;
        ``detail`` (e.g. 'reads=2 rows=True') lands in comb_path as a free-text note.
        """
        if self._conn is None or self._run_id is None:
            return
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO worker_calls"
                    " (run_id, stage, axis_id, provider, model,"
                    "  in_chars, out_chars, est_tokens, real_tokens,"
                    "  latency_s, comb_path, ok, err, status, started_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self._run_id, stage, axis_id, "local", mechanism,
                     in_chars, out_chars, 0, None,
                     latency_s, detail, 1, "", "done", started_at))
                self._conn.commit()
        except Exception as e:
            logger.warning("Ledger: record_local failed: %s", e)

    def finish_run(self, honey_path: str = "", axes_n: int = 0, rounds: int = 0,
                   conflicts_n: int = 0, remaining_n: int = 0,
                   parse_errs: int = 0, status: str = "done") -> None:
        """Update the runs row with final aggregates and status."""
        if self._conn is None or self._run_id is None:
            return
        elapsed_s = time.time() - self._start_ts
        total_in = sum(c["in_chars"] for c in self._calls)
        total_out = sum(c["out_chars"] for c in self._calls)
        total_est = sum(c["est"] for c in self._calls)
        # Sum real tokens only over calls that reported them; stays NULL when no
        # provider did (e.g. a copilot-only run), preserving prior behavior.
        reals = [c["real"] for c in self._calls if c.get("real") is not None]
        total_real = sum(reals) if reals else None
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE runs SET axes_n=?, rounds=?, conflicts_n=?, remaining_n=?, parse_errs=?,"
                    " total_in_chars=?, total_out_chars=?, total_est_tokens=?, total_real_tokens=?,"
                    " elapsed_s=?, honey_path=?, status=? WHERE id=?",
                    (axes_n, rounds, conflicts_n, remaining_n, parse_errs,
                     total_in, total_out, total_est, total_real,
                     elapsed_s, honey_path, status, self._run_id))
                self._conn.commit()
        except Exception as e:
            logger.warning("Ledger: finish_run failed: %s", e)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


class NullLedger:
    """No-op ledger used when ledger.enabled=False or open fails."""
    def start_run(self, *a, **kw): pass
    def begin_call(self, *a, **kw): return None
    def mark_running(self, *a, **kw): pass
    def finish_call(self, *a, **kw): pass
    def record_call(self, *a, **kw): pass
    def record_local(self, *a, **kw): pass
    def finish_run(self, *a, **kw): pass
    def close(self): pass


def open_ledger(enabled: bool, db_path: str) -> "Ledger | NullLedger":
    """Factory: returns a Ledger or NullLedger. Non-fatal."""
    if not enabled:
        return NullLedger()
    try:
        return Ledger(db_path)
    except Exception as e:
        logger.warning("Ledger: open_ledger failed, using no-op: %s", e)
        return NullLedger()
