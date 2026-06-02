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
    model_queen TEXT, model_swarm TEXT,
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
    FOREIGN KEY(run_id) REFERENCES runs(id));
"""


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
        self._connect()

    def _connect(self) -> None:
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.executescript(_DDL)
            self._conn.commit()
        except Exception as e:
            logger.warning("Ledger: failed to connect/init %s: %s", self._db_path, e)
            self._conn = None

    def start_run(self, seed: str, codebase: str, model_queen: str, model_swarm: str,
                  ts: str | None = None) -> None:
        """Insert a runs row with status='running'."""
        if self._conn is None:
            return
        ts = ts or datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO runs (ts, seed, work_type, codebase, model_queen, model_swarm, status)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (ts, seed, "investigate", codebase, model_queen, model_swarm, "running"))
                self._conn.commit()
                self._run_id = cur.lastrowid
        except Exception as e:
            logger.warning("Ledger: start_run failed: %s", e)

    def record_call(self, stage: str, axis_id: str, provider: str, model: str,
                    prompt: str, output: str, latency_s: float,
                    comb_path: str = "", ok: bool = True, err: str = "",
                    real_tokens: int | None = None) -> None:
        """Insert a worker_calls row. Computes in_chars, out_chars, est_tokens from text.

        ``real_tokens``: EXACT total token count when the provider reports it
        (e.g. deepinfra via response.usage). None for copilot, which exposes no
        token counts — leaving the column NULL as before.
        """
        if self._conn is None or self._run_id is None:
            return
        in_chars = len(prompt)
        out_chars = len(output)
        est_tokens = estimate_tokens(prompt) + estimate_tokens(output)
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO worker_calls"
                    " (run_id, stage, axis_id, provider, model,"
                    "  in_chars, out_chars, est_tokens, real_tokens,"
                    "  latency_s, comb_path, ok, err)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self._run_id, stage, axis_id, provider, model,
                     in_chars, out_chars, est_tokens, real_tokens,
                     latency_s, comb_path, int(ok), err))
                self._conn.commit()
                self._calls.append({"in_chars": in_chars, "out_chars": out_chars,
                                    "est": est_tokens, "real": real_tokens})
        except Exception as e:
            logger.warning("Ledger: record_call failed: %s", e)

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
    def record_call(self, *a, **kw): pass
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
