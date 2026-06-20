"""Hermetic inputs for converge's deterministic causal/provenance guards."""
from __future__ import annotations

import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from hive.converge import (
    ConvergeResult,
    _counterfactual_complete_guard,
    _data_stamp_guard,
    _dropped_peer_guard,
    _evidence_sufficiency_guard,
    _field_provenance_guard,
    _http_datasource_provenance_guard,
    _premise_refuted_guard,
    _trace_grounding_guard,
)


@dataclass(frozen=True)
class CorpusCase:
    id: str
    summary: str
    captured: bool
    result: ConvergeResult
    located: list[dict[str, Any]]
    windows: list[dict[str, Any]]
    guard: str
    expect: dict[str, Any]
    # Live producer source to materialize under a temp ``code_root`` before the guard runs
    # (relpath -> source). Needed by the HTTP-datasource plumbing-abstain path, whose gate
    # (``_looks_like_datasource``) only fires when the winning-path producer's code is
    # READABLE — ``if ptext and not _looks_like_datasource(ptext)``. The permanent corpus
    # otherwise calls the guard with ``code_root=None`` (empty ``ptext``), which silently
    # BYPASSES the gate (0032 NR0003 §3). None → keep the historical ``code_root=None`` call.
    producer_files: dict[str, str] | None = None


def _located(axis_id: str, file: str, lines: str, reason: str,
             vtype: str | None = None) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "located": True,
        "file": file,
        "lines": lines,
        "reason": reason,
    }
    if vtype:
        verdict["type"] = vtype
    return {
        "axis_id": axis_id,
        "title": axis_id.replace("_", " ").title(),
        "verdict": verdict,
        "votes": [],
        "candidates": [],
        "coverage": {},
    }


def _result(
    file: str,
    lines: str,
    *,
    converged: bool = True,
    data_dependent: bool = False,
    trace: str = "the attributed code is consistent with the symptom",
    counterfactual: str = "",
    refuted_peers: list[dict[str, Any]] | None = None,
    path: list[dict[str, Any]] | None = None,
    data_backed: bool = False,
    guard_inputs: dict[str, Any] | None = None,
) -> ConvergeResult:
    return ConvergeResult(
        converged=converged,
        path=path or [{"node": "candidate", "file": file, "lines": lines}],
        attributed_defect={
            "node": "candidate",
            "file": file,
            "lines": lines,
            "why": "frozen weak-model attribution",
        },
        causal_check={
            "verdict": "consistent",
            "data_dependent": data_dependent,
            "data_state_assumptions": [],
            "trace": trace,
            "counterfactual": counterfactual,
            "refuted_peers": refuted_peers or [],
            "need_data_state": [],
            "data_reads": [],
        },
        summary="frozen weak-model output",
        data_state_backed=data_backed,
        raw={"guard_inputs": dict(guard_inputs or {})},
    )


HEAD_DECOY = "server/sql/queries/queries.json"
HEAD_PRODUCER = "server/modules/flow_gate/documents/routers/documents.py"
HEAD_LOCATED = [
    _located("SQL_HEAD", HEAD_DECOY, "127-129", "query is lexically similar to head"),
    _located(
        "HEAD_FIELD",
        HEAD_PRODUCER,
        "375-397",
        "producer assigns the wrong workflow_head_type response value",
    ),
]
HEAD_WINDOWS = [
    {
        "file": HEAD_PRODUCER,
        "lines": "375-397",
        "via": "field-producer",
        "field": "workflow_head_type",
        "symbol": "serialize_document",
        "text": (
            "NON_HEAD_TYPES = {'review', 'archive'}\n"
            "head_type = step.type if step.type not in NON_HEAD_TYPES else previous.type\n"
            "out['workflow_head_type'] = head_type\n"
        ),
    }
]


M035_LOCATED = [
    _located("SQL_HEAD", HEAD_DECOY, "127-129", "query appears to choose the wrong head"),
]


M036_FE = "client/src/main/components/NewRequirementModal.vue"
M036_ROUTE = "server/modules/flow_gate/api/v1/legacy_misc_routes.py"
M036_SERVICE = "server/modules/flow_gate/process_service.py"
M036_DATASOURCE = "server/modules/flow_gate/store.py"
M036_DECOY = "server/modules/flow_gate/api/v1/list_routes.py"
M036_LOCATED = [
    _located("MODULE_ROUTE", M036_DECOY, "101-144", "lexically similar module route"),
    _located(
        "PROJECT_SOURCE",
        M036_DATASOURCE,
        "1021-1035",
        "executed datasource hardcodes the module field empty",
    ),
]
M036_WINDOWS = [
    {
        "file": M036_FE,
        "lines": "239-306",
        "text": (
            "const res = await getRequest('/api/v1/projects')\n"
            "currentModules = res.modules\n"
            "if (currentModules.length > 0) renderModules(currentModules)\n"
        ),
    },
    {
        "file": M036_ROUTE,
        "lines": "82-87",
        "via": "http-binding",
        "url": "/api/v1/projects",
        "symbol": "get_projects",
        "text": (
            "# RESOLVED BINDING: client /api/v1/projects\n"
            "async def get_projects():\n"
            "    return project_service.load_projects()\n"
        ),
    },
    {
        "file": M036_SERVICE,
        "lines": "2090-2106",
        "via": "call-chain",
        "symbol": "load_projects",
        "text": (
            "def load_projects():\n"
            "    return project_store.fetch_projects()\n"
        ),
    },
    {
        "file": M036_DATASOURCE,
        "lines": "1021-1035",
        "via": "call-chain",
        "symbol": "fetch_projects",
        "text": (
            "def fetch_projects():\n"
            "    return db.execute(\"SELECT project_id, '' AS module FROM projects\")\n"
        ),
    },
    {
        "file": M036_DECOY,
        "lines": "101-144",
        "text": "def list_modules():\n    return db.execute('SELECT module FROM modules')\n",
    },
]


N180_DATA = "server/app/document_store.py"
N180_RENDER = "client/src/response_mapper.py"
N180_LOCATED = [
    _located("DATA", N180_DATA, "40-51", "database row exists"),
    _located(
        "RENDER",
        N180_RENDER,
        "18-25",
        "handler emits module_id while the front end reads module",
    ),
]


M017_FILE = "server/app/workflow_store.py"
M017_LOCATED = [
    _located("DATA_ORDER", M017_FILE, "70-82", "stored rank may select the wrong row"),
]


N170_REACHABLE = "server/app/reachable_route.py"
N170_PRODUCER = "server/app/payload_builder.py"
N170_UPSTREAM = "server/app/status_store.py"
N170_LOCATED = [
    _located("ROUTE", N170_REACHABLE, "15-24", "route is reached by the request"),
    _located(
        "PRODUCER",
        N170_PRODUCER,
        "44-52",
        "payload builder emits the symptom field",
    ),
]
N170_WINDOWS = [
    {
        "file": N170_PRODUCER,
        "lines": "44-52",
        "via": "field-producer",
        "field": "display_status",
        "symbol": "build_payload",
        "text": (
            "def build_payload(record):\n"
            "    payload['display_status'] = normalize(record.state)\n"
            "    return payload\n"
        ),
    }
]
N170_NEG_LOCATED = [
    _located("UPSTREAM", N170_UPSTREAM, "9-17", "upstream status lookup is wrong"),
    N170_LOCATED[1],
]
N170_NEG_WINDOWS = [
    {
        **N170_WINDOWS[0],
        "text": (
            "from server.app import status_store\n"
            "def build_payload(record):\n"
            "    payload['display_status'] = status_store.lookup(record.id)\n"
            "    return payload\n"
        ),
    }
]


# ── design_change carve-out loci ──
# A design_change verdict means the judge vetted the locus as the node a change must land on
# (reporter expectation = ground truth, code faithful-to-design). The provenance guards must
# not re-point AWAY from it to a merely-structural producer/datasource. Here the rejected
# design lives in head_policy.py while a separate serializer mechanically copies the field.
DC_SITE = "server/app/head_policy.py"
DC_PRODUCER = "server/app/response_serializer.py"
DC_FIELD_WINDOWS = [
    {
        "file": DC_PRODUCER,
        "lines": "20-31",
        "via": "field-producer",
        "field": "workflow_head_type",
        "symbol": "serialize_document",
        "text": "out['workflow_head_type'] = policy_result\n",
    }
]


# Synthetic loci for the prompt-side / count-based facets (P0, P4, P5). Structurally
# correct but clearly fake; no real coordinates are documented for these shapes.
P0_ATTR = "server/app/query.py"
P0_PEER = "client/app/render.ts"


# ── R0001 / group 0032: forced winning-path PLUMBING-decoy injection ──────────────
# DETERMINISTIC regression net for the converge HTTP-datasource gate's plumbing-abstain
# (``_winningpath_ds_targets`` + ``_looks_like_datasource``, committed ab057a4). Source
# material is CAPTURED VERBATIM from real swarm-OFF runs — NOT hand-authored — to remove the
# teaching-to-the-test bias R0001 #2 warns about:
#   • ``get_store`` @ db/connection.py:229-234 — run 480 (smoke/0030_ts0002); the decoy the
#     pre-gate guard re-pointed onto and held converged (golden_0082 _result_note: headline
#     locus = db/connection.py != dispose_group event-write → found=false).
#   • the decoy SHIFTED from db/projects.py (run 477) to db/connection.py (run 480) — golden
#     calls the residual "robust and locus-independent", so a POSITION variant is grounded in
#     real data, not invention (R0001 #4 family: getter/store/connection location/shape).
# The committed gate abstains on these pure accessors so the attribution STAYS on the correct
# dispose_group handler (golden _found_rule). Disabling the gate re-points onto the decoy —
# the firing/removal asymmetry the meta test asserts (R0001 #3: gate無력화 → RED).
DISPOSE_HANDLER = "server/modules/flow_gate/process_service.py"  # golden _found_rule locus

# Verbatim: FlowGate-dev/FlowGate/server/modules/flow_gate/db/connection.py:229-234 (run 480).
_CAP_GET_STORE = (
    "def get_store() -> FlowGateStore:\n"
    '    """Return the singleton FlowGateStore."""\n'
    "    global STORE\n"
    "    if STORE is None:\n"
    "        STORE = FlowGateStore()\n"
    "    return STORE\n"
)
# Sibling accessor shapes in the same family (connection/getter variants). Each is pure
# plumbing: it returns a handle, produces NO collection/row data, so _looks_like_datasource
# is False and the gate must abstain exactly as for get_store.
_CAP_GET_CONNECTION = "def get_connection():\n    return self._conn\n"
_CAP_GET_CONN = "def get_conn():\n    return _POOL.handle\n"

# The forced FE edge: a gated collection (``length > 0``) filled from an HTTP URL whose
# winning-path producer is the injected plumbing decoy. Field stems {types, type} are absent
# from every accessor body above, so condition (b) "producer omits the field" holds and the
# verdict turns purely on condition (c), the datasource-shape gate.
FORCED_FE = "client/src/main/components/DesignTypePicker.vue"
FORCED_URL = "/api/v1/q"


def _forced_fe_windows() -> list[dict[str, Any]]:
    return [{
        "file": FORCED_FE, "lines": "30-44",
        "text": (
            "const res = await getRequest('/api/v1/q')\n"
            "designTypes.value = res.data.types ?? []\n"
            "if (designTypes.length > 0) renderTypes()\n"
        ),
    }]


def _forced_inject_located(decoy_file: str, decoy_lines: str) -> list[dict[str, Any]]:
    """Correct dispose_group handler + a forced HTTP_WINNING_PATH plumbing decoy on the URL."""
    return [
        _located("DISPOSE_ROUTE", DISPOSE_HANDLER, "2091-2150",
                 "winning HTTP path reaches dispose_group (correct handler)"),
        {
            "axis_id": f"HTTP_WINNING_PATH:{FORCED_URL}",
            "title": f"winning HTTP response producer for {FORCED_URL}",
            "verdict": {"located": True, "file": decoy_file, "lines": decoy_lines,
                        "reason": "deterministic winning request-path producer (plumbing decoy)"},
            "votes": [], "candidates": [], "coverage": {},
        },
    ]


def _forced_inject_result() -> ConvergeResult:
    return _result(DISPOSE_HANDLER, "2091-2150",
                   trace="dispose_group is on the executed path and produces the 500")


DECOY_CONN = "server/modules/flow_gate/db/connection.py"
DECOY_PROJ = "server/modules/flow_gate/db/projects.py"   # run-477 position variant
DECOY_STORE = "server/modules/flow_gate/db/store.py"


CASES = [
    CorpusCase(
        id="head",
        summary="field producer must replace the confident SQL decoy",
        captured=True,
        result=_result(
            HEAD_DECOY,
            "127-129",
            data_dependent=True,
            trace="the named SQL query is reachable and rows exist",
        ),
        located=HEAD_LOCATED,
        windows=HEAD_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": HEAD_PRODUCER,
            "stamp": "field_provenance_repointed",
            "outcome": "RE-POINT",
        },
    ),
    CorpusCase(
        id="head-confirm",
        summary="grounded producer restores a prior unrelated demotion",
        captured=True,
        result=_result(
            HEAD_PRODUCER,
            "375-397",
            converged=False,
            data_dependent=True,
            trace="a prior domain guard demoted the producer attribution",
        ),
        located=HEAD_LOCATED,
        windows=HEAD_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": HEAD_PRODUCER,
            "stamp": "field_provenance_confirmed",
            "outcome": "CONFIRM",
        },
    ),
    CorpusCase(
        id="M035",
        summary="collapsed read chain refutes the data premise",
        captured=True,
        result=_result(
            HEAD_DECOY,
            "127-129",
            data_dependent=True,
            guard_inputs={"db_available": True, "chain_broke": True},
        ),
        located=M035_LOCATED,
        windows=[],
        guard="premise_refuted",
        expect={
            "converged": False,
            "attributed_file": HEAD_DECOY,
            "stamp": "data_premise_refuted",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="M035-neg",
        summary="resolved read chain must not refute the premise",
        captured=False,
        result=_result(
            HEAD_DECOY,
            "127-129",
            data_dependent=True,
            guard_inputs={"db_available": True, "chain_broke": False},
        ),
        located=M035_LOCATED,
        windows=[],
        guard="premise_refuted",
        expect={
            "converged": True,
            "attributed_file": HEAD_DECOY,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    CorpusCase(
        id="M036",
        summary="executed HTTP datasource must replace an off-path route decoy",
        captured=True,
        result=_result(M036_DECOY, "101-144", trace="list_modules looks relevant"),
        located=M036_LOCATED,
        windows=M036_WINDOWS,
        guard="http_datasource",
        expect={
            "converged": True,
            "attributed_file": M036_DATASOURCE,
            "stamp": "http_datasource_provenance_repointed",
            "outcome": "RE-POINT",
        },
    ),
    CorpusCase(
        id="M036-neg",
        summary="HTTP datasource guard cannot invent an unlocated target",
        captured=False,
        result=_result(M036_DECOY, "101-144", trace="list_modules looks relevant"),
        located=[M036_LOCATED[0]],
        windows=M036_WINDOWS,
        guard="http_datasource",
        expect={
            "converged": True,
            "attributed_file": M036_DECOY,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    CorpusCase(
        id="N180",
        summary="data-certified result must not drop an unrefuted render peer",
        captured=False,
        result=_result(
            N180_DATA,
            "40-51",
            data_dependent=True,
            data_backed=True,
            trace="document_store.py confirms that the database row exists",
        ),
        located=N180_LOCATED,
        windows=[],
        guard="dropped_peer",
        expect={
            "converged": False,
            "attributed_file": N180_DATA,
            "stamp": "dropped_peer",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="N180-neg",
        summary="an explicitly addressed render peer must not trigger demotion",
        captured=False,
        result=_result(
            N180_DATA,
            "40-51",
            data_dependent=True,
            data_backed=True,
            trace=(
                "document_store.py confirms the row; response_mapper.py was checked and "
                "already reads the emitted module_id key"
            ),
        ),
        located=N180_LOCATED,
        windows=[],
        guard="dropped_peer",
        expect={
            "converged": True,
            "attributed_file": N180_DATA,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    CorpusCase(
        id="M017",
        summary="data-dependent consistency without a live row stamp is unsafe",
        captured=False,
        result=_result(
            M017_FILE,
            "70-82",
            data_dependent=True,
            guard_inputs={"db_available": True},
        ),
        located=M017_LOCATED,
        windows=[],
        guard="data_stamp",
        expect={
            "converged": False,
            "attributed_file": M017_FILE,
            "stamp": "data_unstamped",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="M017-neg",
        summary="a live-row-backed data verdict keeps its earned stamp",
        captured=False,
        result=_result(
            M017_FILE,
            "70-82",
            data_dependent=True,
            data_backed=True,
            guard_inputs={"db_available": True},
        ),
        located=M017_LOCATED,
        windows=[],
        guard="data_stamp",
        expect={
            "converged": True,
            "attributed_file": M017_FILE,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    CorpusCase(
        id="N170",
        summary="field production outranks mere route reachability",
        captured=False,
        result=_result(
            N170_REACHABLE,
            "15-24",
            trace="the route is on the executed request path",
        ),
        located=N170_LOCATED,
        windows=N170_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": N170_PRODUCER,
            "stamp": "field_provenance_repointed",
            "outcome": "RE-POINT",
        },
    ),
    CorpusCase(
        id="N170-neg",
        summary="an upstream dependency on the producer path remains eligible",
        captured=False,
        result=_result(
            N170_UPSTREAM,
            "9-17",
            trace="the upstream status lookup supplies the emitted value",
        ),
        located=N170_NEG_LOCATED,
        windows=N170_NEG_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": N170_UPSTREAM,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    # ── P0 counterfactual completeness (domain-agnostic generalization of N180) ──
    CorpusCase(
        id="counterfactual",
        summary="a consistent verdict that silently drops a distinct peer must abstain",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            counterfactual="correcting query.py changes the selected row",
        ),
        located=[
            _located("QUERY", P0_ATTR, "10-20", "query hypothesis"),
            _located("RENDER", P0_PEER, "30-40", "binding hypothesis"),
        ],
        windows=[],
        guard="counterfactual",
        expect={
            "converged": False,
            "attributed_file": P0_ATTR,
            "stamp": "unrefuted_peer",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="counterfactual-neg",
        summary="an explicitly refuted peer leaves the consistent verdict intact",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            counterfactual="correcting query.py changes the selected row",
            refuted_peers=[{"file": P0_PEER, "lines": "30-40",
                            "why_not": "the consumer reads the key the handler emits"}],
        ),
        located=[
            _located("QUERY", P0_ATTR, "10-20", "query hypothesis"),
            _located("RENDER", P0_PEER, "30-40", "binding hypothesis"),
        ],
        windows=[],
        guard="counterfactual",
        expect={
            "converged": True,
            "attributed_file": P0_ATTR,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    # ── P4 trace grounding (generic prose that never engages the attributed locus) ──
    CorpusCase(
        id="trace-grounding",
        summary="a generic consistent trace that names no attributed locus must abstain",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            trace="the code is consistent with the reported symptom",
        ),
        located=[_located("QUERY", P0_ATTR, "10-20", "query hypothesis")],
        windows=[],
        guard="trace_grounding",
        expect={
            "converged": False,
            "attributed_file": P0_ATTR,
            "stamp": "trace_ungrounded",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="trace-grounding-neg",
        summary="a trace that names the attributed file basename is grounded",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            trace="query.py selects the stale row and reproduces the symptom",
        ),
        located=[_located("QUERY", P0_ATTR, "10-20", "query hypothesis")],
        windows=[],
        guard="trace_grounding",
        expect={
            "converged": True,
            "attributed_file": P0_ATTR,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    # ── P5 evidence sufficiency (thin, wholly ungrounded floor-level guess) ──
    CorpusCase(
        id="sufficiency",
        summary="a floor-count consistent guess with no grounding must abstain",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            counterfactual="correcting query.py removes the symptom",
            refuted_peers=[{"file": P0_PEER, "lines": "30-40", "why_not": "not causal"}],
        ),
        located=[
            _located("QUERY", P0_ATTR, "10-20", "query hypothesis"),
            _located("RENDER", P0_PEER, "30-40", "binding hypothesis"),
        ],
        windows=[],
        guard="sufficiency",
        expect={
            "converged": False,
            "attributed_file": P0_ATTR,
            "stamp": "low_confidence",
            "outcome": "DEMOTE",
        },
    ),
    CorpusCase(
        id="sufficiency-neg",
        summary="a field-producer window grounds the choice and keeps convergence",
        captured=False,
        result=_result(
            P0_ATTR, "10-20",
            counterfactual="correcting query.py removes the symptom",
            refuted_peers=[{"file": P0_PEER, "lines": "30-40", "why_not": "not causal"}],
        ),
        located=[
            _located("QUERY", P0_ATTR, "10-20", "query hypothesis"),
            _located("RENDER", P0_PEER, "30-40", "binding hypothesis"),
        ],
        windows=[{"file": P0_ATTR, "lines": "10-20", "via": "field-producer",
                  "field": "result_value", "text": "out['result_value'] = row"}],
        guard="sufficiency",
        expect={
            "converged": True,
            "attributed_file": P0_ATTR,
            "stamp": None,
            "outcome": "SILENT",
        },
    ),
    # ── design_change carve-out (field-provenance must not re-point a vetted design site) ──
    CorpusCase(
        id="design-change-field",
        summary="a judge-tagged design_change site is preserved, not re-pointed to the producer",
        captured=False,
        result=_result(
            DC_SITE, "60-78",
            trace="head_policy.py implements the head-selection design the reporter rejects",
        ),
        located=[
            _located("HEAD_POLICY", DC_SITE, "60-78",
                     "faithfully implements the rejected head-selection design",
                     vtype="design_change"),
            _located("SERIALIZER", DC_PRODUCER, "20-31",
                     "mechanically serializes workflow_head_type into the response"),
        ],
        windows=DC_FIELD_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": DC_SITE,
            "stamp": "design_change_preserved",
            "outcome": "PRESERVE",
        },
    ),
    CorpusCase(
        id="design-change-field-neg",
        summary="without the design_change tag the same site re-points (carve-out is narrow)",
        captured=False,
        result=_result(
            DC_SITE, "60-78",
            trace="head_policy.py looks relevant to the head value",
        ),
        located=[
            _located("HEAD_POLICY", DC_SITE, "60-78", "looks relevant to the head value"),
            _located("SERIALIZER", DC_PRODUCER, "20-31",
                     "mechanically serializes workflow_head_type into the response"),
        ],
        windows=DC_FIELD_WINDOWS,
        guard="field_provenance",
        expect={
            "converged": True,
            "attributed_file": DC_PRODUCER,
            "stamp": "field_provenance_repointed",
            "outcome": "RE-POINT",
        },
    ),
    # ── design_change carve-out (HTTP datasource must not re-point a vetted design site) ──
    CorpusCase(
        id="design-change-http",
        summary="a design_change-tagged attribution is preserved, not re-pointed to the datasource",
        captured=False,
        result=_result(M036_DECOY, "101-144", trace="this route encodes the rejected design"),
        located=[
            _located("MODULE_ROUTE", M036_DECOY, "101-144",
                     "faithfully encodes the module-listing design the reporter rejects",
                     vtype="design_change"),
            M036_LOCATED[1],
        ],
        windows=M036_WINDOWS,
        guard="http_datasource",
        expect={
            "converged": True,
            "attributed_file": M036_DECOY,
            "stamp": "design_change_preserved",
            "outcome": "PRESERVE",
        },
    ),
    # ── R0001 forced winning-path plumbing-decoy family (captured, code_root-backed) ──
    # ABSTAIN cases (R0001 #3 GREEN): readable captured plumbing → gate abstains → the
    # attribution STAYS on dispose_group, NO re-point. The meta test proves each flips to a
    # re-point when the gate is disabled (gate無력화 → RED).
    CorpusCase(
        id="plumbing-get_store-connection",
        summary="captured run-480 get_store decoy on the winning path → gate abstains, "
                "attribution stays on dispose_group",
        captured=True,
        result=_forced_inject_result(),
        located=_forced_inject_located(DECOY_CONN, "1-6"),
        windows=_forced_fe_windows(),
        guard="http_datasource",
        producer_files={DECOY_CONN: _CAP_GET_STORE},
        expect={
            "converged": True,
            "attributed_file": DISPOSE_HANDLER,
            "stamp": None,
            "outcome": "ABSTAIN",
        },
    ),
    CorpusCase(
        id="plumbing-get_store-projects",
        summary="same get_store plumbing shape at the run-477 position (db/projects.py) → "
                "abstain holds (locus-independent, R0001 #4)",
        captured=True,
        result=_forced_inject_result(),
        located=_forced_inject_located(DECOY_PROJ, "1-6"),
        windows=_forced_fe_windows(),
        guard="http_datasource",
        producer_files={DECOY_PROJ: _CAP_GET_STORE},
        expect={
            "converged": True,
            "attributed_file": DISPOSE_HANDLER,
            "stamp": None,
            "outcome": "ABSTAIN",
        },
    ),
    CorpusCase(
        id="plumbing-get_connection",
        summary="get_connection accessor variant (shape variation, R0001 #4) → abstain",
        captured=False,
        result=_forced_inject_result(),
        located=_forced_inject_located(DECOY_CONN, "1-2"),
        windows=_forced_fe_windows(),
        guard="http_datasource",
        producer_files={DECOY_CONN: _CAP_GET_CONNECTION},
        expect={
            "converged": True,
            "attributed_file": DISPOSE_HANDLER,
            "stamp": None,
            "outcome": "ABSTAIN",
        },
    ),
    CorpusCase(
        id="plumbing-get_conn-store",
        summary="get_conn accessor at db/store.py (getter+location variation) → abstain",
        captured=False,
        result=_forced_inject_result(),
        located=_forced_inject_located(DECOY_STORE, "1-2"),
        windows=_forced_fe_windows(),
        guard="http_datasource",
        producer_files={DECOY_STORE: _CAP_GET_CONN},
        expect={
            "converged": True,
            "attributed_file": DISPOSE_HANDLER,
            "stamp": None,
            "outcome": "ABSTAIN",
        },
    ),
    # EMPTY-PTEXT FAIL-OPEN lock (R0001 NR0003 §3/§7-4): the SAME injection with NO readable
    # producer code (producer_files=None → code_root=None → ptext=="") BYPASSES the gate and
    # re-points onto the plumbing decoy. This is the latent hole the live run-480 miss rode
    # (the gate predates the run, but the bypass survives it). Pinned here so the current
    # fail-open is explicit and a future corpus author cannot add a code_root-less plumbing
    # case and mistake the resulting bypass for a passing abstain.
    CorpusCase(
        id="plumbing-empty-ptext-failopen",
        summary="no readable producer code → gate bypassed → re-points onto the decoy "
                "(documents the empty-ptext fail-open; NOT a desired outcome)",
        captured=False,
        result=_forced_inject_result(),
        located=_forced_inject_located(DECOY_CONN, "1-6"),
        windows=_forced_fe_windows(),
        guard="http_datasource",
        producer_files=None,
        expect={
            "converged": True,
            "attributed_file": DECOY_CONN,
            "stamp": "http_datasource_provenance_repointed",
            "outcome": "RE-POINT (fail-open)",
        },
    ),
]


def run_case(case: CorpusCase) -> ConvergeResult:
    """Run one corpus case through its real guard using fresh mutable inputs."""
    res = deepcopy(case.result)
    located = deepcopy(case.located)
    windows = deepcopy(case.windows)
    inputs = (res.raw or {}).get("guard_inputs", {})

    if case.guard == "field_provenance":
        fp_windows = [w for w in windows if w.get("via") == "field-producer"]
        return _field_provenance_guard(res, located, fp_windows)
    if case.guard == "premise_refuted":
        return _premise_refuted_guard(
            res,
            db_available=bool(inputs.get("db_available")),
            chain_broke=bool(inputs.get("chain_broke")),
        )
    if case.guard == "http_datasource":
        if case.producer_files:
            # Materialize the captured producer source so the gate's _looks_like_datasource
            # check has readable ``ptext`` (the plumbing-abstain path is dead under
            # code_root=None — 0032 NR0003 §3). The temp tree lives only for the guard call.
            with tempfile.TemporaryDirectory() as root:
                for rel, src in case.producer_files.items():
                    path = os.path.join(root, rel.replace("/", os.sep))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write(src)
                return _http_datasource_provenance_guard(
                    res, located, windows, code_root=root
                )
        return _http_datasource_provenance_guard(
            res, located, windows, code_root=None
        )
    if case.guard == "dropped_peer":
        return _dropped_peer_guard(res, located, data_backed=res.data_state_backed)
    if case.guard == "data_stamp":
        return _data_stamp_guard(
            res,
            db_available=bool(inputs.get("db_available")),
            data_backed=res.data_state_backed,
        )
    if case.guard == "counterfactual":
        return _counterfactual_complete_guard(res, located)
    if case.guard == "trace_grounding":
        return _trace_grounding_guard(res, code_root=None)
    if case.guard == "sufficiency":
        return _evidence_sufficiency_guard(
            res, located, windows,
            min_located=int(inputs.get("min_located", 2)),
            data_backed=res.data_state_backed,
        )
    raise ValueError(f"unknown corpus guard: {case.guard}")
