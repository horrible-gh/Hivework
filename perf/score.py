#!/usr/bin/env python3
"""Module-unit scorer for the T901/TR901 sweep — "can this stage's model be lowered?"

This is NOT a '수정완료' (end-to-end fix) scorer. The sweep's question is per-STAGE:
when we raise/lower ONE role's model, does THAT stage still do its job? We answer it
WITHOUT applying anything to the target tree (the whole sweep is read-only): we read
the stage's own output artifact and measure how well it pointed at the RIGHT place —
the golden loci (TR901's known-correct files). No pytest, no apply, no copilot tail
beyond what the stage under test naturally runs.

Per stage, the artifact and the projection scored:

  queen     verdict.json   union of axes' search_plan.file_globs (what queen chose to probe)
  judge     verdict.json   verdicts[].verdict.file + candidates (what judge localised)
  scout     verdict.json   same as judge (scout reinforces judge's evidence)
  converge  verdict.json   attributed_defect.file + path[].file (what converge stitched/attributed)
  specify   edit_spec.json  edits[].file (where specify authored edits)
  review    edit_spec.json  edits[].file MINUS effectiveness.ineffective_ids (what review let stand)
  swarm     honey.md        golden basenames the honey names/locates (assemble's rendered findings)
  assemble  honey.md        same as swarm

The score is GOLDEN-LOCUS RECALL: of the golden files (the right places), how many did
this stage's output identify. Core loci (the 3 functional fixes) are scored apart from
the support loci (i18n). The driver compares each stage's down/base/up recall: a stage
"tolerates lowering" when its `down` recall is no worse than its `base` recall.

Golden loci (from golden/manifest.json), all basenames distinct so basename match is safe:
  core:    process_service.py · ToastContainer.vue · NewRequirementModal.vue
  support: ko.ts · en.ts · ja.ts
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _load_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _base(path: str) -> str:
    """Basename of a possibly ``./``-prefixed, back-slashed, repo-relative path."""
    return os.path.basename(str(path or "").replace("\\", "/").rstrip("/"))


def golden_loci(manifest: dict) -> dict:
    """Resolve the golden manifest into core/support/all sets of FILE BASENAMES.

    Core = the weight:"core" loci (the functional fixes). Support = i18n. We compare
    by basename because verdict/edit-spec paths vary in ``./`` prefixing and relative
    roots, while the golden basenames are mutually distinct (no collision risk).
    """
    loci = manifest.get("loci", {})
    core: set[str] = set()
    support: set[str] = set()
    for key, spec in loci.items():
        if key == "i18n_keys":
            for f in (spec.get("files") or {}):
                support.add(_base(f))
            continue
        bucket = core if spec.get("weight") == "core" else support
        if spec.get("file"):
            bucket.add(_base(spec["file"]))
    return {"core": core, "support": support, "all": core | support}


# ── Per-artifact locus extraction ───────────────────────────────────────────

def loci_from_verdict(v: dict) -> dict:
    """Project a verdict.json into the per-stage candidate basename sets."""
    verdicts = v.get("verdicts") or []
    queen_globs: set[str] = set()
    judge_located: set[str] = set()
    for ax in verdicts:
        # queen_globs measures the QUEEN's own decomposition — so exclude the
        # deterministically INJECTED SEED_ANCHOR axis, whose globs are always the
        # seed-named files regardless of the queen model (investigate.py
        # _prioritize_axes). Counting it would floor queen recall at 1.0 and make the
        # down/base/up comparison blind. Other axes are the queen's real output.
        if ax.get("axis_id") != "SEED_ANCHOR":
            for g in (ax.get("search_plan") or {}).get("file_globs") or []:
                queen_globs.add(_base(g))
        ver = ax.get("verdict") or {}
        if ver.get("located") and ver.get("file"):
            judge_located.add(_base(ver["file"]))
        for c in ax.get("candidates") or []:
            if c.get("file"):
                judge_located.add(_base(c["file"]))

    conv = v.get("converge") or {}
    converge_attr: set[str] = set()
    ad = conv.get("attributed_defect")
    if isinstance(ad, dict) and ad.get("file"):
        converge_attr.add(_base(ad["file"]))
    for d in conv.get("additional_defects") or []:
        if isinstance(d, dict) and d.get("file"):
            converge_attr.add(_base(d["file"]))
    converge_path: set[str] = {_base(n.get("file")) for n in (conv.get("path") or [])
                               if isinstance(n, dict) and n.get("file")}
    winning_producer: set[str] = {
        _base(n.get("file")) for n in (conv.get("winning_path") or [])
        if isinstance(n, dict) and n.get("file") and n.get("role") == "producer"}
    return {
        "queen_globs": queen_globs,
        "judge_located": judge_located,
        "converge_attributed": converge_attr | converge_path,
        "winning_producer": winning_producer,
    }


def loci_from_edit_spec(spec: dict) -> dict:
    """Project an edit_spec.json into authored / surviving (review-kept) basenames."""
    edits = spec.get("edits") or []
    ineffective = set((spec.get("effectiveness") or {}).get("ineffective_ids") or [])
    authored: set[str] = set()
    surviving: set[str] = set()           # what review let stand (not ineffective)
    for e in edits:
        b = _base(e.get("file"))
        if not b:
            continue
        authored.add(b)
        if e.get("id") not in ineffective:
            surviving.add(b)
    deferred = {_base(d.get("evidence", [""])[0]) if d.get("evidence") else ""
                for d in (spec.get("deferred") or [])}
    return {"authored": authored, "surviving": surviving,
            "deferred": {d for d in deferred if d}}


def loci_from_honey(text: str, golden_all: set[str]) -> set[str]:
    """Which golden basenames the honey markdown names (crude recall for run path)."""
    return {b for b in golden_all if re.search(re.escape(b), text)}


# ── Per-stage scoring ────────────────────────────────────────────────────────

# Which artifact + projection each stage is scored on. ``primary`` is the locus set
# whose golden-recall is THE number for the stage; the others ride along as context.
_STAGE_PROJECTION = {
    "queen":    ("verdict",  "queen_globs"),
    "judge":    ("verdict",  "judge_located"),
    "scout":    ("verdict",  "judge_located"),
    "converge": ("verdict",  "converge_attributed"),
    "specify":  ("edit_spec", "authored"),
    "review":   ("edit_spec", "surviving"),
    "swarm":    ("honey",    "honey"),
    "assemble": ("honey",    "honey"),
}


def _recall(located: set[str], golden: set[str]) -> float:
    if not golden:
        return 0.0
    return round(len(located & golden) / len(golden), 3)


def score_cell(stage: str, rep_dir: str, golden: dict) -> dict:
    """Score one cell-run's stage output by golden-locus recall (no tree mutation).

    Reads whichever artifact the stage produces from ``rep_dir`` and returns the
    located basenames + core/full recall. ``measurement: null`` when the artifact is
    absent (e.g. scout never fired, or the pipeline produced nothing).
    """
    g = golden_loci(golden)
    kind, proj = _STAGE_PROJECTION.get(stage, ("verdict", "judge_located"))
    located: set[str] = set()
    projections: dict[str, list[str]] = {}
    artifact = None

    if kind == "edit_spec":
        spec = _load_json(os.path.join(rep_dir, "verdict.edit_spec.json"))
        if spec is not None:
            artifact = "edit_spec"
            ls = loci_from_edit_spec(spec)
            projections = {k: sorted(vv) for k, vv in ls.items()}
            located = ls.get(proj, set())
    elif kind == "honey":
        text = _read(os.path.join(rep_dir, "honey.md"))
        # run path may also emit an edit_spec when --specify is on; prefer it if present.
        spec = _load_json(os.path.join(rep_dir, "honey.edit_spec.json"))
        if spec is not None:
            artifact = "edit_spec"
            ls = loci_from_edit_spec(spec)
            projections = {k: sorted(vv) for k, vv in ls.items()}
            located = ls.get("authored", set())
        elif text:
            artifact = "honey"
            located = loci_from_honey(text, g["all"])
            projections = {"honey": sorted(located)}
    else:  # verdict
        v = _load_json(os.path.join(rep_dir, "verdict.json"))
        if v is not None:
            artifact = "verdict"
            ls = loci_from_verdict(v)
            projections = {k: sorted(vv) for k, vv in ls.items()}
            located = ls.get(proj, set())

    if artifact is None:
        return {"measurement": None, "stage": stage, "reason": "no artifact produced"}

    return {
        "measurement": "recall",
        "stage": stage,
        "artifact": artifact,
        "projection": proj,
        "located": sorted(located),
        "core_hit": sorted(located & g["core"]),
        "core_recall": _recall(located, g["core"]),
        "full_recall": _recall(located, g["all"]),
        "golden_core": sorted(g["core"]),
        "golden_all": sorted(g["all"]),
        "projections": projections,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a cell-run's stage output by golden-locus recall")
    ap.add_argument("--stage", required=True, help="pipeline stage of the varied role")
    ap.add_argument("--rep-dir", required=True, help="perf/results/<cell>/<rep> dir")
    ap.add_argument("--golden", default=os.path.join(HERE, "golden", "manifest.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    golden = _load_json(args.golden) or {}
    res = score_cell(args.stage, args.rep_dir, golden)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
