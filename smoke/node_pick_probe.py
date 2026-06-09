"""Decisive probe: does role-separation (model picks symptom-relevant FE node →
codemap traces it) reach the BE-root? Per case → 1 or 0.

The codemap half is already proven (right node → BE-root). This measures the ONE
unmeasured variable: given the casual symptom + the queen's real FE candidates,
does a model pick a node that traces to the BE-root? No full pipeline — decompose
(real candidates) + one pick call + static trace per case.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hive import codemap as cm
from hive.secrets import load_secrets
from hive.config import load_config
from hive.decompose import run_decompose
from hive.be_root import _live_handler_for, _SNAKE_RE
from hive.providers import call_worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPE = os.path.join(ROOT, "recipes", "recipe_code_bug.md")
BR = r"C:\workspace\projects\FlowGate-dev\branches"

CASES = [
    {"id": "M035", "wt": os.path.join(BR, "20260608_M035"),
     "seed": "워크플로 결정 했는데 아직도 진행중으로 나오고, 다음이 메모인데 메모도 아니고… 액션바도 그렇고 이상해~",
     "be_root": ["documents/routers/documents.py", "workflow_head_type"]},
    {"id": "M036", "wt": os.path.join(BR, "20260608_M036"),
     "seed": "요건정의 작성할 때 모듈 선택이 없다. 나오게 해줘",
     "be_root": ["project_settings", "db/projects.py"]},
    {"id": "M037", "wt": os.path.join(BR, "20260607"),
     "seed": "이거 왜 한 그룹에 R문서가 두개가 작성되지? 이상하네. 토스트도 이상하고… 고쳤으면 좋겠다.",
     "be_root": ["process_service"]},
]

PICK_PROMPT = """A user reported this symptom in a Vue(frontend)+FastAPI(Python) app:
SYMPTOM: {seed}

Below are candidate frontend files the investigator surfaced, with the API
endpoints each calls and notable data fields each reads. ONE of these is the
real entry point for the symptom — the control/screen the user is talking about.

{cands}

Pick the SINGLE most relevant frontend file, and the SINGLE API endpoint OR data
field through which its symptom is produced by the backend. Think about which
screen/control the user means, not keyword overlap.
Reply with ONLY this JSON (no prose):
{{"file": "<one path above>", "endpoint": "<one /api/... it calls or empty>", "field": "<one snake_case field it reads or empty>"}}"""


def fe_files(result):
    out = []
    for t in result.get("tasks", []):
        for g in (t.get("search_plan") or {}).get("file_globs", []):
            g = str(g)
            if g.lower().endswith((".vue", ".ts")) and "*" not in g:
                out.append(g)
    return list(dict.fromkeys(out))[:12]


def candidates_block(code_root, files):
    lines = []
    for f in files:
        eps = cm.endpoints_in_file(code_root, f) or []
        try:
            with open(os.path.join(code_root, f), encoding="utf-8", errors="replace") as fh:
                fields = sorted({x for x in _SNAKE_RE.findall(fh.read())
                                 if x.count("_") >= 2})[:6]
        except OSError:
            fields = []
        lines.append(f"- {f}\n    endpoints: {eps[:5]}\n    fields: {fields}")
    return "\n".join(lines)


def trace(code_root, pick):
    found = set()
    ep = (pick.get("endpoint") or "").strip()
    if ep:
        hf = _live_handler_for(cm, code_root, ep)
        if hf:
            found.add(hf)
            for c in (cm.handler_callees(code_root, hf, "") or []):
                if c.get("module_file"):
                    found.add(c["module_file"])
    fld = (pick.get("field") or "").strip()
    if fld:
        for p in (cm.field_producers(code_root, fld) or [])[:4]:
            if p.get("file"):
                found.add(p["file"])
    return sorted(p.replace("\\", "/") for p in found if str(p).endswith(".py"))


def main():
    load_secrets()
    cfg = load_config()
    q = cfg.queen
    pk = {"read_only": cfg.copilot.read_only}
    tok = cfg.copilot.token or (os.environ.get(cfg.copilot.token_env)
                                if cfg.copilot.token_env else None)
    if tok:
        pk["copilot_token"] = tok
    results = []
    for c in CASES:
        os.chdir(ROOT)
        dec = run_decompose(seed_text=c["seed"], recipe_path=RECIPE,
                            codebase_root=c["wt"], model=q.model,
                            provider=q.provider, provider_kwargs=pk,
                            retries=q.retries, timeout=q.worker_timeout())
        files = fe_files(dec)
        cands = candidates_block(c["wt"], files)
        prompt = PICK_PROMPT.format(seed=c["seed"], cands=cands)
        out = call_worker(q.provider, q.model, prompt, cwd=None,
                          timeout=120, **pk).stdout
        try:
            s = out[out.index("{"):out.rindex("}") + 1]
            pick = json.loads(s)
        except Exception:
            pick = {}
        traced = trace(c["wt"], pick)
        hit = any(b.lower() in " ".join(traced).lower() for b in c["be_root"])
        results.append((c["id"], hit, pick, traced))
        print(f"\n{c['id']}: PICK={pick}\n   traced={traced}\n   "
              f">>> {'1 (BE-root reached)' if hit else '0 (missed)'}  be_root={c['be_root']}")
    print(f"\n{'='*50}\nMEASURED RESULT (1 or 0 per case)\n{'='*50}")
    for cid, hit, _, _ in results:
        print(f"  {cid}: {1 if hit else 0}")
    print(f"  TOTAL: {sum(1 for _,h,_,_ in results if h)}/{len(results)}")


if __name__ == "__main__":
    main()
