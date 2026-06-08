# T901 / TR901 — Hive 모델 퍼포먼스 스윕

각 부품(스테이지)의 모델을 현재 기준에서 올리거나 내렸을 때도 그 단계가 올바른 위치를
짚는지 측정하는 하네스. OFAT(한 번에 한 역할만 바꿈) 방식이며, 수정 적용 여부가 아니라
**모듈 단위 골든-로커스 재현율**을 본다.

- **대상 작업**: T901 = M037 (그룹 내 R 중복 방지 + 토스트 표출)
- **대상 코드**: `C:\workspace\projects\FlowGate-dev\branches\20260607` (수정이 되돌려진 버그 존재 상태)
- **시드 정본**: `…/FlowGate/410_tasks/T901_prevent_duplicate_r_in_group_and_fix_toast.md`
- **골든 로커스**: `perf/golden/manifest.json`

## 왜 두 경로인가

한 파이프라인이 8개 스테이지를 다 돌지 않는다:

| 경로 | CLI | 이 경로에서 채점하는 스테이지 |
|---|---|---|
| investigate | `investigate` (`specify/review`만 `--specify`) | **queen · judge · converge · scout · specify · review** |
| run | `run --specify` (edit-spec로 채점) | **swarm · assemble** |

queen/judge/converge/scout는 `investigate` 산출물인 `verdict.json`만 채점한다. specify/review는
`investigate --specify`가 만든 `verdict.edit_spec.json`을 채점한다. swarm/assemble은 `run --specify`가
만든 `honey.edit_spec.json`을 채점한다(honey 본문만 grep하면 assemble이 시드를 그대로 박아 넣어 시드가
명명한 골든 파일이 항상 잡혀 재현율이 1.0에 고정되므로, 모델에 민감한 edit-spec authored 로커스를 쓴다).
apply는 하지 않는다.

## 채점 = 골든-로커스 재현율

셀마다 해당 단계의 자기 산출물에서 골든 파일 베이스네임을 얼마나 짚었는지 측정한다. 대상 트리는
읽기 전용이며, `apply --write`, pytest, red→green 수정완료 판정은 실행하지 않는다.

| 단계 | 산출물 | 채점 투영 |
|---|---|---|
| queen | `verdict.json` | axes의 `search_plan.file_globs` 합집합 |
| judge | `verdict.json` | `verdicts`의 located + candidates |
| scout | `verdict.json` | judge와 동일. 미발화 시 `measurement=null` 가능 |
| converge | `verdict.json` | `attributed_defect` + `path` |
| specify | `verdict.edit_spec.json` | `edits[].file` |
| review | `verdict.edit_spec.json` | authored - `ineffective_ids` |
| swarm | `honey.md` | honey가 명명한 골든 베이스네임 |
| assemble | `honey.md` | swarm과 동일 |

요약은 stage별로 down/base/up의 평균 `core_recall`을 나란히 보여준다. `down >= base`면 `↓OK`,
미만이면 `↓REGRESS`, `up > base`면 `↑+`, 같으면 `↑=`로 표시한다. `measurement=null`은 `n/a`.

## 두 가지 교정

1. **gpt-5-mini는 deepinfra(openai)에 없다.** 표의 `…|openai|gpt-5-mini|보통` 행은 연결 실패하므로
   `provider=copilot`으로 돌린다(현 베이스라인과 동일). matrix에서 provider만 바꾸면 원복.
2. **scout(reinforce)는 조건부 발화.** coverage_risk 플래그 + 빈 FIND 축에서만 뜬다. scout 셀은
   `reinforce.enabled=true`로 켜두되, M037에서 안 뜨면 측정 불가(`measurement=null`)로 남는다.

## 실행

```powershell
# 0) (선택) 프로파일만 미리 생성 — 무료, config/hive.config.perf-*.json 24개 + perf-smoke
python perf/gen_profiles.py

# 1) 연결 확인 — 전 역할 openai/gpt-oss-20b, 쓰기 없음
python perf/run_sweep.py --smoke

# 2) 계획 점검 — 아무것도 실행 안 함
python perf/run_sweep.py --dry-run

# 3) 실제 스윕 — 유료. 셀×3회 직렬, 매 런 사이 브랜치 리셋
python perf/run_sweep.py
#    부분만:  python perf/run_sweep.py --only judge-up,judge-base,judge-down
```

실행 환경 예 (토큰은 `hive.py`가 `~/.hivework/.env`에서 자동 로드 — 별도 시크릿 호출 불필요):

```powershell
.venv\Scripts\python perf\run_sweep.py --smoke
```

결과: `perf/results/<cell>/<rep>/` (verdict/honey, log, result.json) + 셀별 `ledger.db` +
롤업 `perf/results/summary.md` / `summary.json`. summary는 셀마다 끝날 때 체크포인트로 갱신된다.

## 안전

- 본 스윕(유료)은 사용자 승인 전 실행 금지.
- 모든 `git reset --hard`는 **가드** 통과 후에만: 대상이 git work tree이고 경로에 브랜치 leaf
  (`20260607`)가 들어 있어야 한다. 아니면 즉시 중단한다.
- `--smoke` / `--dry-run`은 대상 트리를 변형하지 않는다.
- 실제 스윕은 apply를 하지 않지만, specify 효과성 게이트의 라이브 RED 프로브가 임시 파일을 남길 수 있어
  위생 목적으로 매 런 전후 브랜치 작업트리를 baseline_sha로 되돌린다.

## 규모·비용

- 셀: investigate 18 + run 6 = **24 × 3회 = 72런**.
- apply·pytest 없음. 예상 비용은 약 **$2~2.5**.
- 셀별 `ledger.db`로 호출수·토큰·문자 실측. 정확한 $는 단가표 적용(가격은 코드에 박지 않음).

## 파일

- `matrix.json` — 표 → 24셀 (경로·역할·provider·model·레벨). **이 스윕의 정본 계획.**
- `gen_profiles.py` — 셀별 `config/hive.config.perf-*.json` 생성 (`--clean`으로 제거).
- `run_sweep.py` — 무인 드라이버 (`--smoke` / `--dry-run` / `--only` / `--repeats`).
- `score.py` — 골든-로커스 재현율 채점기 (단독 실행 가능).
- `golden/manifest.json` — 골든 로커스.
