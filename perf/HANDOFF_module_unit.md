# perf/ 스윕 — 모듈 단위 전환 인수인계 (Claude→GPT)

이 문서 하나로 이어받을 수 있게 정리. 결정은 끝났고, 코드 일부만 남았다.

## 0. 한 줄

T901/TR901 모델 퍼포먼스 스윕을 **풀-파이프라인 + 수정완료(red→green) 채점**에서
**모듈 단위 + 골든-로커스 재현율 채점**으로 바꾼다. **이미 합의된 설계다. 재논의 금지.**

## 1. 무엇을 측정하나 (사용자 확정)

> "이건 모델을 내려도 되나 안되나를 테스트하는 거지, 수정이 되나 안되나를 보는 게 아니다."

- OFAT: 베이스라인 고정, **한 역할만** down/base/up 으로 스왑(matrix.json 그대로).
- 각 셀은 **그 단계의 자기 출력 단계까지만** 돌리고, 그 출력이 **골든 로커스(올바른 위치)를
  짚었는지**(재현율)로 채점한다. **apply 안 함, 수정완료(red→green) 안 함, 대상 트리 읽기만.**
- "이 단계 모델 내려도 되나" = 같은 단계의 `down` 재현율이 `base` 재현율보다 안 나빠지면 OK.
- copilot specify/review 꼬리를 단계마다 끌고 가지 않으므로 비용이 어제 풀런의 **~절반(추정 $2~2.5)**.
  사용자 "절반정도면 감수 가능" 승인함.

## 2. 골든 로커스 (golden/manifest.json, 베이스네임으로 매칭 — 전부 유일)

- **core 3**: `process_service.py` · `ToastContainer.vue` · `NewRequirementModal.vue`
- **support 3**: `ko.ts` · `en.ts` · `ja.ts`
- core_recall / full_recall 두 수치로 본다.

## 3. 단계별 — 무슨 명령으로 돌리고 무슨 산출물을 채점하나

| 단계 | path | 실행 명령 | 산출물 | 채점 투영(score.py) |
|---|---|---|---|---|
| queen | investigate | `investigate`(--specify **없이**) | verdict.json | `queen_globs` (axes의 file_globs 합집합) |
| judge | investigate | `investigate`(없이) | verdict.json | `judge_located` (verdicts located+candidates) |
| scout | investigate | `investigate`(없이, reinforce on) | verdict.json | `judge_located` (안 뜨면 measurement=null) |
| converge | investigate | `investigate`(없이) | verdict.json | `converge_attributed` (attributed_defect+path) |
| specify | investigate | `investigate --specify` | verdict.edit_spec.json | `authored` (edits[].file) |
| review | investigate | `investigate --specify` | verdict.edit_spec.json | `surviving` (authored − ineffective_ids) |
| swarm | run | `run`(--specify **없이**) | honey.md | honey가 명명한 골든 베이스네임 |
| assemble | run | `run`(없이) | honey.md | 동일 |

핵심: **converge는 run_investigate 안에서 ≥2 located면 자동 실행** → queen/judge/converge/scout는
`--specify` 없이도 verdict.json에 converge 귀속이 들어온다(specify 비용 0). `--specify`는 그 뒤
specify+review를 덧붙일 뿐. 그래서 queen/judge/converge/scout 셀은 싸다(~$0.024/rep).

## 4. 이미 끝난 것

- **`perf/score.py` 전면 교체 완료**(이번 커밋 대상, 미실행). 더는 pytest/apply를 안 한다.
  새 공개 API: `score_cell(stage:str, rep_dir:str, golden:dict) -> dict`.
  반환: `{measurement:"recall"|None, core_recall, full_recall, located[], core_hit[],
  projection, projections{...}}`. CLI: `python perf/score.py --stage judge --rep-dir <dir>`.
  - `golden_loci(manifest)` / `loci_from_verdict` / `loci_from_edit_spec` / `loci_from_honey` 헬퍼.
  - 경로는 `./`·역슬래시 무관 **베이스네임**으로 골든과 대조(충돌 없음).

## 5. 남은 것 (GPT가 할 일)

### (A) `perf/run_sweep.py` — 가장 큰 작업

현재는 `reset → 풀 파이프라인 → apply --write --verify → scorer.evaluate(수정완료) → reset`.
이걸 다음으로 바꾼다:

1. **`cell_commands()`**: `apply_cmd` 제거. 파이프라인 명령만 만든다.
   - `path=="investigate"`: `investigate --seed … --codebase … --docs … --out <rep>/verdict.json`,
     그리고 **stage ∈ {specify, review} 일 때만 `--specify` 추가**.
   - `path=="run"`: `run --seed … --recipe <repo>/recipes/recipe_code_bug.md --codebase … --out <rep>/honey.md` (**--specify 없이**).
2. **`run_cell()`**: apply 단계와 `scorer.evaluate(...)` 호출 제거. 대신 파이프라인 실행 후
   `score = scorer.score_cell(cell["stage"], rep_dir, golden_dict)` 호출.
   `result["verdict"]`는 더는 PASS/FAIL이 아니라 `score`(measurement/core_recall/…)를 담는다.
   - **git reset 은 위생 목적으로 유지**(specify 효과성 게이트의 라이브 RED 프로브가 server/tests에
     임시 파일을 남길 수 있음). `_reset_branch` 가드 그대로. 단 이제 트리 변형은 그게 전부.
   - 골든은 한 번 로드: `golden_dict = json.load(open(GOLDEN))`.
3. **`write_summary()`**: **STAGE별로 그룹**. 각 단계에 대해 down/base/up 셀의 평균 core_recall(3 reps)을
   나란히 출력하고, 판정 컬럼: `down_recall ≥ base_recall → "↓OK"`, 미만 → `"↓REGRESS"`,
   `up_recall > base_recall → "↑+"`, 같으면 `"↑="`. measurement=null(예: scout 미발화)은 `n/a`.
   - 표 헤더 예: `| stage | down | base | up | ↓verdict | ↑verdict | reps |`.
4. `run_smoke()`는 이미 propose-only라 거의 그대로 OK(원하면 끝에 `score_cell`로 한 번 찍어 확인).

### (B) `perf/matrix.json` — 작게

- `_about` 문구를 "수정완료 채점"→"모듈 단위 골든-로커스 재현율 채점"으로 갱신.
- 셀/레벨/프로바이더/baseline_roles/corrections는 **그대로**. (구조 안 바꿈.)
- `golden`/`recipe`/`target` 키 유지. apply 관련 가정만 문서에서 빠지면 됨.

### (C) `perf/README.md` — 채점/실행/규모 절을 모듈 단위로 갱신

- "합격 기준 = 수정완료" 절 → "채점 = 골든-로커스 재현율(단계별, 읽기 전용)"로 교체.
- 실행 순서는 동일(`gen_profiles.py` → `--smoke` → `--dry-run` → 본 스윕).
- 규모/비용: 24셀×3, **apply·pytest 없음**, 추정 ~$2~2.5, scout는 M037에서 미발화 가능(null).

## 6. 안전·실행 규칙 (불변)

- **본 스윕(유료)은 사용자 승인 전 실행 금지.** smoke/dry-run만 무승인 OK.
- smoke 모델 = `openai/gpt-oss-20b` (matrix.smoke_model). 연결 확인용.
- `gpt-5-mini`는 deepinfra에 없음 → 해당 "보통" 행은 **provider=copilot**(matrix corrections 참조).
- 실행 환경: copilot 토큰 + .venv 필요. `cmd /c call %USERPROFILE%\.ai_launcher_secrets.bat && .venv\Scripts\python perf\run_sweep.py --smoke`.
- 대상 브랜치 `FlowGate-dev/branches/20260607`(되돌린 버그 상태), baseline_sha `fc2a26e`.
  reset 가드: work tree ∧ 경로에 `20260607` 포함 아니면 중단.

## 7. 절대 하지 말 것

- 수정완료(red→green)·apply 를 되살리지 말 것 — 사용자가 명시적으로 "수정 여부가 아니다"라고 함.
- 단계별로 "정답 산출물 골든"을 새로 발명하지 말 것 — 채점 기준은 **공통**(골든-로커스 재현율) 하나.
- 모델을 더 키워서 때우지 말 것. 측정만 한다.
