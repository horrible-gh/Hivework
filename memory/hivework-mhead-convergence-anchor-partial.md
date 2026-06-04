---
name: hivework-mhead-convergence-anchor-partial
description: M-head 케이스 첫 honey↔proposal 수렴 + specify anchor widening·apply partial atomicity 두 무료수정, needs_pm/재조사비용 진단
metadata:
  type: project
---

M-head 케이스(`_parse_doc_workflow` head_type 선택이 NON_HEAD_TYPES={R,M,Q}로 M 스킵 → DS가 head, 5/30 회귀 커밋 1ebb861b)에서 **처음으로 honey↔proposal 완전 일치**(인과 consistent). 그동안 게이트들(인과·라이브그라운딩·재조사라우팅)이 한 점에 수렴한 첫 관측.

**세 가지 닫음(2026-06-04, 664 passed, 미커밋, 무료·결정적):**

1. **Q1 needs_pm 누수=신기루**: 코드에선 이미 은퇴(specify.py:554 termination==needs_pm → needs_reinvestigation 강제coerce, VALID_TERMINATION에 없음). 리포트에 보이는 "needs_pm/PM에 올림"은 **운영AI의 한국어 내러티브 어휘**일 뿐 Hive 출력 아님. 코드 잔존은 test_parse·test_conflict_scan의 의도적 legacy 픽스처(historical capture)뿐. → 코드변경 불필요.

2. **Q2 조사 처음부터 재실행=명령선택 문제**: 표준 `hive specify --honey <파일>`은 제공된 honey를 재사용·재조사 안 함(reinvestigation 루프 없음). 운영AI가 `hive investigate --seed ... --specify`(체인경로, decompose→retrieve→judge→honey→specify 전부 재실행)를 골랐기 때문. 게다가 config `reinvestigation.live=true`(max_rounds=2)가 체인경로에서만 발화 → NR이면 최대 3배 비용. **Hive 의도가 아니라 명령+플래그 조합이 재실행시킴**. 비용 곤란하면: ①제공 honey엔 `specify --honey`/`apply` 쓰기 ②또는 reinvestigation.live off.

3. **Q3 두 결함 수정**:
   - **specify `_disambiguate_anchors`**(specify.py, run_specify에서 `_verify_anchors_live` 직전 호출): 동일 비유니크 anchor를 공유하는 sibling edit들(두 브랜치 같은 리터럴)을 인접 라이브 라인으로 widening해 각각 유니크 타깃화. 안전조건=같은 (file,anchor)·occurrence수==edit수·**replacement_new 전부 동일**(매핑 무관)일 때만; replacement 다르면 매핑 못 믿어 downgrade경로로. `_unique_line_window`(라인경계로 ±확장하며 count==1까지). 이후 verify가 유니크 확인→verified 유지.
   - **apply partial atomicity**(apply.py build_proposal): `_is_test_file`(tests/·test_*·*_test·*.spec.*·*.test.* 판별) + 규칙=**source(비테스트) edit 중 하나라도 unwritable이면 모든 test edit을 writable에서 hold**(held_reason 스탬프). 코드 실패+테스트만 적용된 깨진 반쪽 방지(코드없는 동작을 테스트가 기대=아무것도 안한것보다 나쁨). over-hold는 안전.

잔여=유료 라이브 재검증(M-head를 `specify --honey`로 돌려 anchor widening이 실제 ready_to_apply 내는지)+커밋(전체 미커밋). 관련 [[hivework-t892-defects-fix]] [[hivework-n177-seed-target-misscrape]] [[feedback-no-pm-handoff-ceremony]].
