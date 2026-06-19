# `runs.jsonl` — hive 사이클 텔레메트리 스키마

성능지표 레포트(`report.py`)의 입력 계약. **런 1회 = JSON 한 줄**(JSON Lines).
하이브가 사이클마다 한 줄을 append 하면, 그 파일 하나가 곧 시계열이 된다.
이 파일은 적재(emit) 코드가 아니라 **렌더러가 기대하는 형태**만 못박는다 — 적재 결선은 후속 작업.

레코드의 거의 모든 필드는 기존 `hive_ledger.db`(`runs` / `worker_calls` 테이블)에서
그대로 투영된다. 매핑은 각 필드 옆에 `← ledger.…`로 표기.

```jsonc
{
  "run_id": "run451",                       // 고유 런 식별자 (정렬·라벨용)
  "ts": "2026-06-19T16:42:00+09:00",        // ISO8601 ← runs.ts
  "seed": "B0001",                          // 시드/작업 식별 ← runs.seed
  "work_type": "investigate",               // investigate | run ← runs.work_type
  "codebase": "FlowGate-dev",               // 대상 레포 ← runs.codebase
  "models": { "queen": "gpt-5-mini",        // ← runs.model_queen
              "swarm": "gpt-oss-120b" },    // ← runs.model_swarm

  // ── 수확 퍼널: 위에서 아래로 단조 감소하는 6단계 카운트 ──────────────
  //    채팅 합의: 축 시도 → comb 발화 → comb-형태(진짜 finding)
  //              → 결론 전환 → 제출 → 통과
  "funnel": {
    "axes_attempted":       10,             // ← runs.axes_n
    "comb_fired":            9,              // comb_path 비어있지 않은 worker_call 수
    "comb_shaped":           1,             // is_comb_shaped 통과(진짜 finding) 수
    "conclusion_converted":  1,             // 결론으로 전환된 수
    "submitted":             1,             // 제출(edit-spec/honey)된 수
    "passed":                0              // 실제 수정→테스트 통과 수
  },

  // ── 축별 분해: 어느 축이 수확하고 어느 축이 0인가 ─────────────────────
  "axes": [
    { "axis_id": "A1", "label": "process_service.py",
      "fired": true, "findings": 0, "shaped": 0, "longest_chain_s": 41.2 }
    // ... 축마다 1개. ← worker_calls 의 axis_id/comb_path/latency_s 집계
  ],

  // ── 비용: provider별 분리 — 과금모델 2종(R0001) ────────────────────────
  //   크레딧계(copilot)  = 호출수 × 크레딧단가(1cr=$0.01) 정액. 토큰 무관.
  //   토큰계(deepinfra/openai호환) = 실토큰 × 단가(입력/출력 분리).
  //   credits/usd 는 분리 축: 크레딧계는 credits 로, 토큰계는 tokens 로 과금.
  "cost": {
    "by_provider": {
      "copilot":   { "tokens": 16928, "calls": 4, "credits": 0, "usd": 0.0 },
      "deepinfra": { "tokens": 9740,  "calls": 6, "credits": 0, "usd": 0.117 }
    }
    // tokens ← est_tokens 합, calls ← worker_call 수, usd ← 과금모델별 산정.
    // 단가·과금모델은 코드에 박지 않음 → prices.json(+_billing 블록).
  },

  // ── 사이클 결과 + 지연 ────────────────────────────────────────────────
  "cycle": {
    "fixes_landed": 0,                      // 실제 랜딩+통과한 수정 수 (= funnel.passed)
    "fixes_total":  2,                      // 시도한 수정 총수
    "wall_clock_s": 424.0                   // ← runs.elapsed_s
  },

  // ── 정확도(골든셋): NR0004 의 5버그 채점. 없으면 생략 가능 ──────────────
  //    "심은 N개 중 X개 찾음 · 헛다리 Y번 · 고쳐서 통과 Z개"
  "golden": {
    "seeded":          5,                   // 되감아 심은 버그 수
    "recalled":        1,                   // 그중 하이브가 찾은 수 (recall 분자)
    "false_positives": 0,                   // 안 심은 곳을 버그라 우긴 수 (헛다리)
    "verified_fixed":  0,                   // 찾은 것 중 고쳐서 테스트 통과한 수
    "per_bug": [                            // 버그별 채점(레벨 = 난이도 사다리)
      { "id": "0096", "level": 1, "found": true,  "fixed": false },
      { "id": "0077", "level": 2, "found": false, "fixed": false },
      { "id": "0082", "level": 3, "found": false, "fixed": false },  // R0014-B: 0064 폐기(정답 미합의·3회 재오픈→채점불가) → 0082(dispose FK·원인명확·교차레이어)로 교체
      { "id": "0062", "level": 4, "found": false, "fixed": false },
      { "id": "0059", "level": 5, "found": false, "fixed": false }
    ]
  }
}
```

## 렌더러가 계산하는 파생 지표 (저장하지 않음)

| 지표 | 정의 | 의미 |
|---|---|---|
| 재현율(recall) | `golden.recalled / golden.seeded` | **메인 정확도 지표(R0014-D)** — 심은 버그를 얼마나 꼼꼼히 찾나 |
| 정밀도(precision) | `recalled / (recalled + false_positives)` | **메인 정확도 지표(R0014-D)** — 얼마나 믿을 만한가(헛다리 안 하나) |
| 통과율(pass rate) | `fixes_landed / fixes_total` | **북극성** — 실제로 고쳐서 통과한 비율 |
| 수율(yield) | `comb_shaped / axes_attempted` | 보조 지표(R0014-D 강등) — 수확 퍼널의 건강도. 메인 카드에서 내려 퍼널 섹션으로 |
| 런당 비용 | `Σ provider.usd` | |
| 수율당 비용 | `usd / comb_shaped` | 효율 지표($/finding) |
| 수정당 비용 | `usd / fixes_landed` | $/fix |

## 견고성 규약

- 누락 필드는 0/빈값으로 흡수한다(부분 적재 중에도 레포트가 깨지지 않게).
- `golden` 블록이 없으면 정확도 섹션은 자동 생략된다(골든셋 미투입 런 허용).
- 런은 `ts` 오름차순으로 정렬해 추세를 그린다. 마지막 런이 "최신 사이클" 상세가 된다.
