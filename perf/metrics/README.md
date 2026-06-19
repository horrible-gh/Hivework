# perf/metrics — 성능지표 HTML 레포트

하이브 사이클 텔레메트리(`runs.jsonl`)를 읽어 **자기완결 정적 HTML** 한 장으로 굽는다.
서버·템플릿엔진·CDN·외부 의존성 **0** (Python stdlib + 인라인 SVG). `file://` 더블클릭으로 열린다.
로컬 실행 컨셉(CH0002)에 맞춘 가벼운 경로.

## 쓰는 법

```bash
# 실데이터
python perf/metrics/report.py runs.jsonl -o report.html

# 번들 샘플로 미리보기 (입력 없어도 됨)
python perf/metrics/report.py --demo

# 렌더 후 브라우저로 바로 열기
python perf/metrics/report.py runs.jsonl --open
```

입력 형식은 [`SCHEMA.md`](SCHEMA.md) (런 1회 = JSON 한 줄). 대부분 필드는
`hive_ledger.db`(`runs`/`worker_calls`)에서 그대로 투영된다. 적재(emit) 결선은 후속 작업이고,
이 도구는 "jsonl이 이미 있다"를 전제로 한 **렌더러**다.

## 레포트 구성

| 섹션 | 답하는 질문 | 우선순위 |
|---|---|---|
| 요약 카드 | 통과한 수정 / 수율 / 비용 / 정확도 | — |
| 수확 퍼널 | 이번 사이클은 **어디서 무너졌나** (6단계 전환율) | ② 왜 |
| 정확도(골든셋) | 심은 것 중 **몇 개 찾고 헛다리 몇 번** (NR0004 5버그) | ① 결과 |
| 추세 | 수율·통과율·정밀도 런 누적 (회귀 눈으로 포착) | ① 결과 |
| 비용 분해 | provider별 $ + **수율당/수정당 비용** | ③ 효율 |
| 축별 분해 | 어느 축이 수확하고 어느 축이 0인가 | ② 왜 |

**북극성 = 통과한 수정 수 / 비용.** 나머지 지표는 "왜 그 수가 0이냐"의 선행지표.

## 견고성

- 누락 필드는 0/빈값 흡수, 깨진 라인은 stderr 경고 후 스킵(레포트는 계속).
- `golden` 블록 없으면 정확도 섹션 자동 생략.
- 빈 파일·단일 런·골든 무투입 런 전부 graceful 렌더.

## 파일

- `report.py` — 렌더러(stdlib only, 인라인 SVG). 단독 실행.
- `SCHEMA.md` — `runs.jsonl` 레코드 계약 + 파생지표 정의.
- `runs.sample.jsonl` — 6런 샘플(run446→453, 무수확→수정통과 진행). `--demo` 입력.
