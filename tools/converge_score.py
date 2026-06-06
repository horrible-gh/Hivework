"""Print a deterministic scorecard for converge's post-model guard corpus."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.converge_corpus import CASES, run_case  # noqa: E402


KNOWN_STAMPS = {
    "dropped_peer",
    "data_premise_refuted",
    "data_unstamped",
    "field_provenance_repointed",
    "field_provenance_confirmed",
    "http_datasource_provenance_repointed",
    "http_datasource_provenance_confirmed",
    "unrefuted_peer",
    "counterfactual_incomplete",
    "trace_ungrounded",
    "low_confidence",
}


def _matches(case, result) -> bool:
    if result.converged is not case.expect["converged"]:
        return False
    if (result.attributed_defect or {}).get("file") != case.expect["attributed_file"]:
        return False
    causal_check = result.causal_check or {}
    stamp = case.expect.get("stamp")
    if stamp:
        return stamp in causal_check
    return KNOWN_STAMPS.isdisjoint(causal_check)


def main() -> int:
    rows = []
    false_positive = 0
    false_negative = 0

    for case in CASES:
        expected_fire = bool(case.expect.get("stamp"))
        try:
            result = run_case(case)
            correct = _matches(case, result)
        except Exception:
            correct = False

        if correct:
            verdict = "correct"
        elif expected_fire:
            verdict = "false-negative"
            false_negative += 1
        else:
            verdict = "false-positive"
            false_positive += 1

        mark = "✓" if correct else "X"
        rows.append(
            (
                case.id,
                "yes" if case.captured else "no",
                case.guard,
                f"{case.expect['outcome']} {mark}",
                verdict,
            )
        )

    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(("case", "captured", "guard", "outcome", "verdict"))
    ]
    header = ("case", "captured", "guard", "outcome", "verdict")
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(header)))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

    correct_count = len(rows) - false_positive - false_negative
    print()
    print(
        f"TOTAL: {correct_count}/{len(rows)} correct  "
        f"({false_positive} false-positive, {false_negative} false-negative)"
    )
    return 0 if correct_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
