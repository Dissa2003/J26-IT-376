from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import EvaluationDatasetError, evaluate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate reviewed policy-verification cases.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("evaluation_dataset.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_dataset(args.dataset)
    except (EvaluationDatasetError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0 if result["passed_cases"] == result["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
