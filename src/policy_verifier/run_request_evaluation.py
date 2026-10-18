from __future__ import annotations

import argparse
import json
from pathlib import Path

from .request_evaluation import evaluate_request_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate independent policy requests.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("evaluation_dataset_v3.json"),
    )
    args = parser.parse_args()
    result = evaluate_request_dataset(args.dataset)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
