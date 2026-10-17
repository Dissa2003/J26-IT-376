"""Command line entry point.

    python -m tokenization_ai.evaluation --out-dir <dir> --write-manifest   # freeze hashes, no scoring
    python -m tokenization_ai.evaluation --out-dir <dir>                    # score and write reports
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .runner import (
    DEFAULT_MAX_LENGTH,
    PROTOCOL_VERSION,
    fixture_hashes,
    render_markdown,
    run_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tokenization_ai.evaluation")
    parser.add_argument("--out-dir", required=True, type=Path, help="where to write reports (outside the repo)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--write-manifest", action="store_true", help="write PRE_RUN_MANIFEST.json and exit")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_manifest:
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "written_before_first_scored_run": True,
            "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": fixture_hashes(),
        }
        path = args.out_dir / "PRE_RUN_MANIFEST.json"
        if path.exists():
            print(f"refusing to overwrite existing manifest: {path}")
            return 2
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return 0

    report = run_evaluation(max_length=args.max_length)
    manifest_path = args.out_dir / "PRE_RUN_MANIFEST.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        report["data"]["matches_pre_run_manifest"] = recorded == report["data"]["fixture_sha256"]
    (args.out_dir / "kpi_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (args.out_dir / "kpi_report.md").write_text(render_markdown(report), encoding="utf-8")
    k = report["results"]
    print("KPI 1 valid alignment :", k["kpi1_input_alignment"]["valid_event_alignment"]["numerator"],
          "/", k["kpi1_input_alignment"]["valid_event_alignment"]["denominator"])
    print("KPI 2 sequence match  :", k["kpi2_token_mapping"]["sequence_exact_match"]["numerator"],
          "/", k["kpi2_token_mapping"]["sequence_exact_match"]["denominator"])
    print("KPI 3 tensor correct  :", k["kpi3_aligned_tensor_correctness"]["tensor_correctness"]["numerator"],
          "/", k["kpi3_aligned_tensor_correctness"]["tensor_correctness"]["denominator"])
    print("wrote", args.out_dir / "kpi_report.json", "and kpi_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
