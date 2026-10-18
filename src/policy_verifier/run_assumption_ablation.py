"""Run consistency-assumption ablations over the reviewed evaluation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .evaluation import _is_v2_record, _load_dataset, _inputs
from .parser import PolicyParser, UnsupportedPolicyError
from .verifier import PolicyVerificationEngine

SETTINGS: tuple[tuple[str, bool, bool], ...] = (
    ("both_on", True, True),
    ("witness_off", False, True),
    ("exclusivity_off", True, False),
    ("both_off", False, False),
)


def _expected(record: dict[str, Any]) -> str:
    if _is_v2_record(record):
        return record["expected_solver_outcome"]
    expected = record["expected"]
    if expected.get("unsupported") is True or expected.get("parsed") is False:
        return "UNSUPPORTED"
    return expected["solver_outcome"]


def _query_type(record: dict[str, Any]) -> str | None:
    if _is_v2_record(record):
        return record.get("query_type")
    return record.get("query_type", record.get("expected", {}).get("query_type"))


def _is_consistency_ablation_case(record: dict[str, Any]) -> bool:
    if _query_type(record) == "consistency":
        return True
    expected = record.get("expected", {})
    return bool(
        expected.get("conflict") is True
        and len(_inputs(record)) > 1
    )


def _actual(
    record: dict[str, Any],
    engine: PolicyVerificationEngine,
    setting: tuple[str, bool, bool],
) -> str:
    parser = PolicyParser()
    try:
        policies = [
            parser.parse_text(text, f"{record['id']}-{index}")
            for index, text in enumerate(_inputs(record))
        ]
    except UnsupportedPolicyError:
        return "UNSUPPORTED"

    name, witness, exclusivity = setting
    del name
    if _is_consistency_ablation_case(record):
        return engine.check_consistency(
            policies,
            assume_role_witness=witness,
            assume_mutually_exclusive_roles=exclusivity,
        ).status.value

    query = record.get("query", record)
    if _is_v2_record(record) or record.get("query"):
        return engine.check_entailment(
            query["premises"], query["conclusion"]
        ).status.value

    return engine.check_entailment(
        f"{policies[0].subject}(Alice)",
        f"Can{policies[0].action}(Alice, {policies[0].resource})",
    ).status.value


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        dataset = _load_dataset(path)
        for record in dataset["cases"]:
            records.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "record": record,
                    "expected": _expected(record),
                }
            )
    return records


def _dependency_explanation(values: dict[str, str]) -> str:
    changed_witness = values["both_on"] != values["witness_off"]
    changed_exclusivity = values["both_on"] != values["exclusivity_off"]
    if changed_witness and changed_exclusivity:
        return (
            "Depends on both assumptions; disabling either one changes the result."
        )
    if changed_witness:
        return "Depends on the witness assumption only."
    if changed_exclusivity:
        return "Depends on the exclusivity assumption only."
    return "Changes only when both assumptions are disabled together."


def run(paths: list[Path]) -> str:
    records = _load_records(paths)
    engine = PolicyVerificationEngine()
    rows: list[dict[str, Any]] = []
    for item in records:
        actual = {
            name: _actual(item["record"], engine, setting)
            for setting in SETTINGS
            for name in [setting[0]]
        }
        rows.append(
            {
                "id": item["id"],
                "expected": item["expected"],
                "actual": actual,
            }
        )

    changed = [
        row
        for row in rows
        if len(set(row["actual"].values())) > 1
    ]
    consistency_rows = [
        row
        for item, row in zip(records, rows)
        if _is_consistency_ablation_case(item["record"])
    ]
    if len(consistency_rows) == 0:
        raise AssertionError("No consistency cases were evaluated.")
    consistency_check = next(
        (row for row in consistency_rows if row["id"] == "consistency-01"),
        None,
    )
    if consistency_check is None:
        raise AssertionError("Baseline consistency-01 was not evaluated.")
    if (
        consistency_check["actual"]["both_on"] != "UNSAT"
        or consistency_check["actual"]["both_off"] != "SAT"
    ):
        raise AssertionError(
            "Baseline consistency-01 must be UNSAT with both axioms on "
            "and SAT with both axioms off."
        )
    change_counts = {
        setting[0]: sum(
            row["actual"][setting[0]] != row["actual"]["both_on"]
            for row in consistency_rows
        )
        for setting in SETTINGS
        if setting[0] != "both_on"
    }

    lines = [
        "# Policy verification assumption ablation",
        "",
        "This report evaluates the original baseline dataset, the v2 dataset, "
        "and their combined 54 cases under four consistency-assumption settings.",
        "",
        f"Consistency cases evaluated: {len(consistency_rows)}. "
        f"Changes versus both axioms on: witness off {change_counts['witness_off']}, "
        f"exclusivity off {change_counts['exclusivity_off']}, "
        f"both off {change_counts['both_off']}.",
        "",
        "| Case ID | Expected | Both on | Witness off | Exclusivity off | Both off |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = row["actual"]
        lines.append(
            f"| `{row['id']}` | {row['expected']} | {values['both_on']} | "
            f"{values['witness_off']} | {values['exclusivity_off']} | "
            f"{values['both_off']} |"
        )

    lines.extend(
        [
            "",
            "## Cases whose result changes",
            "",
            "| Case ID | Results by setting | Assumption dependency |",
            "|---|---|---|",
        ]
    )
    if not changed:
        lines.append("| None | No result changes | None |")
    else:
        for row in changed:
            values = row["actual"]
            rendered = ", ".join(f"{name}={values[name]}" for name, _, _ in SETTINGS)
            lines.append(
                f"| `{row['id']}` | {rendered} | "
                f"{_dependency_explanation(values)} |"
            )
    lines.extend(
        [
            "",
            "No labels were modified. `UNSUPPORTED` means the original dataset "
            "case was explicitly marked unsupported.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        dest="datasets",
        help="Dataset path; may be supplied more than once.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    datasets = args.datasets or [
        root / "evaluation_dataset.json",
        root / "evaluation_dataset_v2.json",
    ]
    report = run(datasets)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
