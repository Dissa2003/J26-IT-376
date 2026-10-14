"""Deterministic generator for the seeded / designed evaluation fixtures.

Expected answers come from :func:`spec.reference_encode`, never from the
production tokenizer. Output is frozen to ``fixtures/*.json`` and a test
requires that regenerating with the documented seeds reproduces those files
byte for byte.

    python -m tokenization_ai.evaluation.generate            # write fixtures
    python -m tokenization_ai.evaluation.generate --check    # compare, no write
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from . import spec

PROTOCOL_VERSION = "1.0.0"
DEV_SEED = 1101
EVAL_SEED = 2202
DEV_SIZES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40]
EVAL_SIZES = [2, 3, 5, 8, 10, 12, 15, 20, 30, 40, 60, 80]

FIXTURE_DIR = Path(__file__).parent / "fixtures"

_WORDS = [
    "alpha", "bravo", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "island",
    "jasper", "kestrel", "lantern", "meadow", "nectar", "orchid", "pebble", "quartz",
    "raven", "saffron", "tundra", "umber", "velvet", "willow", "xenon", "yarrow", "zephyr",
    "amber", "birch", "cobalt", "dune", "elm", "fjord", "grove", "heron", "iris", "juniper",
    "kelp", "lotus", "maple", "nimbus",
]
_KEYS = [f"fld_{i:02d}" for i in range(100)]
_SEGMENTS = ["orders", "users", "items", "reports", "search", "products", "v1", "inventory"]
_METHODS = ["GET", "POST", "PUT", "DELETE"]
_HEADER_POOL = {
    "accept": ["application/json", "text/plain"],
    "content-type": ["application/json"],
    "user-agent": ["demo-agent/1.0", "demo-agent/2.0"],
}


def _scalar(rng: random.Random) -> Any:
    kind = rng.choice(["str", "int", "float", "bool", "null"])
    if kind == "str":
        return rng.choice(_WORDS)
    if kind == "int":
        return rng.randint(0, 9999)
    if kind == "float":
        return round(rng.uniform(0, 100), 2)
    if kind == "bool":
        return rng.choice([True, False])
    return None


def _value(rng: random.Random, depth: int) -> Any:
    kind = rng.choice(["scalar", "scalar", "scalar", "dict", "list"] if depth < 2 else ["scalar"])
    if kind == "dict":
        return {key: _value(rng, depth + 1) for key in rng.sample(_KEYS, rng.randint(2, 3))}
    if kind == "list":
        return [_scalar(rng) for _ in range(rng.randint(2, 3))]
    return _scalar(rng)


def _fused(context: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    meta = context["metadata"]
    return {
        "event_id": context["event_id"],
        "method": meta["method"],
        "path": meta["path"],
        "headers": meta.get("headers", {}),
        "payload": context["payload"],
        "content_type": None,
        "client_ip": None,
        "threat_score": None,
        "rules": [
            {"rule_id": r["rule_id"], "attributes": {k: v for k, v in r.items() if k != "rule_id"}}
            for r in rules
        ],
    }


def _case(case_id: str, description: str, group: str, context: dict, rules: list[dict], tensor: str) -> dict:
    tokens, roles, masked = spec.reference_encode(
        context["metadata"]["method"],
        context["metadata"]["path"],
        context["metadata"].get("headers", {}),
        context["payload"],
        rules,
    )
    return {
        "id": case_id,
        "group": group,
        "description": description,
        "input": {
            "format": "envelope",
            "context": context,
            "rule_info": {"event_id": context["event_id"], "rules": rules} if rules else None,
        },
        "expect": {
            "fused": _fused(context, rules),
            "tokens": spec.to_compact(tokens, roles),
            "masked_paths": masked,
            "tensor": tensor,
        },
    }


def _seeded(seed: int, sizes: list[int], prefix: str, group: str) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for index, size in enumerate(sizes, start=1):
        case_id = f"{prefix}-{index:02d}"
        segments = [rng.choice(_SEGMENTS) for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.5:
            segments.append(str(rng.randint(1000, 9999)))
        headers: dict[str, str] = {}
        for name in rng.sample(sorted(_HEADER_POOL), rng.randint(0, 3)):
            headers[name] = rng.choice(_HEADER_POOL[name])
        if rng.random() < 0.4:
            headers["authorization"] = f"Bearer syn-{rng.randint(100, 999)}"
        payload = {key: _value(rng, 0) for key in rng.sample(_KEYS, size)}
        if rng.random() < 0.4:
            payload["password"] = f"pw-{rng.randint(100, 999)}"
        rules = []
        for j in range(rng.randint(0, 2)):
            rule = {"rule_id": f"MOCK-G-{index:02d}-{j}", "severity": rng.choice(["low", "medium", "high"])}
            if rng.random() < 0.5:
                rule["limit"] = rng.randint(1, 100)
            rules.append(rule)
        context = {
            "event_id": case_id,
            "metadata": {
                "method": rng.choice(_METHODS),
                "path": "/" + "/".join(segments),
                "headers": headers,
            },
            "payload": payload,
        }
        cases.append(
            _case(case_id, f"seeded case with {size} top-level payload fields", group, context, rules, "ok")
        )
    return cases


def _designed() -> list[dict]:
    nine_rules = [{"rule_id": f"MOCK-D-{i}", "severity": "high"} for i in range(9)]
    truncated = _case(
        "G-D1-truncation-with-nine-rules",
        "nine rules leave room for one payload field only; the whole rule segment must survive truncation",
        "designed",
        {
            "event_id": "G-D1-truncation-with-nine-rules",
            "metadata": {"method": "GET", "path": "/designed"},
            "payload": {f"fld_{i:02d}": rng_word for i, rng_word in enumerate(_WORDS[:6])},
        },
        nine_rules,
        "ok",
    )
    ten_rules = [{"rule_id": f"MOCK-D-{i}", "severity": "high"} for i in range(10)]
    capacity = _case(
        "G-D2-capacity-with-ten-rules",
        "framing plus ten rules exceeds max_length even with an empty payload; a capacity rejection is the specified outcome",
        "designed",
        {
            "event_id": "G-D2-capacity-with-ten-rules",
            "metadata": {"method": "GET", "path": "/designed"},
            "payload": {},
        },
        ten_rules,
        "capacity_error",
    )
    return [truncated, capacity]


def build() -> dict[str, dict]:
    """Return ``{filename: document}`` for the generated fixture files."""
    return {
        "dev_generated.json": {
            "protocol_version": PROTOCOL_VERSION,
            "group": "dev_generated",
            "seed": DEV_SEED,
            "note": "Used ONLY to fit the evaluation vocabulary. Never scored.",
            "cases": _seeded(DEV_SEED, DEV_SIZES, "dev-g", "dev_generated"),
        },
        "generated_eval.json": {
            "protocol_version": PROTOCOL_VERSION,
            "group": "generated_eval",
            "seed": EVAL_SEED,
            "note": "Expectations from spec.reference_encode (independent encoder), not from the production tokenizer.",
            "cases": _seeded(EVAL_SEED, EVAL_SIZES, "gen-eval", "generated_eval") + _designed(),
        },
    }


def serialize(document: dict) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with frozen files; write nothing")
    args = parser.parse_args(argv)
    status = 0
    for name, document in build().items():
        path = FIXTURE_DIR / name
        text = serialize(document)
        if args.check:
            same = path.exists() and path.read_text(encoding="utf-8") == text
            print(f"{name}: {'identical' if same else 'DIFFERS'}")
            status |= 0 if same else 1
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
