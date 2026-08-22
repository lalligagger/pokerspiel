#!/usr/bin/env python3
"""Resolve JSON config values into Docker/env exports for the solver runtime.

Usage:
  python3 config_env.py --config cfg/solve_config_debug.json --format shell
  python3 config_env.py --config cfg/solve_config_debug.json --format docker
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ENV_KEYS = [
    "POKERSPIEL_SOLVER",
    "POKERSPIEL_PRESET",
    "POKERSPIEL_RANGE_SAMPLES",
    "POKERSPIEL_POSTFLOP_SAMPLES",
    "POKERSPIEL_MIN_ITERATIONS",
    "POKERSPIEL_CHECKPOINT_EVERY",
    "POKERSPIEL_ITERATIONS",
    "POKERSPIEL_MAX_ITERATIONS",
    "POKERSPIEL_MEMORY_THRESHOLD",
    "POKERSPIEL_OUTPUT_JSON",
]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_env(config: dict) -> dict:
    mapping = {
        "POKERSPIEL_SOLVER": config.get("solver"),
        "POKERSPIEL_PRESET": config.get("preset"),
        "POKERSPIEL_RANGE_SAMPLES": config.get("range_samples"),
        "POKERSPIEL_POSTFLOP_SAMPLES": config.get("postflop_samples"),
        "POKERSPIEL_MIN_ITERATIONS": config.get("min_iterations"),
        "POKERSPIEL_CHECKPOINT_EVERY": config.get("checkpoint_every") if config.get("checkpoint_every") is not None else config.get("stability_checkpoint"),
        "POKERSPIEL_ITERATIONS": config.get("iterations"),
        "POKERSPIEL_MAX_ITERATIONS": config.get("iterations"),
        "POKERSPIEL_MEMORY_THRESHOLD": config.get("memory_threshold"),
        "POKERSPIEL_OUTPUT_JSON": config.get("output_json"),
    }
    result = {}
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        result[key] = str(value)
    return result


def shell_exports(env_map: dict) -> str:
    pieces = []
    for key in ENV_KEYS:
        value = env_map.get(key)
        if value is None:
            continue
        pieces.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(pieces)


def docker_args(env_map: dict) -> str:
    pieces = []
    for key in ENV_KEYS:
        value = env_map.get(key)
        if value is None:
            continue
        pieces.append(f"-e {key}={shlex.quote(value)}")
    return " ".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve JSON config into solver env exports")
    parser.add_argument("--config", required=True, help="Path to solver JSON config")
    parser.add_argument("--format", choices=("shell", "docker"), default="shell")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:  # pragma: no cover - CLI validation path
        print(str(exc), file=sys.stderr)
        return 2

    env_map = resolve_env(config)
    if args.format == "shell":
        print(shell_exports(env_map))
    else:
        print(docker_args(env_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
