#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print project_state.json")
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "project_state.json"),
        help="Path to state JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.state_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    print(json.dumps(json.loads(path.read_text()), indent=2))


if __name__ == "__main__":
    main()
