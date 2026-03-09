#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molsim.data import DatasetManager
from molsim.state import ProjectStateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure dataset exists in ./data by triggering loader/download.")
    parser.add_argument("--dataset-id", type=str, default="qm9", help="Dataset id (qm9, zinc)")
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(PROJECT_ROOT / "project_state.json"),
        help="Path to state JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = DatasetManager(project_root=PROJECT_ROOT)

    dataset = manager.load(args.dataset_id)
    size = len(dataset)

    store = ProjectStateStore.from_path(args.state_path)
    state = store.load()
    state.setdefault("dataset_plan", {})
    state["dataset_plan"]["ensured_dataset"] = args.dataset_id.lower()
    state["dataset_plan"]["ensured_samples"] = int(size)
    store.append_event(state, f"Ensured dataset '{args.dataset_id.lower()}' in data/ folder.")
    store.save(state)

    print(f"Dataset ensured: {args.dataset_id.lower()} (samples={size})")


if __name__ == "__main__":
    main()
