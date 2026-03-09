#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molsim.data import DatasetManager, DatasetProfiler
from molsim.state import ProjectStateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a dataset and persist summary artifact.")
    parser.add_argument("--dataset-id", type=str, default="qm9", help="Dataset id, e.g. qm9 or zinc.")
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="",
        help="Optional output path. Default: artifacts/data_profile_<dataset-id>.json",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(PROJECT_ROOT / "project_state.json"),
        help="Path to state JSON file.",
    )
    parser.add_argument("--sample-size", type=int, default=256, help="Number of samples to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = (
        Path(args.artifact_path).resolve()
        if args.artifact_path
        else (PROJECT_ROOT / "artifacts" / f"data_profile_{args.dataset_id.lower()}.json").resolve()
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(project_root=PROJECT_ROOT)
    profiler = DatasetProfiler(sample_size=args.sample_size)

    dataset = manager.load(args.dataset_id)
    profile = profiler.profile(dataset)
    profile["dataset_id"] = args.dataset_id.lower()

    artifact_path.write_text(json.dumps(profile, indent=2) + "\n")

    store = ProjectStateStore.from_path(args.state_path)
    state = store.load()
    state.setdefault("artifacts", {})
    state["artifacts"][f"data_profile_{args.dataset_id.lower()}"] = str(artifact_path)
    store.append_event(state, f"Profiled dataset '{args.dataset_id.lower()}' and wrote {artifact_path.name}.")
    store.save(state)

    print(f"Wrote profile: {artifact_path}")
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
