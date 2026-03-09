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

from molsim.data import DatasetCatalog, DatasetSelector
from molsim.goals import GoalRegistry
from molsim.state import ProjectStateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap project goal/dataset/metric plan.")
    parser.add_argument(
        "--goal",
        type=str,
        default="qm9_property_regression",
        help="Goal id from GoalRegistry.default()",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(PROJECT_ROOT / "project_state.json"),
        help="Path to state JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    goals = GoalRegistry.default()
    goal = goals.get(args.goal)

    selector = DatasetSelector(catalog=DatasetCatalog.default())
    ranking = selector.rank_for_goal(goal=goal, project_root=PROJECT_ROOT)
    top_choice = ranking[0]

    store = ProjectStateStore.from_path(args.state_path)
    state = store.load()

    state["project_name"] = state.get("project_name", "molecular_interaction_simulation")
    state["current_stage"] = "stage_1_foundation"
    state["goal"] = {
        "id": goal.goal_id,
        "description": goal.description,
        "task_type": goal.task_type.value,
        "target_field": goal.target_field,
        "requires_3d_geometry": goal.requires_3d_geometry,
    }
    state["dataset_plan"] = {
        "primary": top_choice.dataset.display_name,
        "primary_id": top_choice.dataset.dataset_id,
        "local_available": top_choice.local_available,
        "selection_reasons": list(top_choice.reasons),
    }
    state["metrics_plan"] = {
        "primary": list(goal.primary_metrics),
        "secondary": list(goal.secondary_metrics),
    }
    state.setdefault("milestones", {})
    state["milestones"]["stage_1_goal_metric_dataset_defined"] = "completed"
    state["milestones"].setdefault("stage_2_baselines", "pending")
    state["milestones"].setdefault("stage_3_spatial_fusion", "pending")
    state["milestones"].setdefault("stage_4_interaction_supervision", "pending")
    state["milestones"].setdefault("stage_5_productionization", "pending")

    store.append_event(state, f"Bootstrapped state for goal '{goal.goal_id}' with dataset '{top_choice.dataset.dataset_id}'.")
    store.save(state)

    print("Project bootstrap complete.")
    print(f"Goal: {goal.goal_id}")
    print(f"Primary dataset: {top_choice.dataset.display_name} ({top_choice.dataset.dataset_id})")
    print(f"Metrics: {', '.join(goal.primary_metrics)} | secondary: {', '.join(goal.secondary_metrics)}")
    print("Top dataset ranking:")
    for idx, choice in enumerate(ranking[:3], start=1):
        print(
            f"  {idx}. {choice.dataset.display_name:<12} score={choice.score:<2} "
            f"local={choice.local_available} reasons={'; '.join(choice.reasons)}"
        )

    print("\nCurrent state snapshot:")
    print(json.dumps(store.load(), indent=2))


if __name__ == "__main__":
    main()
