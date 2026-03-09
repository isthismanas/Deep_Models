from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    RECONSTRUCTION = "reconstruction"


@dataclass(frozen=True)
class GoalDefinition:
    goal_id: str
    task_type: TaskType
    description: str
    requires_3d_geometry: bool
    target_field: str | None
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]


class GoalRegistry:
    def __init__(self, goals: dict[str, GoalDefinition]) -> None:
        self._goals = goals

    @classmethod
    def default(cls) -> "GoalRegistry":
        goals = {
            "qm9_property_regression": GoalDefinition(
                goal_id="qm9_property_regression",
                task_type=TaskType.REGRESSION,
                description="Predict QM9 quantum properties with spatially-aware graph models.",
                requires_3d_geometry=True,
                target_field="gap",
                primary_metrics=("rmse", "mae"),
                secondary_metrics=("r2",),
            ),
            "voxel_reconstruction": GoalDefinition(
                goal_id="voxel_reconstruction",
                task_type=TaskType.RECONSTRUCTION,
                description="Reconstruct voxelized density fields from molecular graph encodings.",
                requires_3d_geometry=True,
                target_field=None,
                primary_metrics=("voxel_mse",),
                secondary_metrics=("voxel_overlap",),
            ),
            "interaction_screening": GoalDefinition(
                goal_id="interaction_screening",
                task_type=TaskType.BINARY_CLASSIFICATION,
                description="Predict molecular interaction likelihood for paired molecules.",
                requires_3d_geometry=True,
                target_field="interaction_label",
                primary_metrics=("roc_auc", "pr_auc"),
                secondary_metrics=("f1", "balanced_accuracy"),
            ),
        }
        return cls(goals=goals)

    def get(self, goal_id: str) -> GoalDefinition:
        if goal_id not in self._goals:
            known = ", ".join(sorted(self._goals))
            raise KeyError(f"Unknown goal_id '{goal_id}'. Known goals: {known}")
        return self._goals[goal_id]

    def list_goal_ids(self) -> list[str]:
        return sorted(self._goals.keys())
