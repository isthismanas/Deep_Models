from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from molsim.data.catalog import DatasetCatalog, DatasetSpec
from molsim.goals import GoalDefinition


@dataclass(frozen=True)
class DatasetChoice:
    dataset: DatasetSpec
    score: int
    reasons: tuple[str, ...]
    local_available: bool


class DatasetSelector:
    def __init__(self, catalog: DatasetCatalog) -> None:
        self.catalog = catalog

    def rank_for_goal(self, goal: GoalDefinition, project_root: str | Path = ".") -> list[DatasetChoice]:
        choices: list[DatasetChoice] = []

        for spec in self.catalog.list_specs():
            local_available = self.catalog.is_local_available(spec.dataset_id, project_root=project_root)
            score = 0
            reasons: list[str] = []

            if goal.task_type in spec.tasks_supported:
                score += 4
                reasons.append("supports task type")

            if goal.requires_3d_geometry and spec.has_3d_geometry:
                score += 4
                reasons.append("has 3D geometry")

            if goal.target_field and goal.target_field in spec.label_fields:
                score += 2
                reasons.append("contains target field")

            if local_available:
                score += 5
                reasons.append("already available locally")

            choices.append(
                DatasetChoice(
                    dataset=spec,
                    score=score,
                    reasons=tuple(reasons) if reasons else ("no direct match",),
                    local_available=local_available,
                )
            )

        return sorted(choices, key=lambda c: c.score, reverse=True)
