from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from molsim.goals import TaskType


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    has_3d_geometry: bool
    tasks_supported: tuple[TaskType, ...]
    local_root: str
    size_hint: str
    label_fields: tuple[str, ...]
    notes: str


class DatasetCatalog:
    def __init__(self, specs: dict[str, DatasetSpec]) -> None:
        self._specs = specs

    @classmethod
    def default(cls) -> "DatasetCatalog":
        specs = {
            "qm9": DatasetSpec(
                dataset_id="qm9",
                display_name="QM9",
                has_3d_geometry=True,
                tasks_supported=(TaskType.REGRESSION, TaskType.RECONSTRUCTION),
                local_root="data/QM9",
                size_hint="~134k molecules",
                label_fields=(
                    "mu",
                    "alpha",
                    "homo",
                    "lumo",
                    "gap",
                    "u0",
                    "u298",
                    "h298",
                    "g298",
                    "cv",
                ),
                notes="Fast local iteration; strong starter dataset for 3D molecule modeling.",
            ),
            "zinc": DatasetSpec(
                dataset_id="zinc",
                display_name="ZINC",
                has_3d_geometry=False,
                tasks_supported=(TaskType.REGRESSION,),
                local_root="data/ZINC",
                size_hint="up to millions (depending split)",
                label_fields=("logP", "QED", "synthetic_accessibility"),
                notes="Useful for large-scale 2D pretraining and ablations.",
            ),
            "geom_drugs": DatasetSpec(
                dataset_id="geom_drugs",
                display_name="GEOM-Drugs",
                has_3d_geometry=True,
                tasks_supported=(TaskType.REGRESSION, TaskType.RECONSTRUCTION, TaskType.BINARY_CLASSIFICATION),
                local_root="data/GEOM",
                size_hint="millions of conformers",
                label_fields=("conformer_energy", "property_targets", "interaction_label"),
                notes="Good follow-up for richer 3D conformer diversity.",
            ),
        }
        return cls(specs=specs)

    def get(self, dataset_id: str) -> DatasetSpec:
        if dataset_id not in self._specs:
            known = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown dataset_id '{dataset_id}'. Known datasets: {known}")
        return self._specs[dataset_id]

    def list_specs(self) -> list[DatasetSpec]:
        return [self._specs[key] for key in sorted(self._specs.keys())]

    def is_local_available(self, dataset_id: str, project_root: str | Path = ".") -> bool:
        spec = self.get(dataset_id)
        root = Path(project_root).resolve() / spec.local_root
        return root.exists()
