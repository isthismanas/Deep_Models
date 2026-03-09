from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetProfiler:
    sample_size: int = 256

    def profile(self, dataset: Any) -> dict[str, Any]:
        if len(dataset) == 0:
            raise ValueError("Cannot profile an empty dataset")

        first = dataset[0]
        sample_count = min(self.sample_size, len(dataset))
        sample = [dataset[i] for i in range(sample_count)]

        node_counts = [int(d.num_nodes) for d in sample if getattr(d, "num_nodes", None) is not None]
        edge_counts = [int(d.edge_index.shape[1]) for d in sample if getattr(d, "edge_index", None) is not None]

        has_pos = bool(getattr(first, "pos", None) is not None)
        has_z = bool(getattr(first, "z", None) is not None)
        has_y = bool(getattr(first, "y", None) is not None)

        y_dim = None
        if has_y:
            y = first.y
            y_dim = int(y.shape[-1]) if len(y.shape) > 1 else 1

        x_dim = None
        if getattr(first, "x", None) is not None:
            x_dim = int(first.x.shape[-1])

        return {
            "num_samples": int(len(dataset)),
            "num_features": x_dim,
            "has_pos": has_pos,
            "has_atomic_numbers": has_z,
            "has_targets": has_y,
            "target_dim": y_dim,
            "sampled_examples": int(sample_count),
            "avg_nodes": float(np.mean(node_counts)) if node_counts else None,
            "avg_edges": float(np.mean(edge_counts)) if edge_counts else None,
            "min_nodes": int(np.min(node_counts)) if node_counts else None,
            "max_nodes": int(np.max(node_counts)) if node_counts else None,
            "min_edges": int(np.min(edge_counts)) if edge_counts else None,
            "max_edges": int(np.max(edge_counts)) if edge_counts else None,
        }
