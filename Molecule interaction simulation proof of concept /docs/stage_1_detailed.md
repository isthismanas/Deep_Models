# Stage 1 Detailed (Refreshed for Lean Codebase)

## 1) Stage 1 Objective

Stage 1 defines the research frame and measurable criteria before scaling experiments.

Primary question:

- Does explicit spatial supervision (`graph -> voxel`) provide value beyond a standard graph->scalar baseline?

Current codebase scope after cleanup:

- No orchestration/state modules.
- Two active training entrypoints:
  - `scripts/train_qm9_baseline.py`
  - `scripts/train_graph_to_voxel.py`

## 2) Formal Learning Objectives

### 2.1 Scalar baseline objective (active)

Given molecular graph \(G_i = (V_i, E_i, X_i)\), predict one QM9 target \(y_i\) (for example `gap`).

$$
\hat{y}_i = f_\theta(G_i)
$$

$$
\min_\theta \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

### 2.2 Spatial reconstruction objective (active)

Given molecular graph \(G_i\), predict a voxelized spatial field \(\hat{V}_i\) that matches target voxel field \(V_i\):

$$
\hat{V}_i = h_\phi(G_i)
$$

$$
\min_\phi \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{v \in \Omega} (\hat{V}_{i,v} - V_{i,v})^2
$$

where \(\Omega\) is the 3D voxel grid.

## 3) Metrics

### 3.1 Baseline regression metrics

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2}
$$

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
$$

$$
R^2 = 1 - \frac{\sum_i(\hat{y}_i-y_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

### 3.2 Spatial metrics

$$
\text{VoxelMSE}=\frac{1}{|\Omega|}\sum_{v\in\Omega}(\hat{V}_v-V_v)^2
$$

$$
\text{VoxelOverlap}=\frac{|\{v:V_v\ge t\}\cap\{v:\hat{V}_v\ge t\}|+\epsilon}{|\{v:V_v\ge t\}\cup\{v:\hat{V}_v\ge t\}|+\epsilon}
$$

Interpretation:

- `VoxelMSE` measures dense value fidelity.
- `VoxelOverlap` measures occupied-region agreement and is sensitive to geometric support quality.

## 4) Data and Target Construction

Dataset in active scripts:

- QM9 via PyTorch Geometric.

Target formation:

- Baseline path: use `QM9TargetAdapter` to select one property from QM9 label tensor.
- Spatial path: voxelize atomic positions (`data.pos`) using Gaussian splatting.
- Optional refinement: if `--mol2-dir` is provided and matching files exist, mol2 coordinates become the voxel target source.

## 5) Experimental Comparison Protocol

Run both scripts under matched sample budget/seed where possible.

1. Baseline:

```bash
python3 scripts/train_qm9_baseline.py --target gap --epochs 3 --max-samples 2000
```

2. Spatial:

```bash
python3 scripts/train_graph_to_voxel.py --epochs 3 --max-samples 2000 --grid-size 16
```

Outputs are JSON artifacts under `artifacts/` and should be compared across:

- baseline: RMSE/MAE/R2
- spatial: VoxelMSE/VoxelOverlap

## 6) Stage 1 Exit Criteria (Current)

Stage 1 is considered complete if:

1. Objectives are mathematically defined.
2. Metrics are explicitly tied to objectives.
3. Data-to-target mapping is clear for both baseline and spatial paths.
4. At least one successful run per path produces artifacts for inspection.

Status: complete (lean scope).
