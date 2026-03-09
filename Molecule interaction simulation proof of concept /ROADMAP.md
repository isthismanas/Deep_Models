# Spatial Molecular GNN Roadmap

## Objective
Build a production-grade molecular modeling pipeline that captures spatial information beyond standard 2D message passing.

## Stage 1: Foundation (Current)
1. Define clear modeling goals and task boundaries.
2. Define success metrics for each task family.
3. Select primary and secondary datasets.
4. Implement reusable data/goal/state modules.

Deliverable:
- A clean Python codebase with reproducible dataset + metric configuration.

## Stage 2: Baseline Modeling
1. Implement strong baselines:
- 2D GNN baseline (graph-only)
- 3D equivariant GNN baseline
2. Add reproducible train/eval loops.
3. Track metrics with consistent splits.

Deliverable:
- Benchmark table with baseline performance and variance.

## Stage 3: Spatial Enrichment
1. Add voxel/field branch as auxiliary representation.
2. Implement fusion strategies (late fusion, cross-attention, gated fusion).
3. Run ablation studies:
- with/without voxel branch
- different resolutions and sigma values
- compute/performance trade-off

Deliverable:
- Evidence that spatial branch improves target metrics over baselines.

## Stage 4: Interaction/Reactivity Tasking
1. Add supervised interaction targets (not synthetic random labels).
2. Define pair-construction protocol and negative sampling.
3. Evaluate ranking/classification metrics for interaction likelihood.

Deliverable:
- Real interaction benchmark with calibration analysis.

## Stage 5: Research to Industry Readiness
1. Model cards and data cards.
2. Packaging and inference API.
3. Runtime profiling and memory budget targets.
4. Regression tests and CI.

Deliverable:
- Deployable and audited pipeline with documented constraints.

## Initial Metrics Plan
- Regression: `RMSE`, `MAE`, `R2`.
- Binary interaction: `ROC-AUC`, `PR-AUC`, `F1`, `BalancedAccuracy`.
- Reconstruction: `VoxelMSE` and overlap-based score.
