# Stage 1 Detailed: Goals, Metrics, Dataset Strategy, and State

## 1) Problem framing
We are building a molecular learning system that captures structural information better than pure 2D graph models.

Primary Stage 1 goal:
- Define a measurable and reproducible research setup before heavy model work.

Core question:
- Does adding spatial context improve predictive quality enough to justify extra compute?

## 2) Formal objective definitions

### 2.1 Regression objective (current primary)
Given molecular graph `G_i = (V_i, E_i, X_i)` and optional geometry
`P_i in R^(|V_i| x 3)`, learn:

```text
f_theta(G_i, P_i) -> y_i
```

for target property `y_i` (currently QM9 `gap`).

Training objective:

```text
min_theta (1/N) * sum_{i=1..N} (f_theta(G_i, P_i) - y_i)^2
```

### 2.2 Interaction objective (future stage)
For molecule pairs `(i, j)`, predict interaction probability:

```text
p_ij = sigma(g_phi(z_i, z_j))
```

with supervised binary label `t_ij in {0, 1}`.

Loss:

```text
L_BCE = -(1/M) * sum_{(i,j)} [ t_ij*log(p_ij) + (1 - t_ij)*log(1 - p_ij) ]
```

## 3) Metric definitions

### 3.1 Regression
- RMSE:

```text
RMSE = sqrt( (1/N) * sum_{i=1..N} (y_hat_i - y_i)^2 )
```

- MAE:

```text
MAE = (1/N) * sum_{i=1..N} abs(y_hat_i - y_i)
```

- `R^2`:

```text
R^2 = 1 - [ sum_i (y_hat_i - y_i)^2 ] / [ sum_i (y_i - y_bar)^2 ]
```

### 3.2 Binary interaction (planned)
- ROC-AUC
- PR-AUC
- F1
- Balanced Accuracy

### 3.3 Reconstruction (planned)
- Voxel MSE:

```text
VoxelMSE = (1/K) * sum_{k=1..K} (v_hat_k - v_k)^2
```

## 4) Dataset decision logic
Dataset ranking score in code is additive:

```text
S(d | g) =
  4 * I(task match)
  + 4 * I(3D available)
  + 2 * I(target present)
  + 5 * I(local availability)
```

Current selected primary dataset: `QM9`.

Reasons:
1. Local availability and reproducibility.
2. Fast iteration with graph + geometry fields.
3. Suitable for initial regression benchmarking.

## 5) Data split policy
For dataset size `N` and fractions `(p_tr, p_val, p_te)`:

```text
N_tr   = floor(p_tr * N)
N_val  = floor(p_val * N)
N_test = N - N_tr - N_val
```

with seeded random permutation for reproducibility.

## 6) State and traceability
Persistent project state is stored in `project_state.json`.

Minimum state fields:
1. current stage
2. goal and metrics
3. selected dataset
4. milestone status
5. event log
6. generated artifacts

This allows us to continue work with consistent context across sessions.

## 7) Stage 1 completion checklist
1. Goal taxonomy implemented.
2. Metric computation utilities implemented.
3. Dataset catalog and selector implemented.
4. Dataset profile artifact generated.
5. State logging enabled.

Status: complete.
