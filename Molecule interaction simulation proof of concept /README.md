# Molecular Interaction Simulation (Codebase Version)

This repository is moving from notebook prototyping to reusable Python modules.

Current focus:
- Define project goals and measurable metrics.
- Select datasets based on task fit and local availability.
- Persist project state for context-aware iteration.

## Quick start

```bash
python3 scripts/bootstrap_project.py --goal qm9_property_regression
python3 scripts/show_state.py
python3 scripts/ensure_dataset.py --dataset-id qm9
python3 scripts/train_qm9_baseline.py --target gap --epochs 3 --max-samples 2000
```

## Structure

- `src/molsim/goals.py`: goal/task definitions and metric plan.
- `src/molsim/metrics.py`: reusable metric evaluators.
- `src/molsim/data/`: dataset catalog, ranking, loading, and split helpers.
- `src/molsim/models/`: baseline neural architectures.
- `src/molsim/training/`: train/eval loop utilities.
- `src/molsim/state.py`: JSON-backed project state persistence.
- `docs/stage_1_detailed.md`: detailed Stage 1 methodology with equations.
- `ROADMAP.md`: staged implementation roadmap.
- `AGENTS.md`: local repo operating instructions.
- `project_state.json`: running project state.
