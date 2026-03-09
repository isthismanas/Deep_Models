# AGENTS.md

## Scope
This file defines local operating rules for coding agents in this repository.

## Current project phase
- Phase: `stage_1_foundation`
- Primary focus: goals, metrics, and dataset strategy.

## Rules
1. Only modify files inside this repository folder.
2. Prefer reusable Python modules over notebooks.
3. Keep all task definitions explicit in code (`src/molsim/goals.py`).
4. Keep metric logic centralized (`src/molsim/metrics.py`).
5. Keep dataset selection logic centralized (`src/molsim/data/selector.py`).
6. Persist progress updates in `project_state.json` after meaningful milestones.

## State protocol
When completing a milestone, update:
- `project_state.json.current_stage`
- `project_state.json.milestones`
- `project_state.json.last_update`
- `project_state.json.event_log`

## Near-term milestones
1. Implement baseline training/evaluation runner.
2. Add experiment config files.
3. Add tests for dataset selection and metric computation.
