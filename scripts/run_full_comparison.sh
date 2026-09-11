#!/bin/bash
# Runs the full multi-seed, multi-algorithm comparison: trains each agent
# (+ baselines) for TOTAL_STEPS across SEEDS, evaluates each checkpoint on
# held-out test products, then aggregates into results/comparison_summary.csv
# and results/figures/*.png.
set -e
cd "$(dirname "$0")/.."

TOTAL_STEPS=${TOTAL_STEPS:-15000}
SEEDS=${SEEDS:-"0 1 2"}

for seed in $SEEDS; do
  for agent in dqn ddpg sac ppo; do
    echo "=== training $agent seed=$seed ==="
    python -m training.train --agent "$agent" --seed "$seed" --config "configs/$agent.yaml" --steps "$TOTAL_STEPS"
  done
  for agent in random static; do
    echo "=== training baseline $agent seed=$seed ==="
    python -m training.train --agent "$agent" --seed "$seed" --steps "$TOTAL_STEPS" --action-mode continuous
  done
done

for seed in $SEEDS; do
  for agent in dqn ddpg sac ppo; do
    echo "=== evaluating $agent seed=$seed ==="
    python -m evaluation.evaluate --agent "$agent" --seed "$seed" --config "configs/$agent.yaml"
  done
  for agent in random static; do
    echo "=== evaluating baseline $agent seed=$seed ==="
    python -m evaluation.evaluate --agent "$agent" --seed "$seed" --action-mode continuous
  done
done

echo "=== comparing ==="
python -m evaluation.compare

echo "=== DONE ==="
