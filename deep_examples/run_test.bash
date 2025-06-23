#!/usr/bin/env bash
# run_all.sh — Execute RL notebooks in parallel with SEED=1,2,3
set -euo pipefail

notebooks=(
  reinforce.ipynb
  reinforce_2_A2C.ipynb
  reinforce_baseline.ipynb
)

seeds=(1 2 3)

for seed in "${seeds[@]}"; do
  for nb in "${notebooks[@]}"; do
    (
      echo "▶️  Running $nb with SEED=$seed"
      SEED=$seed \
      jupyter execute "$nb" \
        --output "${nb%.ipynb}_seed${seed}.ipynb"
      echo "✅  Finished $nb with SEED=$seed"
    ) &
  done
done

wait
echo "🎉  All notebook runs completed."
