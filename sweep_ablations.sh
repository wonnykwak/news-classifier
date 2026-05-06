#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
CSV="Newsheadlines/merged_headlines.csv"
EPOCHS=15
VS=0.15
SEED=17
for lr in 1e-5 2e-5 3e-5; do
  for bs in 32 64; do
    for mlen in 16 32 64 128; do
      for wd in 0 0.01; do
        dir="checkpoints/lr${lr}_bs${bs}_mlen${mlen}_wd${wd}_vs${VS}_e${EPOCHS}"
        python model/train_distilbert.py \
          --csv "$CSV" \
          --epochs "$EPOCHS" \
          --batch-size "$bs" \
          --learning-rate "$lr" \
          --val-size "$VS" \
          --seed "$SEED" \
          --max-len "$mlen" \
          --weight-decay "$wd" \
          --out "${dir}/model.pt"
      done
    done
  done
done
