#!/bin/bash
# Multi-seed JAX training for all models on MIND-small.
# Usage: bash scripts/run_multiseed_jax.sh 2>&1 | tee /tmp/multiseed_all.log

set -e

MODELS=("nrms" "naml" "lstur" "pprec" "crown" "digat" "glory")

echo "============================================"
echo "NewsReX Multi-Seed JAX Training"
echo "Models: ${MODELS[*]}"
echo "Seeds: 42, 123, 456"
echo "Started at: $(date)"
echo "============================================"

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "[$(date)] Starting: $model"
    echo "============================================"

    uv run python src/train.py \
        experiment=mind/$model \
        framework=jax \
        multi_seed.enabled=true \
        logging.progress_backend=tqdm \
        logging.enable_wandb=true \
        2>&1 | tee /tmp/multiseed_${model}.log

    echo "[$(date)] Finished: $model"
done

echo ""
echo "============================================"
echo "ALL MULTI-SEED RUNS COMPLETE at $(date)"
echo "============================================"
