#!/bin/bash
# SLURM array job: task i of k processes the graph shards with shard % k == i, so tasks never
# overlap and write disjoint files (done.shard<i>of<k>.jsonl, tensors/shard_XXXX.npz).
# All tasks share one output_dir; summarize after the array finishes:
#   jid=$(sbatch --parsable scripts/slurm_array.sbatch configs/server_large.yaml)
#   sbatch --dependency=afterany:$jid --wrap "python -m fptsim summarize --run results/random_mix_v1"
# Choose shard_size so that num_graphs / shard_size >= number of array tasks.
#SBATCH --job-name=fptsim
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120          # SIGTERM 2 min before the limit: stop cleanly, resume later
#SBATCH --output=logs/fptsim_%A_%a.out
#SBATCH --requeue

set -euo pipefail
CONFIG="${1:-configs/server_large.yaml}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
export PYTHONUNBUFFERED=1

# module load python cuda   # site-specific
# source .venv/bin/activate

# exec so that SLURM's SIGTERM reaches python directly
exec python -m fptsim run --config "$CONFIG" --resume \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
