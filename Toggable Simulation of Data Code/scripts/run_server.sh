#!/usr/bin/env bash
# Unattended run on a GPU server; survives logout. Run from the project root:
#   bash scripts/run_server.sh configs/server_large.yaml 0          # one GPU (index 0)
#   bash scripts/run_server.sh configs/server_large.yaml 0,1,2,3    # set devices: [0,1,2,3] in the config
# Stop cleanly with:  kill -TERM <pid>   (finishes the current batch; resume later with the same command)
set -euo pipefail

CONFIG="${1:-configs/server_large.yaml}"
GPUS="${2:-0}"
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
python -m fptsim run --config "$CONFIG" --dry-run 2>&1 | tee "logs/dryrun_${STAMP}.log"

nohup python -m fptsim run --config "$CONFIG" --resume > "logs/nohup_${STAMP}.log" 2>&1 &
PID=$!
echo "$PID" > "logs/fptsim_${STAMP}.pid"
echo "started fptsim (pid $PID) on CUDA_VISIBLE_DEVICES=$GPUS"
echo "  follow: tail -f logs/nohup_${STAMP}.log   (also <output_dir>/run.log)"
echo "  stop:   kill -TERM $PID"
