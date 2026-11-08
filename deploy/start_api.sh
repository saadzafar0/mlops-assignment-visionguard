#!/bin/bash
set -eu

tensorflow_model_server \
  --model_name=visionguard \
  --model_base_path=/app/models/serving \
  --rest_api_port=8501 \
  --rest_api_num_threads=2 \
  --tensorflow_intra_op_parallelism=1 \
  --tensorflow_inter_op_parallelism=2 \
  --grpc_max_threads=2 &
model_pid=$!
web_pid=""

cleanup() {
  kill "$model_pid" ${web_pid:+"$web_pid"} 2>/dev/null || true
}
trap cleanup EXIT TERM INT

python /app/deploy/wait_for_model.py
gunicorn --bind "${BIND_HOST:-0.0.0.0}:${PORT:-8000}" --workers 1 --threads 1 \
  --timeout 180 --access-logfile - --error-logfile - 'visionguard.api:create_app()' &
web_pid=$!
set +e
wait -n "$model_pid" "$web_pid"
exit_code=$?
set -e
exit "$exit_code"
