#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${GENERATOR_MODEL_PATH:-deepseek-ai/deepseek-coder-1.3b-base}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4096}"
MAX_CROSSFILE_TOKENS="${MAX_CROSSFILE_TOKENS:-3072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
VLLM_PORT="${VLLM_PORT:-8000}"
while (($#)); do
    case "$1" in
        --max-context-tokens|--max-crossfile-tokens|--max-new-tokens)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --max-context-tokens) MAX_CONTEXT_TOKENS="$2" ;;
                --max-crossfile-tokens) MAX_CROSSFILE_TOKENS="$2" ;;
                --max-new-tokens) MAX_NEW_TOKENS="$2" ;;
            esac
            shift 2 ;;
        --help|-h)
            echo "Usage: bash serve.sh [--max-context-tokens 4096] [--max-crossfile-tokens 3072] [--max-new-tokens 50]"
            echo "These budgets size the service; pass the same budgets to benchmark_cli.py."
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
for value in "$MAX_CONTEXT_TOKENS" "$MAX_NEW_TOKENS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo "Input/output limits must be positive integers" >&2; exit 2; fi
done
if [[ ! "$MAX_CROSSFILE_TOKENS" =~ ^(0|[1-9][0-9]*)$ ]] || ((MAX_CROSSFILE_TOKENS > MAX_CONTEXT_TOKENS)); then
    echo "Cross-file budget must be between 0 and the input limit" >&2; exit 2
fi
REQUIRED_MODEL_LEN=$((MAX_CONTEXT_TOKENS + MAX_NEW_TOKENS))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$REQUIRED_MODEL_LEN}"
if [[ ! "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || ((MAX_MODEL_LEN < REQUIRED_MODEL_LEN)); then
    echo "MAX_MODEL_LEN must cover input plus output ($REQUIRED_MODEL_LEN tokens)" >&2; exit 2
fi
# The service limit covers prompt plus completion. Evidence packing and the
# actual generation cap are enforced by the experiment runners.
VLLM_CONFIGURE_LOGGING=0 exec vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --dtype half \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
