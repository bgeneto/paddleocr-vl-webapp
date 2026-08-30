#!/bin/sh
# Download GGUFs into /models once, then start llama-server from local files.
# Do not pass --hf-repo / --mmproj-url into llama-server: those re-fetch on every
# start (HF hub cache is not persisted; etag mismatches delete and re-download).
set -eu

MODEL="${LLAMA_ARG_MODEL:-/models/PaddleOCR-VL-1.6-Q4_K_M.gguf}"
MMPROJ="${LLAMA_ARG_MMPROJ:-/models/mmproj-Q8_0.gguf}"
REPO="${LLAMA_HF_REPO:-LunarOilRig/PaddleOCR-VL-1.6-GGUF-Q4}"
FILE="${LLAMA_HF_FILE:-PaddleOCR-VL-1.6-Q4_K_M.gguf}"
MODEL_URL="${LLAMA_MODEL_URL:-https://huggingface.co/${REPO}/resolve/main/${FILE}}"
MMPROJ_URL="${LLAMA_MMPROJ_URL:-https://huggingface.co/LunarOilRig/PaddleOCR-VL-1.6-GGUF-Q4/resolve/main/mmproj-Q8_0.gguf}"

download() {
    dest=$1
    url=$2
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ]; then
        echo "Using cached $dest"
        return 0
    fi
    echo "Downloading $url -> $dest"
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "Using HF_TOKEN for authenticated Hugging Face download"
        curl -L --fail --retry 5 -C - \
            -H "Authorization: Bearer ${HF_TOKEN}" \
            -o "${dest}.part" "$url"
    else
        curl -L --fail --retry 5 -C - -o "${dest}.part" "$url"
    fi
    mv "${dest}.part" "$dest"
}

# Reuse mmproj already pulled into the old llama.cpp cache volume, if present.
if [ ! -s "$MMPROJ" ] && [ -s /root/.cache/llama.cpp/mmproj-Q8_0.gguf ]; then
    echo "Copying mmproj from llama.cpp cache volume"
    cp /root/.cache/llama.cpp/mmproj-Q8_0.gguf "$MMPROJ"
fi

download "$MODEL" "$MODEL_URL"
download "$MMPROJ" "$MMPROJ_URL"

unset LLAMA_ARG_HF_REPO LLAMA_ARG_HF_FILE LLAMA_ARG_MMPROJ_URL LLAMA_ARG_MODEL_URL || true
export LLAMA_ARG_MODEL="$MODEL"
export LLAMA_ARG_MMPROJ="$MMPROJ"

# llama-server splits --ctx-size across slots (n_ctx_slot = n_ctx / n_parallel).
# LLAMA_CTX_SIZE is the desired per-slot window; total is capped by LLAMA_CTX_MAX.
N_PARALLEL="${LLAMA_ARG_N_PARALLEL:-8}"
CTX_PER_SLOT="${LLAMA_CTX_SIZE:-8192}"
CTX_MAX="${LLAMA_CTX_MAX:-131072}"
TOTAL=$((CTX_PER_SLOT * N_PARALLEL))
if [ "$TOTAL" -gt "$CTX_MAX" ]; then
    echo "Requested n_ctx ${TOTAL} (${CTX_PER_SLOT} x ${N_PARALLEL} slots) exceeds LLAMA_CTX_MAX=${CTX_MAX}; capping"
    TOTAL=$CTX_MAX
fi
CTX_SLOT=$((TOTAL / N_PARALLEL))
echo "llama.cpp context: n_ctx=${TOTAL} (${N_PARALLEL} slots, n_ctx_slot=${CTX_SLOT})"
export LLAMA_ARG_CTX_SIZE="$TOTAL"

exec /app/llama-server "$@"
