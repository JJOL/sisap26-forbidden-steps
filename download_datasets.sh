#!/usr/bin/env bash
# download_datasets.sh
#
# Downloads all SISAP 2026 benchmark datasets from HuggingFace and writes the
# accompanying config.json files that datasets.py / search.py expect.
#
# Usage:
#   ./download_datasets.sh [--small-only]
#
#   --small-only   Download only the small development datasets (fast, < 1 GB).
#                  Skips the large full-scale datasets (wikipedia ~15 GB, nq ~7 GB).
#
# After running this script every dataset is ready to use:
#   python search.py --task task1 --dataset wikipedia-small
#   python search.py --task task2 --dataset llama-dev
#   python search.py --task task3 --dataset fiqa-dev

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HF_BASE="https://huggingface.co/datasets/SISAP-Challenges/SISAP2026/resolve/main"

SMALL_ONLY=false
for arg in "$@"; do
    [[ "$arg" == "--small-only" ]] && SMALL_ONLY=true
done

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  already present: $dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  downloading $(basename "$dest") ..."
    # Use wget if available, fall back to curl
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi
    echo "  done: $dest"
}

write_config() {
    local path="$1"
    local content="$2"
    mkdir -p "$(dirname "$path")"
    if [[ -f "$path" ]]; then
        echo "  config exists: $path"
    else
        printf '%s\n' "$content" > "$path"
        echo "  wrote:  $path"
    fi
}

# ---------------------------------------------------------------------------
# Task 1 – K-nearest neighbor graph  (k=15, dot product, normalized vectors)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 1: K-nearest neighbor graph ==="

# --- wikipedia-small (development dataset, ~682 MB) ---
echo ""
echo "-- wikipedia-small (682 MB) --"
download \
    "$HF_BASE/benchmark-dev-wikipedia-bge-m3-small.h5" \
    "data/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"

write_config "data/wikipedia-small/config.json" \
'{
    "task": "task1",
    "data": "train",
    "gt_I": ["allknn", "knns"],
    "k": 15,
    "dataset_name": "wikipedia-small",
    "filename": "benchmark-dev-wikipedia-bge-m3-small.h5"
}'

# --- wikipedia (full evaluation dataset, ~15 GB) ---
if [[ "$SMALL_ONLY" == false ]]; then
    echo ""
    echo "-- wikipedia (full, ~15 GB) --"
    download \
        "$HF_BASE/benchmark-dev-wikipedia-bge-m3.h5" \
        "data/wikipedia/benchmark-dev-wikipedia-bge-m3.h5"

    write_config "data/wikipedia/config.json" \
'{
    "task": "task1",
    "data": "train",
    "gt_I": ["allknn", "knns"],
    "k": 15,
    "dataset_name": "wikipedia",
    "filename": "benchmark-dev-wikipedia-bge-m3.h5"
}'
else
    echo ""
    echo "-- wikipedia (full) skipped (--small-only) --"
fi

# ---------------------------------------------------------------------------
# Task 2 – Maximum Inner Product Search  (k=30, dot product, not normalized)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 2: Maximum Inner Product Search ==="

# --- llama-dev (development + evaluation dataset, ~134 MB) ---
echo ""
echo "-- llama-dev (134 MB) --"
download \
    "$HF_BASE/llama-dev.h5" \
    "data/llama-dev/llama-dev.h5"

write_config "data/llama-dev/config.json" \
'{
    "task": "task2",
    "data": "train",
    "queries": "test/queries",
    "gt_I": "test/knns",
    "k": 30,
    "dataset_name": "llama-dev",
    "filename": "llama-dev.h5"
}'

# ---------------------------------------------------------------------------
# Task 3 – Sparse high-dimensional vectors  (k=30, dot product, SPLADE-v3)
# ---------------------------------------------------------------------------

echo ""
echo "=== Task 3: Sparse vector search ==="

# --- fiqa-dev (small development dataset, ~188 MB) ---
echo ""
echo "-- fiqa-dev (188 MB) --"
download \
    "$HF_BASE/fiqa-dev.h5" \
    "data/fiqa-dev/fiqa-dev.h5"

write_config "data/fiqa-dev/config.json" \
'{
    "task": "task3",
    "data": "train",
    "queries": "otest/queries",
    "gt_I": "otest/knns",
    "k": 30,
    "dataset_name": "fiqa-dev",
    "sparse": true,
    "filename": "fiqa-dev.h5"
}'

# --- nq (full evaluation dataset, ~6.9 GB) ---
if [[ "$SMALL_ONLY" == false ]]; then
    echo ""
    echo "-- nq (full, ~6.9 GB) --"
    download \
        "$HF_BASE/nq.h5" \
        "data/nq/nq.h5"

    write_config "data/nq/config.json" \
'{
    "task": "task3",
    "data": "train",
    "queries": "otest/queries",
    "gt_I": "otest/knns",
    "k": 30,
    "dataset_name": "nq",
    "sparse": true,
    "filename": "nq.h5"
}'
else
    echo ""
    echo "-- nq (full) skipped (--small-only) --"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "All done. Datasets available:"
echo ""
echo "  Task 1 (all-kNN graph, k=15):"
echo "    wikipedia-small   data/wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"
if [[ "$SMALL_ONLY" == false ]]; then
echo "    wikipedia         data/wikipedia/benchmark-dev-wikipedia-bge-m3.h5"
fi
echo ""
echo "  Task 2 (MIPS, k=30):"
echo "    llama-dev         data/llama-dev/llama-dev.h5"
echo ""
echo "  Task 3 (sparse search, k=30):"
echo "    fiqa-dev          data/fiqa-dev/fiqa-dev.h5"
if [[ "$SMALL_ONLY" == false ]]; then
echo "    nq                data/nq/nq.h5"
fi
echo ""
echo "Run search.py with any of these dataset names, e.g.:"
echo "  python search.py --task task1 --dataset wikipedia-small"
echo "  python search.py --task task2 --dataset llama-dev"
echo "  python search.py --task task3 --dataset fiqa-dev"
