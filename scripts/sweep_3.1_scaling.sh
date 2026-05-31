#!/usr/bin/env bash
set -e
COMMON="--cuda --epochs 10 --train_batch_size 16 --eval_batch_size 16 \
        --dropout 0.1 --grad_clip 1.0 --max_sql 256 \
        --results_dir ../results/part_b"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

run_exp() {
    local tag="$1"; shift
    echo ""
    echo "========== 3.1 $tag =========="
    python3 "$SRC_DIR/train.py" $COMMON --tag "$tag" "$@"
}

# ─── 5 model sizes (vary dim, layers, heads; tune lr per size) ────────────────
#   dim  layers  heads  lr     tag
#   64   2       2      1e-3   scaling_s64
#   96   3       3      8e-4   scaling_s96
#   128  4       4      1e-3   scaling_s128
#   160  5       4      5e-4   scaling_s160
#   192  6       4      3e-4   scaling_s192

while read -r dim layers heads lr tag; do
    run_exp "$tag" \
        --emb_dim "$dim" --num_layers "$layers" --num_heads "$heads" --lr "$lr"
done <<EOF
64  2  2  1e-3  scaling_s64
96  3  3  8e-4  scaling_s96
128 4  4  1e-3  scaling_s128
160 5  4  5e-4  scaling_s160
192 6  4  3e-4  scaling_s192
EOF

