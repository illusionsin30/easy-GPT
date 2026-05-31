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
    echo "========== 3.2 $tag =========="
    python3 "$SRC_DIR/train.py" $COMMON --tag "$tag" "$@"
}

# ─── Enhanced architecture (QK Norm + Attention Gate + Value Embedding) ──────
#   Trained at 5 model sizes with the same grid as §3.1 baseline.
#
#   dim  layers  heads  lr     tag
#   64   2       2      1e-3   arch_all3_d64
#   96   3       3      8e-4   arch_all3_d96
#   128  4       4      1e-3   arch_all3_d128
#   160  5       4      5e-4   arch_all3_d160
#   192  6       4      3e-4   arch_all3_d192

ARCH_FLAGS="--qk_norm --attn_gate --value_emb"

while read -r dim layers heads lr tag; do
    run_exp "$tag" $ARCH_FLAGS \
        --emb_dim "$dim" --num_layers "$layers" --num_heads "$heads" --lr "$lr"
done <<EOF
64  2  2  1e-3  arch_all3_d64
96  3  3  8e-4  arch_all3_d96
128 4  4  1e-3  arch_all3_d128
160 5  4  5e-4  arch_all3_d160
192 6  4  3e-4  arch_all3_d192
EOF

