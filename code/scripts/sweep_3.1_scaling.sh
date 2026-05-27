#!/usr/bin/env bash
set -e
# ═══════════════════════════════════════════════════════════════════════════════
# Part B — 3.1 Scaling Laws
# ═══════════════════════════════════════════════════════════════════════════════
#   Study 1: Loss vs. Non-Embedding Parameters (power-law)
#     Train baseline model at 5 sizes. Fixed: dataset, training steps, batch.
#     Results → log-log plot with linear fit.
#
#   Study 2: Loss vs. Token Position (context length effect)
#     train.py already records position_groups in the results JSON for each run.
#     Use the best baseline model for the position-loss plot.
# ═══════════════════════════════════════════════════════════════════════════════

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

# ─── Summary ─────────────────────────────────────────────────────────────────
RESULT_DIR="$SCRIPT_DIR/../results/part_b"
echo ""
echo "========== 3.1 Scaling sweep done =========="
echo ""
printf "%-16s %14s %14s %14s\n" "Tag" "Non-Emb Params" "Best Valid PPL" "Best Valid Loss"
printf "%-16s %14s %14s %14s\n" "---" "--------------" "--------------" "--------------"
for dim in 64 96 128 160 192; do
    f="$RESULT_DIR/results_scaling_s${dim}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json
d = json.load(open('$f'))
print(f\"{'s'+str($dim):<16s} {d['config']['non_embedding_params']:>14,d}  {d['best_valid_ppl']:>14.2f}  {d['valid_epoch_loss'][d['best_epoch']-1]:>14.4f}\")
"
    else
        printf "%-16s %14s\n" "s${dim}" "(not found)"
    fi
done
