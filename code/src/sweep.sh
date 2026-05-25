#!/usr/bin/env bash
set -e

COMMON="--cuda --epochs 10 --eval_batch_size 16"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========== Experiment A: small baseline =========="
python3 "$SCRIPT_DIR/train.py" $COMMON \
  --tag expA --emb_dim 64 --num_layers 2 --num_heads 2 --lr 1e-3 --train_batch_size 16 --dropout 0.1

echo "========== Experiment B: wider =========="
python3 "$SCRIPT_DIR/train.py" $COMMON \
  --tag expB --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3 --train_batch_size 16 --dropout 0.1

echo "========== Experiment C: wider + lower lr =========="
python3 "$SCRIPT_DIR/train.py" $COMMON \
  --tag expC --emb_dim 256 --num_layers 4 --num_heads 4 --lr 3e-4 --train_batch_size 16 --dropout 0.1

echo "========== Experiment D: bigger batch + more dropout =========="
python3 "$SCRIPT_DIR/train.py" $COMMON \
  --tag expD --emb_dim 128 --num_layers 4 --num_heads 4 --lr 3e-4 --train_batch_size 32 --dropout 0.2

echo "========== Experiment E: large model =========="
python3 "$SCRIPT_DIR/train.py" $COMMON \
  --tag expE --emb_dim 256 --num_layers 6 --num_heads 8 --lr 1e-4 --train_batch_size 16 --dropout 0.2

echo "========== All experiments done =========="
for tag in expA expB expC expD expE; do
  echo -n "$tag: "
  cat "$SCRIPT_DIR/results_${tag}.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"best_valid_ppl={d['best_valid_ppl']:.2f} @ epoch {d['best_epoch']}\")" 2>/dev/null || echo "not found"
done
