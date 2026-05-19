#! /usr/bin/bash

python train.py \
    --cuda \
    --epochs 3 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --max_sql 35 \
    --seed 1234 \
    --lr 1e-3 \
    --num_layers 2 \
    --num_heads 4 \
    --emb_dim 256 \
    --gpu_id 0
