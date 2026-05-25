# coding: utf-8
import argparse
import json
import math
import os
import sys

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import model

# ─── CLI Arguments ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N')
parser.add_argument('--eval_batch_size', type=int, default=16, metavar='N')
parser.add_argument('--max_sql', type=int, default=256, help='sequence length')
parser.add_argument('--seed', type=int, default=1234, help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
parser.add_argument('--tag', type=str, default='default', help='experiment tag for output files')

args = parser.parse_args()

# ─── Output directories ─────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # code/src/
RESULT_DIR = os.path.join(SCRIPT_DIR, '..', 'results')    # code/results/
CKPT_DIR   = os.path.join(SCRIPT_DIR, '..', '..', 'checkpoints')  # checkpoints/
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ─── Setup ────────────────────────────────────────────────────────────────────

torch.manual_seed(args.seed)

use_gpu = args.cuda
if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Config: layers={args.num_layers}, heads={args.num_heads}, dim={args.emb_dim}, "
      f"lr={args.lr}, dropout={args.dropout}, batch={args.train_batch_size}, "
      f"max_sql={args.max_sql}, epochs={args.epochs}")

# ─── Data ─────────────────────────────────────────────────────────────────────

batch_size = {'train': args.train_batch_size, 'valid': args.eval_batch_size}
data_loader = data.Corpus(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ptb"),
                          batch_size, args.max_sql)

# ─── Model ────────────────────────────────────────────────────────────────────

lm = model.CausalLMM(
    vocab_size=len(data_loader.vocabulary),
    dim=args.emb_dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    max_seq_len=args.max_sql,
    dropout=args.dropout,
)
lm = lm.to(device)

total_params = sum(p.numel() for p in lm.parameters())
print(f"Model params: {total_params:,}")

optimizer = optim.AdamW(lm.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
criterion = nn.CrossEntropyLoss()

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate():
    data_loader.set_valid()
    lm.eval()
    total_loss = 0.0
    total_tokens = 0
    idx = 0
    print("Validating...")
    while True:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            logits = lm(data)
            loss = criterion(logits.view(-1, logits.size(-1)), target)
            total_loss += loss.item() * target.size(0)
            total_tokens += target.size(0)
            idx += 1
            if end_flag:
                break
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"  Valid loss: {avg_loss:.4f}, perplexity: {ppl:.2f}")
    return ppl, avg_loss

# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    data_loader.set_train()
    lm.train()
    step_losses = []
    idx = 0
    while True:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = lm(data)
        loss = criterion(logits.view(-1, logits.size(-1)), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), args.grad_clip)
        optimizer.step()

        step_losses.append(loss.item())
        if (idx + 1) % 50 == 0:
            print(f"  Step {idx + 1}, loss: {loss.item():.4f}")
        idx += 1
        if end_flag:
            break

    avg_loss = sum(step_losses) / len(step_losses)
    ppl = math.exp(avg_loss)
    print(f"  Train avg loss: {avg_loss:.4f}, perplexity: {ppl:.2f}")
    return ppl, avg_loss, step_losses

# ─── Main Loop ────────────────────────────────────────────────────────────────

all_step_losses = []
train_epoch_ppl = []
train_epoch_loss = []
valid_epoch_ppl = []
valid_epoch_loss = []
best_valid_ppl = float('inf')

for epoch in range(1, args.epochs + 1):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{args.epochs}  (lr: {scheduler.get_last_lr()[0]:.2e})")
    print(f"{'='*50}")

    t_ppl, t_loss, step_losses = train()
    train_epoch_ppl.append(t_ppl)
    train_epoch_loss.append(t_loss)
    all_step_losses.extend(step_losses)

    v_ppl, v_loss = evaluate()
    valid_epoch_ppl.append(v_ppl)
    valid_epoch_loss.append(v_loss)

    if v_ppl < best_valid_ppl:
        best_valid_ppl = v_ppl
        best_epoch = epoch
        # 保存最佳模型
        torch.save(lm.state_dict(), os.path.join(CKPT_DIR, f"best_model_{args.tag}.pt"))
        print(f"  ★ New best valid PPL: {best_valid_ppl:.2f} (epoch {best_epoch})")

    scheduler.step()

# ─── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Training complete. Best valid PPL: {best_valid_ppl:.2f} @ epoch {best_epoch}")
print(f"{'='*50}")

# ─── Save results ─────────────────────────────────────────────────────────────

results = {
    'tag': args.tag,
    'config': {
        'emb_dim': args.emb_dim, 'num_layers': args.num_layers,
        'num_heads': args.num_heads, 'lr': args.lr, 'dropout': args.dropout,
        'batch_size': args.train_batch_size, 'max_sql': args.max_sql,
        'epochs': args.epochs, 'grad_clip': args.grad_clip,
        'total_params': total_params,
    },
    'train_epoch_ppl': train_epoch_ppl,
    'train_epoch_loss': train_epoch_loss,
    'valid_epoch_ppl': valid_epoch_ppl,
    'valid_epoch_loss': valid_epoch_loss,
    'best_valid_ppl': best_valid_ppl,
    'best_epoch': best_epoch,
}

with open(os.path.join(RESULT_DIR, f"results_{args.tag}.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {os.path.join(RESULT_DIR, f'results_{args.tag}.json')}" )

# ─── Plot loss curves ────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Step-level training loss
    axes[0].plot(all_step_losses, alpha=0.6, linewidth=0.5)
    # 添加平滑曲线 (窗口均值)
    window = min(50, len(all_step_losses) // 10) if len(all_step_losses) > 50 else 1
    if window > 1:
        smoothed = [sum(all_step_losses[max(0,i-window):i+1]) / (i-max(0,i-window)+1)
                    for i in range(len(all_step_losses))]
        axes[0].plot(smoothed, color='red', linewidth=1.5, label=f'avg({window})')
        axes[0].legend()
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss (per step)')
    axes[0].grid(True, alpha=0.3)

    # 2) Epoch-level perplexity
    epochs_range = range(1, args.epochs + 1)
    axes[1].plot(epochs_range, train_epoch_ppl, 'b-o', label='Train PPL', markersize=4)
    axes[1].plot(epochs_range, valid_epoch_ppl, 'r-o', label='Valid PPL', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Train & Valid Perplexity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Epoch-level loss
    axes[2].plot(epochs_range, train_epoch_loss, 'b-o', label='Train Loss', markersize=4)
    axes[2].plot(epochs_range, valid_epoch_loss, 'r-o', label='Valid Loss', markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Train & Valid Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f"loss_curves_{args.tag}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Loss curves saved to {save_path}")
except ImportError:
    print("matplotlib not available, skipping plot")
