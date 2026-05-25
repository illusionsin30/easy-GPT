# Transformer Decoder with RoPE — Implementation Plan

**Goal:** 手动实现带 RoPE 的 Transformer Decoder 语言模型，在 PTB 数据集上训练并调参。

**Architecture:** Pre-norm Transformer Decoder，RoPE 旋转位置编码，SwiGLU FFN，因果掩码注意力。全部手写，仅使用 `nn.Linear`、`nn.LayerNorm`、`nn.Embedding` 等基础模块。

**Tech Stack:** PyTorch 2.12, matplotlib（画 loss 曲线）

---

## File Structure

```
src/
├── data.py      # 不动
├── model.py     # 重写：RoPE, Attention, SwiGLU, TransformerBlock, CausalLMM
└── train.py     # 重写：loss 记录、画图、超参、梯度裁剪、学习率调度
```

---

### Task 1: 实现 `model.py` — RoPE 旋转位置编码

**Files:**
- Rewrite: `src/model.py`

- [ ] **Step 1: 编写 `precompute_freqs_cis` 函数**

  生成旋转角度的复数表示。对于 head_dim 维度（必须是偶数），计算每个位置每个维度对的频率：

  ```python
  def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
      # freqs[i] = 1 / (theta^(2i/dim)), i = 0,1,...,dim/2-1
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
      t = torch.arange(max_seq_len)
      freqs = torch.outer(t, freqs)  # (max_seq_len, dim/2)
      freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式 e^{iθ}
      return freqs_cis
  ```

- [ ] **Step 2: 编写 `apply_rotary_emb` 函数**

  将 RoPE 应用到 Q 和 K 上：

  ```python
  def apply_rotary_emb(xq, xk, freqs_cis):
      # xq, xk: (seq_len, B, num_heads, head_dim)
      # 将 xq reshape 为复数: (..., head_dim/2) 视为复数
      xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
      xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
      # freqs_cis: (seq_len, head_dim/2) → broadcast
      freqs_cis = freqs_cis[:xq.shape[0]]  # 截取当前 seq_len
      freqs_cis = freqs_cis.unsqueeze(1).unsqueeze(2)  # (seq_len, 1, 1, head_dim/2)
      xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # 回到实数
      xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
      return xq_out.type_as(xq), xk_out.type_as(xk)
  ```

- [ ] **Step 3: 运行验证脚本确认形状正确**

  ```bash
  cd src && LD_LIBRARY_PATH=/nix/store/ybp235ps7m4yd85v0pgvqkhd4xmxf6jq-gcc-14.3.0-lib/lib python3 -c "
  import torch
  from model import precompute_freqs_cis, apply_rotary_emb
  freqs = precompute_freqs_cis(32, 256)
  print('freqs shape:', freqs.shape)  # (256, 16)
  xq = torch.randn(10, 4, 2, 32)
  xk = torch.randn(10, 4, 2, 32)
  q, k = apply_rotary_emb(xq, xk, freqs)
  print('q shape:', q.shape, 'k shape:', k.shape)  # (10, 4, 2, 32)
  print('RoPE OK')
  "
  ```

---

### Task 2: 实现 `model.py` — Multi-Head Self-Attention

**Files:**
- Modify: `src/model.py`（在 Task 1 基础上追加）

- [ ] **Step 1: 编写 `CausalSelfAttention` 类**

  ```python
  class CausalSelfAttention(nn.Module):
      def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
          super().__init__()
          self.num_heads = num_heads
          self.head_dim = dim // num_heads
          assert dim % num_heads == 0
          self.wq = nn.Linear(dim, dim, bias=False)
          self.wk = nn.Linear(dim, dim, bias=False)
          self.wv = nn.Linear(dim, dim, bias=False)
          self.wo = nn.Linear(dim, dim, bias=False)
          self.attn_dropout = nn.Dropout(dropout)
          self.resid_dropout = nn.Dropout(dropout)
          # 注册预计算的 RoPE freqs
          freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
          self.register_buffer('freqs_cis', freqs_cis)
          # 因果掩码
          mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
          self.register_buffer('mask', mask)

      def forward(self, x):
          B, T, C = x.shape  # 注意 train.py 传入是 (seq_len, B, dim)
          # 上面注释是错的——实际 forward 接收的 x 形状取决于调用方式
          # model.py 的 forward 接收 input_ids: (seq_len, B)
          # 经 embedding 后 x: (seq_len, B, dim)
          # 为了和标准做法一致，我们转置为 (B, T, C)
          # 不，保持 (seq_len, B, dim) 的约定更安全，因为 data.py 就是这个形状
          # 但 attention 通常用 (B, T, C)
          # 方案：在 forward 里转置
          seq_len, B, C = x.shape
          x = x.transpose(0, 1)  # → (B, seq_len, dim)

          q = self.wq(x).view(B, seq_len, self.num_heads, self.head_dim)
          k = self.wk(x).view(B, seq_len, self.num_heads, self.head_dim)
          v = self.wv(x).view(B, seq_len, self.num_heads, self.head_dim)

          # 应用 RoPE
          q, k = apply_rotary_emb(q, k, self.freqs_cis)

          # 转置为 (B, num_heads, seq_len, head_dim) 用于 attention 计算
          q = q.transpose(1, 2)
          k = k.transpose(1, 2)
          v = v.transpose(1, 2)

          # Scaled dot-product attention with causal mask
          scale = self.head_dim ** -0.5
          scores = torch.matmul(q, k.transpose(-2, -1)) * scale
          scores = scores.masked_fill(self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0), float('-inf'))
          attn = torch.softmax(scores, dim=-1)
          attn = self.attn_dropout(attn)

          out = torch.matmul(attn, v)  # (B, num_heads, seq_len, head_dim)
          out = out.transpose(1, 2).contiguous().view(B, seq_len, C)
          out = self.wo(out)
          out = self.resid_dropout(out)

          return out.transpose(0, 1)  # 回到 (seq_len, B, dim)
  ```

- [ ] **Step 2: 运行验证脚本**

  ```bash
  cd src && LD_LIBRARY_PATH=/nix/store/ybp235ps7m4yd85v0pgvqkhd4xmxf6jq-gcc-14.3.0-lib/lib python3 -c "
  import torch
  from model import CausalSelfAttention
  attn = CausalSelfAttention(dim=64, num_heads=2)
  x = torch.randn(10, 4, 64)  # (seq_len=10, B=4, dim=64)
  out = attn(x)
  print('Attention output shape:', out.shape)  # (10, 4, 64)
  print('Attention OK')
  "
  ```

---

### Task 3: 实现 `model.py` — SwiGLU FFN

**Files:**
- Modify: `src/model.py`

- [ ] **Step 1: 编写 `SwiGLUFFN` 类**

  SwiGLU 公式：`output = W_down(SiLU(W_gate(x)) * W_up(x))`
  隐藏维度 = `4 * dim`（或 `8/3 * dim` 向上取 64 的倍数，这里简单用 `4 * dim`）

  ```python
  class SwiGLUFFN(nn.Module):
      def __init__(self, dim, hidden_dim=None, dropout=0.1):
          super().__init__()
          if hidden_dim is None:
              hidden_dim = 4 * dim
          self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
          self.w_up = nn.Linear(dim, hidden_dim, bias=False)
          self.w_down = nn.Linear(hidden_dim, dim, bias=False)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          # x: (seq_len, B, dim)
          return self.dropout(self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x)))
  ```

- [ ] **Step 2: 验证**

  ```bash
  cd src && LD_LIBRARY_PATH=/nix/store/ybp235ps7m4yd85v0pgvqkhd4xmxf6jq-gcc-14.3.0-lib/lib python3 -c "
  import torch
  from model import SwiGLUFFN
  ffn = SwiGLUFFN(dim=64)
  x = torch.randn(10, 4, 64)
  out = ffn(x)
  print('FFN output shape:', out.shape)  # (10, 4, 64)
  print('SwiGLU OK')
  "
  ```

---

### Task 4: 实现 `model.py` — TransformerBlock + CausalLMM 组装

**Files:**
- Modify: `src/model.py`

- [ ] **Step 1: 编写 `TransformerBlock` 类**

  Pre-norm 架构：

  ```python
  class TransformerBlock(nn.Module):
      def __init__(self, dim, num_heads, max_seq_len=256, ffn_hidden_dim=None, dropout=0.1):
          super().__init__()
          self.attn_norm = nn.LayerNorm(dim)
          self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout)
          self.ffn_norm = nn.LayerNorm(dim)
          self.ffn = SwiGLUFFN(dim, ffn_hidden_dim, dropout)

      def forward(self, x):
          # x: (seq_len, B, dim)
          x = x + self.attn(self.attn_norm(x))
          x = x + self.ffn(self.ffn_norm(x))
          return x
  ```

- [ ] **Step 2: 重写 `CausalLMM.__init__`**

  ```python
  class CausalLMM(nn.Module):
      def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8,
                   max_seq_len=256, ffn_hidden_dim=None, dropout=0.1):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, dim)
          self.drop = nn.Dropout(dropout)
          self.layers = nn.ModuleList([
              TransformerBlock(dim, num_heads, max_seq_len, ffn_hidden_dim, dropout)
              for _ in range(num_layers)
          ])
          self.norm = nn.LayerNorm(dim)
          self.head = nn.Linear(dim, vocab_size, bias=False)
          self.init_weights()
  ```

- [ ] **Step 3: 重写 `CausalLMM.forward`**

  ```python
      def forward(self, input_ids):
          x = self.drop(self.embedding(input_ids))  # (seq_len, B, dim)
          for layer in self.layers:
              x = layer(x)
          x = self.norm(x)
          x = self.head(x)
          return x
  ```

- [ ] **Step 4: 实现 `init_weights`**

  ```python
      def init_weights(self):
          for module in self.modules():
              if isinstance(module, nn.Linear):
                  nn.init.normal_(module.weight, mean=0.0, std=0.02)
                  if module.bias is not None:
                      nn.init.zeros_(module.bias)
              elif isinstance(module, nn.Embedding):
                  nn.init.normal_(module.weight, mean=0.0, std=0.02)
  ```

- [ ] **Step 5: 端到端验证**

  ```bash
  cd src && LD_LIBRARY_PATH=/nix/store/ybp235ps7m4yd85v0pgvqkhd4xmxf6jq-gcc-14.3.0-lib/lib python3 -c "
  import torch
  from model import CausalLMM
  model = CausalLMM(vocab_size=1000, dim=64, num_layers=2, num_heads=2, max_seq_len=32)
  input_ids = torch.randint(0, 1000, (16, 4))  # (seq_len=16, B=4)
  out = model(input_ids)
  print('Output shape:', out.shape)  # (16, 4, 1000)
  print('Model params:', sum(p.numel() for p in model.parameters()))
  print('CausalLMM OK')
  "
  ```

---

### Task 5: 重写 `train.py` — 训练循环 + Loss 记录

**Files:**
- Rewrite: `src/train.py`

- [ ] **Step 1: 改造训练循环**

  核心改动：
  - 增加 `--lr`、`--dropout` 参数
  - 训练时记录每步 loss 到列表
  - 增加 **梯度裁剪** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
  - 增加 **学习率调度**：CosineAnnealingLR 或 ReduceLROnPlateau
  - `train()` 返回 `(perplexity, step_losses)`
  - `evaluate()` 返回 `(perplexity, avg_loss)`
  - 训练结束后保存 loss 数据到 JSON

- [ ] **Step 2: 编写画图函数**

  ```python
  import json, matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  def plot_curves(train_step_losses, train_epoch_ppl, valid_epoch_ppl, save_path):
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
      # 左图: step-level train loss
      ax1.plot(train_step_losses)
      ax1.set_xlabel('Step')
      ax1.set_ylabel('Loss')
      ax1.set_title('Training Loss (per step)')
      ax1.grid(True)
      # 右图: epoch-level perplexity
      epochs = range(1, len(train_epoch_ppl) + 1)
      ax2.plot(epochs, train_epoch_ppl, 'b-o', label='Train PPL')
      ax2.plot(epochs, valid_epoch_ppl, 'r-o', label='Valid PPL')
      ax2.set_xlabel('Epoch')
      ax2.set_ylabel('Perplexity')
      ax2.set_title('Train & Valid Perplexity')
      ax2.legend()
      ax2.grid(True)
      plt.tight_layout()
      plt.savefig(save_path, dpi=150)
      print(f'Loss curves saved to {save_path}')
  ```

- [ ] **Step 3: 运行快速冒烟测试（1 epoch，小模型）**

  ```bash
  cd /home/pulcerto/Workspace/PRML/assignment2/code/src && \
  LD_LIBRARY_PATH=/nix/store/ybp235ps7m4yd85v0pgvqkhd4xmxf6jq-gcc-14.3.0-lib/lib \
  python3 train.py --epochs 1 --num_layers 2 --num_heads 2 --emb_dim 64 --max_sql 64 --train_batch_size 8
  ```

  预期：训练正常跑完，打印 train/valid perplexity，生成 `loss_curves.png`。

- [ ] **Step 4: 提交 model.py + train.py**

  ```bash
  cd /home/pulcerto/Workspace/PRML/assignment2/code
  git add src/model.py src/train.py
  git commit -m "feat: implement Transformer decoder with RoPE, SwiGLU, pre-norm"
  ```

---

### Task 6: 超参搜索 + 最佳模型训练

**Files:**
- Create: `src/sweep.py`（可选，或直接手跑多组）

- [ ] **Step 1: 定义搜索空间并跑实验**

  推荐手动跑以下组合（CPU 上训练，每组约几分钟）：

  | 实验 | emb_dim | layers | heads | lr   | batch | dropout |
  |------|---------|--------|-------|------|-------|---------|
  | A    | 64      | 2      | 2     | 1e-3 | 16    | 0.1     |
  | B    | 128     | 4      | 4     | 1e-3 | 16    | 0.1     |
  | C    | 256     | 4      | 4     | 3e-4 | 16    | 0.1     |
  | D    | 128     | 4      | 4     | 3e-4 | 32    | 0.2     |
  | E    | 256     | 6      | 8     | 1e-4 | 16    | 0.2     |

  每组跑 10 epochs，记录最终 valid perplexity。

  命令示例：
  ```bash
  # 实验 B
  python3 train.py --epochs 10 --emb_dim 128 --num_layers 4 --num_heads 4 --lr 1e-3 \
      --train_batch_size 16 --dropout 0.1 --tag expB
  ```

- [ ] **Step 2: 选出最佳配置，跑 20-30 epochs 的完整训练**

  ```bash
  # 假设实验 C 最佳
  python3 train.py --epochs 30 --emb_dim 256 --num_layers 4 --num_heads 4 --lr 3e-4 \
      --train_batch_size 16 --dropout 0.1 --tag best
  ```

- [ ] **Step 3: 保存最终结果**

  将 `loss_curves_best.png`、`results_best.json` 用于报告。

---

## 实现顺序总结

```
Task 1: RoPE (precompute_freqs_cis + apply_rotary_emb)
Task 2: CausalSelfAttention (含 RoPE + causal mask)
Task 3: SwiGLUFFN
Task 4: TransformerBlock + CausalLMM 组装
Task 5: train.py 重写 (loss 记录 + 画图 + 梯度裁剪)
Task 6: 超参搜索 + 最终训练
```

每个 Task 完成后都有验证步骤，确保组件正确再继续。
