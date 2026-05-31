import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RoPE: Rotary Positional Embedding ───────────────────────────────────────

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    预计算旋转角度的复数表示。
    freqs_cis[t, i] = e^{i * t / theta^(2i/dim)}

    Args:
        dim: head_dim (每个注意力头的维度)
        max_seq_len: 最大序列长度
        theta: 基础频率，默认 10000
    Returns:
        freqs_cis: (max_seq_len, dim//2) 复数张量
    """
    # freqs[i] = 1 / (theta^(2i/dim)), i = 0,1,...,dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{iθ}
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    将 RoPE 应用到 Q 和 K 上。

    Args:
        xq: (..., seq_len, num_heads, head_dim)
        xk: (..., seq_len, num_heads, head_dim)
        freqs_cis: (max_seq_len, head_dim//2) 复数张量
    Returns:
        xq_out, xk_out: 与输入形状相同的实数张量
    """
    seq_len = xq.shape[1]
    # 将最后 dim 维 reshape 为 (dim//2, 2)，然后视为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 截取当前序列长度，广播到 (1, seq_len, 1, head_dim//2)
    fc = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * fc).flatten(-2)
    xk_out = torch.view_as_real(xk_ * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ─── Causal Self-Attention with RoPE ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=512, dropout=0.1,
                 use_qk_norm=False, use_attn_gate=False, use_value_emb=False,
                 vocab_size=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_qk_norm = use_qk_norm
        self.use_attn_gate = use_attn_gate
        self.use_value_emb = use_value_emb

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # QK Norm: per-head LayerNorm on Q and K before attention
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

        # Attention Gate: learnable per-head scalar gate (SDPA Elementwise)
        if use_attn_gate:
            self.attn_gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # Value Embedding: separate embedding added to V
        if use_value_emb:
            assert vocab_size is not None, "vocab_size required for value_emb"
            self.value_embedding = nn.Embedding(vocab_size, dim)

        # 预计算 RoPE 频率和因果掩码
        self.register_buffer('freqs_cis', precompute_freqs_cis(self.head_dim, max_seq_len))
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x, input_ids=None):
        """
        Args:
            x: (seq_len, B, dim)
            input_ids: (seq_len, B) optional, for value_emb variant
        Returns:
            (seq_len, B, dim)
        """
        seq_len, B, C = x.shape

        # 转置为 (B, seq_len, dim) 做线性投影
        x_t = x.transpose(0, 1)

        q = self.wq(x_t).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x_t).view(B, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x_t).view(B, seq_len, self.num_heads, self.head_dim)

        # Value Embedding: add separate embedding to V
        if self.use_value_emb:
            v_emb = self.value_embedding(input_ids.transpose(0, 1))  # (B, seq_len, dim)
            v_emb = v_emb.view(B, seq_len, self.num_heads, self.head_dim)
            v = v + v_emb

        # QK Norm: normalize Q and K per-head before RoPE
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE: 输入形状 (B, seq_len, num_heads, head_dim)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        # 转置为 (B, num_heads, seq_len, head_dim) 用于 attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention + causal mask
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0),
            float('-inf')
        )
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)

        # Attention Gate: per-head sigmoid gate (applied before head concatenation)
        if self.use_attn_gate:
            out = out * torch.sigmoid(self.attn_gate)

        out = out.transpose(1, 2).contiguous().view(B, seq_len, C)
        out = self.wo(out)
        out = self.resid_dropout(out)

        return out.transpose(0, 1)  # 回到 (seq_len, B, dim)


# ─── SwiGLU Feed-Forward Network ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        SwiGLU: output = W_down( SiLU(W_gate(x)) * W_up(x) )
        Args:
            x: (seq_len, B, dim)
        """
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ─── Transformer Block (Pre-Norm) ────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=512, ffn_hidden_dim=None, dropout=0.1,
                 use_qk_norm=False, use_attn_gate=False, use_value_emb=False,
                 vocab_size=None):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len, dropout,
                                        use_qk_norm, use_attn_gate, use_value_emb,
                                        vocab_size)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden_dim, dropout)

    def forward(self, x, input_ids=None):
        """Pre-norm: x + SubLayer(LayerNorm(x))"""
        x = x + self.attn(self.attn_norm(x), input_ids)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── Causal Language Model ───────────────────────────────────────────────────

class CausalLMM(nn.Module):
    def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8,
                 max_seq_len=512, ffn_hidden_dim=None, dropout=0.1,
                 use_qk_norm=False, use_attn_gate=False, use_value_emb=False):
        super(CausalLMM, self).__init__()
        self.dim = dim
        self.use_value_emb = use_value_emb
        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, max_seq_len, ffn_hidden_dim, dropout,
                             use_qk_norm, use_attn_gate, use_value_emb, vocab_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.init_weights()

    def num_non_embedding_params(self):
        """Non-embedding parameter count: total params minus the main token embedding."""
        embed_params = sum(p.numel() for p in self.embedding.parameters())
        return sum(p.numel() for p in self.parameters()) - embed_params

    def init_weights(self):
        """标准 Transformer 权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (seq_len, B) token 索引
        Returns:
            logits: (seq_len, B, vocab_size)
        """
        x = self.drop(self.embedding(input_ids))  # (seq_len, B, dim)
        for layer in self.layers:
            x = layer(x, input_ids)
        x = self.norm(x)
        x = self.head(x)
        return x
