"""Masked Absorbing-State Diffusion Language Model (LLaDA-style).

Forward process: randomly mask tokens at ratio t ~ Uniform(0, 1).
Reverse process: predict original tokens from masked sequence (bidirectional).
Sampling: iterative unmasking — start from all [MASK], gradually decode.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import precompute_freqs_cis, apply_rotary_emb, SwiGLUFFN


# ─── Bidirectional Self-Attention (no causal mask) ───────────────────────────

class BidirectionalAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer('freqs_cis',
                             precompute_freqs_cis(self.head_dim, max_seq_len))

    def forward(self, x):
        seq_len, B, C = x.shape
        x_t = x.transpose(0, 1)  # (B, seq_len, dim)
        q = self.wq(x_t).view(B, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x_t).view(B, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x_t).view(B, seq_len, self.num_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # no causal mask → bidirectional
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, C)
        out = self.wo(out)
        out = self.resid_dropout(out)
        return out.transpose(0, 1)  # (seq_len, B, dim)


# ─── Bidirectional Transformer Block ──────────────────────────────────────────

class BidirectionalBlock(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len=256, ffn_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = BidirectionalAttention(dim, num_heads, max_seq_len, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ─── Masked Diffusion Language Model ──────────────────────────────────────────

class MaskedDiffusionLM(nn.Module):
    def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8,
                 max_seq_len=256, ffn_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size  # includes [MASK] token at index vocab_size-1
        self.dim = dim
        self.mask_id = vocab_size - 1

        self.embedding = nn.Embedding(vocab_size, dim)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            BidirectionalBlock(dim, num_heads, max_seq_len, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """input_ids: (seq_len, B) with some positions = mask_id."""
        x = self.drop(self.embedding(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)  # (seq_len, B, vocab_size)
        # Zero out [MASK] — model should never predict it
        logits = logits.clone()
        logits[:, :, self.mask_id] = -float('inf')
        return logits

    @torch.no_grad()
    def generate(self, seq_len, batch_size, steps=32, device='cpu', top_p=0.9):
        """Iterative unmasking sampling.

        Start from all-[MASK], at each step predict all masked positions,
        keep the most confident ones, re-mask the rest.
        """
        tokens = torch.full((seq_len, batch_size), self.mask_id,
                            dtype=torch.long, device=device)
        mask = torch.ones(seq_len, batch_size, dtype=torch.bool, device=device)

        for step in range(steps):
            # Number of tokens to unmask this step (linear schedule)
            n_unmask = int(seq_len * batch_size * (step + 1) / steps)
            n_currently_masked = mask.sum().item()
            n_to_unmask = max(1, n_unmask - (seq_len * batch_size - n_currently_masked))

            logits = self.forward(tokens)  # (seq_len, B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Confidence: max probability for each masked position
            confidence = probs.max(dim=-1).values  # (seq_len, B)
            confidence[~mask] = -float('inf')  # already unmasked → skip

            # Select top-k most confident masked positions
            flat_conf = confidence.view(-1)
            _, top_indices = torch.topk(flat_conf, min(n_to_unmask, n_currently_masked))

            # Sample from predicted distribution at selected positions
            flat_probs = probs.view(-1, self.vocab_size)
            sampled = torch.multinomial(flat_probs[top_indices], 1).squeeze(-1)

            flat_tokens = tokens.view(-1)
            flat_tokens[top_indices] = sampled
            mask.view(-1)[top_indices] = False

            if mask.sum() == 0:
                break

        return tokens

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
