"""LLaDA: Large Language Diffusion Models (Nie et al., 2025).

Masked absorbing-state discrete diffusion on text tokens.
- Forward: tokens transition to [MASK] according to a linear schedule.
- Reverse: bidirectional Transformer predicts p(x_0 | x_t).
- Sampling: iterative decoding with confidence-based remasking.

Reference: https://github.com/ML-GSAI/LLaDA
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_lm import BidirectionalBlock


class LLaDA(nn.Module):
    """Masked diffusion language model following LLaDA (Nie et al., 2025)."""

    def __init__(self, vocab_size, dim=256, num_layers=4, num_heads=8,
                 max_seq_len=256, ffn_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size   # includes [MASK] at last index
        self.mask_id = vocab_size - 1
        self.dim = dim

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
        """input_ids: (seq_len, B), some positions = mask_id."""
        x = self.drop(self.embedding(input_ids))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))  # (seq_len, B, vocab_size)

    def diffuse(self, x0):
        """Mask tokens at random ratio t ~ U(0, 1) per sequence.

        Returns (x_t, mask, t) where mask indicates which positions
        were replaced with [MASK] and t is the per-sequence ratio.
        """
        B = x0.size(1)
        # LLaDA samples t uniformly from [eps, 1) to avoid t=0
        t = torch.rand(B, device=x0.device) * 0.999 + 0.001
        mask = torch.rand_like(x0, dtype=torch.float, device=x0.device) < t.unsqueeze(0)
        xt = x0.clone()
        xt[mask] = self.mask_id
        return xt, mask, t

    def training_loss(self, x0):
        """Compute weighted diffusion loss on a batch.
        [MASK] token is excluded from prediction targets.
        """
        xt, mask, t = self.diffuse(x0)
        logits = self.forward(xt)  # (seq_len, B, vocab)
        logits = logits.clone()
        logits[:, :, self.mask_id] = -float('inf')

        vocab_eff = self.vocab_size - 1
        loss_per_token = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x0.view(-1),
            reduction='none',
            ignore_index=-100,
        ).view_as(x0)

        weight = 1.0 / t.unsqueeze(0).clamp(min=0.05)
        weighted_loss = (loss_per_token * mask.float() * weight).sum()
        n_masked = mask.sum().clamp(min=1)
        return weighted_loss / n_masked

    @torch.no_grad()
    def eval_loss(self, x0, num_masks=5):
        """
        Compute unweighted pseudo-ppl by averaging over several mask ratios.
        Uses fixed mask ratios {0.1, 0.3, 0.5, 0.7, 0.9} for stable evaluation,
        without the 1/t weight factor.
        """
        ratios = torch.linspace(0.1, 0.9, num_masks, device=x0.device)
        total_loss = 0.0
        total_masked = 0
        B = x0.size(1)

        for r in ratios:
            t = torch.full((B,), r, device=x0.device)
            mask = torch.rand_like(x0, dtype=torch.float, device=x0.device) < t.unsqueeze(0)
            xt = x0.clone()
            xt[mask] = self.mask_id

            logits = self.forward(xt)
            logits = logits.clone()
            logits[:, :, self.mask_id] = -float('inf')
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                x0.view(-1),
                reduction='none',
            ).view_as(x0)

            total_loss += (loss_per_token * mask.float()).sum().item()
            total_masked += mask.sum().item()

        return total_loss / max(total_masked, 1)

    @torch.no_grad()
    def generate(self, seq_len, batch_size=1, steps=128, temperature=1.0,
                 remask_low_conf=True, device='cpu'):
        """
        LLaDA-style iterative decoding with remasking.
        Algorithm (per generation step):
          1. Predict p(x0 | xt) for all masked positions.
          2. Determine how many tokens n_k to keep this step from the schedule.
          3. Select n_k most confident predictions and unmask them.
          4. Optionally remask the least confident unmasked tokens.
        """
        tokens = torch.full((seq_len, batch_size), self.mask_id,
                            dtype=torch.long, device=device)
        unmasked = torch.zeros(seq_len, batch_size, dtype=torch.bool, device=device)

        for s in range(1, steps + 1):
            frac = math.cos(math.pi / 2 * (1.0 - s / steps))
            target_unmasked = max(1, int(seq_len * batch_size * frac))
            n_to_unmask = max(1, target_unmasked - unmasked.sum().item())

            logits = self.forward(tokens) / temperature
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)  # (seq_len, B)
            conf[unmasked] = -float('inf')

            flat_conf = conf.view(-1)
            n_available = (~unmasked).sum().item()
            n_pick = min(n_to_unmask, n_available)
            _, top_idx = torch.topk(flat_conf, n_pick)

            flat_probs = probs.view(-1, self.vocab_size)
            sampled = torch.multinomial(flat_probs[top_idx], 1).squeeze(-1)

            flat_tokens = tokens.view(-1)
            flat_tokens[top_idx] = sampled
            unmasked.view(-1)[top_idx] = True
            if remask_low_conf and s < steps * 0.9:
                n_remask = int(n_to_unmask * 0.1)  # remask 10% as noise
                if n_remask > 0:
                    unmasked_flat = unmasked.view(-1)
                    unmasked_indices = unmasked_flat.nonzero(as_tuple=True)[0]
                    if len(unmasked_indices) > 0:
                        remask_conf = conf.view(-1)[unmasked_indices]
                        _, low_idx = torch.topk(remask_conf, n_remask, largest=False)
                        flat_tokens[unmasked_indices[low_idx]] = self.mask_id
                        unmasked.view(-1)[unmasked_indices[low_idx]] = False

        return tokens

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
