import math
import torch
import torch.nn as nn


# RoPE & Causal Mask reference: llama implementation of transformers
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device,
    past_key_values_length: int = 0
):
    # mask_shape [bsz, 1, tgt_len, src_len]
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int):
    # mask shape: [bsz, 1, tgt_len, src_len]
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_embed(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryEmbed(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len,
        attn_scale,
        device,
        theta=10000
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = theta
        self.attn_scale = attn_scale
        self.device = device
        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attn_scale
        sin = emb.sin() * self.attn_scale

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attn_heads: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.head_dim = hidden_dim // num_attn_heads
        self.attn_scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, self.head_dim * self.num_attn_heads, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * self.num_attn_heads, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * self.num_attn_heads, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: bool = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False
    ):
        # [bsz, seq_len, hidden_dim]
        bsz, seq_len, _ = hidden_states.shape

        # [bsz, num_heads, seq_len, head_dim]
        query = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(bsz, seq_len, self.num_attn_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_embed(query, key, cos, sin)

        kv_seq_len = key.shape[-2]
        if past_kv is not None:
            kv_seq_len += past_kv[0].shape[-2]
            key = torch.cat([past_kv[0], key], dim=2)
            value = torch.cat([past_kv[1], value], dim=2)
        
        past_kv = (key, value) if use_cache else None

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
            # numerical stability
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(bsz, seq_len, self.hidden_dim)

        attn_out = self.o_proj(attn_out)
        
        return attn_out, attn_weights, past_kv


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.gate = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down = nn.Linear(mlp_dim, hidden_dim, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down(self.act_fn(self.gate(x)) * self.up(x))
    
    
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        num_attn_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        bias: bool = False,
        device=None
    ):
        super().__init__()
        self.hidden_dim = dim
        self.rotary_emb = RotaryEmbed(dim // num_attn_heads, max_seq_len, device=device, attn_scale=1.0)
        self.attn = Attention(dim, num_attn_heads, dropout=dropout, bias=bias)
        self.mlp = MLP(dim, mlp_ratio)
        self.pre_norm = nn.LayerNorm(dim)
        self.post_attn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        hidden_states,
        attn_mask=None,
        position_ids=None,
        past_kv=None,
        use_cache=False
    ):
        res = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states, _, past_kv = self.attn(
            hidden_states,
            position_embeddings,
            attn_mask,
            past_kv,
            use_cache=use_cache
        )
        hidden_states = res + hidden_states

        res = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = res + hidden_states

        return (hidden_states, past_kv)


class CausalLMM(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a stack of Transformer blocks and an output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The Transformer blocks have input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(
        self,
        vocab_size,
        dim=256,
        max_seq_len=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        bias=False,
        device=None
    ):
        super(CausalLMM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim)

        # TODO: Construct you Transformer model
        self.layers = nn.ModuleList([
            Block(dim, max_seq_len, num_attn_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, bias=bias, device=device) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        # TODO: Init model weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids,
        attn_mask=None,
        position_ids=None,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]]=None,
        labels=None,
        use_cache=None,
    ):
        x = self.embedding(input_ids)

        # TODO: Write code here
        use_cache = use_cache if use_cache is not None else True
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        past_kv_length = 0
        if past_kvs is not None:
            past_kv_length = past_kvs[0][0].shape[2]
        
        seq_len_with_past = seq_len + past_kv_length
        if position_ids is None:
            position_ids = torch.arange(
                past_kv_length, seq_len_with_past, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        
        if attn_mask is None:
            attn_mask = torch.ones(
                (bsz, seq_len_with_past), dtype=torch.bool, device=device
            )
        
        causal_mask = _make_causal_mask((bsz, seq_len), x.dtype, device, past_kv_length)
        expand_attn_mask = _expand_mask(attn_mask, x.dtype, tgt_len=seq_len)
        attn_mask = causal_mask + expand_attn_mask
        new_past_kvs = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, past_kv = layer(
                x,
                attn_mask=attn_mask,
                position_ids=position_ids,
                past_kv=past_kv,
                use_cache=use_cache
            )
            if use_cache:
                new_past_kvs.append(past_kv)
        
        x = self.norm(x)
        x = self.head(x)

        return {
            "logits": x,
            "past_key_values": new_past_kvs
        }
