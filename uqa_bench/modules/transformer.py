import math
import json
import types
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    num_layers: int = 4
    max_seq_length: int = 8192
    hidden_size: int = 512
    num_heads: int = 4
    ffn_multiple_of: int = 256
    norm_eps: float = 1e-5
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    initializer_range: float = 0.02
    ffn_dim: int = -1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    qkv_bias: bool = False
    kv_heads: int = 0

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        if self.kv_heads <= 0:
            self.kv_heads = self.num_heads
        if self.ffn_dim == -1:
            ff_mult = 8 / 3
            mult_of = self.ffn_multiple_of
            ff_dim = int(ff_mult * self.hidden_size)
            ff_dim = mult_of * ((ff_dim + mult_of - 1) // mult_of)
            self.ffn_dim = ff_dim

    def save_to_json(self, json_path: str):
        dict_obj = asdict(self)
        json_str = json.dumps(dict_obj, indent=4, sort_keys=True)
        with open(json_path, "w") as f:
            f.write(json_str)
    
    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            dic = json.load(f)
        kwargs = {k: v for k, v in dic.items() if hasattr(cls, k)}
        return cls(**kwargs)

def build_rotary_embeddings(
    max_len: int,
    dim: int,
    base: int = 10000
) -> Tuple[torch.Tensor, torch.Tensor]:
    freq = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inv_freq = 1.0 / freq
    t = torch.arange(max_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=1)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=x.ndim - 1)
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    output = (x * cos) + (rotate_half(x) * sin)
    return output


def apply_rotary_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return apply_rotary(x.to(torch.float32), cos, sin).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        output = self.weight * out
        return output


class GLU(nn.Module):
    def __init__(self, in_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.up_proj_x = nn.Linear(in_dim, ff_dim, bias=False)
        self.up_proj_g = nn.Linear(in_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, in_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, g = self.up_proj_x(x), self.up_proj_g(x)
        h = F.silu(g) * x
        output = self.down_proj(h)
        return output


class AttentionWithRoPE(nn.Module):
    """Multi-head(or query) causal attention with rotary positional embedding."""

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        kv_heads: int,
        hidden_size: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert num_heads % kv_heads == 0
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.qkv_bias = qkv_bias
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_groups = self.num_heads // self.kv_heads
        self.q_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.kv_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.kv_heads * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        # Shape: [bsz, leng, hidden_dim]
        q: torch.Tensor = self.q_proj(hidden_states)
        # Shape: [bsz, leng, hidden_dim_kv]
        k: torch.Tensor = self.k_proj(hidden_states)
        v: torch.Tensor = self.v_proj(hidden_states)

        # Shape: [bsz, num_heads, q_len, head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [bsz, kv_heads, q_len, head_dim]
        k = k.view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_fp32(q, cos, sin)
        k = apply_rotary_fp32(k, cos, sin)

        # Shape: [bsz, num_heads, q_len, head_dim]
        if self.kv_groups != 1:
            # NOTE: PyTorch 2.5 support GQA.
            k = torch.repeat_interleave(k, dim=1, repeats=self.kv_groups)
            v = torch.repeat_interleave(v, dim=1, repeats=self.kv_groups)

        p = self.dropout if self.training else 0

        # Shape: [bsz, num_heads, q_len, head_dim]
        is_causal = True if attn_mask is None else False
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=p, is_causal=is_causal
        )
        # Shape: [bsz, q_len, hidden_size]
        output = output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        output = self.o_proj(output)

        return output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        num_layers: int,
        num_heads: int,
        kv_heads: int,
        hidden_size: int,
        ffn_dim: int,
        attention_dropout: int,
        hidden_dropout: int,
        norm_eps: int,
        qkv_bias: bool,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.qkv_bias = qkv_bias
        self.initializer_range = initializer_range
        self.attn = AttentionWithRoPE(
            layer_idx, num_heads, kv_heads, hidden_size, attention_dropout,
            qkv_bias=qkv_bias,
        )
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.mlp = GLU(hidden_size, ffn_dim)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        std = self.initializer_range

        def init(m: nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def inito(m):
            nn.init.normal_(m.weight, mean=0.0, std=std /
                            math.sqrt(2 * self.num_layers))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for p in [
            self.attn.q_proj, self.attn.k_proj, self.attn.v_proj,
            self.mlp.up_proj_x, self.mlp.up_proj_g
        ]:
            init(p)
            # A workaround for FSDP parameter initialization on the meta device.
            p.reset_parameters = types.MethodType(init, p)

        for p in [self.attn.o_proj, self.mlp.down_proj]:
            inito(p)
            # A workaround for FSDP parameter initialization on the meta device.
            p.reset_parameters = types.MethodType(inito, p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, cos, sin, attn_mask)
        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states


class TransformerPlus(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.hidden_size
        self.layers = self.build_layers(config)
        self.head_dim = config.hidden_size // config.num_heads
        cos = torch.empty(
            config.max_seq_length, self.head_dim,
            dtype=torch.float32
        )
        sin = torch.empty_like(cos)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        cos, sin = build_rotary_embeddings(
            self.config.max_seq_length, self.head_dim
        )
        self.cos.copy_(cos)
        self.sin.copy_(sin)

    def build_layers(self, config: TransformerConfig):
        layers = nn.ModuleList([
            TransformerLayer(
                i,
                config.num_layers,
                config.num_heads,
                config.kv_heads,
                config.hidden_size,
                config.ffn_dim,
                config.attention_dropout,
                config.hidden_dropout,
                config.norm_eps,
                config.qkv_bias,
                config.initializer_range,
            )
            for i in range(config.num_layers)
        ])
        return layers

    def get_rel_pos(self, offset: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[offset:offset + seq_len].view(
            1, 1, seq_len, self.head_dim
        )
        sin = self.sin[offset:offset + seq_len].view(
            1, 1, seq_len, self.head_dim
        )
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_lengths: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[1]
        cos, sin = self.get_rel_pos(0, seq_length)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cos, sin
            )

        return hidden_states
