"""A modern decoder-only Transformer (GPT) implemented from scratch.

Components implemented by hand:
  - token embedding with weight tying to the output head
  - RMSNorm (pre-norm)
  - Rotary Position Embeddings (RoPE)
  - multi-head causal self-attention (with an optional KV cache for generation)
  - SwiGLU feed-forward block
  - autoregressive sampling with temperature / top-k
No `transformers` model classes are used: only torch primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    mlp_ratio: float = 8 / 3      # SwiGLU hidden size factor (~2.67x)
    rope_base: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def build_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    """Return (cos, sin) of shape (seq_len, head_dim) for rotary embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.dropout_p = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # scaled dot-product attention with a causal mask (manual)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Tq,Tk)
        Tq, Tk = att.shape[-2], att.shape[-1]
        causal = torch.ones(Tq, Tk, device=x.device, dtype=torch.bool).tril(diagonal=Tk - Tq)
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v                                  # (B,H,Tq,D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        hidden = 32 * ((hidden + 31) // 32)          # round to a multiple of 32
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        a, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = build_rope(T, self.cfg.n_embd // self.cfg.n_head,
                              self.cfg.rope_base, idx.device, self.tok_emb.weight.dtype)
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive sampling with a KV cache."""
        self.eval()
        head_dim = self.cfg.n_embd // self.cfg.n_head
        caches = [None] * len(self.blocks)
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            T = cur.shape[1]
            cos, sin = build_rope(pos + T, head_dim, self.cfg.rope_base,
                                  idx.device, self.tok_emb.weight.dtype)
            cos, sin = cos[pos:pos + T], sin[pos:pos + T]
            x = self.tok_emb(cur)
            for i, block in enumerate(self.blocks):
                x, caches[i] = block(x, cos, sin,
                                     kv_cache=(caches[i] if caches[i] else (None, None)))
            logits = self.lm_head(self.norm(x))[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            pos += T
            cur = nxt
        return idx

    def configure_optimizers(self, weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
