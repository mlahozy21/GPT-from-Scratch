"""KV-cache correctness test.

The single real correctness risk in cached generation is that the incremental,
cached forward pass produces logits that differ from a full (uncached) forward
over the same prefix. This test drives the model's cached attention path one
token at a time (mirroring `GPT.generate`) and asserts that, at every position,
the next-token logits match a full uncached `forward` over the prefix so far to
atol ~1e-4. Tiny config, CPU only.
"""
import sys

import torch

sys.path.insert(0, "src")

from minigpt.model import GPT, GPTConfig, build_rope  # noqa: E402


def _cached_logits(model, idx):
    """Run the cached forward path token-by-token over `idx`.

    Returns a tensor (T, vocab) where row t is the next-token logits the model
    produced after consuming idx[:, :t+1] using the KV cache.
    """
    cfg = model.cfg
    head_dim = cfg.n_embd // cfg.n_head
    caches = [None] * len(model.blocks)
    pos = 0
    out = []
    for t in range(idx.shape[1]):
        cur = idx[:, t:t + 1]
        T = cur.shape[1]
        cos, sin = build_rope(pos + T, head_dim, cfg.rope_base,
                              idx.device, model.tok_emb.weight.dtype)
        cos, sin = cos[pos:pos + T], sin[pos:pos + T]
        x = model.tok_emb(cur)
        for i, block in enumerate(model.blocks):
            x, caches[i] = block(x, cos, sin,
                                 kv_cache=(caches[i] if caches[i] else (None, None)))
        logits = model.lm_head(model.norm(x))[:, -1, :]
        out.append(logits)
        pos += T
    return torch.cat(out, dim=0)  # (T, vocab) for batch size 1


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=37, block_size=24, n_layer=3, n_head=3, n_embd=48)
    model = GPT(cfg).eval()

    torch.manual_seed(123)
    T = 16
    idx = torch.randint(0, cfg.vocab_size, (1, T))

    with torch.no_grad():
        # Full, uncached forward over the whole prefix at once.
        full_logits, _ = model(idx)          # (1, T, vocab)
        full_logits = full_logits[0]         # (T, vocab)

        # Incremental cached forward, one token at a time.
        cached_logits = _cached_logits(model, idx)  # (T, vocab)

    assert cached_logits.shape == full_logits.shape
    max_diff = (cached_logits - full_logits).abs().max().item()
    assert torch.allclose(cached_logits, full_logits, atol=1e-4), (
        f"cached logits diverge from full forward (max abs diff {max_diff:.2e})"
    )


def test_generate_greedy_is_deterministic_via_cache():
    """top_k=1 greedy generation must be reproducible (cache path is stable)."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=29, block_size=20, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    a = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=1)
    b = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=1)
    assert torch.equal(a, b)
