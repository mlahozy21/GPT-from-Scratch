"""Fast CPU smoke tests for the from-scratch GPT (architecture-level invariants)."""
import sys

import pytest
import torch

sys.path.insert(0, "src")

from minigpt.model import GPT, GPTConfig  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    return GPT(cfg).eval(), cfg


def test_forward_shapes_and_finite_loss(tiny_model):
    model, cfg = tiny_model
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_causality(tiny_model):
    """Changing a future token must not change past logits (causal mask works)."""
    model, cfg = tiny_model
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % cfg.vocab_size  # perturb ONLY the last token
    with torch.no_grad():
        la, _ = model(x)
        lb, _ = model(y)
    assert torch.allclose(la[0, :-1], lb[0, :-1], atol=1e-5)
    assert not torch.allclose(la[0, -1], lb[0, -1], atol=1e-5)


def test_generate_length_and_range(tiny_model):
    model, cfg = tiny_model
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=8)
    assert out.shape == (1, 12)
    assert int(out.max()) < cfg.vocab_size and int(out.min()) >= 0


def test_overfits_a_fixed_batch():
    """A few optimizer steps must reduce the loss on a fixed batch (training works)."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=32)
    model = GPT(cfg)
    opt = model.configure_optimizers(lr=3e-3)
    x = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
    model.train()
    _, first = model(x, x)
    for _ in range(30):
        _, loss = model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.8


def test_num_params_positive(tiny_model):
    model, _ = tiny_model
    assert model.num_params() > 0
