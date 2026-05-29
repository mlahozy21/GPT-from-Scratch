"""Validate the from-scratch GPT on a synthetic copy task (CPU-friendly).

Sequences look like:  p_1 ... p_L  <SEP>  p_1 ... p_L
The model can only predict the part after <SEP> by attending to the prefix, so
reaching high accuracy proves attention + the whole pipeline work.
"""

import sys
sys.path.insert(0, "src")
import torch

from minigpt.model import GPT, GPTConfig

VOCAB, L, SEP = 20, 8, 19          # tokens 0..18 are content, 19 is <SEP>
BLOCK = 2 * L + 1


def batch(bs, device):
    p = torch.randint(0, VOCAB - 1, (bs, L), device=device)
    sep = torch.full((bs, 1), SEP, device=device)
    seq = torch.cat([p, sep, p], dim=1)             # (bs, 2L+1)
    x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
    y[:, :L] = -1                                   # only score the copied part
    return x, y, p


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(GPTConfig(vocab_size=VOCAB, block_size=BLOCK,
                          n_layer=2, n_head=2, n_embd=64)).to(dev)
    print(f"params (non-embedding): {model.num_params():,}")
    opt = model.configure_optimizers(lr=3e-3)
    model.train()
    for step in range(300):
        x, y, _ = batch(64, dev)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 60 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")

    # Evaluate exact-copy accuracy via generation (uses the KV cache).
    x, _, p = batch(200, dev)
    prompt = x[:, :L + 1]                            # prefix + <SEP>
    out = model.generate(prompt, max_new_tokens=L, temperature=1.0, top_k=1)
    gen = out[:, L + 1:]
    acc = (gen == p).float().mean().item()
    print(f"\ncopy accuracy: {acc:.1%}")
    assert loss.item() < 0.5 and acc > 0.9, "smoke test FAILED"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
