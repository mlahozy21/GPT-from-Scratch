"""Tokenise a Hugging Face text dataset into a flat uint16 token stream, and
sample training blocks from it via a memory-mapped array (nanoGPT-style)."""

from __future__ import annotations

import os

import numpy as np
import torch


def get_tokenizer():
    """GPT-2 BPE (tiktoken if available, else the HF tokenizer)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda s: enc.encode_ordinary(s), enc.n_vocab, enc.eot_token
    except Exception:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        return (lambda s: tok.encode(s), tok.vocab_size,
                tok.eos_token_id or tok.vocab_size - 1)


def prepare(dataset="roneneldan/TinyStories", text_field="text", out_dir="data",
            max_docs=None, val_fraction=0.005):
    """Stream a dataset, BPE-encode it and write train.bin / val.bin (uint16)."""
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    encode, _, eot = get_tokenizer()
    ds = load_dataset(dataset, split="train", streaming=True)

    tokens = []
    for i, ex in enumerate(ds):
        if max_docs and i >= max_docs:
            break
        tokens.extend(encode(ex[text_field]))
        tokens.append(eot)
    arr = np.array(tokens, dtype=np.uint16)
    n_val = int(len(arr) * val_fraction)
    arr[:-n_val].tofile(os.path.join(out_dir, "train.bin"))
    arr[-n_val:].tofile(os.path.join(out_dir, "val.bin"))
    print(f"Wrote {len(arr) - n_val:,} train and {n_val:,} val tokens to {out_dir}/")


class TokenStream:
    """Random fixed-length blocks from a memory-mapped uint16 token file."""

    def __init__(self, path: str, block_size: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(
            self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), \
                   y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
