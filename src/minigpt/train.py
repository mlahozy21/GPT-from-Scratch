"""Training loop for the from-scratch GPT: AMP, gradient accumulation, cosine LR
with warmup, gradient clipping, periodic evaluation (loss + perplexity)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .data import TokenStream
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "outputs"
    max_iters: int = 5000
    batch_size: int = 32
    grad_accum: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    seed: int = 42


def _lr(it, cfg: TrainConfig):
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, stream, cfg, device):
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for i in range(cfg.eval_iters):
        x, y = stream.batch(cfg.batch_size, device)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    mean = losses.mean().item()
    return mean, math.exp(mean)


def train(model_cfg: GPTConfig, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    train_stream = TokenStream(f"{cfg.data_dir}/train.bin", model_cfg.block_size)
    val_stream = TokenStream(f"{cfg.data_dir}/val.bin", model_cfg.block_size)

    model = GPT(model_cfg).to(device)
    print(f"Model: {model.num_params()/1e6:.1f}M non-embedding params")
    opt = model.configure_optimizers(cfg.weight_decay, cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and not use_bf16))

    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = _lr(it, cfg)

        if it % cfg.eval_interval == 0:
            vloss, vppl = evaluate(model, val_stream, cfg, device)
            print(f"iter {it:5d} | val loss {vloss:.4f} | val ppl {vppl:7.2f} "
                  f"| {time.time()-t0:.0f}s")

        for micro in range(cfg.grad_accum):
            x, y = train_stream.batch(cfg.batch_size, device)
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if device == "cuda" else torch.autocast(device_type="cpu", dtype=torch.bfloat16))
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    import os
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model_cfg.__dict__},
               f"{cfg.out_dir}/ckpt.pt")
    final_loss, final_ppl = evaluate(model, val_stream, cfg, device)
    print(f"Done. Final val loss {final_loss:.4f} | ppl {final_ppl:.2f}")
    return model, final_loss, final_ppl
