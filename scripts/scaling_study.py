"""Scaling study: train several model sizes for a fixed budget and plot the
final validation loss against the (non-embedding) parameter count."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minigpt.model import GPTConfig
from minigpt.train import TrainConfig, train

# (name, n_layer, n_head, n_embd)
SIZES = [
    ("tiny",   4, 4, 128),
    ("small",  6, 6, 256),
    ("medium", 8, 8, 512),
    ("large", 12, 12, 768),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="scaling.csv")
    args = ap.parse_args()

    rows = []
    for name, n_layer, n_head, n_embd in SIZES:
        print(f"\n===== {name}: L{n_layer} H{n_head} D{n_embd} =====")
        mcfg = GPTConfig(block_size=args.block_size, n_layer=n_layer,
                         n_head=n_head, n_embd=n_embd)
        tcfg = TrainConfig(data_dir=args.data_dir, out_dir=f"outputs/{name}",
                           max_iters=args.max_iters, batch_size=args.batch_size)
        model, vloss, vppl = train(mcfg, tcfg)
        rows.append({"name": name, "params_M": round(model.num_params() / 1e6, 2),
                     "val_loss": round(vloss, 4), "val_ppl": round(vppl, 2)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    xs = [r["params_M"] for r in rows]; ys = [r["val_loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-")
    for r in rows:
        ax.annotate(r["name"], (r["params_M"], r["val_loss"]))
    ax.set_xscale("log"); ax.set_xlabel("non-embedding parameters (M)")
    ax.set_ylabel("validation loss"); ax.set_title("Scaling: loss vs model size")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig("figures/scaling.png", dpi=150)
    print("\nSaved scaling.csv and figures/scaling.png"); [print(r) for r in rows]
