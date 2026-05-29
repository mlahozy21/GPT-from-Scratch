"""Train the from-scratch GPT on a prepared token stream."""
import argparse
from minigpt.model import GPTConfig
from minigpt.train import TrainConfig, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-interval", type=int, default=500)
    args = ap.parse_args()

    mcfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                     n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    tcfg = TrainConfig(data_dir=args.data_dir, out_dir=args.out_dir,
                       max_iters=args.max_iters, batch_size=args.batch_size,
                       grad_accum=args.grad_accum, lr=args.lr,
                       eval_interval=args.eval_interval)
    train(mcfg, tcfg)
