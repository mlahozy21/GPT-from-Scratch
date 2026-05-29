"""Tokenise a Hugging Face dataset into train.bin / val.bin."""
import argparse
from minigpt.data import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--max-docs", type=int, default=200000,
                    help="Limit documents for a quick build (0 = all).")
    args = ap.parse_args()
    prepare(args.dataset, args.text_field, args.out_dir, args.max_docs or None)
