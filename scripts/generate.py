"""Generate text from a trained checkpoint."""
import argparse
import torch
from minigpt.data import get_tokenizer
from minigpt.model import GPT, GPTConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/ckpt.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode, decode = enc.encode_ordinary, enc.decode
    except Exception:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        encode, decode = tok.encode, tok.decode

    idx = torch.tensor([encode(args.prompt)], device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(decode(out[0].tolist()))
