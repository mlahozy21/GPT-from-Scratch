# Training a Transformer (GPT) from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlahozy21/GPT-from-Scratch/blob/main/notebooks/train_gpt.ipynb)

A **decoder-only Transformer language model implemented from scratch** in PyTorch —
only tensor primitives, no pretrained or `transformers` model classes — with the
components used in current LLMs:

- **RMSNorm** (pre-norm) · **Rotary Position Embeddings (RoPE)** · **SwiGLU** MLP
- causal multi-head self-attention with a **KV cache** for fast generation
- weight tying, AdamW + decoupled weight decay, cosine LR with warmup, grad clipping,
  gradient accumulation and mixed precision.

Everything is in `src/minigpt/model.py` (~200 lines) and a small training loop.

## Implementation is validated

`scripts/smoke_test.py` trains a tiny model on a **copy task** (predict a sequence
that only appears earlier in the context — solvable only via attention). It reaches
**98.4% exact-copy accuracy** in ~300 steps on CPU, confirming attention, RoPE,
masking and cached generation are correct. A short write-up is in
[`paper/report.pdf`](paper/report.pdf).

## Run

One click in Colab (badge above), or locally with a GPU:

```bash
pip install -e .

python scripts/smoke_test.py                       # validate the implementation (~30 s, CPU ok)
python scripts/prepare_data.py --max-docs 200000   # BPE-tokenise TinyStories -> data/*.bin
python scripts/train.py --n-layer 8 --n-head 8 --n-embd 512 --max-iters 5000
python scripts/generate.py --prompt "Once upon a time"
python scripts/scaling_study.py --max-iters 3000   # loss vs model size -> figures/scaling.png
```

Model size, context length, optimisation and data are all configurable via flags;
scale them to your compute budget.

## Repository layout

```
.
├── src/minigpt/
│   ├── model.py      # the GPT: RMSNorm, RoPE, attention(+KV cache), SwiGLU, generate
│   ├── data.py       # BPE tokenisation -> uint16 stream + block sampler
│   └── train.py      # training loop (AMP, cosine LR, grad accum, eval/perplexity)
├── scripts/          # smoke_test, prepare_data, train, generate, scaling_study
├── notebooks/train_gpt.ipynb
├── paper/report.tex (+ report.pdf)
└── figures/
```

## References

Vaswani et al. (2017) *Attention Is All You Need* · Su et al. (2021) *RoFormer/RoPE* ·
Zhang & Sennrich (2019) *RMSNorm* · Shazeer (2020) *GLU Variants* ·
Kaplan et al. (2020) *Scaling Laws*. Data: TinyStories (Eldan & Li, 2023).

## License

Released under the MIT License — see `LICENSE`.
