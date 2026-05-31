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

## Results (trained run)

Training a **25.3M-parameter** model (8 layers, 8 heads, dim 512, context 256) on
**TinyStories** (44.7M training tokens) for 5,000 iterations (~11 min on one GPU):

| | start | final |
|---|------:|------:|
| validation loss | 10.92 | **1.54** |
| validation perplexity | 55,262 | **4.67** |

Sample (`prompt = "Once upon a time"`):

> *Once upon a time, there were two friends, Jack and Mia. They were playing in the
> park, having lots of fun. ... "Me too! Let's play again soon!" And so they did, and
> they played until it was time to go home, happy and tired from their special day.*

The model produces coherent, on-distribution short stories — confirming the
from-scratch implementation trains a real language model end to end.

## Run

One click in Colab (badge above), or locally with a GPU:

```bash
pip install -e .

python scripts/smoke_test.py                       # validate the implementation (~30 s, CPU ok)
python scripts/prepare_data.py --max-docs 200000   # BPE-tokenise TinyStories -> data/*.bin
python scripts/train.py --n-layer 8 --n-head 8 --n-embd 512 --max-iters 5000
python scripts/generate.py --prompt "Once upon a time"
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
