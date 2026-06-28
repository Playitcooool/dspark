# DSpark

**D**raft model with **S**peculative decoding using a **Markov** transition head.

A lightweight speculative-decoding draft model for Qwen3.5 (or any LLM with a compatible API).

## Architecture

```
input tokens  ─►  base model (frozen)  ─►  last hidden state (B, H)
                                              │
                                  ┌───────────┼───────────┐
                                  ▼           ▼           ▼
                           DraftHeads   MarkovHead  ConfidenceHead
                           (×N heads)      │            (×N)
                                  │         │              │
                                  ▼         ▼              ▼
                              (B,N,H) + (B,N,H)      P(accept_k)
                                  │
                                  ▼
                           vocab proj (B,N,V)
                                  │
                                  ▼
                           softmax(logits)
```

1. **DraftHeads** — N parallel SiLU-activated adapters predicting positions 1…N ahead from the *same* hidden state (Medusa-style).
2. **MarkovHead** — Bilinear transition bias from adjacent token embeddings `(x_{t-1}, x_t)`. Added to draft logits to capture bigram patterns.
3. **ConfidenceHead** — MLP predicting `P(accept_k)` per draft position, for deciding how many tokens to accept.

## Setup

```bash
pip install torch transformers datasets accelerate
```

## Training

```bash
python -m dspark.train
```

Adjust `TrainConfig` in `train.py` for your hardware (batch size, gradient accumulation, context length).

## Evaluation

```bash
python -m dspark.eval
```

Reports:
- Base-model perplexity vs DSpark first-draft-position perplexity
- Per-position marginal & cumulative acceptance probability
- Speculative-decoding speedup (tokens/sec)

## Parameters

| Module       | Trainable params | Details |
|-------------|-----------------|---------|
| Base model  | 0 (frozen)       | Qwen3.5-0.8B (752.4M) |
| DraftHeads  | ~5.2M            | 5× adapter `Linear(H,H) → SiLU` |
| MarkovHead  | ~2.1M            | `Linear(2H,H) → SiLU → shared vocab proj` |
| Confidence  | ~0.13M           | `LayerNorm → Linear(H,128) → SiLU → Linear(128,N)` |
| **Total**   | **~7.5M**        | on top of 752.4M frozen |

## How it works

**Training:**
1. Base model processes context + N future tokens (causal mask prevents leakage).
2. Draft heads predict each future position from the last context hidden state.
3. Markov head adds a transition bias computed from ground-truth adjacent token pairs.
4. CE loss on `(draft_logits + markov_bias)` vs the ground-truth tokens.
5. Confidence head is trained with BCE against the oracle acceptance probability `∑_v min(draft(v), base(v))`.

**Inference (speculative decoding):**
1. Draft heads predict N tokens in parallel from the context hidden state.
2. Greedy-decode step 1 tokens → Markov head computes transition biases.
3. Re-rank logits with markov bias → re-sample tokens.
4. Confidence head decides how many to attempt.
5. Standard speculative-decoding verification against the base model.

## License

MIT
