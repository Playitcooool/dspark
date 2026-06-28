"""Evaluate DSpark: perplexity, acceptance rate, and speculative-decoding speedup."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import DSparkModel


@dataclass
class EvalConfig:
    model_path: str = "/Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B/"
    checkpoint: str = ""  # empty = untrained heads
    num_drafts: int = 5
    context_len: int = 512
    batch_size: int = 4
    num_batches: int = 20          # how many batches for perplexity eval
    # Speculative decoding
    max_new_tokens: int = 128
    spec_decode_rounds: int = 5


# ═══════════════════════════════════════════════════════════════ helpers


def _ensure_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<|endoftext|>"
    return tok


def _load_dataloader(tokenizer, cfg: EvalConfig):
    """Return a small validation DataLoader from wikitext-103-raw."""
    from torch.utils.data import DataLoader
    from .train import ParquetTextStream, _collate, _ensure_wikitext

    _, valid_path = _ensure_wikitext()

    stream = ParquetTextStream(valid_path, tokenizer, cfg.context_len, cfg.num_drafts)
    return DataLoader(stream, batch_size=cfg.batch_size, collate_fn=_collate)


# ═══════════════════════════════════════════════════════════════ perplexity


@torch.no_grad()
def eval_perplexity(model: DSparkModel, dataloader, cfg: EvalConfig):
    """Compare base-model PPL vs DSpark (draft-head 0 + markov) PPL."""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    base_nll, dspark_nll, count = 0.0, 0.0, 0

    for i, batch in enumerate(dataloader):
        if i >= cfg.num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = input_ids.shape

        # ── Base-model perplexity (standard LM) ──────────────────────────────
        out = model.base_model(input_ids)
        # Standard LM loss on shift-1 labels
        base_logits = out.logits  # (B, T, V)
        lm_labels = labels[:, :T]  # (B, T) — ground-truth for next-token pred
        # cross_entropy internally shifts: logits[:, :-1] vs labels[:, 1:]
        base_nll += F.cross_entropy(
            base_logits[:, :-1, :].reshape(-1, model.vocab_size),
            lm_labels[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        count += (T - 1) * B

        # ── DSpark perplexity via draft_predict ────────────────────────────
        _, _, draft_logits = model.draft_predict(input_ids)
        # draft_logits: (B, N, V) — take position 0
        dspark_nll += F.cross_entropy(
            draft_logits[:, 0, :].reshape(-1, model.vocab_size),
            labels[:, T].reshape(-1),
            reduction="sum",
        ).item()

    base_ppl = math.exp(base_nll / count)
    dspark_ppl = math.exp(dspark_nll / (cfg.num_batches * cfg.batch_size))
    print(f"Base PPL:       {base_ppl:.2f}  (over {count} tokens)")
    print(f"DSpark PPL (@1): {dspark_ppl:.2f}  (over {cfg.num_batches * cfg.batch_size} positions)")
    return base_ppl, dspark_ppl


# ═══════════════════════════════════════════════════════════════ acceptance rate


@torch.no_grad()
def eval_acceptance(model: DSparkModel, dataloader, cfg: EvalConfig):
    """Measure per-position acceptance probability of draft tokens."""
    model.eval()
    device = next(model.parameters()).device

    total_accept = torch.zeros(cfg.num_drafts)
    total_pos = 0

    for i, batch in enumerate(dataloader):
        if i >= cfg.num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = input_ids.shape

        # Get base logits at draft positions (base model on full sequence)
        full_out = model.base_model(labels)
        base_at = full_out.logits[:, T - 1:T - 1 + cfg.num_drafts, :]  # (B, N, V)

        # Get draft logits from diffusion inference
        _, _, draft_logits = model.draft_predict(input_ids)

        # Acceptance = sum_v min(draft_softmax(v), base_softmax(v))
        dp = F.softmax(draft_logits.float(), dim=-1)
        bp = F.softmax(base_at.float(), dim=-1)
        accept_probs = torch.min(dp, bp).sum(dim=-1)  # (B, N)

        total_accept += accept_probs.sum(dim=0).cpu()
        total_pos += B

    avg_accept = total_accept / total_pos
    print("Per-position acceptance probability:")
    for k in range(cfg.num_drafts):
        cum = avg_accept[:k + 1].prod().item() if k > 0 else avg_accept[0].item()
        print(f"  pos {k}: marginal={avg_accept[k]:.4f}  cumulative={cum:.4f}")
    print(f"  Expected accepted tokens per round: {min(1, avg_accept[0].item()):.4f}")
    return avg_accept


# ═══════════════════════════════════════════════════════════════ speculative decoding


@torch.no_grad()
def spec_decode_step(model: DSparkModel, context: torch.LongTensor,
                     max_drafts: int = 5,
                     temperature: float = 0.0,
                     threshold: float = 0.05,
                     ) -> tuple[list[int], int]:
    """One round of speculative decoding.

    Args:
        context: (1, T) input tokens.
        max_drafts: max draft tokens to generate.
        temperature: sampling temperature (0 = greedy).
        threshold: minimum confidence to consider a draft token.
    Returns:
        (accepted_ids, n_accepted) where accepted_ids are the new tokens.
    """
    device = context.device
    T = context.shape[1]
    N = max_drafts

    # 1. Draft
    draft_tokens, accept_probs, final_logits = model.draft_predict(
        context, temperature=temperature)

    # 2. How many to try based on confidence
    n_try = N
    for k in range(N):
        if accept_probs[0, k].item() < threshold:
            n_try = k
            break
    if n_try == 0:
        return [], 0

    # 3. Verify each draft token against the base model
    accepted = []
    for k in range(n_try):
        tok = draft_tokens[0, k].item()

        # Run base model on context + accepted so far
        extended = torch.cat([context, torch.tensor([accepted], device=device).unsqueeze(0)
                             if accepted else context[:, :0]], dim=1)
        base_out = model.base_model(extended)
        base_logits = base_out.logits[0, -1, :]  # (V,)

        # Draft distribution
        draft_logits_k = final_logits[0, k, :]
        draft_prob = F.softmax(draft_logits_k.float(), dim=-1)[tok].item()
        base_prob = F.softmax(base_logits.float(), dim=-1)[tok].item()

        # Acceptance test (speculative decoding)
        r = min(1.0, base_prob / max(draft_prob, 1e-10))
        if torch.rand(1).item() < r:
            accepted.append(tok)
        else:
            # Rejection: sample from residual distribution
            # We skip resampling for simplicity (just accept fewer tokens)
            break

    return accepted, len(accepted)


@torch.no_grad()
def eval_speed(model: DSparkModel, context: torch.LongTensor,
               max_new: int = 128, num_drafts: int = 5):
    """Compare tokens/sec: baseline vs DSpark speculative decoding."""
    device = context.device
    B = context.shape[0]

    # ── Baseline: standard autoregressive ────────────────────────────────────
    model.base_model.eval()
    torch.mps.empty_cache()

    input_ids = context.clone()
    t0 = time.perf_counter()
    for _ in range(max_new):
        out = model.base_model(input_ids)
        next_token = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    baseline_time = time.perf_counter() - t0

    # ── DSpark speculative ──────────────────────────────────────────────────
    model.eval()
    torch.mps.empty_cache()

    input_ids = context.clone()
    total_generated = 0
    t0 = time.perf_counter()
    while total_generated < max_new:
        n_remain = max_new - total_generated
        n_draft = min(num_drafts, n_remain)

        draft_tokens, accept_probs, final_logits = model.draft_predict(input_ids)
        accept_probs = accept_probs[0]

        # Determine how many to try
        n_try = n_draft
        for k in range(n_draft):
            if accept_probs[k].item() < 0.05:
                n_try = k
                break
        if n_try == 0:
            # Fall back to single base-model step
            out = model.base_model(input_ids)
            tok = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, tok], dim=-1)
            total_generated += 1
            continue

        # Verify
        accepted = []
        for k in range(n_try):
            tok = draft_tokens[0, k].item()
            ext = torch.cat([input_ids, torch.tensor([accepted], device=device).unsqueeze(0)
                            if accepted else input_ids[:, :0]], dim=1)
            base_out = model.base_model(ext)
            bl = base_out.logits[0, -1, :]
            dp = F.softmax(final_logits[0, k, :].float(), dim=-1)[tok].item()
            bp = F.softmax(bl.float(), dim=-1)[tok].item()
            r = min(1.0, bp / max(dp, 1e-10))
            if torch.rand(1).item() < r:
                accepted.append(tok)
            else:
                break

        if accepted:
            toks = torch.tensor([accepted], device=device)
            input_ids = torch.cat([input_ids, toks], dim=-1)
            total_generated += len(accepted)
        else:
            # Rejected at first position → fallback
            out = model.base_model(input_ids)
            tok = out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, tok], dim=-1)
            total_generated += 1

    spec_time = time.perf_counter() - t0
    n_draft_rounds = max(1, max_new / num_drafts)  # approximate

    print(f"\n── Speed ──")
    print(f"  Baseline:     {max_new / baseline_time:.1f} tok/s  ({baseline_time*1000:.0f}ms)")
    print(f"  DSpark spec:  {max_new / spec_time:.1f} tok/s  ({spec_time*1000:.0f}ms)")
    print(f"  Speedup:      {baseline_time / spec_time:.2f}×")
    return baseline_time, spec_time


# ═══════════════════════════════════════════════════════════════ main


def main():
    cfg = EvalConfig()
    print("Loading model …")
    model = DSparkModel(cfg.model_path, num_drafts=cfg.num_drafts)
    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="mps")
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  loaded checkpoint: {cfg.checkpoint}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.print_summary()

    print("\nLoading data …")
    tokenizer = _ensure_tokenizer(cfg.model_path)
    loader = _load_dataloader(tokenizer, cfg)

    print("\n── Perplexity ──")
    eval_perplexity(model, loader, cfg)

    print("\n── Acceptance Rate ──")
    eval_acceptance(model, loader, cfg)

    # Get a single longer context for speed test
    batch = next(iter(loader))
    ctx = batch["input_ids"][:1].to(device)
    eval_speed(model, ctx, max_new=cfg.max_new_tokens, num_drafts=cfg.num_drafts)


if __name__ == "__main__":
    main()
