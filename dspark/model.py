"""DSpark: Parallel draft model with Markov head and confidence head.

Architecture (inference flow):
  1. Frozen base model produces last-token hidden state  (B, H)
  2. N parallel DraftHeads predict lookahead tokens       (B, N, V_logits)
  3. Greedy-decode step 2 tokens → MarkovHead computes
     transition bias from adjacent (prev, cur) pairs      (B, N, V_bias)
  4. Final logits = draft_logits + markov_bias            (B, N, V)
  5. ConfidenceHead predicts P(accept_k) per position     (B, N)

Training uses ground-truth labels for the Markov bias pairs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ════════════════════════════════════════════════════════ components


class NoiseSchedule:
    """Cosine noise schedule for multinomial diffusion.

    α(t) = cos²( (t/T + s) / (1 + s) · π/2 )  ︱  α(0) ≡ 1
    """
    def __init__(self, T: int = 1000, s: float = 0.008):
        self.T = T
        self.s = s
        self._f0 = math.cos((s / (1.0 + s)) * (math.pi / 2.0)) ** 2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Return α(t) for integer timesteps t ∈ [0, T].

        t can be a scalar, 1D tensor, or any shape — output matches.
        """
        x = (t.float() / self.T + self.s) / (1.0 + self.s)
        return torch.cos(x * (math.pi / 2.0)) ** 2 / self._f0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep encoding (same as Transformer position encoding)."""
    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        half = hidden_size // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(0, half, dtype=torch.float) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        """t: (B,) timesteps → (B, hidden_size) encoding."""
        emb = t.unsqueeze(-1).float() * self.freqs.unsqueeze(0)  # (B, half)
        enc = torch.cat([emb.sin(), emb.cos()], dim=-1)           # (B, half*2)
        if self.hidden_size % 2 != 0:
            enc = torch.cat([enc, torch.zeros_like(enc[:, :1])], dim=-1)
        return enc


class TimeProjection(nn.Module):
    """Map sinusoidal timestep encoding → (B, H)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embed = SinusoidalTimeEmbedding(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        return self.mlp(self.embed(t))  # (B, H)


class DiffDraftHead(nn.Module):
    """Per-position denoiser MLP for discrete token diffusion.

    Given noisy token embeddings + conditioning h_T + timestep encoding,
    predicts the clean hidden state at each draft position.
    """
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * expansion, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * expansion, hidden_size, bias=False),
        )

    def forward(self, noisy_embed: torch.Tensor, h_T: torch.Tensor,
                t_enc: torch.Tensor) -> torch.Tensor:
        """(B,N,H), (B,H), (B,H) → (B,N,H) clean hidden states."""
        x = noisy_embed + h_T.unsqueeze(1) + t_enc.unsqueeze(1)
        return self.net(x)


class MarkovHead(nn.Module):
    """Bilinear transition bias from (prev_embed, cur_embed) pairs.

    Learns a context-dependent bias that is *added* to draft logits,
    capturing bigram-like transition patterns in embedding space.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, bias=False),
            nn.SiLU(),
        )

    def forward(self, prev_embeds: torch.Tensor,
                cur_embeds: torch.Tensor) -> torch.Tensor:
        """(B, N, H), (B, N, H) → (B, N, H)  markov bias."""
        return self.net(torch.cat([prev_embeds, cur_embeds], dim=-1))


class ConfidenceHead(nn.Module):
    """Predicts P(accept_k) ∈ [0, 1] for each draft position."""

    def __init__(self, hidden_size: int, num_drafts: int, inner: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.SiLU(),
            nn.Linear(inner, num_drafts),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, H) → (B, N)  acceptance probabilities."""
        return self.net(hidden)


# ════════════════════════════════════════════════════════ main model


class DSparkModel(nn.Module):
    """DSpark speculative-decoding draft model for Qwen3.5 / similar LLMs."""

    def __init__(self, base_model_name: str, num_drafts: int = 5):
        super().__init__()
        self.num_drafts = num_drafts
        self.num_diff_steps = 8  # DDIM steps at inference

        # -- frozen base model -------------------------------------------------
        dtype = torch.bfloat16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            trust_remote_code=True,
        )
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

        # Resolve config (Qwen3.5 nests under text_config)
        cfg = self.base_model.config
        tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
        self.hidden_size: int = tc.hidden_size
        self.vocab_size: int = tc.vocab_size
        self._pad_id = getattr(tc, "pad_token_id", None) or 0

        # Shared vocab projection — just a reference to the frozen lm_head
        self._lm_head = self.base_model.lm_head  # Linear(H, V), frozen

        # Embedding lookup (shared, frozen)
        self.embed = self.base_model.get_input_embeddings()

        # -- trainable heads (same dtype as base model to avoid MPS mixed-dtype) -
        head_dtype = dtype
        self.noise_schedule = NoiseSchedule(T=1000, s=0.008)
        self.time_proj = TimeProjection(self.hidden_size).to(head_dtype)
        self.diff_draft_head = DiffDraftHead(self.hidden_size, expansion=4).to(head_dtype)
        self.markov_head = MarkovHead(self.hidden_size).to(head_dtype)
        self.confidence_head = ConfidenceHead(self.hidden_size, num_drafts).to(head_dtype)

    # ── public helpers ────────────────────────────────────────────────────────

    def _vocab_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project (…, H) → (…, V) via the frozen lm_head."""
        return self._lm_head(x.to(self._lm_head.weight.dtype))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters that should receive gradients."""
        params = []
        params.extend(self.diff_draft_head.parameters())
        params.extend(self.time_proj.parameters())
        params.extend(self.markov_head.parameters())
        params.extend(self.confidence_head.parameters())
        return params

    # ── forward / training ────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train step.

        When *labels* is provided, the base model runs on the **full** sequence
        ``labels`` (T+N tokens) so it can produce oracle logits at every draft
        position for the confidence-head target.  The last hidden state is taken
        from position ``T-1`` (the last context token) so draft heads never see
        future tokens through the causal mask.

        Args:
            input_ids: (B, T) — context tokens (marks where draft starts).
            labels: (B, T+N) — context + N future tokens.  When given the
                    markov head uses ground-truth pairs and the dict includes
                    ``accept_targets``.
        Returns:
            dict with keys:
              draft_hidden   (B, N, H)
              logits         (B, N, V)  — draft_logits + markov_bias
              accept_probs   (B, N)
              accept_targets (B, N)     — only when *labels* is given
        """
        B, T = input_ids.shape
        N = self.num_drafts
        has_labels = labels is not None

        # 1. Base model — run on the full sequence when labels are available
        with torch.no_grad():
            if has_labels:
                # labels: (B, T+N) = context + N draft tokens
                # Causal mask prevents position T-1 from seeing positions T…T+N-1,
                # so hidden_states[-1][:, T-1, :] is the pure-context representation.
                out = self.base_model(
                    labels,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = out.hidden_states[-1][:, T - 1, :]   # (B, H)
                # Base logits at positions T-1 … T+N-2 predict tokens T … T+N-1
                base_at_draft = out.logits[:, T - 1:T - 1 + N, :]  # (B, N, V)
            else:
                out = self.base_model(
                    input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = out.hidden_states[-1][:, -1, :]      # (B, H)
                base_at_draft = None

        # 2. Draft heads  (fully parallel)
        draft_hidden = self.draft_heads(last_hidden)  # (B, N, H)

        # 3. Markov bias  (training: ground-truth pairs)
        if has_labels:
            #   prev_ids = [x_{T-1}, x_T,    …, x_{T+N-2}]   (B, N)
            #   cur_ids  = [x_T,     x_{T+1}, …, x_{T+N-1}]   (B, N)
            prev_ids = torch.cat([input_ids[:, -1:],
                                  labels[:, T:T + N - 1]], dim=1)   # (B, N)
            cur_ids = labels[:, T:T + N]                              # (B, N)
            markov_bias_h = self.markov_head(self.embed(prev_ids),
                                             self.embed(cur_ids))     # (B, N, H)
        else:
            markov_bias_h = 0.0

        # 4. Final logits
        logits = self._vocab_proj(draft_hidden + markov_bias_h)  # (B, N, V)

        # 5. Confidence head
        accept_probs = self.confidence_head(last_hidden)  # (B, N)

        result = dict(draft_hidden=draft_hidden, logits=logits,
                      accept_probs=accept_probs)

        # 6. Acceptance targets  (expected P(accept) under speculative decoding)
        if has_labels and base_at_draft is not None:
            with torch.no_grad():
                dp = F.softmax(logits.float(), dim=-1)
                bp = F.softmax(base_at_draft.float(), dim=-1)
                # Expected acceptance = sum_v min(draft(v), base(v))
                result["accept_targets"] = torch.min(dp, bp).sum(dim=-1)  # (B, N)

        return result

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def draft_predict(self, input_ids: torch.LongTensor,
                      attention_mask: torch.Tensor | None = None,
                      temperature: float = 0.0,
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce N draft tokens, confidence scores, and full logits.

        Returns (draft_tokens, accept_probs, final_logits):
          draft_tokens  (B, N)  — argmax / sampled tokens
          accept_probs  (B, N)  — confidence head output
          final_logits  (B, N, V)
        """
        B = input_ids.shape[0]
        N = self.num_drafts

        # -- base model forward ------------------------------------------------
        out = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1][:, -1, :]  # (B, H)

        # -- step 1: parallel draft (no markov bias) ---------------------------
        draft_h = self.draft_heads(last_hidden)          # (B, N, H)
        draft_logits = self._vocab_proj(draft_h)          # (B, N, V)

        # Greedy decode full draft (needed for markov bias pairs)
        if temperature <= 0.0:
            draft_tokens = draft_logits.argmax(dim=-1)   # (B, N)
        else:
            draft_tokens = torch.multinomial(
                F.softmax(draft_logits.float() / temperature, dim=-1)
                .view(-1, self.vocab_size), 1).view(B, N)

        # -- step 2: markov bias from greedy token pairs -----------------------
        # Context: [x_{T-1}, x_T^, x_{T+1}^, …, x_{T+N-1}^]
        # The ^ denotes greedy-decoded draft tokens.
        all_tokens = torch.cat([input_ids[:, -1:], draft_tokens], dim=1)  # (B,N+1)
        prev_ids = all_tokens[:, :-1]    # (B, N)
        cur_ids = all_tokens[:, 1:]      # (B, N)
        markov_bias_h = self.markov_head(self.embed(prev_ids),
                                         self.embed(cur_ids))  # (B, N, H)

        # -- final logits ------------------------------------------------------
        final_logits = self._vocab_proj(draft_h + markov_bias_h)  # (B, N, V)

        # -- confidence --------------------------------------------------------
        accept_probs = self.confidence_head(last_hidden)  # (B, N)

        return draft_tokens, accept_probs, final_logits

    # ── utilities ─────────────────────────────────────────────────────────────

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = self.num_trainable()
        frozen = total - trainable
        print(f"DSparkModel  total:{total/1e6:.1f}M  "
              f"trainable:{trainable/1e6:.1f}M  "
              f"frozen:{frozen/1e6:.1f}M")
        print(f"  DiffDraftHead:  {self.hidden_size}→{self.hidden_size*4}→{self.hidden_size} (shared MLP)")
        print(f"  MarkovHead:  {self.hidden_size}×2→{self.hidden_size}→V")
        print(f"  Confidence:  {self.hidden_size}→128→{self.num_drafts} sigmoid")
