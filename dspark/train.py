"""Train the DSpark draft heads, markov head, and confidence head."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from .model import DSparkModel


@dataclass
class TrainConfig:
    # Paths
    model_path: str = "/Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B/"
    data_cache: str = "/Volumes/Samsung/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
    output_dir: str = "./checkpoints"

    # Architecture
    num_drafts: int = 5
    context_len: int = 512        # T — base model context window

    # Training
    batch_size: int = 4
    grad_acc_steps: int = 8       # effective batch = 4 × 8 = 32
    max_steps: int = 5000
    warmup_steps: int = 200
    lr: float = 5e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    save_every: int = 1000
    log_every: int = 25
    valid_every: int = 500

    # Loss weights
    ce_weight: float = 1.0        # cross-entropy on draft+markov logits
    confidence_weight: float = 1.0  # BCE on acceptance probabilities

    # Data
    max_valid_batches: int = 50

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# ═══════════════════════════════════════════════════════════════ data


class WikiTextStream(IterableDataset):
    """Stream wikitext chunks from a pre-tokenized iterable dataset.

    Yields dicts:
      input_ids: (T,)  — context tokens
      labels:    (T+N,) — full sequence (context + N draft targets)
    """

    def __init__(self, ds_iter, tokenizer, context_len: int,
                 num_drafts: int):
        self.ds_iter = ds_iter
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.full_len = context_len + num_drafts  # T+N
        self._buffer: list[int] = []

    def __iter__(self):
        acc = self._buffer.copy()
        for example in self.ds_iter:
            text = example["text"]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            acc.extend(ids)
            while len(acc) >= self.full_len:
                chunk = acc[:self.full_len]
                yield {"input_ids": chunk[:self.context_len],
                       "labels": chunk}
                acc = acc[self.context_len:]
        self._buffer = acc


def get_dataloaders(tokenizer, cfg: TrainConfig):
    """Return (train_loader, valid_loader)."""
    from datasets import Dataset, load_from_disk

    # Load from local cache path
    train_path = os.path.join(cfg.data_cache, "wikitext-train.arrow")
    valid_path = os.path.join(cfg.data_cache, "wikitext-validation.arrow")
    train_ds = Dataset.from_file(train_path)
    valid_ds = Dataset.from_file(valid_path)

    all_text = [ex["text"] for ex in train_ds]
    valid_text = [ex["text"] for ex in valid_ds]

    def _make_loader(texts, shuffle: bool):
        class _ListIter:
            def __init__(self, lst):
                self.lst = lst
                self.idx = 0
            def __iter__(self):
                self.idx = 0
                return self
            def __next__(self):
                if self.idx >= len(self.lst):
                    raise StopIteration
                val = {"text": self.lst[self.idx]}
                self.idx += 1
                return val

        ds = WikiTextStream(
            _ListIter(texts), tokenizer,
            cfg.context_len, cfg.num_drafts,
        )
        return DataLoader(ds, batch_size=cfg.batch_size,
                          collate_fn=_collate)

    return _make_loader(all_text, shuffle=False), _make_loader(valid_text, shuffle=False)


def _collate(batch):
    """Pad/collate a list of dict items into a single batch."""
    ii = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long)
                      for b in batch])
    lb = torch.stack([torch.tensor(b["labels"], dtype=torch.long)
                      for b in batch])
    return {"input_ids": ii, "labels": lb}


# ═══════════════════════════════════════════════════════════════ training


class Trainer:
    def __init__(self, model: DSparkModel, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.device = next(p for p in model.trainable_parameters()).device

        # Optimizer — only the trainable params
        self.opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt,
            max_lr=cfg.lr,
            total_steps=cfg.max_steps,
            pct_start=cfg.warmup_steps / cfg.max_steps,
        )

        self.step = 0
        self.best_valid_loss = float("inf")
        self._log_keys = ["loss", "ce_loss", "conf_loss", "accept_rate", "ppl"]

    def _save(self, path: str):
        torch.save({
            "step": self.step,
            "model_state": {k: v for k, v in self.model.state_dict().items()
                           if k.startswith(("draft_heads.", "markov_head.",
                                           "confidence_head."))},
            "opt_state": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, path)
        print(f"  → checkpoint saved at step {self.step}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.opt.load_state_dict(ckpt["opt_state"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        print(f"  ← checkpoint loaded (step {self.step})")

    # ── single batch ─────────────────────────────────────────────────────────

    def train_step(self, batch) -> dict[str, float]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        out = self.model(input_ids, labels=labels)

        # CE loss: draft + markov prediction vs ground-truth next tokens
        # logits: (B, N, V), targets: labels at positions T..T+N-1 (B, N)
        N = self.cfg.num_drafts
        T = input_ids.shape[1]
        targets = labels[:, T:T + N]  # (B, N)
        ce_loss = F.cross_entropy(
            out["logits"].reshape(-1, out["logits"].size(-1)),
            targets.reshape(-1),
        )

        # BCE loss: confidence head vs oracle acceptance probability
        conf_loss = F.binary_cross_entropy(
            out["accept_probs"].float(),
            out["accept_targets"].float(),
        )

        loss = self.cfg.ce_weight * ce_loss + self.cfg.confidence_weight * conf_loss

        loss.backward()

        # Stats
        with torch.no_grad():
            acc = out["accept_targets"].float().mean().item()
            ppl = math.exp(min(ce_loss.item(), 20))

        return dict(loss=loss.item(), ce_loss=ce_loss.item(),
                    conf_loss=conf_loss.item(), accept_rate=acc, ppl=ppl)

    @torch.no_grad()
    def valid_step(self, batch) -> dict[str, float]:
        return self.train_step(batch)  # same computation, no gradients

    # ── full epochs ──────────────────────────────────────────────────────────

    def train(self, train_loader, valid_loader):
        self.model.train()
        t0 = time.time()

        while self.step < self.cfg.max_steps:
            for batch in train_loader:
                if self.step >= self.cfg.max_steps:
                    break

                # -- forward + backward ---------------------------------------
                stats = self.train_step(batch)

                # Gradient accumulation
                if (self.step + 1) % self.cfg.grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.trainable_parameters(),
                        self.cfg.max_grad_norm,
                    )
                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad()

                # -- logging --------------------------------------------------
                if self.step % self.cfg.log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = (self.step * self.cfg.batch_size *
                                   (self.cfg.context_len + self.cfg.num_drafts) / elapsed)
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"step {self.step:>5d} | loss {stats['loss']:.4f} "
                        f"ce {stats['ce_loss']:.4f} conf {stats['conf_loss']:.4f} "
                        f"accept {stats['accept_rate']:.3f} ppl {stats['ppl']:.1f} "
                        f"lr {lr:.2e} t/s {tok_per_sec:.0f}"
                    )

                # -- validation -----------------------------------------------
                if self.step > 0 and self.step % self.cfg.valid_every == 0:
                    self._validate(valid_loader)

                # -- checkpoint -----------------------------------------------
                if self.step > 0 and self.step % self.cfg.save_every == 0:
                    self._save(os.path.join(self.cfg.output_dir,
                                            f"dspark_{self.step}.pt"))

                self.step += 1

        # Final save
        self._save(os.path.join(self.cfg.output_dir, "dspark_final.pt"))

    def _validate(self, valid_loader):
        self.model.eval()
        losses = []
        for i, batch in enumerate(valid_loader):
            if i >= self.cfg.max_valid_batches:
                break
            losses.append(self.valid_step(batch))
        avg = {k: sum(d[k] for d in losses) / len(losses)
               for k in self._log_keys}
        print(
            f"  ── valid ──  loss {avg['loss']:.4f}  ce {avg['ce_loss']:.4f}  "
            f"conf {avg['conf_loss']:.4f}  accept {avg['accept_rate']:.3f}  "
            f"ppl {avg['ppl']:.1f}"
        )
        if avg["loss"] < self.best_valid_loss:
            self.best_valid_loss = avg["loss"]
            self._save(os.path.join(self.cfg.output_dir, "dspark_best.pt"))
        self.model.train()


# ═══════════════════════════════════════════════════════════════ main


def main():
    cfg = TrainConfig()

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    print("Building model …")
    model = DSparkModel(cfg.model_path, num_drafts=cfg.num_drafts)
    model.print_summary()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    print("Preparing data …")
    train_loader, valid_loader = get_dataloaders(tokenizer, cfg)

    print("Training …")
    trainer = Trainer(model, cfg)
    trainer.train(train_loader, valid_loader)
    print("Done.")


if __name__ == "__main__":
    main()
