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

WIKITEXT_CACHE = os.path.expanduser("~/.cache/dspark/wikitext-103-raw")

# Local data dir (project-relative) checked first
_LOCAL_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _ensure_wikitext() -> tuple[str, str]:
    """Locate wikitext-103-raw parquet files, downloading if needed.

    Checks in order:
      1. ``data/`` (project-relative) for raw shard files (merges to cache)
      2. ``~/.cache/dspark/wikitext-103-raw/`` for merged parquet files
      3. Downloads from HF Hub (via HF_ENDPOINT) if neither found

    Returns (train_path, valid_path) — paths to merged parquet files.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_path = os.path.join(WIKITEXT_CACHE, "train.parquet")
    valid_path = os.path.join(WIKITEXT_CACHE, "validation.parquet")

    # ── 1. Check project-local data/ ────────────────────────────────────────
    local_shards = sorted([
        os.path.join(_LOCAL_DATA, f)
        for f in os.listdir(_LOCAL_DATA)
        if f.startswith("train-") and f.endswith(".parquet")
    ]) if os.path.isdir(_LOCAL_DATA) else []
    local_valid = (
        os.path.join(_LOCAL_DATA, "validation-00000-of-00001.parquet")
        if os.path.isdir(_LOCAL_DATA) and os.path.exists(
            os.path.join(_LOCAL_DATA, "validation-00000-of-00001.parquet"))
        else None
    )
    if local_shards and local_valid:
        print(f"  found {len(local_shards)} train shard(s) + validation in data/")
        os.makedirs(WIKITEXT_CACHE, exist_ok=True)
        tables = [pq.read_table(s) for s in local_shards]
        pq.write_table(pa.concat_tables(tables), train_path)
        import shutil
        shutil.copy2(local_valid, valid_path)

    # ── 2. Download if still missing ────────────────────────────────────────
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        os.makedirs(WIKITEXT_CACHE, exist_ok=True)

        # Try HF mirror first (faster in CN); override with HF_ENDPOINT env var
        endpoint = os.environ.get("HF_ENDPOINT",
                                  "https://hf-mirror.com")
        tmpdir = os.path.join(WIKITEXT_CACHE, ".dl")
        os.makedirs(tmpdir, exist_ok=True)
        shards = [
            ("wikitext-103-raw-v1/train-00000-of-00002.parquet",
             os.path.join(tmpdir, "train-00.parquet")),
            ("wikitext-103-raw-v1/train-00001-of-00002.parquet",
             os.path.join(tmpdir, "train-01.parquet")),
            ("wikitext-103-raw-v1/validation-00000-of-00001.parquet",
             os.path.join(tmpdir, "valid.parquet")),
        ]
        import subprocess, shutil
        for src, dst in shards:
            if os.path.exists(dst):
                print(f"  {src} already cached")
                continue
            url = f"{endpoint}/datasets/wikitext/resolve/main/{src}"
            print(f"  downloading {src} …")
            subprocess.run(
                ["wget", "-c", "-t", "10", "--timeout=60", "-O", dst, url],
                check=True, capture_output=True)
            sz = os.path.getsize(dst)
            print(f"    -> {sz / 1e6:.1f} MB")

        # Merge train shards
        import pyarrow.parquet as pq
        t1 = pq.read_table(os.path.join(tmpdir, "train-00.parquet"))
        t2 = pq.read_table(os.path.join(tmpdir, "train-01.parquet"))
        pq.write_table(pa.concat_tables([t1, t2]), train_path)
        os.rename(os.path.join(tmpdir, "valid.parquet"), valid_path)
        shutil.rmtree(tmpdir)

    return train_path, valid_path


class ParquetTextStream(IterableDataset):
    """Stream wikitext chunks lazily from a parquet file — no preloading."""

    def __init__(self, path, tokenizer, context_len: int,
                 num_drafts: int):
        self.path = path
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.full_len = context_len + num_drafts
        self._buffer: list[int] = []

    def __iter__(self):
        import pyarrow.parquet as pq
        acc = self._buffer.copy()
        pf = pq.ParquetFile(self.path)
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text is None:
                    continue
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                acc.extend(ids)
                while len(acc) >= self.full_len:
                    chunk = acc[:self.full_len]
                    yield {"input_ids": chunk[:self.context_len],
                           "labels": chunk}
                    acc = acc[self.context_len:]
        self._buffer = acc


def get_dataloaders(tokenizer, cfg: TrainConfig):
    """Return (train_loader, valid_loader) from wikitext-103-raw."""
    print("Locating wikitext-103-raw …")
    train_path, valid_path = _ensure_wikitext()
    train_ds = ParquetTextStream(train_path, tokenizer,
                                  cfg.context_len, cfg.num_drafts)
    valid_ds = ParquetTextStream(valid_path, tokenizer,
                                  cfg.context_len, cfg.num_drafts)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, collate_fn=_collate),
        DataLoader(valid_ds, batch_size=cfg.batch_size, collate_fn=_collate),
    )


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
        from tqdm import tqdm
        pbar = tqdm(total=self.cfg.max_steps, desc="train", unit="step",
                    bar_format="{l_bar}{bar:10}{r_bar}")

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

                # -- progress bar ----------------------------------------------
                elapsed = time.time() - t0
                tok_per_sec = ((self.step + 1) * self.cfg.batch_size *
                               (self.cfg.context_len + self.cfg.num_drafts) /
                               max(elapsed, 1))
                pbar.set_postfix(
                    loss=f"{stats['loss']:.3f}",
                    ce=f"{stats['ce_loss']:.2f}",
                    conf=f"{stats['conf_loss']:.3f}",
                    accept=f"{stats['accept_rate']:.2f}",
                    ppl=f"{stats['ppl']:.0f}",
                    tok_s=f"{tok_per_sec:.0f}",
                )
                pbar.update(1)

                # -- validation -----------------------------------------------
                if self.step > 0 and self.step % self.cfg.valid_every == 0:
                    self._validate(valid_loader)

                # -- checkpoint -----------------------------------------------
                if self.step > 0 and self.step % self.cfg.save_every == 0:
                    self._save(os.path.join(self.cfg.output_dir,
                                            f"dspark_{self.step}.pt"))

                self.step += 1

        pbar.close()
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
