#!/usr/bin/env python3
"""
Train: Joint CNN-token + kmer-token Transformer regressor using WebDataset shards.

Two input modes:

A) Directory mode (per-sample, backwards compatible):
   --wds_dir <DIR>
   Expects:
     <DIR>/train-*.tar
     <DIR>/val-*.tar
   Writes checkpoint to:
     <DIR>/joint_model_wds.pt   (default behavior)

B) Shard-list mode (global training):
   --train_shards train_shards.txt --val_shards val_shards.txt
   Each file must contain one shard path per line.
   Writes checkpoint to:
     ./joint_model_wds.pt       (unless --out_ckpt is set)

NEW:
  - Saves a "best checkpoint" by validation MSE if --out_ckpt_best is provided
    (or defaults alongside out_ckpt).
"""

from __future__ import annotations

import argparse
import io
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds


# -------------------------
# Model
# -------------------------

class SignalToTokensCNN(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(2),  # 400 -> 200
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),  # 200 -> 100
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),  # 100 -> 50
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = signal.unsqueeze(1)
        x = self.conv(x).transpose(1, 2)
        return self.norm(x)


class JointSignalKmerTransformer(nn.Module):
    def __init__(
        self,
        k: int = 9,
        vocab_size: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 4,
        dropout: float = 0.1,
        max_sig_tokens: int = 128,
    ):
        super().__init__()
        self.k = k
        self.sig_cnn = SignalToTokensCNN(d_model=d_model)
        self.kmer_emb = nn.Embedding(vocab_size, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Embedding(1 + max_sig_tokens + k, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, signal: torch.Tensor, kmer_ids: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        b = signal.shape[0]
        sig_tok = self.sig_cnn(signal)
        Lsig = sig_tok.shape[1]

        kmer_tok = self.kmer_emb(kmer_ids)
        x = torch.cat([self.cls.expand(b, -1, -1), sig_tok, kmer_tok], dim=1)

        pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
        x = self.enc(x + self.pos(pos_ids))

        cls = x[:, 0, :]
        kmer_center = x[:, 1 + Lsig + (self.k // 2), :]
        mean = x.mean(dim=1)

        fused = torch.cat([cls, kmer_center, mean], dim=-1)
        logit = self.head(fused).squeeze(-1)
        if return_logits:
            return logit
        return torch.sigmoid(logit)


def build_model() -> nn.Module:
    """Construct the model architecture for eval/checkpoint loading."""
    return JointSignalKmerTransformer(k=9)


# -------------------------
# WebDataset decoding
# -------------------------

def _torch_load_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu")


def decode_sample(sample: dict):
    sig = _torch_load_bytes(sample["signal.pth"])
    km = _torch_load_bytes(sample["kmer.pth"])
    y = _torch_load_bytes(sample["y.pth"])
    return sig, km, y


def make_loader(shards: List[str], batch_size: int, shuffle: bool, num_workers: int):
    # Clamp workers so we never have more workers than shards
    nw = int(num_workers)
    if nw > 0:
        nw = min(nw, max(1, len(shards)))

    ds = wds.WebDataset(
        shards,
        shardshuffle=0 if not shuffle else 100,
        empty_check=False,  # don't crash if a worker sees no samples
    )
    ds = ds.map(decode_sample)
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batched(batch_size, partial=False)
    return wds.WebLoader(ds, num_workers=nw, batch_size=None)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss()
    losses = []
    for sig, km, y in loader:
        sig = sig.to(device).float()
        km = km.to(device).long()
        y = y.to(device).float().view(-1)
        pred = model(sig, km)
        loss = mse(pred, y)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


# -------------------------
# Shard discovery helpers
# -------------------------

def _read_shard_list(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Shard list file not found: {path}")
    out: List[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _discover_shards_from_wds_dir(wds_dir: str) -> Tuple[List[str], List[str]]:
    train_shards = sorted(glob(str(Path(wds_dir) / "train-*.tar")))
    val_shards = sorted(glob(str(Path(wds_dir) / "val-*.tar")))
    return train_shards, val_shards


def main() -> None:
    ap = argparse.ArgumentParser()

    # Input mode A (back-compat)
    ap.add_argument("--wds_dir", default=None, help="Directory containing train-*.tar and val-*.tar")

    # Input mode B (global)
    ap.add_argument("--train_shards", default=None, help="Text file listing train shard paths (one per line)")
    ap.add_argument("--val_shards", default=None, help="Text file listing val shard paths (one per line)")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)

    # Output
    ap.add_argument(
        "--out_ckpt",
        default=None,
        help="Where to write FINAL checkpoint. If omitted: wds_dir/joint_model_wds.pt (dir mode) or ./joint_model_wds.pt (list mode).",
    )
    ap.add_argument(
        "--out_ckpt_best",
        default=None,
        help="Where to write BEST-by-val checkpoint. If omitted, defaults to <out_ckpt>.best.pt",
    )

    args = ap.parse_args()

    # Decide mode
    using_dir = args.wds_dir is not None
    using_lists = (args.train_shards is not None) or (args.val_shards is not None)

    if using_dir and using_lists:
        raise SystemExit("Provide EITHER --wds_dir OR (--train_shards AND --val_shards), not both.")

    if using_dir:
        train_shards, val_shards = _discover_shards_from_wds_dir(args.wds_dir)
        if not train_shards or not val_shards:
            raise SystemExit(f"Could not find train/val shards under: {args.wds_dir}")

        out_ckpt = args.out_ckpt or str(Path(args.wds_dir) / "joint_model_wds.pt")

    else:
        if not args.train_shards or not args.val_shards:
            raise SystemExit("Shard-list mode requires --train_shards and --val_shards.")
        train_shards = _read_shard_list(args.train_shards)
        val_shards = _read_shard_list(args.val_shards)
        if not train_shards:
            raise SystemExit(f"No train shards found in {args.train_shards}")
        if not val_shards:
            raise SystemExit(f"No val shards found in {args.val_shards}")

        out_ckpt = args.out_ckpt or "joint_model_wds.pt"

    out_ckpt_best = args.out_ckpt_best or (out_ckpt + ".best.pt")

    train_loader = make_loader(train_shards, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_shards, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    print("device:", device)
    print("train shards:", len(train_shards), "val shards:", len(val_shards))
    print("out_ckpt:", out_ckpt)
    print("out_ckpt_best:", out_ckpt_best)

    best_val = float("inf")
    best_epoch: Optional[int] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for sig, km, y in train_loader:
            sig = sig.to(device).float()
            km = km.to(device).long()
            y = y.to(device).float().view(-1)

            pred = model(sig, km)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(losses)) if losses else float("nan")
        va = eval_epoch(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_mse={tr:.6f} | val_mse={va:.6f}")

        # Save best-by-val checkpoint
        if np.isfinite(va) and va < best_val:
            best_val = float(va)
            best_epoch = int(epoch)
            Path(out_ckpt_best).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_ckpt_best)
            print(f"saved_best: {out_ckpt_best} (epoch={best_epoch:02d} val_mse={best_val:.6f})")

    # Save final checkpoint
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print("saved:", out_ckpt)

    if best_epoch is not None:
        print(f"best_epoch: {best_epoch:02d} best_val_mse={best_val:.6f} best_ckpt={out_ckpt_best}")


if __name__ == "__main__":
    main()