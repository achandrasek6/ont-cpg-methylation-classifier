"""
Train a joint CNN-token + kmer-token Transformer regressor on labeled CpG chunks.

Expected Parquet columns:
  - signal: array-like length 400 (often numpy.ndarray)
  - kmer_ids: array-like length 9 (often numpy.ndarray), tokens A=0,C=1,G=2,T=3,N=4
  - meth_frac: float in [0,1]

Example input file (your 1-fast5 prototype):
  outputs/training/one/<RUN>/train_labeled_k9.parquet

Run:
  python train_joint_cnn_kmer_transformer_regressor.py \
    --data outputs/training/one/20210510_1127_X4_FAQ32498_b90eaed8/train_labeled_k9.parquet \
    --epochs 5 \
    --batch_size 64

Notes:
  - This is a small prototype trainer (no fancy logging, no distributed).
  - For tiny datasets, we do a random train/val split.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Model
# -------------------------

class SignalToTokensCNN(nn.Module):
    """Convert a 1D signal window into a sequence of tokens."""
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
        # signal: (B,400) -> (B,1,400)
        x = signal.unsqueeze(1)
        x = self.conv(x)            # (B, d_model, Lsig) Lsig~50
        x = x.transpose(1, 2)       # (B, Lsig, d_model)
        x = self.norm(x)
        return x


class JointSignalKmerTransformer(nn.Module):
    """Joint Transformer over CNN-derived signal tokens + kmer tokens."""
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
        self.d_model = d_model

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

    def forward(self, signal: torch.Tensor, kmer_ids: torch.Tensor) -> torch.Tensor:
        b = signal.shape[0]
        sig_tok = self.sig_cnn(signal)      # (B, Lsig, d_model)
        Lsig = sig_tok.shape[1]

        need_pos = 1 + Lsig + self.k
        if need_pos > self.pos.num_embeddings:
            raise ValueError(
                f"Position embedding too small: need {need_pos}, have {self.pos.num_embeddings}. "
                "Increase max_sig_tokens."
            )

        kmer_tok = self.kmer_emb(kmer_ids)  # (B, k, d_model)

        cls_tok = self.cls.expand(b, -1, -1)
        x = torch.cat([cls_tok, sig_tok, kmer_tok], dim=1)  # (B, T, d_model)

        pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(b, -1)
        x = x + self.pos(pos_ids)

        x = self.enc(x)  # (B, T, d_model)

        cls = x[:, 0, :]
        kmer_center = x[:, 1 + Lsig + (self.k // 2), :]
        mean = x.mean(dim=1)

        fused = torch.cat([cls, kmer_center, mean], dim=-1)
        logit = self.head(fused).squeeze(-1)
        y_hat = torch.sigmoid(logit)  # (B,) in [0,1]
        return y_hat


# -------------------------
# Data
# -------------------------

class CpGChunkDataset(Dataset):
    """Dataset wrapping a parquet with signal/kmer_ids/meth_frac."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        r = self.df.iloc[idx]
        sig = np.asarray(r["signal"], dtype=np.float32)      # (400,)
        km = np.asarray(r["kmer_ids"], dtype=np.int64)       # (9,)
        y = float(r["meth_frac"])
        return sig, km, y


def collate(batch: List[Tuple[np.ndarray, np.ndarray, float]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sig = torch.tensor(np.stack([b[0] for b in batch], axis=0), dtype=torch.float32)
    km = torch.tensor(np.stack([b[1] for b in batch], axis=0), dtype=torch.int64)
    y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return sig, km, y


# -------------------------
# Train
# -------------------------

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_split(n: int, val_frac: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(math.floor(n * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Split(train_idx=train_idx, val_idx=val_idx)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    mse = nn.MSELoss(reduction="mean")
    for sig, km, y in loader:
        sig, km, y = sig.to(device), km.to(device), y.to(device)
        pred = model(sig, km)
        loss = mse(pred, y)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Parquet path with signal+kmer_ids+meth_frac")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    needed = {"signal", "kmer_ids", "meth_frac"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in parquet: {sorted(missing)}")

    # Basic sanity filters (accept numpy arrays)
    df = df.dropna(subset=["meth_frac"]).copy()
    df = df[df["signal"].map(lambda x: hasattr(x, "__len__") and len(x) == 400)]
    df = df[df["kmer_ids"].map(lambda x: hasattr(x, "__len__") and len(x) == 9)]
    df = df.reset_index(drop=True)

    if len(df) < 10:
        raise SystemExit(f"Too few rows after filtering: {len(df)}")

    split = make_split(len(df), args.val_frac, args.seed)
    train_ds = CpGChunkDataset(df.iloc[split.train_idx])
    val_ds = CpGChunkDataset(df.iloc[split.val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate, drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointSignalKmerTransformer(k=9).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    print(f"Loaded rows: {len(df)} | train: {len(train_ds)} | val: {len(val_ds)} | device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for sig, km, y in train_loader:
            sig, km, y = sig.to(device), km.to(device), y.to(device)
            pred = model(sig, km)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = eval_epoch(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_mse={tr:.6f} | val_mse={va:.6f}")

    out = Path(args.data).with_suffix("").as_posix() + "_joint_model.pt"
    torch.save(model.state_dict(), out)
    print("saved:", out)


if __name__ == "__main__":
    train()