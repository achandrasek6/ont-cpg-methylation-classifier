"""
Smoke-test training: Joint CNN-token + kmer-token Transformer regressor using WebDataset shards.

Reads samples from:
  - train-*.tar and val-*.tar in a directory

Each sample must contain:
  - signal.pth (torch float32 [400])
  - kmer.pth   (torch int64 [9])
  - y.pth      (torch float32 scalar)

Run:
  python scripts/train_joint_cnn_kmer_transformer_wds_smoketest.py \
    --wds_dir outputs/wds_smoke/<RUN> \
    --epochs 10 --batch_size 128
"""

from __future__ import annotations

import argparse
import io
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds


# -------------------------
# Model (same as before)
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

    def forward(self, signal: torch.Tensor, kmer_ids: torch.Tensor) -> torch.Tensor:
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
        return torch.sigmoid(logit)


# -------------------------
# WebDataset decoding helpers
# -------------------------

def _torch_load_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu")


def decode_sample(sample: dict):
    """
    Convert raw bytes sample to (signal, kmer, y) torch tensors.
    WebDataset provides keys like 'signal.pth', 'kmer.pth', 'y.pth' as bytes.
    """
    sig = _torch_load_bytes(sample["signal.pth"])
    km = _torch_load_bytes(sample["kmer.pth"])
    y = _torch_load_bytes(sample["y.pth"])
    return sig, km, y


def make_loader(shards, batch_size: int, shuffle: bool, num_workers: int):
    ds = wds.WebDataset(shards, shardshuffle=0 if not shuffle else 100)  # avoid warning
    ds = ds.map(decode_sample)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batched(batch_size, partial=False)
    return wds.WebLoader(ds, num_workers=num_workers, batch_size=None)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    mse = nn.MSELoss()
    losses = []
    for sig, km, y in loader:
        sig = sig.to(device).float()          # (B,400)
        km = km.to(device).long()             # (B,9)
        y = y.to(device).float().view(-1)     # (B,)
        pred = model(sig, km)
        loss = mse(pred, y)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    train_shards = sorted(glob(str(Path(args.wds_dir) / "train-*.tar")))
    val_shards = sorted(glob(str(Path(args.wds_dir) / "val-*.tar")))
    if not train_shards or not val_shards:
        raise SystemExit(f"Could not find train/val shards under: {args.wds_dir}")

    train_loader = make_loader(train_shards, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_shards, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointSignalKmerTransformer(k=9).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.MSELoss()

    print("device:", device)
    print("train shards:", len(train_shards), "val shards:", len(val_shards))

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

    out = str(Path(args.wds_dir) / "joint_model_wds.pt")
    torch.save(model.state_dict(), out)
    print("saved:", out)


if __name__ == "__main__":
    main()
def build_model() -> nn.Module:
    """Construct the model architecture for eval/checkpoint loading."""
    return JointSignalKmerTransformer(k=9)
