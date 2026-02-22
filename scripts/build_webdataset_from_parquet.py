"""
Build WebDataset shards (.tar) from a labeled parquet produced by extract_coord_chunks_wgbs.py.

Input parquet must contain:
  - signal: array-like length 400 (numpy.ndarray)
  - kmer_ids: array-like length 9 (numpy.ndarray)
  - meth_frac: float

Outputs:
  - <out_dir>/train-000000.tar, train-000001.tar, ...
  - <out_dir>/val-000000.tar, val-000001.tar, ...

Each sample is stored as three torch-serialized tensors:
  - <key>.signal.pth  (float32, shape [400])
  - <key>.kmer.pth    (int64, shape [9])
  - <key>.y.pth       (float32, shape [])

This is a smoke-test-friendly format, and loads via webdataset.torch_loads.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import webdataset as wds


def torch_bytes(obj) -> bytes:
    """Serialize a torch object to bytes (for WebDataset .pth entries)."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=1024, help="samples per shard")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    need = {"signal", "kmer_ids", "meth_frac"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in parquet: {sorted(missing)}")

    # accept numpy arrays
    df = df.dropna(subset=["meth_frac"]).copy()
    df = df[df["signal"].map(lambda x: hasattr(x, "__len__") and len(x) == 400)]
    df = df[df["kmer_ids"].map(lambda x: hasattr(x, "__len__") and len(x) == 9)]
    df = df.reset_index(drop=True)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    if len(df) < 10:
        raise SystemExit(f"Too few rows after filtering: {len(df)}")

    # deterministic split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * args.val_frac))
    val_idx = set(idx[:n_val].tolist())

    train_pattern = str(out_dir / "train-%06d.tar")
    val_pattern = str(out_dir / "val-%06d.tar")

    with wds.ShardWriter(train_pattern, maxcount=args.shard_size) as wtrain, \
         wds.ShardWriter(val_pattern, maxcount=args.shard_size) as wval:
        for i in range(len(df)):
            r = df.iloc[i]

            sig = torch.tensor(np.asarray(r["signal"], dtype=np.float32))      # [400]
            km  = torch.tensor(np.asarray(r["kmer_ids"], dtype=np.int64))      # [9]
            y   = torch.tensor(float(r["meth_frac"]), dtype=torch.float32)     # []

            sample = {
                "__key__": f"{i:09d}",
                "signal.pth": torch_bytes(sig),
                "kmer.pth": torch_bytes(km),
                "y.pth": torch_bytes(y),
            }

            if i in val_idx:
                wval.write(sample)
            else:
                wtrain.write(sample)

    print("Wrote WebDataset shards to:", out_dir)
    print("Train shards pattern:", train_pattern)
    print("Val shards pattern:", val_pattern)
    print("Rows total:", len(df), "val:", n_val, "train:", len(df) - n_val)


if __name__ == "__main__":
    main()