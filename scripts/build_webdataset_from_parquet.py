#!/usr/bin/env python3
"""
Build WebDataset shards (.tar) from labeled parquets produced by extract_coord_chunks_wgbs.py.

Two modes:

A) Legacy mode (single parquet + random val split):
   --parquet <labeled.parquet> --val_frac 0.2

B) Holdout mode (explicit train/val parquets, e.g., chr1-19 vs chr20):
   --train_parquet <train_labeled.parquet> --val_parquet <val_labeled.parquet>

Input parquet(s) must contain:
  - signal: array-like length 400 (numpy.ndarray)
  - kmer_ids: array-like length 9 (numpy.ndarray)
  - meth_frac: float

Outputs:
  - <out_dir>/train-000000.tar, train-000001.tar, ...
  - <out_dir>/val-000000.tar,   val-000001.tar, ...
  - Optional (if --calib_frac > 0): <out_dir>/calib-000000.tar, calib-000001.tar, ...

Calibration split behavior:
  - If --calib_frac > 0, we split the VAL set into:
      * calib: frac = calib_frac (min 1 row)
      * val:   remaining
    Train set is unchanged.
  - By default, calib is created via STRATIFIED split on meth_frac quantile bins,
    to ensure high/low methylation regions are represented in calib (prevents
    calibration overfitting to midrange).

Each sample is stored as three torch-serialized tensors:
  - <key>.signal.pth  (float32, shape [400])
  - <key>.kmer.pth    (int64, shape [9])
  - <key>.y.pth       (float32, shape [])
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import webdataset as wds


def torch_bytes(obj) -> bytes:
    """Serialize a torch object to bytes (for WebDataset .pth entries)."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _load_and_filter(parquet_path: str) -> pd.DataFrame:
    """
    Load a labeled parquet dataset from either:
      - a single parquet file, OR
      - a directory containing part-*.parquet files

    Then enforce required columns and basic row-level sanity filters.
    """
    p = Path(parquet_path)

    if p.is_dir():
        parts = sorted(p.glob("part-*.parquet"))
        if not parts:
            raise SystemExit(f"No part-*.parquet files found in {parquet_path}")
        # simple + reliable: concatenate parts
        df0 = pd.concat((pd.read_parquet(x) for x in parts), ignore_index=True)
    else:
        df0 = pd.read_parquet(parquet_path)

    need = {"signal", "kmer_ids", "meth_frac"}
    missing = need - set(df0.columns)
    if missing:
        raise SystemExit(f"Missing columns in parquet: {sorted(missing)} in {parquet_path}")

    n0 = len(df0)

    df1 = df0.dropna(subset=["meth_frac"]).copy()
    n1 = len(df1)

    df2 = df1[df1["signal"].map(lambda x: hasattr(x, "__len__") and len(x) == 400)]
    n2 = len(df2)

    df3 = df2[df2["kmer_ids"].map(lambda x: hasattr(x, "__len__") and len(x) == 9)]
    n3 = len(df3)

    df3 = df3.reset_index(drop=True)

    print(f"[FILTER] {parquet_path}")
    print(f"  rows raw={n0} after_dropna={n1} after_signal_len={n2} after_kmer_len={n3}")

    return df3


def _stratified_subsample(
    df: pd.DataFrame,
    max_rows: int,
    seed: int,
    bins: int,
) -> pd.DataFrame:
    """
    Subsample df to max_rows while roughly preserving the meth_frac distribution.
    Uses quantile-based bins (equal-frequency bins) for stability.
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    rng = np.random.default_rng(seed)

    y = df["meth_frac"].astype(float).to_numpy()
    try:
        b = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    except Exception:
        b = pd.cut(y, bins=bins, labels=False, include_lowest=True)

    b = np.asarray(b)
    keep_idx = []

    unique_bins, counts = np.unique(b[~pd.isna(b)], return_counts=True)
    total = counts.sum()
    if total == 0:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        return df.iloc[idx[:max_rows]].reset_index(drop=True)

    targets = {int(k): max(1, int(round(max_rows * (c / total)))) for k, c in zip(unique_bins, counts)}

    cur = sum(targets.values())
    keys = list(targets.keys())
    while cur != max_rows and keys:
        if cur > max_rows:
            k = max(keys, key=lambda kk: targets[kk])
            if targets[k] > 1:
                targets[k] -= 1
                cur -= 1
            else:
                keys.remove(k)
        else:
            k = keys[cur % len(keys)]
            targets[k] += 1
            cur += 1

    for k, t in targets.items():
        idx = np.where(b == k)[0]
        if len(idx) == 0:
            continue
        if len(idx) <= t:
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=t, replace=False).tolist())

    keep_idx = np.asarray(keep_idx)
    rng.shuffle(keep_idx)
    keep_idx = keep_idx[:max_rows]

    return df.iloc[keep_idx].reset_index(drop=True)


def _split_df(df: pd.DataFrame, frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Non-stratified split (kept for reference / fallback)."""
    if frac <= 0.0 or len(df) == 0:
        return df.iloc[0:0].copy(), df.copy()

    frac = float(max(0.0, min(1.0, frac)))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_a = int(round(len(df) * frac))
    n_a = max(1, min(len(df) - 1, n_a)) if len(df) > 1 else 1

    a_idx = idx[:n_a]
    b_idx = idx[n_a:]

    a = df.iloc[a_idx].reset_index(drop=True)
    b = df.iloc[b_idx].reset_index(drop=True)
    return a, b


def _stratified_split(
    df: pd.DataFrame,
    frac: float,
    seed: int,
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by meth_frac quantile bins:
      - within each bin, take approx frac into calib
      - remainder stays in eval val
    Guarantees both splits see high/low meth_frac regions.

    If bins collapse (low unique), falls back to non-stratified.
    """
    if frac <= 0.0 or len(df) == 0:
        return df.iloc[0:0].copy(), df.copy()

    frac = float(max(0.0, min(1.0, frac)))
    rng = np.random.default_rng(seed)

    y = df["meth_frac"].astype(float).to_numpy()
    try:
        b = pd.qcut(y, q=bins, labels=False, duplicates="drop")
    except Exception:
        b = pd.cut(y, bins=bins, labels=False, include_lowest=True)

    b = np.asarray(b)
    valid = ~pd.isna(b)
    if valid.sum() == 0:
        return _split_df(df, frac, seed)

    calib_idx = []
    val_idx = []

    for k in np.unique(b[valid]):
        k = int(k)
        idx_k = np.where(b == k)[0]
        if len(idx_k) == 0:
            continue
        rng.shuffle(idx_k)
        n_c = int(round(len(idx_k) * frac))
        # Keep both sides non-empty when possible
        if len(idx_k) > 1:
            n_c = max(1, min(len(idx_k) - 1, n_c))
        else:
            n_c = 1

        calib_idx.extend(idx_k[:n_c].tolist())
        val_idx.extend(idx_k[n_c:].tolist())

    # If stratification produced an empty side due to tiny bins, fallback
    if len(calib_idx) == 0 or len(val_idx) == 0:
        return _split_df(df, frac, seed)

    calib_idx = np.asarray(calib_idx)
    val_idx = np.asarray(val_idx)
    rng.shuffle(calib_idx)
    rng.shuffle(val_idx)

    calib_df = df.iloc[calib_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return calib_df, val_df


def _write_wds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: Path,
    shard_size: int,
    calib_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, str, Optional[str]]:
    train_pattern = str(out_dir / "train-%06d.tar")
    val_pattern = str(out_dir / "val-%06d.tar")
    calib_pattern = str(out_dir / "calib-%06d.tar") if calib_df is not None else None

    writers = []
    wtrain = wds.ShardWriter(train_pattern, maxcount=shard_size)
    writers.append(wtrain)
    wval = wds.ShardWriter(val_pattern, maxcount=shard_size)
    writers.append(wval)
    wcal = None
    if calib_df is not None:
        wcal = wds.ShardWriter(calib_pattern, maxcount=shard_size)
        writers.append(wcal)

    try:
        for i in range(len(train_df)):
            r = train_df.iloc[i]
            sig = torch.tensor(np.asarray(r["signal"], dtype=np.float32))
            km = torch.tensor(np.asarray(r["kmer_ids"], dtype=np.int64))
            yv = torch.tensor(float(r["meth_frac"]), dtype=torch.float32)
            wtrain.write(
                {
                    "__key__": f"train_{i:09d}",
                    "signal.pth": torch_bytes(sig),
                    "kmer.pth": torch_bytes(km),
                    "y.pth": torch_bytes(yv),
                }
            )

        for i in range(len(val_df)):
            r = val_df.iloc[i]
            sig = torch.tensor(np.asarray(r["signal"], dtype=np.float32))
            km = torch.tensor(np.asarray(r["kmer_ids"], dtype=np.int64))
            yv = torch.tensor(float(r["meth_frac"]), dtype=torch.float32)
            wval.write(
                {
                    "__key__": f"val_{i:09d}",
                    "signal.pth": torch_bytes(sig),
                    "kmer.pth": torch_bytes(km),
                    "y.pth": torch_bytes(yv),
                }
            )

        if calib_df is not None and wcal is not None:
            for i in range(len(calib_df)):
                r = calib_df.iloc[i]
                sig = torch.tensor(np.asarray(r["signal"], dtype=np.float32))
                km = torch.tensor(np.asarray(r["kmer_ids"], dtype=np.int64))
                yv = torch.tensor(float(r["meth_frac"]), dtype=torch.float32)
                wcal.write(
                    {
                        "__key__": f"calib_{i:09d}",
                        "signal.pth": torch_bytes(sig),
                        "kmer.pth": torch_bytes(km),
                        "y.pth": torch_bytes(yv),
                    }
                )
    finally:
        for w in writers:
            w.close()

    return train_pattern, val_pattern, calib_pattern


def main() -> None:
    ap = argparse.ArgumentParser()

    # legacy mode
    ap.add_argument("--parquet", default=None)

    # holdout mode
    ap.add_argument("--train_parquet", default=None)
    ap.add_argument("--val_parquet", default=None)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=1024, help="samples per shard")
    ap.add_argument("--val_frac", type=float, default=0.2)  # used only in legacy mode

    ap.add_argument("--calib_frac", type=float, default=0.0, help="split VAL into calib (frac) and eval-val (rest)")
    ap.add_argument(
        "--calib_stratify_bins",
        type=int,
        default=0,
        help="bins for STRATIFIED calib split (0 -> use stratify_bins; 1 -> disable stratification)",
    )

    ap.add_argument("--seed", type=int, default=7)

    # optional subsampling
    ap.add_argument("--max_rows", type=int, default=0, help="legacy mode: 0 = all rows")
    ap.add_argument("--max_rows_train", type=int, default=0, help="holdout mode: 0 = all rows")
    ap.add_argument("--max_rows_val", type=int, default=0, help="holdout mode: 0 = all rows")
    ap.add_argument("--stratify_bins", type=int, default=10, help="bins for stratified subsampling")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_mode = bool(args.train_parquet and args.val_parquet)

    # Choose bins for calib stratification
    calib_bins = args.calib_stratify_bins
    if calib_bins == 0:
        calib_bins = args.stratify_bins

    def make_calib_split(val_df_full: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        if not args.calib_frac or args.calib_frac <= 0:
            return None, val_df_full
        if calib_bins <= 1:
            calib_df, val_df = _split_df(val_df_full, args.calib_frac, seed=args.seed + 2)
            print(f"[CALIB_SPLIT] NON-STRAT frac={args.calib_frac} calib={len(calib_df)} val={len(val_df)}")
        else:
            calib_df, val_df = _stratified_split(val_df_full, args.calib_frac, seed=args.seed + 2, bins=calib_bins)
            print(f"[CALIB_SPLIT] STRAT bins={calib_bins} frac={args.calib_frac} calib={len(calib_df)} val={len(val_df)}")

        if len(calib_df) < 5 or len(val_df) < 5:
            raise SystemExit(
                f"Too few rows after calib split: calib={len(calib_df)} val={len(val_df)} (adjust calib_frac/bins)"
            )
        return calib_df, val_df

    if holdout_mode:
        train_df = _load_and_filter(args.train_parquet)
        val_df_full = _load_and_filter(args.val_parquet)

        if args.max_rows_train and args.max_rows_train > 0:
            train_df = _stratified_subsample(train_df, args.max_rows_train, args.seed, args.stratify_bins)
        if args.max_rows_val and args.max_rows_val > 0:
            val_df_full = _stratified_subsample(val_df_full, args.max_rows_val, args.seed + 1, args.stratify_bins)

        if len(train_df) < 10:
            raise SystemExit(f"Too few TRAIN rows after filtering: {len(train_df)}")
        if len(val_df_full) < 10:
            raise SystemExit(f"Too few VAL rows after filtering: {len(val_df_full)}")

        calib_df, val_df = make_calib_split(val_df_full)

        train_pattern, val_pattern, calib_pattern = _write_wds(
            train_df=train_df,
            val_df=val_df,
            out_dir=out_dir,
            shard_size=args.shard_size,
            calib_df=calib_df,
        )

        print("Mode: holdout (explicit train/val parquets)")
        print("Wrote WebDataset shards to:", out_dir)
        print("Train shards pattern:", train_pattern)
        if calib_pattern:
            print("Calib shards pattern:", calib_pattern)
            print("Calib rows total:", len(calib_df))
        print("Val shards pattern:", val_pattern)
        print("Train rows total:", len(train_df))
        print("Val rows total:", len(val_df))

    else:
        if not args.parquet:
            raise SystemExit("Must provide --parquet (legacy mode) OR --train_parquet and --val_parquet (holdout mode)")

        df = _load_and_filter(args.parquet)

        if args.max_rows and args.max_rows > 0:
            df = _stratified_subsample(df, args.max_rows, args.seed, args.stratify_bins)

        if len(df) < 10:
            raise SystemExit(f"Too few rows after filtering: {len(df)}")

        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_val = max(1, int(len(df) * args.val_frac))
        val_mask = np.zeros(len(df), dtype=bool)
        val_mask[idx[:n_val]] = True

        train_df = df.loc[~val_mask].reset_index(drop=True)
        val_df_full = df.loc[val_mask].reset_index(drop=True)

        calib_df, val_df = make_calib_split(val_df_full)

        train_pattern, val_pattern, calib_pattern = _write_wds(
            train_df=train_df,
            val_df=val_df,
            out_dir=out_dir,
            shard_size=args.shard_size,
            calib_df=calib_df,
        )

        print("Mode: legacy (single parquet + val_frac split)")
        print("Wrote WebDataset shards to:", out_dir)
        print("Train shards pattern:", train_pattern)
        if calib_pattern:
            print("Calib shards pattern:", calib_pattern)
            print("Calib rows total:", len(calib_df))
        print("Val shards pattern:", val_pattern)
        print("Rows total:", len(df), "val:", len(val_df), "train:", len(train_df))


if __name__ == "__main__":
    main()