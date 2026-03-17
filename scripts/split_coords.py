#!/usr/bin/env python3
"""
Split a coords parquet into multiple smaller parquets by row count.

Why:
  - Makes EXTRACT_LABELED run as many smaller jobs instead of one huge job.
  - Helps avoid OOM / long-running “monster” tasks and improves parallelism.

Input parquet is expected to contain at least columns like: chrom, pos0
(and often meth_frac). All columns are preserved.

Example:
  python3 scripts/split_coords.py \
    --in_parquet sample.coords_covered.parquet \
    --out_dir coords_shards \
    --shard_sites 5000 \
    --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_sites", type=int, default=5000)
    ap.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to shuffle rows before splitting (helps avoid chr/pos ordering skew).",
    )
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling; split in existing row order.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.in_parquet)
    if len(df) == 0:
        raise SystemExit(f"Empty input parquet: {args.in_parquet}")

    shard_sites = max(1, int(args.shard_sites))

    # Shuffle to avoid systematic chrom/pos ordering skew per shard
    if not args.no_shuffle:
        df = df.sample(frac=1.0, random_state=int(args.seed)).reset_index(drop=True)

    n = len(df)
    part = 0

    for start in range(0, n, shard_sites):
        end = min(n, start + shard_sites)
        chunk = df.iloc[start:end].reset_index(drop=True)

        out_path = out_dir / f"part-{part:06d}.parquet"
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, str(out_path), compression="zstd")
        part += 1

    print(
        f"Wrote {part} shards to {out_dir} (rows={n}, shard_sites={shard_sites}, "
        f"shuffled={'no' if args.no_shuffle else 'yes'}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()