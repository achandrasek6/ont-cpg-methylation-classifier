"""
Chunked converter: GM24385_mod_2021.09 all.cpg.bed -> Parquet.

Why chunked:
  - all.cpg.bed is multi-GB text; pandas full read is slow and memory-heavy.
  - This script reads in chunks, filters/transforms, and writes incrementally.

Input:
  --bed: path to all.cpg.bed (tab-separated, no header)

Outputs:
  --out_dir: Parquet dataset directory written incrementally (recommended)
  Optionally, --out_file: consolidate to a single Parquet file at the end

Schema (output):
  chrom: str
  pos0: int64 (0-based CpG position; equals BED start)
  pos1: int64 (1-based CpG position; pos0+1)
  strand: str (+/-)
  coverage: int32/64
  meth_count: int32/64
  unmeth_count: int32/64
  meth_frac: float32 (0..1)
  pct_meth: float32 (0..100)
  flag: int32/64

Filtering:
  - mod == "5mC"
  - canonical chromosomes only by default (chr1..chr22, chrX, chrY, chrM)
  - coverage >= --min_cov
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, Optional

import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.dataset as ds  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires 'pyarrow'. Install with: pip install pyarrow\n"
        f"Import error: {e}"
    )

CANON_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="Path to all.cpg.bed (local file)")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output Parquet dataset directory (will be created/overwritten)",
    )
    ap.add_argument(
        "--out_file",
        default="",
        help="Optional: write a consolidated single Parquet file after building dataset",
    )
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Rows per chunk")
    ap.add_argument("--min_cov", type=int, default=10)
    ap.add_argument("--canonical_only", action="store_true", default=True)
    ap.add_argument("--log_every", type=int, default=1, help="Log every N chunks")
    return ap.parse_args()


def ensure_empty_dir(path: str) -> None:
    """Create an empty directory at `path` (delete if it already exists)."""
    if os.path.exists(path):
        # Be explicit; user asked to rebuild cleanly.
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
        os.rmdir(path)
    os.makedirs(path, exist_ok=True)


def bed_reader(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield pandas DataFrames from a large BED-like TSV using chunked reading."""
    cols = [
        "chrom", "start", "end", "mod", "score", "strand",
        "thickStart", "thickEnd", "itemRgb",
        "coverage", "pct_meth", "meth_count", "unmeth_count", "flag",
    ]
    # Use dtype hints to speed parsing; keep integers nullable until filtered.
    dtype = {
        "chrom": "string",
        "mod": "string",
        "strand": "string",
        "start": "Int64",
        "end": "Int64",
        "coverage": "Int64",
        "pct_meth": "float64",
        "meth_count": "Int64",
        "unmeth_count": "Int64",
        "flag": "Int64",
        # other columns can be skipped by not naming dtypes; they’ll parse but we’ll drop later
        "score": "Int64",
        "thickStart": "Int64",
        "thickEnd": "Int64",
        "itemRgb": "string",
    }
    usecols = cols  # keep them all for now; we’ll drop later

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=cols,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    )


def transform_chunk(df: pd.DataFrame, min_cov: int, canonical_only: bool) -> pd.DataFrame:
    """Filter/transform one chunk into the output schema."""
    df = df[df["mod"] == "5mC"].copy()
    if canonical_only:
        df = df[df["chrom"].isin(CANON_CHROMS)].copy()

    # coverage filter
    df = df[df["coverage"].fillna(0) >= min_cov].copy()

    # compute label fields
    df["meth_frac"] = (df["pct_meth"].astype("float32") / 100.0).astype("float32")
    df["pos0"] = df["start"].astype("int64")
    df["pos1"] = (df["start"].astype("int64") + 1).astype("int64")

    out = df[
        ["chrom", "pos0", "pos1", "strand", "coverage", "meth_count", "unmeth_count", "meth_frac", "pct_meth", "flag"]
    ].copy()

    # Tighten dtypes (optional but helps size)
    out["chrom"] = out["chrom"].astype("string")
    out["strand"] = out["strand"].astype("string")
    # Keep coverage/counts as int64 for safety; parquet will compress well.
    out["pct_meth"] = out["pct_meth"].astype("float32")
    out["flag"] = out["flag"].astype("Int64")

    return out


def main() -> None:
    args = parse_args()

    # Build dataset directory fresh to avoid mixing old partial outputs
    ensure_empty_dir(args.out_dir)

    import pyarrow as pa
    import pyarrow.parquet as pq

    rows_in = 0
    rows_out = 0
    t0 = time.time()
    schema: Optional[pa.Schema] = None

    # We will write one parquet file per chunk under out_dir/part-000000.parquet, etc.
    part_idx = 0

    for chunk_idx, chunk in enumerate(bed_reader(args.bed, args.chunksize), start=1):
        rows_in += len(chunk)
        out = transform_chunk(chunk, args.min_cov, args.canonical_only)
        rows_out += len(out)

        if len(out) > 0:
            table = pa.Table.from_pandas(out, preserve_index=False)
            if schema is None:
                schema = table.schema
            else:
                # Ensure stable schema across chunks
                table = table.cast(schema)

            part_path = os.path.join(args.out_dir, f"part-{part_idx:06d}.parquet")
            pq.write_table(table, part_path, compression="zstd")
            part_idx += 1

        if args.log_every > 0 and (chunk_idx % args.log_every == 0):
            dt = time.time() - t0
            rate = rows_in / dt if dt > 0 else 0
            print(
                f"[chunk {chunk_idx}] read_rows={rows_in:,} kept_rows={rows_out:,} "
                f"parts={part_idx} elapsed={dt:,.1f}s read_rate={rate:,.0f} rows/s"
            )

    dt = time.time() - t0
    print(f"Done. read_rows={rows_in:,} kept_rows={rows_out:,} parts={part_idx} elapsed={dt:,.1f}s")

    # Optional: consolidate dataset to a single file
    if args.out_file:
        import pyarrow.dataset as ds

        dataset = ds.dataset(args.out_dir, format="parquet")
        table = dataset.to_table()  # may take time/memory; optional
        pq.write_table(table, args.out_file, compression="zstd")
        print(f"Wrote consolidated file: {args.out_file}")


if __name__ == "__main__":
    main()