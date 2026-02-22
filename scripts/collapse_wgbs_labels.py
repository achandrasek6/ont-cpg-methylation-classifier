"""
Collapse strand-specific CpG label records into a strand-agnostic site table.

Input:
  outputs/labels/cpg_labels_ds (Parquet dataset) with columns including:
    chrom, pos0, strand, meth_frac, coverage

Output:
  outputs/labels/cpg_labels_sites_ds (Parquet dataset dir)
with one row per (chrom,pos0):
  chrom, pos0, meth_frac, coverage

meth_frac is computed as coverage-weighted mean when possible.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ds", required=True)
    ap.add_argument("--out_ds", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_ds, exist_ok=True)

    dset = ds.dataset(args.in_ds, format="parquet")
    # Pull only needed cols (some of your label ds may not include strand; handle gracefully)
    cols = [c for c in ["chrom", "pos0", "meth_frac", "coverage"] if c in dset.schema.names]
    tbl = dset.to_table(columns=cols)
    pdf = tbl.to_pandas()

    if "coverage" not in pdf.columns:
        pdf["coverage"] = 1

    # coverage-weighted mean meth_frac
    pdf["w"] = pdf["coverage"].astype("float64")
    pdf["wm"] = pdf["meth_frac"].astype("float64") * pdf["w"]

    grp = pdf.groupby(["chrom", "pos0"], as_index=False).agg(
        coverage=("coverage", "sum"),
        wm=("wm", "sum"),
        w=("w", "sum"),
    )
    grp["meth_frac"] = (grp["wm"] / grp["w"]).astype("float32")
    grp = grp.drop(columns=["wm", "w"])

    out_path = os.path.join(args.out_ds, "part-000000.parquet")
    pq.write_table(pa.Table.from_pandas(grp, preserve_index=False), out_path, compression="zstd")
    print("Wrote:", args.out_ds, "rows:", len(grp))


if __name__ == "__main__":
    main()