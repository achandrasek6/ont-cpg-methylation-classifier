"""
Create chromosome-based train/val/test splits from a Parquet dataset directory.

Reads labels from a Parquet dataset (directory containing part-*.parquet) and writes
coordinate split Parquets to outputs/splits/.

Default split:
  - train: chr1–chr19
  - val:   chr20–chr21
  - test:  chr22 + chrX

Outputs:
  outputs/splits/{train,val,test}_coords.parquet
"""

import argparse
import os

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_ds", required=True, help="Parquet dataset directory (e.g., outputs/labels/cpg_labels_ds)")
    ap.add_argument("--out_dir", required=True, help="Output directory (e.g., outputs/splits)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = ds.dataset(args.labels_ds, format="parquet")

    train_chroms = [f"chr{i}" for i in range(1, 20)]
    val_chroms = ["chr20", "chr21"]
    test_chroms = ["chr22", "chrX"]

    split_defs = {
        "train": train_chroms,
        "val": val_chroms,
        "test": test_chroms,
    }

    cols = ["chrom", "pos0", "pos1", "strand", "meth_frac", "coverage"]

    for name, chroms in split_defs.items():
        filt = ds.field("chrom").isin(chroms)
        tbl = dataset.to_table(columns=cols, filter=filt)
        out_path = os.path.join(args.out_dir, f"{name}_coords.parquet")
        pq.write_table(tbl, out_path, compression="zstd")
        print(f"{name}: rows={tbl.num_rows:,} -> {out_path}")


if __name__ == "__main__":
    main()