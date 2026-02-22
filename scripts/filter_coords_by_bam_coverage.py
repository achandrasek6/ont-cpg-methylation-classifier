"""
Filter CpG coords to those with at least one overlapping alignment in a BAM.

This increases yield for smoke tests by ensuring the chosen CpG sites are present
in the subset of reads you actually have.
"""

from __future__ import annotations
import argparse
import pandas as pd
import pysam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sites", type=int, default=0, help="0 = keep all passing sites")
    ap.add_argument("--min_mapq", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_parquet(args.coords)
    bam = pysam.AlignmentFile(args.bam, "rb")

    keep = []
    for r in df.itertuples(index=False):
        chrom = str(getattr(r, "chrom"))
        pos0 = int(getattr(r, "pos0"))
        ok = False
        for aln in bam.fetch(chrom, pos0, pos0 + 1):
            if not aln.is_unmapped and aln.mapping_quality >= args.min_mapq:
                ok = True
                break
        keep.append(ok)

    bam.close()
    out = df[pd.Series(keep)].copy()
    if args.max_sites and len(out) > args.max_sites:
        out = out.sample(n=args.max_sites, random_state=1)

    out.to_parquet(args.out, index=False)
    print("Wrote:", args.out, "rows:", len(out), "of", len(df))


if __name__ == "__main__":
    main()