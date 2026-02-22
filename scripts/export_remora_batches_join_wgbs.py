"""
Export Remora CoreRemoraDataset batches and join WGBS CpG truth labels.

This script:
  1) Iterates Remora batches from CoreRemoraDataset.iter_batches().
  2) Maps each chunk's (read_id, read_focus_base) to a reference coordinate using a BAM:
       - read_id -> BAM alignment
       - focus_base (query index) -> reference pos0 via aln.get_reference_positions(full_length=True)
  3) Normalizes the mapped reference position to the CpG "C" coordinate (handles focus landing on G).
  4) Joins (chrom, cpg_pos0) to WGBS labels dataset (chrom,pos0 -> meth_frac,coverage).
  5) Writes a Parquet file of training-ready examples.

Inputs:
  --remora_ds : outputs/remora_core/smoke_ds
  --bam       : outputs/dorado_smoke/smoke.sorted.bam
  --labels_ds : outputs/labels/cpg_labels_ds
  --ref_fa    : reference FASTA used for alignment (must match BAM contigs; e.g. chr1..)
Output:
  --out_parquet : outputs/training/smoke_labeled.parquet

Important:
  Remora's read_focus_base may be in a trimmed/centered sequence coordinate system.
  If matched_labels is low, adjust --focus_offset (default 0). A common value is +4
  (kmer left context) depending on how focus is defined for your dataset.

CpG normalization logic (reference + strand-agnostic):
  - If ref[pos] == C and ref[pos+1] == G => CpG C is pos
  - If ref[pos] == G and ref[pos-1] == C => CpG C is pos-1
  - Else => not a CpG site (no label)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pysam

from remora.data_chunks import CoreRemoraDataset


def build_wgbs_lookup(
    labels_ds_path: str,
    coords: Iterable[Tuple[str, int]],
) -> Dict[Tuple[str, int], Tuple[float, int]]:
    """Build (chrom,pos0)->(meth_frac,coverage) lookup for just the requested coords."""
    coords_by_chrom: Dict[str, List[int]] = {}
    for chrom, pos0 in coords:
        coords_by_chrom.setdefault(chrom, []).append(pos0)

    labels = ds.dataset(labels_ds_path, format="parquet")
    out: Dict[Tuple[str, int], Tuple[float, int]] = {}

    for chrom, pos_list in coords_by_chrom.items():
        pos_list = sorted(set(pos_list))
        if not pos_list:
            continue
        filt = (ds.field("chrom") == chrom) & ds.field("pos0").isin(pos_list)
        tbl = labels.to_table(columns=["chrom", "pos0", "meth_frac", "coverage"], filter=filt)
        if tbl.num_rows == 0:
            continue
        pdf = tbl.to_pandas()
        for r in pdf.itertuples(index=False):
            out[(r.chrom, int(r.pos0))] = (float(r.meth_frac), int(r.coverage))
    return out


def normalize_to_cpg_c_pos0(ref: pysam.FastaFile, chrom: str, refpos0: int) -> int:
    """
    Normalize an arbitrary reference position to the CpG cytosine coordinate (0-based).

    Returns:
      CpG C pos0 (>=0) if the position is part of a CpG dinucleotide, else -1.
    """
    if not chrom or refpos0 < 0:
        return -1

    try:
        base = ref.fetch(chrom, refpos0, refpos0 + 1).upper()
    except Exception:
        return -1

    if base == "C":
        try:
            nxt = ref.fetch(chrom, refpos0 + 1, refpos0 + 2).upper()
        except Exception:
            return -1
        return refpos0 if nxt == "G" else -1

    if base == "G" and refpos0 > 0:
        try:
            prv = ref.fetch(chrom, refpos0 - 1, refpos0).upper()
        except Exception:
            return -1
        return (refpos0 - 1) if prv == "C" else -1

    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remora_ds", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--labels_ds", required=True)
    ap.add_argument("--ref_fa", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--flush_every_batches", type=int, default=5)
    ap.add_argument(
        "--focus_offset",
        type=int,
        default=0,
        help="Additive offset applied to read_focus_base before BAM mapping",
    )
    args = ap.parse_args()

    cds = CoreRemoraDataset(args.remora_ds)
    cds.return_arrays = ["signal", "read_id", "read_focus_base"]
    cds.init_super_batch_iter()

    # Reference for CpG normalization
    ref = pysam.FastaFile(args.ref_fa)

    # Load BAM alignments into dict (smoke-scale OK)
    aln_by_read: Dict[str, pysam.AlignedSegment] = {}
    bam = pysam.AlignmentFile(args.bam, "rb")
    for aln in bam.fetch(until_eof=True):
        aln_by_read[aln.query_name] = aln
    bam.close()

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    total = 0
    matched = 0

    cached_rows = []
    cached_coords: List[Tuple[str, int]] = []

    def flush() -> None:
        nonlocal writer, total, matched, cached_rows, cached_coords
        if not cached_rows:
            return
        wgbs = build_wgbs_lookup(args.labels_ds, cached_coords)

        table_rows = []
        for row in cached_rows:
            total += 1
            key = (row["chrom"], row["pos0"])
            if row["chrom"] and row["pos0"] >= 0 and key in wgbs:
                mf, cv = wgbs[key]
                row["meth_frac"] = float(mf)
                row["coverage"] = int(cv)
                row["has_label"] = 1
                matched += 1
            else:
                row["meth_frac"] = None
                row["coverage"] = None
                row["has_label"] = 0
            table_rows.append(row)

        table = pa.Table.from_pylist(table_rows)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
        writer.write_table(table)

        cached_rows = []
        cached_coords = []

    it = cds.iter_batches(
        batch_size=args.batch_size,
        max_batches=(None if args.max_batches == 0 else args.max_batches),
    )

    for bidx, batch in enumerate(it, start=1):
        read_ids = batch["read_id"].astype(str)
        focus = np.asarray(batch["read_focus_base"], dtype=np.int64) + args.focus_offset
        sig = np.asarray(batch["signal"], dtype=np.float32)

        for i in range(len(read_ids)):
            rid = read_ids[i]
            aln = aln_by_read.get(rid)

            chrom = ""
            pos0 = -1
            strand = ""

            if aln is not None:
                strand = "-" if aln.is_reverse else "+"
                ref_positions = aln.get_reference_positions(full_length=True)
                qpos = int(focus[i])
                if 0 <= qpos < len(ref_positions):
                    rp = ref_positions[qpos]
                    if rp is not None:
                        chrom = aln.reference_name
                        # Normalize to CpG cytosine coordinate
                        pos0 = normalize_to_cpg_c_pos0(ref, chrom, int(rp))

            if chrom and pos0 >= 0:
                cached_coords.append((chrom, pos0))

            cached_rows.append(
                {
                    "read_id": rid,
                    "chrom": chrom,
                    "pos0": pos0,
                    "strand": strand,
                    "signal": sig[i].reshape(-1).tolist(),  # (1,400) -> (400,)
                }
            )

        if args.flush_every_batches and (bidx % args.flush_every_batches == 0):
            flush()

    flush()
    if writer is not None:
        writer.close()

    ref.close()

    print(f"Wrote: {out_path}")
    print(f"rows={total:,} matched_labels={matched:,} ({(matched/total*100.0 if total else 0):.2f}%)")


if __name__ == "__main__":
    main()