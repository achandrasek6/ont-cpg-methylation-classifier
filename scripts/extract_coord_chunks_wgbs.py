"""
Coordinate-restricted chunk extraction + WGBS labeling using Dorado move tables (with 9-mer context).

This script extracts fixed-length signal windows centered on CpG sites that are
already labeled by WGBS (your coords parquet), producing training-ready examples.

It additionally adds k-mer sequence context (k=9) around the aligned CpG base
in the read's basecalled query sequence:
  - kmer: string length 9 (padded with 'N' at ends)
  - kmer_ids: list[int] where A=0,C=1,G=2,T=3,N/other=4

Key idea:
  - Use BAM alignments to find (read_id, query_pos) overlapping each CpG site.
  - Use Dorado move table tag mv:B:c plus trimmed-samples tag ts:i to map
    query_pos -> raw signal sample index.  (Dorado move table docs:
    https://software-docs.nanoporetech.com/dorado/latest/basecaller/move_table/)
  - Pull a window (default 400 samples) from POD5 raw signal.
  - Attach WGBS label (meth_frac) from the coords parquet.

Inputs:
  --coords_parquet : Parquet with columns: chrom, pos0, meth_frac (pos0 is 0-based CpG C coordinate)
  --bam           : Dorado BAM produced with --emit-moves and aligned with --reference
  --pod5          : POD5 file containing the reads (read_id must match BAM query_name)

Output:
  --out_parquet   : Parquet with:
      chrom, pos0, read_id, strand, qpos, center_sample, meth_frac, signal, kmer, kmer_ids

Notes:
  - This version uses pod5.Reader(...).reads() to build a read_id->signal map (smoke-scale).
  - For full-scale runs, we’ll avoid holding all signals in RAM by streaming per-run/per-shard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam

try:
    import pod5  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires the pod5 Python package. Install with: pip install pod5\n"
        f"Import error: {e}"
    )


K = 9
_KL = K // 2  # 4
_KR = K - _KL - 1  # 4
_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_from_query(seq: str, qpos: int, k: int = 9) -> str:
    """
    Return k-mer centered at qpos from seq; pad with 'N' if out of bounds.
    """
    left = k // 2
    right = k - left - 1
    out = []
    for i in range(qpos - left, qpos + right + 1):
        if i < 0 or i >= len(seq):
            out.append("N")
        else:
            b = seq[i].upper()
            out.append(b if b in _BASE_TO_ID else "N")
    return "".join(out)


def kmer_ids(kmer: str) -> List[int]:
    """Tokenize kmer string into ids A=0,C=1,G=2,T=3,N/other=4."""
    return [(_BASE_TO_ID.get(b, 4)) for b in kmer]


def decode_mv_tag(mv: Iterable[int]) -> Tuple[int, List[int]]:
    """
    Decode Dorado mv tag.

    mv tag format:
      mv:B:c,[block_stride],[signal_block_move_list...]
    with overflow encoding using -128/127 chaining.
    """
    mv_list = list(mv)
    if len(mv_list) < 2:
        raise ValueError("mv tag too short")

    stride = int(mv_list[0])
    raw = mv_list[1:]

    moves: List[int] = []
    i = 0
    while i < len(raw):
        v = int(raw[i])
        if v in (-128, 127):
            acc = v
            j = i + 1
            while j < len(raw):
                nxt = int(raw[j])
                acc += nxt
                j += 1
                if nxt not in (-128, 127):
                    break
            moves.append(acc)
            i = j
        else:
            moves.append(v)
            i += 1

    return stride, moves


def build_base_to_block_index(moves: List[int], expected_bases: int) -> Optional[np.ndarray]:
    """
    Build base_idx -> block_idx mapping from move table.

    For standard simplex DNA basecalling, moves are typically 0/1, where 1 indicates
    a base emission at that block index.
    """
    emit_blocks = [i for i, m in enumerate(moves) if m == 1]
    if len(emit_blocks) < expected_bases:
        return None
    emit_blocks = emit_blocks[:expected_bases]
    return np.asarray(emit_blocks, dtype=np.int64)


def find_query_positions_for_refpos(aln: pysam.AlignedSegment, refpos0: int) -> List[int]:
    """Return all query positions (qpos) that align to a given reference pos0 in this alignment."""
    out: List[int] = []
    for qpos, rpos in aln.get_aligned_pairs(matches_only=False, with_seq=False):
        if rpos == refpos0 and qpos is not None:
            out.append(int(qpos))
    return out


def load_pod5_signals(pod5_path: str) -> Dict[str, np.ndarray]:
    """
    Load a POD5 file into a read_id -> signal mapping.

    Uses pod5.Reader(...).reads() API (compatible with pod5 versions where Reader.read does not exist).
    This is smoke-scale; for large datasets, do not load everything into RAM.
    """
    signals: Dict[str, np.ndarray] = {}
    with pod5.Reader(pod5_path) as reader:
        for r in reader.reads():
            rid = str(r.read_id)
            signals[rid] = r.signal
    return signals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords_parquet", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--pod5", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--window", type=int, default=400, help="Total samples per chunk (default 400)")
    ap.add_argument("--max_reads_per_site", type=int, default=3, help="Limit reads extracted per CpG site")
    ap.add_argument("--min_mapq", type=int, default=10)
    ap.add_argument("--limit_sites", type=int, default=0, help="0 = all, else limit number of CpG sites")
    args = ap.parse_args()

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)

    if args.window % 2 != 0:
        raise SystemExit("--window must be even (e.g., 400)")
    half = args.window // 2

    coords = pd.read_parquet(args.coords_parquet)
    required_cols = {"chrom", "pos0", "meth_frac"}
    missing = required_cols - set(coords.columns)
    if missing:
        raise SystemExit(f"coords_parquet missing columns: {sorted(missing)}")

    if args.limit_sites and args.limit_sites > 0:
        coords = coords.head(args.limit_sites).copy()

    print("Loading POD5 signals into memory...")
    sig_by_read = load_pod5_signals(args.pod5)
    print(f"Loaded signals for {len(sig_by_read):,} reads")

    bam = pysam.AlignmentFile(args.bam, "rb")

    out_rows: List[dict] = []
    skipped_no_signal = 0
    skipped_no_mv = 0
    skipped_oob = 0
    skipped_no_seq = 0
    skipped_qpos_seq_oob = 0

    for row in coords.itertuples(index=False):
        chrom = str(getattr(row, "chrom"))
        pos0 = int(getattr(row, "pos0"))
        meth_frac = float(getattr(row, "meth_frac"))

        extracted_here = 0

        for aln in bam.fetch(chrom, pos0, pos0 + 1):
            if extracted_here >= args.max_reads_per_site:
                break
            if aln.is_unmapped or aln.mapping_quality < args.min_mapq:
                continue

            read_id = aln.query_name
            if read_id not in sig_by_read:
                skipped_no_signal += 1
                continue

            qpos_list = find_query_positions_for_refpos(aln, pos0)
            if not qpos_list:
                continue
            qpos = int(qpos_list[0])

            # Need query sequence for kmer
            seq = aln.query_sequence
            if not seq:
                skipped_no_seq += 1
                continue
            if qpos < 0 or qpos >= len(seq):
                skipped_qpos_seq_oob += 1
                continue

            # mv tag required
            try:
                mv = aln.get_tag("mv")
            except KeyError:
                skipped_no_mv += 1
                continue

            # ts tag is trimmed samples start; if missing assume 0
            try:
                ts = int(aln.get_tag("ts"))
            except KeyError:
                ts = 0

            stride, moves = decode_mv_tag(mv)

            seq_len = aln.query_length
            if seq_len is None or seq_len <= 0:
                continue

            base_to_block = build_base_to_block_index(moves, seq_len)
            if base_to_block is None or qpos >= len(base_to_block):
                continue

            block_idx = int(base_to_block[qpos])
            center = ts + block_idx * stride + (stride // 2)

            start = center - half
            end = center + half

            signal = sig_by_read[read_id]
            if start < 0 or end > int(signal.shape[0]):
                skipped_oob += 1
                continue

            chunk = signal[start:end].astype(np.float32)

            km = kmer_from_query(seq, qpos, k=K)
            km_ids = kmer_ids(km)

            out_rows.append(
                {
                    "chrom": chrom,
                    "pos0": pos0,
                    "read_id": read_id,
                    "strand": "-" if aln.is_reverse else "+",
                    "qpos": qpos,
                    "center_sample": int(center),
                    "meth_frac": float(meth_frac),
                    "kmer": km,
                    "kmer_ids": km_ids,
                    "signal": chunk.tolist(),
                }
            )
            extracted_here += 1

    bam.close()

    if not out_rows:
        raise SystemExit(
            "No chunks extracted. Possible causes:\n"
            "- POD5 read_id does not match BAM query_name\n"
            "- BAM reads overlapping coords lack mv tags\n"
            "- window goes out of bounds\n"
        )

    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, args.out_parquet, compression="zstd")

    print(f"Wrote: {args.out_parquet}")
    print(f"rows={len(out_rows):,} sites={len(coords):,} avg_per_site={len(out_rows)/max(len(coords),1):.3f}")
    print(
        "skipped_no_signal={:,} skipped_no_mv={:,} skipped_oob={:,} skipped_no_seq={:,} skipped_qpos_seq_oob={:,}".format(
            skipped_no_signal, skipped_no_mv, skipped_oob, skipped_no_seq, skipped_qpos_seq_oob
        )
    )


if __name__ == "__main__":
    main()