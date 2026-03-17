#!/usr/bin/env python3
"""
Coordinate-restricted chunk extraction + WGBS labeling using Dorado move tables (with 9-mer context).

This script extracts fixed-length signal windows centered on CpG sites that are
already labeled by WGBS (your coords parquet), producing training-ready examples.

It adds k-mer sequence context (k=9) around the aligned CpG base in the read's
basecalled query sequence:
  - kmer: string length 9 (padded with 'N' at ends)
  - kmer_ids: list[int] where A=0,C=1,G=2,T=3,N/other=4

Key idea:
  - Use BAM alignments to find (read_id, query_pos) overlapping each CpG site.
  - Use Dorado move table tag mv:B:c plus trimmed-samples tag ts:i to map
    query_pos -> raw signal sample index.
  - Pull a window (default 400 samples) from POD5 raw signal.
  - Attach WGBS label (meth_frac) from the coords parquet.

Inputs:
  --coords_parquet : Parquet with columns: chrom, pos0, meth_frac (pos0 is 0-based CpG C coordinate)
  --bam           : Dorado BAM produced with --emit-moves and aligned with --reference
  --pod5          : POD5 file containing the reads (read_id must match BAM query_name)

Outputs (choose exactly one mode):
  A) Single parquet (legacy):
       --out_parquet out.parquet
  B) Partitioned output (recommended):
       --out_dir labeled_parts/  (writes part-000000.parquet, part-000001.parquet, ...)

Parquet schema:
  chrom, pos0, read_id, strand, qpos, center_sample, meth_frac, signal, kmer, kmer_ids

Memory notes:
  - This implementation DOES NOT load all POD5 signals into memory.
  - It performs:
      Pass 1 (BAM): build a bounded list of extraction requests keyed by read_id.
      Pass 2 (POD5): stream reads; when read_id is needed, extract the requested windows and write output.
  - Output writing is chunked (either multiple part-*.parquet files or a streamed single parquet).

Requirements:
  pip install pod5 pysam pandas pyarrow numpy
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

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
_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_from_query(seq: str, qpos: int, k: int = 9) -> str:
    """Return k-mer centered at qpos from seq; pad with 'N' if out of bounds."""
    left = k // 2
    right = k - left - 1
    out: List[str] = []
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


@dataclass(frozen=True)
class Request:
    chrom: str
    pos0: int
    read_id: str
    strand: str
    qpos: int
    center_sample: int
    start: int
    end: int
    meth_frac: float
    kmer: str
    kmer_ids: List[int]


class Writer:
    """
    Incremental parquet writer.
    - If out_dir is set: writes part-*.parquet files of ~part_rows rows.
    - Else: writes a single parquet file incrementally via ParquetWriter.
    """

    def __init__(
        self,
        out_parquet: Optional[str],
        out_dir: Optional[str],
        part_rows: int,
    ) -> None:
        self.out_parquet = out_parquet
        self.out_dir = Path(out_dir) if out_dir else None
        self.part_rows = int(part_rows)

        if (out_parquet is None) == (out_dir is None):
            raise SystemExit("Provide exactly one of --out_parquet OR --out_dir")

        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._pw: Optional[pq.ParquetWriter] = None
            self._part_idx = 0
        else:
            outp = Path(self.out_parquet)  # type: ignore[arg-type]
            outp.parent.mkdir(parents=True, exist_ok=True)
            self._pw = None
            self._part_idx = 0

        self._buf: List[dict] = []
        self.rows_total = 0

    def _rows_to_table(self, rows: List[dict]) -> pa.Table:
        return pa.Table.from_pylist(rows)

    def add_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        self._buf.extend(rows)
        if len(self._buf) >= self.part_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return

        table = self._rows_to_table(self._buf)

        if self.out_dir is not None:
            out_path = self.out_dir / f"part-{self._part_idx:06d}.parquet"
            pq.write_table(table, str(out_path), compression="zstd")
            print(f"[WRITE] {out_path} rows={len(self._buf):,}")
            self._part_idx += 1
        else:
            if self._pw is None:
                # initialize ParquetWriter with the first batch schema
                self._pw = pq.ParquetWriter(self.out_parquet, table.schema, compression="zstd")  # type: ignore[arg-type]
            self._pw.write_table(table)

        self.rows_total += len(self._buf)
        self._buf = []

    def close(self) -> None:
        self.flush()
        if self._pw is not None:
            self._pw.close()
            self._pw = None


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--coords_parquet", required=True)
    ap.add_argument("--bam", required=True)
    ap.add_argument("--pod5", required=True)

    # output mode: choose exactly one
    ap.add_argument("--out_parquet", default=None, help="write one parquet (legacy)")
    ap.add_argument("--out_dir", default=None, help="write partitioned parquets to this directory")
    ap.add_argument("--part_rows", type=int, default=50_000, help="rows per part / flush batch")

    ap.add_argument("--window", type=int, default=400, help="Total samples per chunk (default 400)")
    ap.add_argument("--max_reads_per_site", type=int, default=3, help="Limit reads extracted per CpG site")
    ap.add_argument("--min_mapq", type=int, default=10)
    ap.add_argument("--limit_sites", type=int, default=0, help="0 = all, else limit number of CpG sites")

    args = ap.parse_args()

    if args.window % 2 != 0:
        raise SystemExit("--window must be even (e.g., 400)")
    half = int(args.window) // 2

    coords = pd.read_parquet(args.coords_parquet)
    required_cols = {"chrom", "pos0", "meth_frac"}
    missing = required_cols - set(coords.columns)
    if missing:
        raise SystemExit(f"coords_parquet missing columns: {sorted(missing)}")

    if args.limit_sites and int(args.limit_sites) > 0:
        coords = coords.head(int(args.limit_sites)).copy()

    # -------------------------
    # PASS 1: BAM -> Requests
    # -------------------------
    bam = pysam.AlignmentFile(args.bam, "rb")

    req_by_read: DefaultDict[str, List[Request]] = defaultdict(list)

    skipped_no_mv = 0
    skipped_no_seq = 0
    skipped_qpos_seq_oob = 0
    skipped_bad_mv_map = 0
    skipped_oob_window = 0

    total_sites = 0
    total_reqs = 0

    for row in coords.itertuples(index=False):
        total_sites += 1
        chrom = str(getattr(row, "chrom"))
        pos0 = int(getattr(row, "pos0"))
        meth_frac = float(getattr(row, "meth_frac"))

        extracted_here = 0

        for aln in bam.fetch(chrom, pos0, pos0 + 1):
            if extracted_here >= int(args.max_reads_per_site):
                break
            if aln.is_unmapped or aln.mapping_quality < int(args.min_mapq):
                continue

            read_id = aln.query_name

            qpos_list = find_query_positions_for_refpos(aln, pos0)
            if not qpos_list:
                continue
            qpos = int(qpos_list[0])

            seq = aln.query_sequence
            if not seq:
                skipped_no_seq += 1
                continue
            if qpos < 0 or qpos >= len(seq):
                skipped_qpos_seq_oob += 1
                continue

            try:
                mv = aln.get_tag("mv")
            except KeyError:
                skipped_no_mv += 1
                continue

            try:
                ts = int(aln.get_tag("ts"))
            except KeyError:
                ts = 0

            try:
                stride, moves = decode_mv_tag(mv)
            except Exception:
                skipped_bad_mv_map += 1
                continue

            seq_len = aln.query_length
            if seq_len is None or seq_len <= 0:
                skipped_bad_mv_map += 1
                continue

            base_to_block = build_base_to_block_index(moves, seq_len)
            if base_to_block is None or qpos >= len(base_to_block):
                skipped_bad_mv_map += 1
                continue

            block_idx = int(base_to_block[qpos])
            center = ts + block_idx * stride + (stride // 2)

            start = center - half
            end = center + half
            if start < 0:
                skipped_oob_window += 1
                continue

            km = kmer_from_query(seq, qpos, k=K)
            km_ids = kmer_ids(km)

            req_by_read[read_id].append(
                Request(
                    chrom=chrom,
                    pos0=pos0,
                    read_id=read_id,
                    strand="-" if aln.is_reverse else "+",
                    qpos=qpos,
                    center_sample=int(center),
                    start=int(start),
                    end=int(end),
                    meth_frac=float(meth_frac),
                    kmer=km,
                    kmer_ids=km_ids,
                )
            )
            extracted_here += 1
            total_reqs += 1

    bam.close()

    if total_reqs == 0:
        raise SystemExit(
            "No extraction requests were created from BAM.\n"
            "Possible causes:\n"
            "- BAM does not overlap the coords\n"
            "- mv tags missing\n"
            "- min_mapq too high\n"
        )

    need_reads = set(req_by_read.keys())
    print(f"[PASS1] sites={total_sites:,} requests={total_reqs:,} unique_reads={len(need_reads):,}")
    print(
        "[PASS1] skipped_no_mv={:,} skipped_no_seq={:,} skipped_qpos_seq_oob={:,} skipped_bad_mv_map={:,} skipped_oob_window={:,}".format(
            skipped_no_mv, skipped_no_seq, skipped_qpos_seq_oob, skipped_bad_mv_map, skipped_oob_window
        )
    )

    # -------------------------
    # PASS 2: POD5 -> Extract
    # -------------------------
    writer = Writer(out_parquet=args.out_parquet, out_dir=args.out_dir, part_rows=args.part_rows)

    found_reads = 0
    skipped_no_signal = 0
    skipped_signal_oob = 0

    # Stream POD5 reads; only materialize signals for those we need.
    with pod5.Reader(args.pod5) as reader:
        for r in reader.reads():
            rid = str(r.read_id)
            reqs = req_by_read.get(rid)
            if not reqs:
                continue

            found_reads += 1
            signal = r.signal  # numpy array

            out_rows: List[dict] = []
            for req in reqs:
                if req.end > int(signal.shape[0]):
                    skipped_signal_oob += 1
                    continue

                chunk = signal[req.start : req.end].astype(np.float32)

                out_rows.append(
                    {
                        "chrom": req.chrom,
                        "pos0": req.pos0,
                        "read_id": req.read_id,
                        "strand": req.strand,
                        "qpos": req.qpos,
                        "center_sample": req.center_sample,
                        "meth_frac": req.meth_frac,
                        "kmer": req.kmer,
                        "kmer_ids": req.kmer_ids,
                        "signal": chunk.tolist(),
                    }
                )

            if out_rows:
                writer.add_rows(out_rows)

            # free memory by deleting list for this read (optional)
            del req_by_read[rid]

    # Any remaining reqs correspond to reads not present in this POD5
    if req_by_read:
        skipped_no_signal = sum(len(v) for v in req_by_read.values())

    writer.close()

    if writer.rows_total == 0:
        raise SystemExit(
            "No chunks extracted in PASS2.\n"
            "Most common causes:\n"
            "- POD5 read_id does not match BAM query_name\n"
            "- Requested windows go out of bounds of the signal\n"
        )

    print(f"[PASS2] found_reads={found_reads:,} wrote_rows={writer.rows_total:,}")
    if args.out_dir:
        print(f"[OUT] wrote parts under: {args.out_dir}")
    else:
        print(f"[OUT] wrote parquet: {args.out_parquet}")

    print(
        "[PASS2] skipped_no_signal={:,} skipped_signal_oob={:,}".format(
            skipped_no_signal, skipped_signal_oob
        )
    )


if __name__ == "__main__":
    main()