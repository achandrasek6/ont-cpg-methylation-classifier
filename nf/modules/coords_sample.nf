process COORDS_SAMPLE {
  tag "${run_id}"
  label 'cpu'

  input:
    path coords_parquet
    val  run_id
    val  sample_n

  output:
    path "${run_id}.coords_sample.parquet"

  publishDir "${params.outdir}/derived/runs/${run_id}/coords",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  python3 - <<'PY'
  \"\"\"Sample CpG coordinates from a Parquet file WITHOUT loading the whole file into memory.\"\"\"

  import random
  import pyarrow as pa
  import pyarrow.parquet as pq

  src = "${coords_parquet}"
  out = "${run_id}.coords_sample.parquet"
  n = int(${sample_n})
  seed = 7

  pf = pq.ParquetFile(src)

  # If n <= 0, keep full file (but still do it row-group wise to reduce peak mem)
  if n <= 0:
      tbl = pf.read()
      pq.write_table(tbl, out)
      print(f"Wrote {out} rows={tbl.num_rows} from {src} (full read)")
      raise SystemExit(0)

  # Read row groups in random order until we have >= n rows
  rg_ids = list(range(pf.num_row_groups))
  random.Random(seed).shuffle(rg_ids)

  tables = []
  rows = 0
  for rg in rg_ids:
      t = pf.read_row_group(rg)
      tables.append(t)
      rows += t.num_rows
      if rows >= n:
          break

  tbl = pa.concat_tables(tables, promote_options="default")

  # Downsample exactly to n rows (uniform)
  if tbl.num_rows > n:
      rng = random.Random(seed)
      idx = sorted(rng.sample(range(tbl.num_rows), n))
      tbl = tbl.take(pa.array(idx))

  pq.write_table(tbl, out)
  print(f"Wrote {out} rows={tbl.num_rows} from {src} (row_groups_read={len(tables)}/{pf.num_row_groups})")
  PY
  """
}