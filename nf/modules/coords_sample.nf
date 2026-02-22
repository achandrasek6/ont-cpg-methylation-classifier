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
  \"\"\"Sample CpG coordinates from a Parquet file for local smoke tests.\"\"\"
  import pandas as pd

  src = "${coords_parquet}"
  out = "${run_id}.coords_sample.parquet"
  n = int(${sample_n})

  df = pd.read_parquet(src)

  if n > 0 and len(df) > n:
      df = df.sample(n=n, random_state=7)

  df.to_parquet(out, index=False)
  print(f"Wrote {out} rows={len(df)} from {src}")
  PY
  """
}
