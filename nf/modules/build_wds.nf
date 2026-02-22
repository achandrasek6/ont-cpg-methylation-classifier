process BUILD_WDS {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(labeled_parquet)
    path build_wds_py
    val  dataset_id
    val  shard_size
    val  val_frac
    val  seed

  output:
    tuple val(sample_id), path("wds_out")

  publishDir "${params.outdir}/derived/runs/${sample_id}/wds/window=${params.window}_k=${params.k}_rps=${params.max_reads_per_site}_shard=${shard_size}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail
  mkdir -p wds_out

  python3 "${build_wds_py}" \
    --parquet "${labeled_parquet}" \
    --out_dir wds_out \
    --shard_size ${shard_size} \
    --val_frac ${val_frac} \
    --seed ${seed}
  """
}