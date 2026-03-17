process BUILD_WDS {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id),
          path(train_labeled_in, stageAs: 'train_in'),
          path(val_labeled_in,   stageAs: 'val_in'),
          val(use_holdout)
    path build_wds_py
    val  dataset_id
    val  shard_size
    val  val_frac
    val  calib_frac
    val  seed
    val  max_rows_train
    val  max_rows_val
    val  stratify_bins

  output:
    tuple val(sample_id), path("wds_out", type: 'dir')

  publishDir "${params.outdir}/derived/runs/${sample_id}/wds/window=${params.window}_k=${params.k}_rps=${params.max_reads_per_site}_shard=${shard_size}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail
  mkdir -p wds_out

  if ${use_holdout}; then
    echo "[BUILD_WDS] Holdout mode (explicit train/val)"
    python3 "${build_wds_py}" \\
      --train_parquet "train_in" \\
      --val_parquet   "val_in" \\
      --out_dir wds_out \\
      --shard_size ${shard_size} \\
      --seed ${seed} \\
      --max_rows_train ${max_rows_train} \\
      --max_rows_val ${max_rows_val} \\
      --stratify_bins ${stratify_bins} \\
      --calib_frac ${calib_frac} \\
      --calib_stratify_bins ${stratify_bins}
  else
    echo "[BUILD_WDS] Legacy mode (single input + val_frac split)"
    python3 "${build_wds_py}" \\
      --parquet "train_in" \\
      --out_dir wds_out \\
      --shard_size ${shard_size} \\
      --val_frac ${val_frac} \\
      --seed ${seed} \\
      --max_rows ${max_rows_train} \\
      --stratify_bins ${stratify_bins} \\
      --calib_frac ${calib_frac} \\
      --calib_stratify_bins ${stratify_bins}
  fi
  """
}