process TRAIN_WDS {
  tag "${sample_id}"
  label 'gpu'

  input:
    tuple val(sample_id), path(wds_dir)
    path train_py
    val  dataset_id
    val  exp_name
    val  epochs
    val  batch_size
    val  lr
    val  num_workers

  output:
    tuple val(sample_id), path("${sample_id}.ckpt.pt"), path("${sample_id}.train.log")

  publishDir "${params.outdir}/models/${sample_id}/exp=${exp_name}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  python3 "${train_py}" \\
    --wds_dir "${wds_dir}" \\
    --epochs ${epochs} \\
    --batch_size ${batch_size} \\
    --lr ${lr} \\
    --num_workers ${num_workers} \\
    | tee "${sample_id}.train.log"

  # Trainer writes checkpoint under the WDS directory
  if [ ! -f "${wds_dir}/joint_model_wds.pt" ]; then
    echo "ERROR: Expected checkpoint not found: ${wds_dir}/joint_model_wds.pt" >&2
    echo "Listing ${wds_dir}:" >&2
    ls -lah "${wds_dir}" >&2 || true
    exit 2
  fi

  cp "${wds_dir}/joint_model_wds.pt" "${sample_id}.ckpt.pt"
  """
}