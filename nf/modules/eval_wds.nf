process EVAL_WDS {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(wds_dir)
    tuple val(sample_id2), path(ckpt)
    path eval_py
    path model_py
    val  dataset_id
    val  exp_name
    val  split
    val  batch_size
    val  num_workers
    val  calibrate
    val  calib_bins

  output:
    tuple val(sample_id), path("${sample_id}.eval_metrics.json")

  publishDir "${params.outdir}/models/${sample_id}/exp=${exp_name}",
    mode: params.publish_mode, overwrite: true

  when:
    sample_id == sample_id2

  script:
  """
  set -euo pipefail

  python3 "${eval_py}" \\
    --wds_dir "${wds_dir}" \\
    --ckpt "${ckpt}" \\
    --model_py "${model_py}" \\
    --split "${split}" \\
    --batch_size ${batch_size} \\
    --num_workers ${num_workers} \\
    --calib_bins ${calib_bins} \\
    --out_json "${sample_id}.eval_metrics.json" \\
    ${calibrate ? "--calibrate" : ""}
  """
}