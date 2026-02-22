process CONVERT_POD5 {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(fast5_dir)

  output:
    tuple val(sample_id), path("${sample_id}.pod5")

  publishDir "${params.outdir}/derived/runs/${sample_id}/pod5",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail
  pod5 convert fast5 "${fast5_dir}" --output "${sample_id}.pod5"
  """
}
