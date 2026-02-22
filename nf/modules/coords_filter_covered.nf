process COORDS_FILTER_COVERED {
  tag "${sample_id}"
  label 'cpu'

  /*
    This module expects the caller to have already paired (coords_sample, bam, bai)
    by sample_id using `join`, so it takes ONE tuple input.

    Input payload:
      (sample_id, coords_sample, bam_sorted, bam_bai)

    Output payload (carry BAM/BAI forward so no second BAM consumer is needed):
      (sample_id, coords_covered, bam_sorted, bam_bai)
  */

  input:
    tuple val(sample_id), path(coords_sample), path(bam_sorted), path(bam_bai)
    path filter_py
    val  min_mapq
    val  max_sites

  output:
    tuple val(sample_id),
          path("${sample_id}.coords_covered.parquet"),
          path(bam_sorted),
          path(bam_bai)

  publishDir "${params.outdir}/derived/runs/${sample_id}/coords",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  # Stable BAM basename so pysam finds the matching index
  ln -sf "${bam_sorted}" bam.bam
  ln -sf "${bam_bai}"    bam.bam.bai

  python3 "${filter_py}" \
    --coords "${coords_sample}" \
    --bam bam.bam \
    --out "${sample_id}.coords_covered.parquet" \
    --min_mapq ${min_mapq} \
    --max_sites ${max_sites}
  """
}
