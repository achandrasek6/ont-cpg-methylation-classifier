process EXTRACT_LABELED {
  tag "${sample_id}"
  label 'cpu'

  /*
    IMPORTANT (bug fix):
      - Avoid consuming BAM/BAI in multiple downstream steps.
      - Upstream `nf/main.nf` now builds a single payload channel:
          (sample_id, coords_covered, bam_sorted, bam_bai, pod5)
      - This process therefore takes ONE tuple channel for those artifacts.
  */

  input:
    tuple val(sample_id), path(coords_covered), path(bam_sorted), path(bam_bai), path(pod5)
    path extract_py
    val  window
    val  max_reads_per_site
    val  min_mapq
    val  limit_sites
    val  k              // bookkeeping only; script is fixed at K=9

  output:
    tuple val(sample_id), path("${sample_id}.labeled_k${k}.parquet")

  publishDir "${params.outdir}/derived/runs/${sample_id}/labeled/window=${window}_k=${k}_rps=${max_reads_per_site}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  # Stable BAM basename so pysam sees the matching index
  ln -sf "${bam_sorted}" bam.bam
  ln -sf "${bam_bai}"    bam.bam.bai

  python3 "${extract_py}" \
    --coords_parquet "${coords_covered}" \
    --bam bam.bam \
    --pod5 "${pod5}" \
    --out_parquet "${sample_id}.labeled_k${k}.parquet" \
    --window ${window} \
    --max_reads_per_site ${max_reads_per_site} \
    --min_mapq ${min_mapq} \
    --limit_sites ${limit_sites}
  """
}
