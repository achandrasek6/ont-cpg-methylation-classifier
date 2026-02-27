process EXTRACT_LABELED {
  tag "${sample_id}"
  label 'cpu'

  /*
    VAL extractor (chr-holdout):
      Same as TRAIN, but writes a different output filename so BUILD_WDS can stage
      both train+val labeled parquets without filename collisions.
  */

  input:
    tuple val(sample_id), path(coords_covered), path(bam_sorted), path(bam_bai), path(pod5)
    path extract_py
    val  window
    val  max_reads_per_site
    val  min_mapq
    val  limit_sites
    val  k

  output:
    tuple val(sample_id), path("${sample_id}.val.labeled_k${k}.parquet")

  publishDir "${params.outdir}/derived/runs/${sample_id}/labeled_val/window=${window}_k=${k}_rps=${max_reads_per_site}",
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
    --out_parquet "${sample_id}.val.labeled_k${k}.parquet" \
    --window ${window} \
    --max_reads_per_site ${max_reads_per_site} \
    --min_mapq ${min_mapq} \
    --limit_sites ${limit_sites}
  """
}