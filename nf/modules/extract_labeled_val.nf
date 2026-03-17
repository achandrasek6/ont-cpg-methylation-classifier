process EXTRACT_LABELED_VAL {
  tag "${sample_id}"
  label 'cpu'

  /*
    VAL extractor (chr-holdout):
      Writes partitioned parquet parts to avoid OOM.
      Output is a directory containing part-000000.parquet, part-000001.parquet, ...
      Uses a distinct dirname to avoid collisions with TRAIN output.
  */

  input:
    tuple val(sample_id), path(coords_covered), path(bam_sorted), path(bam_bai), path(pod5)
    path extract_py
    val  window
    val  max_reads_per_site
    val  min_mapq
    val  limit_sites
    val  k
    val  part_rows

  output:
    tuple val(sample_id), path("${sample_id}.val.labeled_k${k}", type: 'dir')

  publishDir "${params.outdir}/derived/runs/${sample_id}/labeled_val/window=${window}_k=${k}_rps=${max_reads_per_site}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  # Stable BAM basename so pysam sees the matching index
  ln -sf "${bam_sorted}" bam.bam
  ln -sf "${bam_bai}"    bam.bam.bai

  OUTDIR="${sample_id}.val.labeled_k${k}"
  mkdir -p "\$OUTDIR"

  python3 "${extract_py}" \
    --coords_parquet "${coords_covered}" \
    --bam bam.bam \
    --pod5 "${pod5}" \
    --out_dir "\$OUTDIR" \
    --part_rows ${part_rows} \
    --window ${window} \
    --max_reads_per_site ${max_reads_per_site} \
    --min_mapq ${min_mapq} \
    --limit_sites ${limit_sites}

  # fail fast if nothing was produced
  test \$(ls -1 "\$OUTDIR"/*.parquet 2>/dev/null | wc -l) -gt 0

  # quick sanity
  ls -lh "\$OUTDIR" | head -n 50
  """
}