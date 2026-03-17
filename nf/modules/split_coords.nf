nextflow.enable.dsl=2

/*
  SPLIT_COORDS
  ------------
  Split a coords_covered parquet into smaller parquet shards so EXTRACT_LABELED
  runs as many small jobs instead of one large job.

  Input:
    tuple(sample_id, coords_covered_parquet, bam_sorted, bam_bai)

  Output:
    tuple(sample_id, coords_shards_dir, bam_sorted, bam_bai)

  NOTE:
    We emit the shard directory as a single path object.
    The caller should flatten that directory into one tuple per shard parquet.
*/

process SPLIT_COORDS {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(coords_cov), path(bam_sorted), path(bam_bai)
    path split_coords_py
    val  shard_sites

  output:
    tuple val(sample_id), path("coords_shards", type: 'dir'), path(bam_sorted), path(bam_bai)

  script:
  """
  set -euo pipefail
  mkdir -p coords_shards

  python3 "${split_coords_py}" \
    --in_parquet "${coords_cov}" \
    --out_dir coords_shards \
    --shard_sites ${shard_sites}

  test \$(find coords_shards -maxdepth 1 -type f -name "part-*.parquet" | wc -l) -gt 0

  ls -lh coords_shards | head -n 50
  """
}