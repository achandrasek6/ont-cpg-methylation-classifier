process COLLECT_WDS_SHARDS {
  tag "global"
  label 'cpu'

  input:
    path(wds_dirs_txt, stageAs: 'wds_dirs.txt')

  output:
    tuple path("train_shards.txt"), path("val_shards.txt"), path("calib_shards.txt")

  publishDir "${params.outdir}/derived/shards/run=${params.run_id ?: 'run'}",
    mode: params.publish_mode, overwrite: true

  script:
  '''
  set -euo pipefail

  : > train_shards.txt
  : > val_shards.txt
  : > calib_shards.txt

  echo "[COLLECT] reading wds dirs from: wds_dirs.txt"
  wc -l wds_dirs.txt || true
  tr -d '\r' < wds_dirs.txt > wds_dirs.clean.txt

  norm_s3() {
    local p="$1"

    # already s3://bucket/key
    if [[ "$p" == s3://* ]]; then
      echo "$p"
      return 0
    fi

    # Nextflow S3 provider path like /bucket/key...
    if [[ "$p" == /* ]]; then
      p="${p#/}"
      local bucket="${p%%/*}"
      local key="${p#*/}"
      if [[ "$bucket" == "$p" ]]; then
        key=""
      fi
      echo "s3://${bucket}/${key}"
      return 0
    fi

    # fallback
    echo "$p"
  }

  s3_bucket_from_uri() {
    # s3://bucket/anything -> bucket
    echo "$1" | sed -E 's#^s3://([^/]+)/.*$#\\1#'
  }

  while IFS= read -r d; do
    d="${d#"${d%%[![:space:]]*}"}"
    d="${d%"${d##*[![:space:]]}"}"
    [ -z "$d" ] && continue

    s3dir="$(norm_s3 "$d")"
    echo "[COLLECT] dir=$s3dir"

    # The aws s3 ls --recursive output column 4 is a *bucket-relative key*.
    # Always rebuild full URI as s3://<bucket>/<key> (do NOT prepend s3dir again).
    bucket="$(s3_bucket_from_uri "$s3dir")"

    aws s3 ls "${s3dir%/}/" --recursive 2>/dev/null \
      | awk '{print $4}' | grep -E 'train-.*\\.tar$' \
      | awk -v b="$bucket" '{print "s3://" b "/" $0}' >> train_shards.txt || true

    aws s3 ls "${s3dir%/}/" --recursive 2>/dev/null \
      | awk '{print $4}' | grep -E 'val-.*\\.tar$' \
      | awk -v b="$bucket" '{print "s3://" b "/" $0}' >> val_shards.txt || true

    aws s3 ls "${s3dir%/}/" --recursive 2>/dev/null \
      | awk '{print $4}' | grep -E 'calib-.*\\.tar$' \
      | awk -v b="$bucket" '{print "s3://" b "/" $0}' >> calib_shards.txt || true

  done < wds_dirs.clean.txt

  sort -u train_shards.txt -o train_shards.txt
  sort -u val_shards.txt   -o val_shards.txt
  sort -u calib_shards.txt -o calib_shards.txt

  echo "[COLLECT] train shards: $(wc -l < train_shards.txt || echo 0)"
  echo "[COLLECT] val   shards: $(wc -l < val_shards.txt   || echo 0)"
  echo "[COLLECT] calib shards: $(wc -l < calib_shards.txt || echo 0)"

  # Fail fast if we accidentally double-prefix again
  if grep -q "/wds_out/ont-cpg/work/" train_shards.txt; then
    echo "ERROR: detected double-prefixed shard paths in train_shards.txt" >&2
    head -n 3 train_shards.txt >&2 || true
    exit 3
  fi

  if [ ! -s train_shards.txt ]; then
    echo "ERROR: train_shards.txt is empty" >&2
    echo "DEBUG wds_dirs.clean.txt:" >&2
    cat wds_dirs.clean.txt >&2 || true
    exit 2
  fi
  if [ ! -s val_shards.txt ]; then
    echo "ERROR: val_shards.txt is empty" >&2
    echo "DEBUG wds_dirs.clean.txt:" >&2
    cat wds_dirs.clean.txt >&2 || true
    exit 2
  fi
  '''
}