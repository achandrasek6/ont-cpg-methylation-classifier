process FIT_CALIB_GLOBAL {
  tag "global"
  label 'gpu'

  input:
    path calib_shards_txt
    path ckpt
    path fit_py
    path model_py
    val  calib_method
    val  batch_size
    val  num_workers

  output:
    path("global.calib.json")

    publishDir "${params.outdir}/calibration/run=${params.run_id ?: 'run'}/method=${calib_method}",
    mode: params.publish_mode, overwrite: false

  script:
  """
  set -euo pipefail

  echo "[CALIB] calib_shards_txt=${calib_shards_txt}"
  echo "[CALIB] ckpt=${ckpt}"
  echo "[CALIB] model_py=${model_py}"
  echo "[CALIB] method=${calib_method} batch_size=${batch_size} num_workers=${num_workers}"

  # Convert s3:// shard URIs into WebDataset-friendly pipe URLs
  awk 'NF{print "pipe:aws s3 cp " \$0 " -"}' "${calib_shards_txt}" > calib_shards.pipe.txt

  echo "[CALIB] pipe shards: \$(wc -l < calib_shards.pipe.txt || echo 0)"
  head -n 2 calib_shards.pipe.txt || true

  if [ ! -s calib_shards.pipe.txt ]; then
    echo "ERROR: calib_shards.pipe.txt is empty (no calib shards to fit on)" >&2
    echo "First lines of calib_shards_txt:" >&2
    head -n 50 "${calib_shards_txt}" >&2 || true
    exit 2
  fi

  python3 "${fit_py}" \\
    --shards_txt calib_shards.pipe.txt \\
    --ckpt "${ckpt}" \\
    --model_py "${model_py}" \\
    --calib_method ${calib_method} \\
    --batch_size ${batch_size} \\
    --num_workers ${num_workers} \\
    --out_json global.calib.json

  if [ ! -s global.calib.json ]; then
    echo "ERROR: global.calib.json not produced or empty" >&2
    ls -lah >&2 || true
    exit 2
  fi

  echo "[CALIB] wrote global.calib.json"
  """
}