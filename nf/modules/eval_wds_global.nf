process EVAL_WDS_GLOBAL {
  tag "${eval_id}"
  label 'cpu'

  input:
    tuple val(eval_id), path(val_shards_txt), path(ckpt), path(calib_json)

    path eval_py
    path model_py
    val  dataset_id
    val  exp_name
    val  eval_batch_size
    val  eval_num_workers
    val  calibrate
    val  calib_method
    val  calib_fit_split
    val  calib_bins

  output:
    tuple val(eval_id), path("${eval_id}.eval_metrics.json")

  publishDir "${params.outdir}/derived/global_eval",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  echo "[EVAL_GLOBAL] eval_id=${eval_id}"
  echo "[EVAL_GLOBAL] shards_txt=${val_shards_txt}"
  echo "[EVAL_GLOBAL] ckpt=${ckpt}"
  echo "[EVAL_GLOBAL] calib_json=${calib_json}"
  echo "[EVAL_GLOBAL] batch_size=${eval_batch_size} num_workers=${eval_num_workers}"

  mkdir -p wds_in

  # Copy shards from S3 -> local
  i=0
  while read -r uri; do
    [ -z "\$uri" ] && continue
    bn=\$(basename "\$uri")
    echo "[EVAL_GLOBAL] copy \$uri -> wds_in/\$bn"
    aws s3 cp "\$uri" "wds_in/\$bn"
    i=\$((i+1))
  done < "${val_shards_txt}"

  echo "[EVAL_GLOBAL] copied \$i shards"
  ls -lah wds_in | head -n 50

  # Build local shard list
  ls -1 wds_in/val-*.tar > val_local.txt
  echo "[EVAL_GLOBAL] local shards:"
  head -n 5 val_local.txt || true

  APPLY_GLOBAL_ARGS=""
  if [ -f "${calib_json}" ] && [ "\$(basename "${calib_json}")" != "NO_CALIB" ]; then
    echo "[EVAL_GLOBAL] Using GLOBAL calibrator: ${calib_json}"
    APPLY_GLOBAL_ARGS="--calib_params_json ${calib_json}"
  else
    echo "[EVAL_GLOBAL] No global calibrator provided."
  fi

  python3 "${eval_py}" \
    --shards_txt "val_local.txt" \
    \$APPLY_GLOBAL_ARGS \
    --ckpt "${ckpt}" \
    --model_py "${model_py}" \
    --split "val" \
    --batch_size ${eval_batch_size} \
    --num_workers ${eval_num_workers} \
    --calib_bins ${calib_bins} \
    --out_json "${eval_id}.eval_metrics.json"
  """
}