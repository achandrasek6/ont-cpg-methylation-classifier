process EVAL_WDS {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(wds_dir), path(ckpt)
    path eval_py
    path model_py
    val  dataset_id
    val  exp_name
    val  split
    val  batch_size
    val  num_workers
    val  calibrate
    val  calib_method
    val  calib_fit_split
    val  calib_bins

  output:
    tuple val(sample_id), path("${sample_id}.eval_metrics.json")

  publishDir "${params.outdir}/metrics/run=${params.run_id ?: 'run'}/sample=${sample_id}/exp=${exp_name}",
    mode: params.publish_mode, overwrite: true

  script:
  """
  set -euo pipefail

  echo "[EVAL] sample_id=${sample_id}"
  echo "[EVAL] wds_dir=${wds_dir}"
  echo "[EVAL] ckpt=${ckpt}"
  echo "[EVAL] split=${split} batch_size=${batch_size} num_workers=${num_workers}"
  echo "[EVAL] calibrate=${calibrate} calib_method=${calib_method} calib_fit_split=${calib_fit_split} calib_bins=${calib_bins}"

  # Show what's in the staged shard dir
  ls -lah "${wds_dir}" || true
  ls -lah "${wds_dir}"/${split}-*.tar || true

  # Clamp workers for eval split
  EVAL_SHARDS=\$(ls -1 "${wds_dir}"/${split}-*.tar 2>/dev/null | wc -l | tr -d ' ')
  NW=${num_workers}
  if [ "\${NW}" -gt 0 ] && [ "\${EVAL_SHARDS}" -gt 0 ] && [ "\${NW}" -gt "\${EVAL_SHARDS}" ]; then
    echo "[EVAL] Clamping eval num_workers \${NW} -> \${EVAL_SHARDS} (shards=\${EVAL_SHARDS})"
    NW=\${EVAL_SHARDS}
  fi

  # Decide calibration args (for eval script)
  CALIB_ARGS=""
  CALIB_FIT_ARGS=""

  if [ "${calib_method}" != "none" ]; then
    CALIB_ARGS="--calib_method ${calib_method}"
    CALIB_FIT_ARGS="--calib_fit_split ${calib_fit_split}"
  else
    # Back-compat: old boolean flag triggers affine_ls inside the script
    if [ "${calibrate}" = "true" ]; then
      CALIB_ARGS="--calibrate"
      CALIB_FIT_ARGS="--calib_fit_split ${calib_fit_split}"
    fi
  fi

  # Optional: clamp workers for calib-fit split too (only if calibration is enabled)
  if [ -n "\${CALIB_FIT_ARGS}" ]; then
    FIT_SHARDS=\$(ls -1 "${wds_dir}"/${calib_fit_split}-*.tar 2>/dev/null | wc -l | tr -d ' ')
    if [ "\${NW}" -gt 0 ] && [ "\${FIT_SHARDS}" -gt 0 ] && [ "\${NW}" -gt "\${FIT_SHARDS}" ]; then
      echo "[EVAL] Clamping calib-fit num_workers \${NW} -> \${FIT_SHARDS} (shards=\${FIT_SHARDS})"
      NW=\${FIT_SHARDS}
    fi
  fi

  # Build *local path* shard lists (status quo: wds_dir is a staged local directory)
  ls -1 "${wds_dir}"/${split}-*.tar 2>/dev/null > eval_shards.paths.txt || true
  if [ ! -s eval_shards.paths.txt ]; then
    echo "ERROR: No shards found for split=${split} under wds_dir=${wds_dir}" >&2
    exit 2
  fi

  CALIB_PATH_ARGS=""
  if [ -n "\${CALIB_FIT_ARGS}" ]; then
    ls -1 "${wds_dir}"/${calib_fit_split}-*.tar 2>/dev/null > calib_fit_shards.paths.txt || true
    if [ ! -s calib_fit_shards.paths.txt ]; then
      echo "ERROR: calib_fit_split=${calib_fit_split} shards not found but calibration enabled" >&2
      exit 2
    fi
    CALIB_PATH_ARGS="--calib_shards_txt calib_fit_shards.paths.txt"
  fi

  echo "[EVAL] eval shards: \$(wc -l < eval_shards.paths.txt || echo 0)"
  head -n 2 eval_shards.paths.txt || true
  if [ -n "\${CALIB_PATH_ARGS}" ]; then
    echo "[EVAL] calib-fit shards: \$(wc -l < calib_fit_shards.paths.txt || echo 0)"
    head -n 2 calib_fit_shards.paths.txt || true
  fi

  python3 "${eval_py}" \\
    --shards_txt "eval_shards.paths.txt" \\
    \${CALIB_PATH_ARGS} \\
    --ckpt "${ckpt}" \\
    --model_py "${model_py}" \\
    --split "${split}" \\
    --batch_size ${batch_size} \\
    --num_workers \${NW} \\
    \${CALIB_ARGS} \\
    \${CALIB_FIT_ARGS} \\
    --calib_bins ${calib_bins} \\
    --out_json "${sample_id}.eval_metrics.json"
  """
}