process TRAIN_WDS_GLOBAL {
  tag "global"
  label 'gpu'

  input:
    path train_shards_txt
    path val_shards_txt
    path train_py
    val  dataset_id
    val  exp_name
    val  epochs
    val  batch_size
    val  lr
    val  num_workers

  output:
    tuple path("global.ckpt.pt"), path("global.train.log")

  publishDir "${params.outdir}/models/run=${dataset_id}/model=global/exp=${exp_name}",
    mode: params.publish_mode, overwrite: false

  script:
  """
  set -euo pipefail

  # Convert s3:// shard URIs into WebDataset-friendly streaming URLs.
  awk 'NF{print "pipe:aws s3 cp " \$0 " -"}' "${train_shards_txt}" > train_shards.pipe.txt
  awk 'NF{print "pipe:aws s3 cp " \$0 " -"}' "${val_shards_txt}"   > val_shards.pipe.txt

  echo "[TRAIN] train shards (pipe): \$(wc -l < train_shards.pipe.txt || echo 0)"
  echo "[TRAIN] val   shards (pipe): \$(wc -l < val_shards.pipe.txt   || echo 0)"
  head -n 2 train_shards.pipe.txt || true
  head -n 2 val_shards.pipe.txt || true

  python3 "${train_py}" \\
    --train_shards "train_shards.pipe.txt" \\
    --val_shards   "val_shards.pipe.txt" \\
    --epochs ${epochs} \\
    --batch_size ${batch_size} \\
    --lr ${lr} \\
    --num_workers ${num_workers} \\
    --out_ckpt "joint_model_wds.pt" \\
    --out_ckpt_best "joint_model_wds.best.pt" \\
    | tee global.train.log

  if [ ! -f "joint_model_wds.best.pt" ]; then
    echo "ERROR: Expected best checkpoint not found: joint_model_wds.best.pt" >&2
    ls -lah >&2 || true
    exit 2
  fi

  # Publish best-by-val as the canonical checkpoint
  cp joint_model_wds.best.pt global.ckpt.pt
  """
}