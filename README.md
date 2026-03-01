# ONT CpG Methylation Classifier

An end-to-end **epigenomics ML system** that learns CpG methylation (and related modification signals) directly from **Oxford Nanopore (ONT) raw signal**. The repo emphasizes both **scientific rigor** (truth alignment, coordinate correctness, leakage-safe evaluation, calibrated probabilities) and **production-grade engineering** (streaming datasets, reproducible runs, cloud-ready training).

---

## What works today (implemented + verified)

### ✅ End-to-end supervised training + evaluation loop (signal + k‑mer → methylation fraction)

A complete, repeatable pipeline exists that:

1. Generates labeled examples at CpG coordinates (dense labels by design)
2. Adds k‑mer sequence context centered at each site (default **k=9**)
3. Builds streamable datasets (**Parquet → WebDataset shards**)
4. Trains a joint **CNN (signal→tokens) + Transformer (signal tokens + k‑mer tokens)** regressor
5. Evaluates on a held-out split and writes metrics JSON
6. Supports **calibration** (affine / temperature / isotonic), including **global calibration** for global training runs

This is the current “golden path” for both local smoke tests and AWS Batch scaling.

---

## Pipeline modes

### Per-sample mode

* Build WDS per sample
* Train a model per sample
* Calibrate per sample per run (optional)

### Global mode (recommended for real training)

* Pool shards across samples
* Train **one global checkpoint**
* Fit **one global calibrator** on pooled `calib-*` shards
* Apply that calibrator to each sample’s eval split

---

## Training / evaluation workflow

### Inputs

* `--samples_csv`: multi-sample FAST5-first input (`sample_id` + `fast5_dir`; optional `pod5` / `bam` / `bai`)
* `--coords_parquet`: candidate CpG coordinates for TRAIN sampling
* Optional **chrom holdout**:

  * `--val_coords_parquet`: validation coordinates (e.g., chr20)
  * (future) `--test_coords_parquet`: true test coordinates (e.g., chr22)
* `--ref_fa`: reference FASTA for alignment + k-mer context
* Dorado model selection + GPU device selection

### DAG (Nextflow)

#### 1) FAST5 → POD5 (optional per sample)

If `pod5` isn’t provided in `samples_csv`, FAST5 directories are converted to POD5.

#### 2) Basecall + align (optional per sample)

If `bam/bai` aren’t provided, Dorado basecall+align produces a coordinate-sorted BAM/BAI.

#### 3) Coordinate sampling + coverage filtering

* Coordinates are sampled from `coords_parquet` (dataset-level).
* Sites are filtered to those covered by each sample’s BAM (MAPQ/coverage gates).

#### 4) Extract labeled raw-signal windows at CpG sites

For each covered CpG site, a fixed window from raw signal is extracted and labeled:

* `signal`: fixed-length window (e.g., **400 samples**)
* `kmer` / `kmer_ids`: k-mer context centered at the site (k=9)
* `meth_frac`: supervision target (methylation fraction)

Example labeled Parquet fields:

* `chrom, pos0, read_id, strand, qpos, center_sample, meth_frac, kmer, kmer_ids, signal`

#### 5) Build WebDataset shards (train/val/calib)

Labeled Parquet is converted to WebDataset `.tar` shards suitable for streaming.

* Random split mode: `--val_frac` + `--calib_frac`
* Chrom-holdout mode: provide `--val_coords_parquet` and the pipeline builds explicit `train-*` and `val-*` shards (and still produces `calib-*` for calibration fitting)

Shard contents (per sample):

* `signal.pth` (float32 [window])
* `kmer.pth` (int64 [k])
* `y.pth` (float32 scalar in [0,1])

#### 6) Train joint CNN + k‑mer Transformer

Trains a regressor on `meth_frac`.

Key features implemented:

* WebDataset loading from local paths or S3 streaming (`pipe:aws s3 cp ... -`)
* Worker clamping vs shard count to avoid dead workers / empty splits
* Best-checkpoint saving (e.g., `joint_model_wds.best.pt`) when enabled

Outputs:

* `*.ckpt.pt` checkpoint
* `*.train.log` train/val curves

#### 7) Evaluate + calibrate

Evaluation runs on an eval split (typically `val`) and writes a structured JSON report.

Metrics:

* **MSE**, **MAE**
* **ECE** with bin stats (reliability table)

Calibration options:

* `affine_ls`: `p_cal = clip01(a*p + b)`
* `temp`: `p_cal = sigmoid(logit(p)/T)`
* `isotonic`: monotone PAV step function `p_cal = f(p)`

Calibration modes supported:

* **Per-eval fit+apply** (fit on `calib` split and apply to `val`)
* **Global apply-only** (fit once → `global.calib.json`, then apply everywhere)

Outputs:

* `*.eval_metrics.json` with raw + calibrated metrics, plus bin summaries
* `global.calib.json` (global mode)

---

## Verified runs

### ✅ Full AWS Batch run: global training + global calibration

Successfully ran:

`CONVERT_POD5 → BASECALL_ALIGN_DORADO → COORDS_SAMPLE → COORDS_FILTER_COVERED → EXTRACT_LABELED → BUILD_WDS → COLLECT_WDS_SHARDS → TRAIN_WDS_GLOBAL → FIT_CALIB_GLOBAL → EVAL_WDS`

Artifacts written to S3 under:

* `outputs/nf_runs/models/run=<RUN_ID>/model=global/exp=<EXP>/global.ckpt.pt`
* `outputs/nf_runs/calibration/run=<RUN_ID>/method=<METHOD>/global.calib.json`
* `outputs/nf_runs/metrics/run=<RUN_ID>/sample=<SAMPLE>/exp=<EXP>/*.eval_metrics.json`

---

## Calibration notes (practical)

* Calibration **often improves ECE** materially even when MSE changes only slightly.
* With small eval sizes, **sparse tails** can show large bin gaps (high-confidence bins with `n<50`).

  * This is expected to diminish as you scale sites × reads/site × samples.
* Temperature scaling (global `T`) is the most stable “default” calibrator.
* Isotonic can be strong but may exhibit step artifacts in very sparse regions.

---

## Evaluation definitions in this repo

* **Train**: used for gradient updates
* **Calib**: used only to fit calibrator parameters (no gradients)
* **Val**: used to report generalization and tune model choices
* **Test (future)**: true holdout used once at the end (e.g., chr22)

Chrom-holdout support exists for validation via `val_coords_parquet`.

---

## Next steps

### 1) Strengthen learning signal (still lightweight)

* Increase `max_reads_per_site` for higher-quality methylation fraction labels
* Increase sites and/or samples per run to stabilize calibration tails
* Add explicit baseline metrics to eval JSON (e.g., constant predictor)

### 2) True holdout testing (chr22)

* Add `test_coords_parquet` path and a third branch (TRAIN/VAL/TEST)
* Update `BUILD_WDS` to write `test-*` shards
* Add `eval_splits=val,test` to evaluate both splits per run

### 3) Scale up on AWS Batch

* Increase shards + throughput (larger `coords_sample_n`, more samples)
* Track run-level metadata and compare runs systematically

---

## Repo layout

```text
configs/     # dataset + train configs (YAML)
nf/          # Nextflow orchestration (end-to-end DAG)
infra/       # Terraform / cloud infra (Batch/ECR/S3/IAM)
scripts/     # runnable scripts (extraction, sharding, training, eval, calibration)
src/         # python package
outputs/     # run artifacts (gitignored)
data/        # local datasets (gitignored)
```

---

## License

Apache-2.0

---

## Acknowledgments

* ONT POD5 ecosystem for raw-signal access
* WebDataset for shard-based streaming
* Dorado for basecall + align
* WGBS-derived methylation fraction supervision
