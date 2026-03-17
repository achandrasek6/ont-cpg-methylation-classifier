# ONT CpG Methylation Classifier

An end-to-end **epigenomics ML system** that learns CpG methylation (and related modification signals) directly from **Oxford Nanopore (ONT) raw signal**. The repo emphasizes both **scientific rigor** (truth alignment, coordinate correctness, leakage-safe evaluation, calibrated probabilities) and **production-grade engineering** (streaming datasets, reproducible runs, cloud-ready training), bringing together **raw-signal genomics, machine learning, and cloud workflow orchestration** in a reproducible end-to-end platform.

---

## 🧬 What works today (implemented + verified)

### ✅ End-to-end supervised training + evaluation loop (signal + k-mer → methylation fraction)

A complete, repeatable pipeline exists that:

1. Generates labeled examples at CpG coordinates (dense labels by design)
2. Adds k-mer sequence context centered at each site (default **k=9**)
3. Builds streamable datasets (**Parquet → WebDataset shards**)
4. Trains a joint **CNN (signal→tokens) + Transformer (signal tokens + k-mer tokens)** regressor
5. Evaluates on a held-out split and writes metrics JSON
6. Supports **calibration** (affine / temperature / isotonic), including **global calibration** for global training runs

This is the current “golden path” for both local smoke tests and AWS Batch scaling.

---

## 🧭 Pipeline modes

### Per-sample mode

* Build WDS per sample
* Train a model per sample
* Calibrate per sample per run (optional)

### Global mode (recommended for real training)

* Pool shards across samples
* Train **one global checkpoint**
* Fit **one global calibrator** on pooled `calib-*` shards
* Apply that calibrator during global evaluation

---

## ⚙️ Training / evaluation workflow

### Inputs

* `--samples_csv`: multi-sample FAST5-first input (`sample_id` + `fast5_dir`; optional `pod5` / `bam` / `bai`)
* `--coords_parquet`: candidate CpG coordinates for TRAIN sampling
* Optional **chrom holdout**:

  * `--val_coords_parquet`: validation coordinates (e.g., chr20)
  * future `--test_coords_parquet`: true test coordinates (e.g., chr22)
* `--ref_fa`: reference FASTA for alignment + k-mer context
* Dorado model selection + GPU device selection

### DAG (Nextflow)

#### 1) FAST5 → POD5 (optional per sample)

If `pod5` is not provided in `samples_csv`, FAST5 directories are converted to POD5.

#### 2) Basecall + align (optional per sample)

If `bam` / `bai` are not provided, Dorado basecall+align produces a coordinate-sorted BAM/BAI.

#### 3) Coordinate sampling + coverage filtering

* Coordinates are sampled from `coords_parquet` at the dataset level.
* Sites are filtered to those covered by each sample’s BAM using MAPQ / coverage gates.

#### 4) Split covered coordinates into parquet shards

Covered coordinate parquet files are split into smaller shard parquets before extraction.

This is now the key scaling mechanism for extraction on AWS Batch:

* each split task writes `part-*.parquet` files
* each extract task consumes **one shard parquet at a time**
* this avoids passing a multi-file list into the extractor and keeps extraction bounded per task

#### 5) Extract labeled raw-signal windows at CpG sites

For each covered CpG site, a fixed window from raw signal is extracted and labeled:

* `signal`: fixed-length window (for example **400 samples**)
* `kmer` / `kmer_ids`: k-mer context centered at the site (`k=9` by default)
* `meth_frac`: supervision target (methylation fraction)

Example labeled parquet fields:

* `chrom, pos0, read_id, strand, qpos, center_sample, meth_frac, kmer, kmer_ids, signal`

Extraction writes partitioned parquet parts into labeled output directories to avoid large monolithic outputs.

#### 6) Build WebDataset shards (train / val / calib)

Labeled parquet is converted to WebDataset `.tar` shards suitable for streaming.

Supported modes:

* random split mode using `--val_frac` and `--calib_frac`
* chrom-holdout mode using `--val_coords_parquet`, where the pipeline builds explicit `train-*` and `val-*` shards and still produces `calib-*` shards for calibration fitting

Shard contents:

* `signal.pth` (float32 `[window]`)
* `kmer.pth` (int64 `[k]`)
* `y.pth` (float32 scalar in `[0,1]`)

#### 7) Train joint CNN + k-mer Transformer

Trains a regressor on `meth_frac`.

Implemented features:

* WebDataset loading from local paths or S3 streaming (`pipe:aws s3 cp ... -`)
* worker clamping vs shard count to avoid dead workers / empty splits
* best-checkpoint saving (`joint_model_wds.best.pt`)
* canonical published checkpoint for global mode (`global.ckpt.pt`) mapped to the best-by-validation checkpoint

Outputs:

* `*.ckpt.pt` checkpoint
* `*.train.log` train / validation curves

#### 8) Evaluate + calibrate

Evaluation runs on an eval split (typically `val`) and writes a structured JSON report.

Metrics:

* **MSE**, **MAE**
* **ECE** with bin-level reliability summaries
* baseline comparison against a constant mean predictor

Calibration options:

* `affine_ls`: `p_cal = clip01(a*p + b)`
* `temp`: `p_cal = sigmoid(logit(p)/T)`
* `isotonic`: monotone PAV step function `p_cal = f(p)`

Calibration modes supported:

* **Per-eval fit + apply** using a `calib` split
* **Global apply-only** using one pooled `global.calib.json`

Outputs:

* `*.eval_metrics.json` with raw + calibrated metrics and reliability bins
* `global.calib.json` in global mode

---

## 🧪 Verified runs

### ✅ Full AWS Batch run: global training + global calibration + global eval

Successfully ran:

`CONVERT_POD5 → BASECALL_ALIGN_DORADO → COORDS_SAMPLE → COORDS_FILTER_COVERED → SPLIT_COORDS → EXTRACT_LABELED → COLLECT_LABELED_DIR → BUILD_WDS → COLLECT_WDS_SHARDS → TRAIN_WDS_GLOBAL → FIT_CALIB_GLOBAL → EVAL_WDS_GLOBAL`

This confirms that the sharded extraction path works end to end on AWS Batch at full training scale.

### Verified full run summary

Run ID:

`real_global_full_20260317_1605`

Training artifacts showed:

* **67 train shards**
* **55 validation shards**
* **17 calibration shards**
* best checkpoint selected at **epoch 3**

Training log:

* epoch 1: train MSE **0.094541**, val MSE **0.099796**
* epoch 2: train MSE **0.088203**, val MSE **0.098708**
* epoch 3: train MSE **0.085543**, val MSE **0.096966**
* epoch 4: train MSE **0.083277**, val MSE **0.097624**
* epoch 5: train MSE **0.081627**, val MSE **0.097892**

Best checkpoint:

* `joint_model_wds.best.pt`
* published downstream as `global.ckpt.pt`

Interpretation:

* the model is clearly learning
* validation improves through epoch 3
* mild overfitting begins after epoch 3
* the global eval path uses the **best checkpoint**, not the final epoch checkpoint

### Verified global calibration

Temperature scaling fit successfully with:

* `T = 1.2702101469039917`

Calibration artifact:

* `global.calib.json`

### Verified held-out global eval

Global eval metrics on held-out validation:

* **n = 19,086**
* **MSE = 0.0990235**
* **MAE = 0.2575294**
* **ECE = 0.0338500**

Calibrated global eval metrics:

* **MSE (calibrated) = 0.0977473**
* **MAE (calibrated) = 0.2657518**
* **ECE (calibrated) = 0.0173364**

Baseline comparison:

* baseline mean-predictor **MSE = 0.1111854**
* baseline mean-predictor **MAE = 0.2914718**

Practical interpretation:

* the model beats a constant mean baseline on both MSE and MAE
* temperature scaling substantially improves calibration quality
* in this run, calibration improves **ECE** and slightly improves **MSE**, while slightly worsening **MAE**

### Current artifact locations

Examples of verified outputs:

* `outputs/nf_runs/models/run=<RUN_ID>/model=global/exp=<EXP>/global.ckpt.pt`
* `outputs/nf_runs/models/run=<RUN_ID>/model=global/exp=<EXP>/global.train.log`
* `outputs/nf_runs/calibration/run=<RUN_ID>/method=<METHOD>/global.calib.json`
* eval metrics currently exist and are verified in the task workdir; publish-path cleanup for global eval is a remaining polish item

---

## 🌡️ Calibration notes (practical)

* Calibration often improves **ECE** materially even when MSE changes only modestly.
* With finite eval sizes, sparse tails can still show larger bin gaps in low-support confidence regions.
* Temperature scaling is currently the most stable default calibrator for global runs.
* Isotonic remains useful but may produce step artifacts in sparse regions.

---

## 📐 Evaluation definitions in this repo

* **Train**: used for gradient updates
* **Calib**: used only to fit calibrator parameters
* **Val**: used to report generalization and tune model choices
* **Test (future)**: true one-time holdout, for example chr22

Chrom-holdout support already exists for validation via `val_coords_parquet`.

---

## 📌 Current conclusions

* The end-to-end cloud pipeline is operational.
* Parquet sharding fixed the earlier extraction-path failure mode by ensuring one shard parquet is passed to each extract task.
* The global CNN + Transformer model is learning nontrivial methylation signal.
* Best-checkpoint selection is functioning correctly.
* Global temperature scaling improves calibration substantially.
* The remaining polish item is to publish `EVAL_WDS_GLOBAL` metrics into the structured outputs tree instead of leaving them only in the work directory.

---

## 🚀 Next steps

### 1) Publish global eval artifacts cleanly

* Update `EVAL_WDS_GLOBAL` so `GLOBAL.eval_metrics.json` is written to the structured run outputs path
* Optionally publish predictions or reliability summaries alongside metrics

### 2) Strengthen learning signal

* Increase `max_reads_per_site` where cost allows
* Increase total sites and sample coverage to stabilize tail calibration bins
* Add more explicit baseline / ablation reporting

### 3) Add a true test holdout branch

* add `test_coords_parquet`
* write `test-*` shards
* evaluate both `val` and `test` in a reproducible run summary

### 4) Systematic model tuning

* compare windows, read caps, shard sizing, and model hyperparameters
* track run-level metrics in a structured experiment table

---

## 🗂️ Repo layout

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

## ⚖️ License

Apache-2.0

---

## 🙏 Acknowledgments

* ONT POD5 ecosystem for raw-signal access
* WebDataset for shard-based streaming
* Dorado for basecall + align
* WGBS-derived methylation fraction supervision
