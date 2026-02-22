# ONT CpG Methylation Classifier

An end-to-end **epigenomics ML system** that aims to learn CpG methylation (and related modification signals) directly from **Oxford Nanopore (ONT) raw signal**. The repo emphasizes both **scientific rigor** (truth alignment, coordinate correctness, leakage-safe evaluation, calibrated probabilities) and **production-grade engineering** (streaming datasets, reproducible runs, cloud-ready training).

---

## What works today (implemented + verified locally)

### ✅ End-to-end supervised training + validation loop (signal + k-mer → methylation fraction)

A complete, repeatable pipeline exists that:

1. Generates training examples at CpG coordinates (dense labels by design)
2. Adds k-mer sequence context (k=9)
3. Builds streamable datasets (Parquet → WebDataset shards)
4. Trains a joint CNN + Transformer model
5. Runs evaluation + calibration and writes metrics JSON

This loop is the current “golden path” for local smoke tests and forms the basis for cloud scaling.

---

## Training/validation workflow (local smoke test)

### Inputs

* `--samples_csv`: multi-sample FAST5-first input (sample_id + fast5_dir; optional pod5/bam/bai)
* `--coords_parquet`: candidate CpG coordinates to sample from
* `--ref_fa`: reference FASTA for alignment/sequence context
* Dorado model + GPU device selection

### Steps

#### 1) FAST5 → POD5 (optional per sample)

If `pod5` isn’t provided in the CSV, FAST5 directories are converted to POD5.

#### 2) Basecall + align (optional per sample)

If `bam/bai` aren’t provided, Dorado basecall+align is run to produce a coordinate-sorted BAM/BAI.

#### 3) Coordinate sampling + coverage filtering

* Coordinates are sampled from `coords_parquet` (dataset-level).
* Sites are filtered to those covered by each sample’s BAM (MAPQ/coverage gates).

#### 4) Extract labeled raw-signal windows at CpG sites

For each covered CpG site, a fixed window from raw signal is extracted and labeled:

* `signal`: fixed-length window (e.g. 400 samples)
* `kmer` / `kmer_ids`: k-mer context centered at the site (k=9)
* `meth_frac`: supervision target (methylation fraction)

Labeled Parquet schema (example):

* `chrom, pos0, read_id, strand, qpos, center_sample, meth_frac, kmer, kmer_ids, signal`

#### 5) Build WebDataset shards (train/val split)

Labeled Parquet is converted to WebDataset tar shards for streaming.

* `--val_frac` controls the holdout split (current smoke mode)

#### 6) Train joint CNN + k-mer Transformer

A regression model is trained on `meth_frac`.

Outputs:

* `*.ckpt.pt` checkpoint
* `*.train.log` train/val curves

#### 7) Evaluate + calibrate

Evaluation runs on the chosen split (currently `val`) and optionally calibrates predictions.

Outputs:

* `*.eval_metrics.json` containing `mse/mae/ece` + calibration parameters and binned reliability stats.

---

## Recent local run (verified)

A full Nextflow DAG has been executed end-to-end on two samples:

* `CONVERT_POD5 → BASECALL_ALIGN_DORADO → COORDS_SAMPLE → COORDS_FILTER_COVERED → EXTRACT_LABELED → BUILD_WDS → TRAIN_WDS → EVAL_WDS`

Example eval output (per-sample):

* `split=val, n=256`
* `mse ~ 0.10–0.11`, `mae ~ 0.26–0.28`
* calibration enabled (`affine_ls`) reduces `ece`

Important observation:

* A constant predictor baseline on the labeled Parquet is strong (e.g., MSE ~0.10). Smoke-test performance at this scale is often baseline-level and sensitive to label noise (e.g., low `max_reads_per_site`).

---

## Evaluation and “validation” in this repo

* Validation = evaluation on a held-out split not used for gradient updates.
* Current smoke setup uses `val_frac` to create a holdout shard split.
* Next step is leakage-safe chromosome/region splits (train/val/test by chrom or blocks).

---

## Next steps

### 1) Strengthen the local loop (still lightweight)

* Best-checkpoint-by-val (early stopping)
* Constant-baseline metrics written into eval JSON
* Modestly increase label quality for learning tests (raise `max_reads_per_site`)
* Move from random `val_frac` to chromosome-wise splits

### 2) Scale the same pipeline on AWS Batch

* Store inputs and outputs in S3 (workDir + publishDir)
* Run each process in containers (ECR)
* Increase sites × reads/site × samples for stable training curves

---

## Repo layout

```text
configs/     # dataset + train configs (YAML)
nf/          # Nextflow orchestration (end-to-end DAG)
infra/       # Terraform / cloud infra (Batch/ECR/S3/IAM)
scripts/     # runnable scripts (extraction, sharding, training, eval)
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
* Dorado for basecall + align with move table support
* WGBS-derived labels / methylation fraction supervision
