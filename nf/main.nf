nextflow.enable.dsl=2

/*
  Cloud-ready multi-sample pipeline (FAST5-first)

  Step 1: FAST5 -> POD5 (optional) -> Dorado BAM (optional)
  Step 2: coords_sample -> coords_filter_covered
  Step 3: split_coords -> extract_labeled (many shards per sample)
  Step 4: collect labeled outputs into per-sample directories
  Step 5: build_wds (supports file OR directory inputs)
  Step 6: train_wds (per-sample or global)
  Step 7: eval_wds (optional)

  Holdout validation:
    - If params.val_coords_parquet is provided:
        params.coords_parquet     = TRAIN coords (e.g. chr1-19)
        params.val_coords_parquet = VAL coords   (e.g. chr20)
      We run Steps 2–4 twice (train + val), then BUILD_WDS uses explicit holdout mode.

  Global calibration (global train mode only):
    - Pool calib shards across all samples
    - Fit ONE calibrator (global.calib.json) on pooled calib shards + global ckpt
    - Apply that calibrator to each sample's eval split (or do one global eval)
*/

include { CONVERT_POD5 }          from './modules/convert_pod5'
include { BASECALL_ALIGN_DORADO } from './modules/basecall_align'

// Aliases because these are called twice (train + val)
include { COORDS_SAMPLE         as COORDS_SAMPLE_TRAIN } from './modules/coords_sample'
include { COORDS_SAMPLE         as COORDS_SAMPLE_VAL   } from './modules/coords_sample'
include { COORDS_FILTER_COVERED as COORDS_FILTER_TRAIN } from './modules/coords_filter_covered'
include { COORDS_FILTER_COVERED as COORDS_FILTER_VAL   } from './modules/coords_filter_covered'
include { SPLIT_COORDS          as SPLIT_COORDS_TRAIN  } from './modules/split_coords'
include { SPLIT_COORDS          as SPLIT_COORDS_VAL    } from './modules/split_coords'

include { EXTRACT_LABELED_TRAIN } from './modules/extract_labeled_train'
include { EXTRACT_LABELED_VAL }   from './modules/extract_labeled_val'

include { BUILD_WDS }        from './modules/build_wds'
include { TRAIN_WDS }        from './modules/train_wds'
include { EVAL_WDS }         from './modules/eval_wds'
include { EVAL_WDS_GLOBAL }  from './modules/eval_wds_global'

include { COLLECT_WDS_SHARDS } from './modules/collect_wds_shards'
include { TRAIN_WDS_GLOBAL }   from './modules/train_wds_global'
include { FIT_CALIB_GLOBAL }   from './modules/fit_calib_global'

/*
  Collect many labeled outputs for a sample into one directory.

  Each EXTRACT_LABELED task emits a directory containing part-*.parquet.
  After groupTuple() we get:
    (sample_id, [dir1, dir2, dir3, ...])

  We do not interpolate that list into bash. We simply scan the staged
  task workdir for parquet parts and copy them into labeled_dir/.
*/
process COLLECT_LABELED_DIR_TRAIN {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(labeled_dirs)

  output:
    tuple val(sample_id), path("labeled_dir", type: 'dir')

  script:
  """
  set -euo pipefail
  mkdir -p labeled_dir
  echo "[COLLECT_LABELED_DIR_TRAIN] sample=${sample_id}"

  find . -maxdepth 5 -type f -name "part-*.parquet" -print0 \
    | xargs -0 -I{} cp -n {} labeled_dir/ || true

  find labeled_dir -maxdepth 1 -type f -name "*.parquet" -print -quit | grep -q .

  echo -n "[COLLECT_LABELED_DIR_TRAIN] wrote "
  find labeled_dir -maxdepth 1 -type f -name "*.parquet" | wc -l
  echo " parquet files"

  ls -lh labeled_dir | head -n 50
  """
}

process COLLECT_LABELED_DIR_VAL {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), path(labeled_dirs)

  output:
    tuple val(sample_id), path("labeled_dir", type: 'dir')

  script:
  """
  set -euo pipefail
  mkdir -p labeled_dir
  echo "[COLLECT_LABELED_DIR_VAL] sample=${sample_id}"

  find . -maxdepth 5 -type f -name "part-*.parquet" -print0 \
    | xargs -0 -I{} cp -n {} labeled_dir/ || true

  find labeled_dir -maxdepth 1 -type f -name "*.parquet" -print -quit | grep -q .

  echo -n "[COLLECT_LABELED_DIR_VAL] wrote "
  find labeled_dir -maxdepth 1 -type f -name "*.parquet" | wc -l
  echo " parquet files"

  ls -lh labeled_dir | head -n 50
  """
}

workflow {

  // -----------------------
  // Required params
  // -----------------------
  if( !params.samples_csv ) {
    log.error "This workflow requires --samples_csv"
    System.exit(1)
  }
  if( !params.outdir ) {
    log.error "Missing --outdir"
    System.exit(1)
  }
  if( !params.coords_parquet ) {
    log.error "Multi-sample mode: missing --coords_parquet"
    System.exit(1)
  }
  if( !params.split_coords_py ) {
    log.error "Missing --split_coords_py (required for sharded extraction)"
    System.exit(1)
  }
  if( !params.filter_coords_py ) {
    log.error "Missing --filter_coords_py"
    System.exit(1)
  }
  if( !params.extract_py ) {
    log.error "Missing --extract_py"
    System.exit(1)
  }

  def dataset_id = params.run_id ?: "dataset"
  def norm = { x -> x != null && x.toString().trim() ? x.toString().trim() : null }

  // -----------------------
  // Normalize script-path params once
  // -----------------------
  def split_coords_py = file(params.split_coords_py.toString())
  def filter_coords_py = file(params.filter_coords_py.toString())
  def extract_py = file(params.extract_py.toString())
  def build_wds_py = params.build_wds_py ? file(params.build_wds_py.toString()) : null
  def train_py = params.train_py ? file(params.train_py.toString()) : null
  def eval_py = params.eval_py ? file(params.eval_py.toString()) : null
  def fit_calib_py = file((params.fit_calib_py ?: "scripts/fit_calibrator_global.py").toString())
  def no_calib_file = file("${projectDir}/nf/assets/NO_CALIB")

  // Extraction knobs
  def shard_sites = params.coords_shard_sites ?: 5000
  def part_rows   = params.extract_part_rows ?: 50000

  // -----------------------
  // Read samples.csv
  // -----------------------
  Channel
    .fromPath(params.samples_csv)
    .splitCsv(header: true)
    .map { r ->
      def sid = norm(r.sample_id)
      if( !sid ) {
        throw new IllegalArgumentException("samples_csv: missing sample_id")
      }

      tuple(
        sid,
        norm(r.fast5_dir),
        norm(r.pod5),
        norm(r.bam),
        norm(r.bai),
        norm(r.ref_fa) ?: norm(params.ref_fa),
        norm(r.dorado_model) ?: norm(params.dorado_model)
      )
    }
    .set { ch_rows }

  // -----------------------
  // Step 1a: POD5 per sample
  // -----------------------
  def ch_pod5_direct = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> pod5 }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> tuple(sid, file(pod5.toString())) }

  def ch_fast5 = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> !pod5 }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model ->
      if( !fast5_dir ) {
        throw new IllegalArgumentException("samples_csv: sample ${sid} must provide either pod5 or fast5_dir")
      }
      tuple(sid, fast5_dir)
    }

  def ch_pod5_from_fast5 = CONVERT_POD5(ch_fast5)
  def ch_pod5 = ch_pod5_direct.mix(ch_pod5_from_fast5)

  ch_pod5.view { t -> "[${dataset_id}] POD5 -> ${t[0]} -> ${t[1]}" }

  // -----------------------
  // Step 1b: BAM/BAI per sample
  // -----------------------
  def ch_bam_direct = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> bam && bai }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model ->
      tuple(sid, file(bam.toString()), file(bai.toString()))
    }

  def ch_need_dorado = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> !(bam && bai) }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model ->
      if( !ref_fa ) {
        throw new IllegalArgumentException("samples_csv: sample ${sid} missing ref_fa (and no global --ref_fa)")
      }
      if( !dorado_model ) {
        throw new IllegalArgumentException("samples_csv: sample ${sid} missing dorado_model (and no global --dorado_model)")
      }
      tuple(sid, file(ref_fa.toString()), dorado_model)
    }

  def ch_pod5_for_dorado = ch_pod5
    .join(ch_need_dorado)
    .map { sid, pod5_path, ref_fa, dorado_model -> tuple(sid, pod5_path, ref_fa, dorado_model) }

  def ch_bam_from_dorado = BASECALL_ALIGN_DORADO(ch_pod5_for_dorado)
  def ch_bam = ch_bam_direct.mix(ch_bam_from_dorado)

  ch_bam.view { t -> "[${dataset_id}] BAM/BAI -> ${t[0]} -> ${t[1]} / ${t[2]}" }

  // Utility channel
  def ch_sample_ids = ch_rows
    .map { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> sid }
    .distinct()

  // -----------------------------
  // Step 2: coords_sample -> coords_filter_covered (TRAIN)
  // -----------------------------
  def ch_coords_sample_train = COORDS_SAMPLE_TRAIN(
    Channel.value(file(params.coords_parquet.toString())),
    Channel.value(dataset_id),
    Channel.value(params.coords_sample_n)
  )

  def ch_coords_sample_train_by_sample = ch_sample_ids
    .combine(ch_coords_sample_train)
    .map { sid, coords_path -> tuple(sid, coords_path) }

  def ch_coords_cov_in_train = ch_coords_sample_train_by_sample
    .join(ch_bam)
    .map { sid, coords_path, bam_path, bai_path -> tuple(sid, coords_path, bam_path, bai_path) }

  ch_coords_cov_in_train.view { t -> "COORDS_COV_TRAIN_IN -> ${t[0]}" }

  def ch_coords_cov_train = COORDS_FILTER_TRAIN(
    ch_coords_cov_in_train,
    Channel.value(filter_coords_py),
    Channel.value(params.min_mapq),
    Channel.value(params.max_sites)
  )

  ch_coords_cov_train.view { t -> "COORDS_COVERED_TRAIN -> ${t[0]} -> ${t[1]}" }

  // -----------------------------
  // Optional VAL coords_sample (chr holdout)
  // -----------------------------
  def use_holdout_val = params.val_coords_parquet ? true : false
  def ch_coords_cov_val = Channel.empty()

  if( use_holdout_val ) {

    def ch_coords_sample_val = COORDS_SAMPLE_VAL(
      Channel.value(file(params.val_coords_parquet.toString())),
      Channel.value("${dataset_id}.val"),
      Channel.value(params.val_sample_n)
    )

    def ch_coords_sample_val_by_sample = ch_sample_ids
      .combine(ch_coords_sample_val)
      .map { sid, coords_path -> tuple(sid, coords_path) }

    def ch_coords_cov_in_val = ch_coords_sample_val_by_sample
      .join(ch_bam)
      .map { sid, coords_path, bam_path, bai_path -> tuple(sid, coords_path, bam_path, bai_path) }

    ch_coords_cov_in_val.view { t -> "COORDS_COV_VAL_IN -> ${t[0]}" }

    ch_coords_cov_val = COORDS_FILTER_VAL(
      ch_coords_cov_in_val,
      Channel.value(filter_coords_py),
      Channel.value(params.min_mapq),
      Channel.value(params.max_sites)
    )

    ch_coords_cov_val.view { t -> "COORDS_COVERED_VAL -> ${t[0]} -> ${t[1]}" }
  }

  // -----------------------------
  // Step 3: split_coords -> extract labeled (TRAIN)
  // -----------------------------
def ch_coords_shards_train_dir = SPLIT_COORDS_TRAIN(
  ch_coords_cov_train,
  Channel.value(split_coords_py),
  Channel.value(shard_sites)
)

def ch_coords_shards_train = ch_coords_shards_train_dir
  .flatMap { sid, shard_dir, bam_path, bai_path ->
    shard_dir.listFiles()
      .findAll { it.name.endsWith('.parquet') }
      .sort { a, b -> a.name <=> b.name }
      .collect { shard_file -> tuple(sid, shard_file, bam_path, bai_path) }
  }

def ch_extract_in_train = ch_coords_shards_train
  .join(ch_pod5)
  .map { sid, coords_shard, bam_path, bai_path, pod5_path ->
    tuple(sid, coords_shard, bam_path, bai_path, pod5_path)
  }

  ch_extract_in_train.view { t ->
    "EXTRACT_TRAIN_IN -> ${t[0]} -> coords=${t[1]} bam=${t[2]} pod5=${t[4]}"
  }

  def ch_labeled_train = EXTRACT_LABELED_TRAIN(
    ch_extract_in_train,
    Channel.value(extract_py),
    Channel.value(params.window),
    Channel.value(params.max_reads_per_site),
    Channel.value(params.min_mapq),
    Channel.value(params.limit_sites),
    Channel.value(params.k),
    Channel.value(part_rows)
  )

  ch_labeled_train.view { "LABELED_TRAIN_SHARD ${it[0]} => ${it[1]}" }

  // -----------------------------
  // Step 3b: split_coords -> extract labeled (VAL)
  // -----------------------------
  def ch_labeled_val = Channel.empty()

  if( use_holdout_val ) {

def ch_coords_shards_val_dir = SPLIT_COORDS_VAL(
  ch_coords_cov_val,
  Channel.value(split_coords_py),
  Channel.value(shard_sites)
)

def ch_coords_shards_val = ch_coords_shards_val_dir
  .flatMap { sid, shard_dir, bam_path, bai_path ->
    shard_dir.listFiles()
      .findAll { it.name.endsWith('.parquet') }
      .sort { a, b -> a.name <=> b.name }
      .collect { shard_file -> tuple(sid, shard_file, bam_path, bai_path) }
  }

def ch_extract_in_val = ch_coords_shards_val
  .join(ch_pod5)
  .map { sid, coords_shard, bam_path, bai_path, pod5_path ->
    tuple(sid, coords_shard, bam_path, bai_path, pod5_path)
  }

    ch_extract_in_val.view { t ->
      "EXTRACT_VAL_IN -> ${t[0]} -> coords=${t[1]} bam=${t[2]} pod5=${t[4]}"
    }

    ch_labeled_val = EXTRACT_LABELED_VAL(
      ch_extract_in_val,
      Channel.value(extract_py),
      Channel.value(params.window),
      Channel.value(params.max_reads_per_site),
      Channel.value(params.min_mapq),
      Channel.value(params.limit_sites),
      Channel.value(params.k),
      Channel.value(part_rows)
    )

    ch_labeled_val.view { "LABELED_VAL_SHARD ${it[0]} => ${it[1]}" }
  }

  // -----------------------------
  // Step 4: collect labeled outputs into per-sample directories
  // -----------------------------
  def ch_labeled_train_group = ch_labeled_train.groupTuple()
  def ch_labeled_train_dir   = COLLECT_LABELED_DIR_TRAIN(ch_labeled_train_group)

  def ch_labeled_val_dir = Channel.empty()
  if( use_holdout_val ) {
    def ch_labeled_val_group = ch_labeled_val.groupTuple()
    ch_labeled_val_dir = COLLECT_LABELED_DIR_VAL(ch_labeled_val_group)
  }

  ch_labeled_train_dir.view { "LABELED_TRAIN_DIR ${it[0]} => ${it[1]}" }
  if( use_holdout_val ) {
    ch_labeled_val_dir.view { "LABELED_VAL_DIR ${it[0]} => ${it[1]}" }
  }

  // -----------------------------
  // Step 5: build WDS
  // -----------------------------
  if( build_wds_py ) {

    def ch_labeled_pair

    if( use_holdout_val ) {
      ch_labeled_pair = ch_labeled_train_dir
        .join(ch_labeled_val_dir)
        .map { sid, train_dir, val_dir -> tuple(sid, train_dir, val_dir, true) }
    } else {
      ch_labeled_pair = ch_labeled_train_dir
        .map { sid, train_dir -> tuple(sid, train_dir, train_dir, false) }
    }

    def ch_wds = BUILD_WDS(
      ch_labeled_pair,
      Channel.value(build_wds_py),
      Channel.value(dataset_id),
      Channel.value(params.wds_shard_size),
      Channel.value(params.val_frac),
      Channel.value(params.calib_frac),
      Channel.value(params.seed),
      Channel.value(params.max_rows_train),
      Channel.value(params.max_rows_val),
      Channel.value(params.stratify_bins)
    )

    ch_wds.view { "WDS ${it[0]} => ${it[1]}" }

    // -----------------------------
    // Step 6: train
    // -----------------------------
    def ch_train = null
    def ch_train_global = null
    def ch_shards = null

    if( train_py ) {

      def train_mode = (params.train_mode ?: 'per_sample')

      if( train_mode == 'global' ) {

        def ch_wds_dirs_txt = ch_wds
          .map { sid, wds_dir -> wds_dir.toString() }
          .collectFile(name: "wds_dirs.txt", newLine: true)

        ch_shards = COLLECT_WDS_SHARDS(ch_wds_dirs_txt)

        def ch_train_shards_txt = ch_shards.map { train_txt, val_txt, calib_txt -> train_txt }
        def ch_val_shards_txt   = ch_shards.map { train_txt, val_txt, calib_txt -> val_txt }

        ch_train_global = TRAIN_WDS_GLOBAL(
          ch_train_shards_txt,
          ch_val_shards_txt,
          Channel.value(train_py),
          Channel.value(dataset_id),
          Channel.value(params.exp_name),
          Channel.value(params.epochs),
          Channel.value(params.batch_size),
          Channel.value(params.lr),
          Channel.value(params.num_workers)
        )

        ch_train_global.view { "TRAIN global => ckpt=${it[0]} log=${it[1]}" }

      } else {

        ch_train = TRAIN_WDS(
          ch_wds,
          Channel.value(train_py),
          Channel.value(dataset_id),
          Channel.value(params.exp_name),
          Channel.value(params.epochs),
          Channel.value(params.batch_size),
          Channel.value(params.lr),
          Channel.value(params.num_workers)
        )

        ch_train.view { "TRAIN ${it[0]} => ckpt=${it[1]} log=${it[2]}" }
      }
    }

    // -----------------------------
    // Step 7: eval
    // -----------------------------
    if( eval_py ) {

      def train_mode = (params.train_mode ?: 'per_sample')
      def eval_mode  = (params.eval_mode ?: 'per_sample')

      if( train_mode == 'global' ) {

        if( !ch_train_global ) {
          throw new IllegalArgumentException("train_mode=global but ch_train_global is null (did TRAIN_WDS_GLOBAL run?)")
        }

        if( !ch_shards ) {
          throw new IllegalArgumentException("train_mode=global but ch_shards is null (did COLLECT_WDS_SHARDS run?)")
        }

        def ch_global_ckpt = ch_train_global.map { ckpt, log -> ckpt }

        def ch_val_shards_txt   = ch_shards.map { train_txt, val_txt, calib_txt -> val_txt }
        def ch_calib_shards_txt = ch_shards.map { train_txt, val_txt, calib_txt -> calib_txt }

        def ch_global_calib = FIT_CALIB_GLOBAL(
          ch_calib_shards_txt,
          ch_global_ckpt,
          Channel.value(fit_calib_py),
          Channel.value(train_py),
          Channel.value(params.calib_method),
          Channel.value(params.eval_batch_size),
          Channel.value(params.eval_num_workers)
        )

        if( eval_mode == 'global' ) {

          def ch_eval_global_in = Channel.value('GLOBAL')
            .combine(ch_val_shards_txt)
            .combine(ch_global_ckpt)
            .combine(ch_global_calib)
            .map { eval_id, val_txt, ckpt, calib_json -> tuple(eval_id, val_txt, ckpt, calib_json) }

          def ch_eval = EVAL_WDS_GLOBAL(
            ch_eval_global_in,
            Channel.value(eval_py),
            Channel.value(train_py),
            Channel.value(dataset_id),
            Channel.value(params.exp_name),
            Channel.value(params.eval_batch_size),
            Channel.value(params.eval_num_workers),
            Channel.value(params.calibrate),
            Channel.value(params.calib_method),
            Channel.value(params.calib_fit_split),
            Channel.value(params.calib_bins)
          )

          ch_eval.view { "EVAL_GLOBAL ${it[0]} => ${it[1]}" }

        } else {

          def ch_wds_with_ckpt = ch_wds.combine(ch_global_ckpt)
          def ch_wds_with_ckpt_and_calib = ch_wds_with_ckpt.combine(ch_global_calib)

          def ch_eval_in = ch_wds_with_ckpt_and_calib
            .map { sid, wds_dir, ckpt, calib_json -> tuple(sid, wds_dir, ckpt, calib_json) }

          def ch_eval = EVAL_WDS(
            ch_eval_in,
            Channel.value(eval_py),
            Channel.value(train_py),
            Channel.value(dataset_id),
            Channel.value(params.exp_name),
            Channel.value(params.eval_split),
            Channel.value(params.eval_batch_size),
            Channel.value(params.eval_num_workers),
            Channel.value(params.calibrate),
            Channel.value(params.calib_method),
            Channel.value(params.calib_fit_split),
            Channel.value(params.calib_bins)
          )

          ch_eval.view { "EVAL ${it[0]} => ${it[1]}" }
        }

      } else {

        if( !ch_train ) {
          throw new IllegalArgumentException("train_mode=per_sample but ch_train is null (did TRAIN_WDS run?)")
        }

        def ch_no_calib = Channel.value(no_calib_file)
        def ch_ckpt = ch_train.map { sid, ckpt, log -> tuple(sid, ckpt) }

        def ch_trip = ch_wds
          .join(ch_ckpt)
          .map { sid, wds_dir, ckpt -> tuple(sid, wds_dir, ckpt) }

        def ch_eval_in = ch_trip
          .combine(ch_no_calib)
          .map { trip, no_calib -> tuple(trip[0], trip[1], trip[2], no_calib) }

        def ch_eval = EVAL_WDS(
          ch_eval_in,
          Channel.value(eval_py),
          Channel.value(train_py),
          Channel.value(dataset_id),
          Channel.value(params.exp_name),
          Channel.value(params.eval_split),
          Channel.value(params.eval_batch_size),
          Channel.value(params.eval_num_workers),
          Channel.value(params.calibrate),
          Channel.value(params.calib_method),
          Channel.value(params.calib_fit_split),
          Channel.value(params.calib_bins)
        )

        ch_eval.view { "EVAL ${it[0]} => ${it[1]}" }
      }
    }
  }

  return
}