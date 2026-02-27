nextflow.enable.dsl=2

/*
  Cloud-ready multi-sample pipeline (FAST5-first):

    Step 1: FAST5 -> POD5 (optional) -> Dorado BAM (optional)
    Step 2: coords_sample -> coords_filter_covered
    Step 3: extract_labeled (coords_covered + BAM/BAI + POD5 -> labeled parquet)
    Step 4: build_wds
    Step 5: train_wds (per-sample or global)
    Step 6: eval_wds (optional)

  NEW (chr-holdout validation):
    - If params.val_coords_parquet is provided:
        params.coords_parquet     = TRAIN coords (e.g., chr1-19)
        params.val_coords_parquet = VAL coords   (e.g., chr20)
      We run Steps 2–3 twice (train + val) and pass both labeled parquets to BUILD_WDS.

  Invariants:
    - Do NOT consume BAM twice.
    - COORDS_FILTER_COVERED input:  (sample_id, coords_sample, bam_sorted, bam_bai)
      output:                        (sample_id, coords_covered, bam_sorted, bam_bai)
    - EXTRACT_LABELED input:         (sample_id, coords_covered, bam_sorted, bam_bai, pod5)
*/

include { CONVERT_POD5 }          from './modules/convert_pod5'
include { BASECALL_ALIGN_DORADO } from './modules/basecall_align'

// Aliases because we call these modules twice (train + val)
include { COORDS_SAMPLE         as COORDS_SAMPLE_TRAIN }         from './modules/coords_sample'
include { COORDS_SAMPLE         as COORDS_SAMPLE_VAL   }         from './modules/coords_sample'
include { COORDS_FILTER_COVERED as COORDS_FILTER_TRAIN }         from './modules/coords_filter_covered'
include { COORDS_FILTER_COVERED as COORDS_FILTER_VAL   }         from './modules/coords_filter_covered'
include { EXTRACT_LABELED       as EXTRACT_LABELED_TRAIN }       from './modules/extract_labeled'
include { EXTRACT_LABELED       as EXTRACT_LABELED_VAL   }       from './modules/extract_labeled'

include { BUILD_WDS }             from './modules/build_wds'
include { TRAIN_WDS }             from './modules/train_wds'
include { EVAL_WDS }              from './modules/eval_wds'

include { COLLECT_WDS_SHARDS } from './modules/collect_wds_shards'
include { TRAIN_WDS_GLOBAL }   from './modules/train_wds_global'

workflow {

  if( !params.samples_csv ) {
    log.error "This workflow requires --samples_csv"
    System.exit(1)
  }

  if( !params.outdir ) {
    log.error "Missing --outdir"
    System.exit(1)
  }

  def dataset_id = params.run_id ?: "dataset"
  def norm = { x -> x != null && x.toString().trim() ? x.toString().trim() : null }

  // Expect CSV columns:
  // sample_id,fast5_dir,pod5,ref_fa,dorado_model,(optional) bam,bai
  Channel
    .fromPath(params.samples_csv)
    .splitCsv(header:true)
    .map { r ->
      def sid = norm(r.sample_id)
      if( !sid ) throw new IllegalArgumentException("samples_csv: missing sample_id")
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
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> tuple(sid, file(pod5)) }

  def ch_fast5 = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> !pod5 }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model ->
      if( !fast5_dir ) throw new IllegalArgumentException("samples_csv: sample ${sid} must provide either pod5 or fast5_dir")
      tuple(sid, file(fast5_dir))
    }

  def ch_pod5_from_fast5 = CONVERT_POD5(ch_fast5)
  def ch_pod5 = ch_pod5_direct.mix(ch_pod5_from_fast5)

  ch_pod5.view { t -> "[${dataset_id}] POD5 -> ${t[0]} -> ${t[1]}" }

  // -----------------------
  // Step 1b: BAM/BAI per sample
  // -----------------------
  def ch_bam_direct = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> bam && bai }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> tuple(sid, file(bam), file(bai)) }

  def ch_need_dorado = ch_rows
    .filter { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> !(bam && bai) }
    .map    { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model ->
      if( !ref_fa ) throw new IllegalArgumentException("samples_csv: sample ${sid} missing ref_fa (and no global --ref_fa)")
      if( !dorado_model ) throw new IllegalArgumentException("samples_csv: sample ${sid} missing dorado_model (and no global --dorado_model)")
      tuple(sid, file(ref_fa), dorado_model)
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
  // Step 2: coords_sample -> coords_filter_covered
  // -----------------------------
  if( !params.coords_parquet ) {
    log.error "Multi-sample mode: missing --coords_parquet"
    System.exit(1)
  }

  // TRAIN coords_sample (chr1-19)
  def ch_coords_sample_train = COORDS_SAMPLE_TRAIN(
    Channel.value(file(params.coords_parquet)),
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
    Channel.value(file(params.filter_coords_py)),
    Channel.value(params.min_mapq),
    Channel.value(params.max_sites)
  )

  ch_coords_cov_train.view { t -> "COORDS_COVERED_TRAIN -> ${t[0]} -> ${t[1]}" }

  // Optional VAL coords_sample (chr20)
  def use_holdout_val = params.val_coords_parquet ? true : false
  def ch_coords_cov_val = Channel.empty()

  if( use_holdout_val ) {

    def ch_coords_sample_val = COORDS_SAMPLE_VAL(
      Channel.value(file(params.val_coords_parquet)),
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
      Channel.value(file(params.filter_coords_py)),
      Channel.value(params.min_mapq),
      Channel.value(params.max_sites)
    )

    ch_coords_cov_val.view { t -> "COORDS_COVERED_VAL -> ${t[0]} -> ${t[1]}" }
  }

  // -----------------------------
  // Step 3: extract labeled (train + optional val)
  // -----------------------------
  def ch_extract_in_train = ch_coords_cov_train
    .join(ch_pod5)
    .map { sid, coords_cov, bam_path, bai_path, pod5_path ->
      tuple(sid, coords_cov, bam_path, bai_path, pod5_path)
    }

  ch_extract_in_train.view { t -> "EXTRACT_TRAIN_IN -> ${t[0]} -> coords=${t[1]} bam=${t[2]} pod5=${t[4]}" }

  def ch_labeled_train = EXTRACT_LABELED_TRAIN(
    ch_extract_in_train,
    Channel.value(file(params.extract_py)),
    Channel.value(params.window),
    Channel.value(params.max_reads_per_site),
    Channel.value(params.min_mapq),
    Channel.value(params.limit_sites),
    Channel.value(params.k)
  )

  ch_labeled_train.view { "LABELED_TRAIN ${it[0]} => ${it[1]}" }

  def ch_labeled_val = Channel.empty()

  if( use_holdout_val ) {
    def ch_extract_in_val = ch_coords_cov_val
      .join(ch_pod5)
      .map { sid, coords_cov, bam_path, bai_path, pod5_path ->
        tuple(sid, coords_cov, bam_path, bai_path, pod5_path)
      }

    ch_extract_in_val.view { t -> "EXTRACT_VAL_IN -> ${t[0]} -> coords=${t[1]} bam=${t[2]} pod5=${t[4]}" }

    ch_labeled_val = EXTRACT_LABELED_VAL(
      ch_extract_in_val,
      Channel.value(file(params.extract_py)),
      Channel.value(params.window),
      Channel.value(params.max_reads_per_site),
      Channel.value(params.min_mapq),
      Channel.value(params.limit_sites),
      Channel.value(params.k)
    )

    ch_labeled_val.view { "LABELED_VAL ${it[0]} => ${it[1]}" }
  }

  // -----------------------------
  // Step 4: build wds (+ Step 5/6 inside)
  // -----------------------------
  if( params.build_wds_py ) {

    def ch_labeled_pair

    if( use_holdout_val ) {
      // join train+val labeled by sample_id
      ch_labeled_pair = ch_labeled_train
        .join(ch_labeled_val)
        .map { sid, train_parquet, val_parquet -> tuple(sid, train_parquet, val_parquet, true) }
    }
    else {
      // legacy mode: stage train parquet in both slots; flag selects legacy path
      ch_labeled_pair = ch_labeled_train
        .map { sid, train_parquet -> tuple(sid, train_parquet, train_parquet, false) }
    }

    def ch_wds = BUILD_WDS(
      ch_labeled_pair,
      Channel.value(file(params.build_wds_py)),
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
    // Step 5: train
    // -----------------------------
    def ch_train = null
    def ch_train_global = null

    if( params.train_py ) {

      def train_mode = (params.train_mode ?: 'per_sample')

      if( train_mode == 'global' ) {

        // Write one wds_dir per line
        def ch_wds_dirs_txt = ch_wds
          .map { sid, wds_dir -> wds_dir.toString() }
          .collectFile(name: "wds_dirs.txt", newLine: true)

        // COLLECT_WDS_SHARDS outputs 3 separate paths (in declared order)
        def ch_shards = COLLECT_WDS_SHARDS(ch_wds_dirs_txt)

        // IMPORTANT: access as out[0]/out[1]/out[2]
        def ch_train_shards_txt = ch_shards.map{ it[0] }
        def ch_val_shards_txt   = ch_shards.map{ it[1] }
        // def ch_calib_shards_txt = ch_shards.map{ it[2] }

        ch_train_global = TRAIN_WDS_GLOBAL(
          ch_train_shards_txt,
          ch_val_shards_txt,
          Channel.value(file(params.train_py)),
          Channel.value(dataset_id),
          Channel.value(params.exp_name),
          Channel.value(params.epochs),
          Channel.value(params.batch_size),
          Channel.value(params.lr),
          Channel.value(params.num_workers)
        )

        // TRAIN_WDS_GLOBAL outputs: (global.ckpt.pt, global.train.log)
        ch_train_global.view { "TRAIN global => ckpt=${it[0]} log=${it[1]}" }

      } else {

        ch_train = TRAIN_WDS(
          ch_wds,
          Channel.value(file(params.train_py)),
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
    // Step 6: eval
    // -----------------------------
    if( params.eval_py ) {

      def ch_eval_in
      def train_mode = (params.train_mode ?: 'per_sample')

      if( train_mode == 'global' ) {

        if( !ch_train_global )
          throw new IllegalArgumentException("train_mode=global but ch_train_global is null (did TRAIN_WDS_GLOBAL run?)")

        // ch_train_global emits (ckpt, log)
        def ch_global_ckpt = ch_train_global.map { ckpt, log -> ckpt }

        ch_eval_in = ch_wds
          .combine(ch_global_ckpt)                 // emits: sid, wds_dir, ckpt
          .map { sid, wds_dir, ckpt -> tuple(sid, wds_dir, ckpt) }

      } else {

        if( !ch_train )
          throw new IllegalArgumentException("train_mode=per_sample but ch_train is null (did TRAIN_WDS run?)")

        def ch_ckpt = ch_train.map { sid, ckpt, log -> tuple(sid, ckpt) }

        ch_eval_in = ch_wds
          .join(ch_ckpt)
          .map { sid, wds_dir, ckpt -> tuple(sid, wds_dir, ckpt) }

      }

      def ch_eval = EVAL_WDS(
        ch_eval_in,
        Channel.value(file(params.eval_py)),
        Channel.value(file(params.train_py)),
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

  return
}