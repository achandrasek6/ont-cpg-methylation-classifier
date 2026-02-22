nextflow.enable.dsl=2

/*
  Local smoke skeleton:
    Step 1: FAST5 -> POD5 (optional) -> Dorado BAM (optional)
    Step 2: coords_sample -> coords_filter_covered
    Step 3: extract_labeled (coords_covered + BAM/BAI + POD5 -> labeled parquet)
    Step 4: build_wds (optional)
    Step 5: train_wds (optional)
    Step 6: eval_wds (optional)

  NEW:
    - FAST5-first multi-sample input via --samples_csv.
    - FAST5 -> POD5 conversion becomes optional per-row (skip when pod5 is provided).

  Notes:
    - Multi-sample mode now runs Steps 1–6 with sample_id threaded through.

  IMPORTANT (bug fixes):
    - Do NOT consume BAM twice.
    - Make sample pairing explicit with `join` before COORDS_FILTER_COVERED.

  This file expects:
    - COORDS_FILTER_COVERED input payload: (sample_id, coords_sample, bam_sorted, bam_bai)
    - COORDS_FILTER_COVERED output:       (sample_id, coords_covered, bam_sorted, bam_bai)
    - EXTRACT_LABELED input payload:      (sample_id, coords_covered, bam_sorted, bam_bai, pod5)
*/

include { CONVERT_POD5 }           from './modules/convert_pod5'
include { BASECALL_ALIGN_DORADO }  from './modules/basecall_align'
include { COORDS_SAMPLE }          from './modules/coords_sample'
include { COORDS_FILTER_COVERED }  from './modules/coords_filter_covered'
include { EXTRACT_LABELED }        from './modules/extract_labeled'
include { BUILD_WDS }              from './modules/build_wds'
include { TRAIN_WDS }              from './modules/train_wds'
include { EVAL_WDS }               from './modules/eval_wds'

workflow {

  // -----------------------------
  // Multi-sample mode (FAST5-first)
  // -----------------------------
  if( params.samples_csv ) {

    if( !params.outdir ) error "Missing --outdir"

    def dataset_id = params.run_id ?: "dataset"
    def norm = { x -> x != null && x.toString().trim() ? x.toString().trim() : null }

    // Expect CSV columns:
    // sample_id,fast5_dir,pod5,ref_fa,dorado_model,(optional) bam,bai
    Channel
      .fromPath(params.samples_csv)
      .splitCsv(header:true)
      .map { r ->
        def sid = norm(r.sample_id)
        if( !sid ) error "samples_csv: missing sample_id"
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
        if( !fast5_dir ) error "samples_csv: sample ${sid} must provide either pod5 or fast5_dir"
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
        if( !ref_fa ) error "samples_csv: sample ${sid} missing ref_fa (and no global --ref_fa)"
        if( !dorado_model ) error "samples_csv: sample ${sid} missing dorado_model (and no global --dorado_model)"
        tuple(sid, file(ref_fa), dorado_model)
      }

    // Join pod5 + (ref_fa,dorado_model) on sample_id
    def ch_pod5_for_dorado = ch_pod5
      .join(ch_need_dorado)
      .map { sid, pod5_path, ref_fa, dorado_model -> tuple(sid, pod5_path, ref_fa, dorado_model) }

    def ch_bam_from_dorado = BASECALL_ALIGN_DORADO(ch_pod5_for_dorado)
    def ch_bam = ch_bam_direct.mix(ch_bam_from_dorado)

    ch_bam.view { t -> "[${dataset_id}] BAM/BAI -> ${t[0]} -> ${t[1]} / ${t[2]}" }

    // -----------------------------
    // Step 2: coords_sample -> coords_filter_covered  (multi-sample)
    // -----------------------------
    if( !params.coords_parquet )
      error "Multi-sample mode: missing --coords_parquet (for now we require coords pipeline)"

    def ch_sample_ids = ch_rows
      .map { sid, fast5_dir, pod5, bam, bai, ref_fa, dorado_model -> sid }
      .distinct()

    // one coords_sample for the whole run (dataset-level)
    def ch_coords_sample = COORDS_SAMPLE(
      Channel.value(file(params.coords_parquet)),
      Channel.value(dataset_id),
      Channel.value(params.coords_sample_n)
    )

    // attach coords_sample to each sample_id
    def ch_coords_sample_by_sample = ch_sample_ids
      .combine(ch_coords_sample)
      .map { sid, coords_path -> tuple(sid, coords_path) }

    // Explicitly pair sample_id across coords + bam
    def ch_coords_cov_in = ch_coords_sample_by_sample
      .join(ch_bam)
      .map { sid, coords_path, bam_path, bai_path -> tuple(sid, coords_path, bam_path, bai_path) }

    ch_coords_cov_in.view { t -> "COORDS_COV_IN -> ${t[0]}" }

    def ch_coords_cov = COORDS_FILTER_COVERED(
      ch_coords_cov_in,
      Channel.value(file(params.filter_coords_py)),
      Channel.value(params.min_mapq),
      Channel.value(params.max_sites)
    )

    ch_coords_cov.view { t -> "COORDS_COVERED -> ${t[0]} -> ${t[1]}" }

    // -----------------------------
    // Step 3: extract labeled
    // -----------------------------
    def ch_extract_in = ch_coords_cov
      .join(ch_pod5)
      .map { sid, coords_cov, bam_path, bai_path, pod5_path ->
        tuple(sid, coords_cov, bam_path, bai_path, pod5_path)
      }

    ch_extract_in.view { t -> "EXTRACT_IN -> ${t[0]} -> coords=${t[1]} bam=${t[2]} pod5=${t[4]}" }

    def ch_labeled = EXTRACT_LABELED(
      ch_extract_in,
      Channel.value(file(params.extract_py)),
      Channel.value(params.window),
      Channel.value(params.max_reads_per_site),
      Channel.value(params.min_mapq),
      Channel.value(params.limit_sites),
      Channel.value(params.k)
    )

    ch_labeled.view { "LABELED ${it[0]} => ${it[1]}" }

    // -----------------------------
    // Step 4: build wds
    // -----------------------------
    if( params.build_wds_py ) {
      def ch_wds = BUILD_WDS(
        ch_labeled,
        Channel.value(file(params.build_wds_py)),
        Channel.value(dataset_id),
        Channel.value(params.wds_shard_size),
        Channel.value(params.val_frac),
        Channel.value(params.seed)
      )

      ch_wds.view { "WDS ${it[0]} => ${it[1]}" }

      // -----------------------------
      // Step 5: train
      // -----------------------------
      if( params.train_py ) {
        def ch_train = TRAIN_WDS(
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

        // -----------------------------
        // Step 6: eval
        // -----------------------------
        if( params.eval_py ) {
          def ch_ckpt = ch_train.map { sid, ckpt, log -> tuple(sid, ckpt) }

          def ch_eval = EVAL_WDS(
            ch_wds,
            ch_ckpt,
            Channel.value(file(params.eval_py)),
            Channel.value(file(params.train_py)),
            Channel.value(dataset_id),
            Channel.value(params.exp_name),
            Channel.value(params.eval_split),
            Channel.value(params.eval_batch_size),
            Channel.value(params.eval_num_workers),
            Channel.value(params.calibrate),
            Channel.value(params.calib_bins)
          )

          ch_eval.view { "EVAL ${it[0]} => ${it[1]}" }
        }
      }
    }

    return
  }

  error "Single-sample mode not included in this drop-in replacement; run with --samples_csv."
}
