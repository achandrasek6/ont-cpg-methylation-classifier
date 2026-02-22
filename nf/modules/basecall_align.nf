process BASECALL_ALIGN_DORADO {
  tag "${sample_id}"
  label 'gpu'

  input:
    tuple val(sample_id), path(pod5), path(ref_fa), val(dorado_model)

  output:
    tuple val(sample_id), path("${sample_id}.sorted.bam"), path("${sample_id}.sorted.bam.bai")

  publishDir "${params.outdir}/derived/runs/${sample_id}/basecalls/model=${dorado_model}",
    mode: params.publish_mode, overwrite: true

  script:
    def dev = (params.dorado_device ?: "cuda:all").toString().trim()
    def device_arg = dev ? "--device ${dev}" : ""

    // optional smoke limiter
    def max_reads = (params.dorado_max_reads ?: 0) as int
    def max_reads_arg = max_reads > 0 ? "--max-reads ${max_reads}" : ""

    """
    set -euo pipefail

    dorado basecaller "${dorado_model}" "${pod5}" \\
      --reference "${ref_fa}" \\
      --emit-moves \\
      --emit-sam \\
      ${device_arg} \\
      ${max_reads_arg} \\
    | samtools sort -@ ${task.cpus} -m 512M -o "${sample_id}.sorted.bam" -

    samtools index "${sample_id}.sorted.bam"
    """
}