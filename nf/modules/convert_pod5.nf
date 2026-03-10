process CONVERT_POD5 {
  tag "${sample_id}"
  label 'cpu'

  input:
    tuple val(sample_id), val(fast5_s3_prefix)

  output:
    tuple val(sample_id), path("${sample_id}.pod5")

  publishDir "${params.outdir}/derived/runs/${sample_id}/pod5",
    mode: params.publish_mode, overwrite: true

  script:
    def max_files = (params.fast5_max_files ?: 0) as int
    def seed      = (params.fast5_sample_seed ?: 42) as int
    def prefix    = fast5_s3_prefix.toString()

    return '''
set -euo pipefail

max_files=''' + max_files + '''
seed=''' + seed + '''

echo "[CONVERT_POD5] sample_id=''' + sample_id + '''"
echo "[CONVERT_POD5] fast5_s3_prefix=''' + prefix + '''"
echo "[CONVERT_POD5] fast5_max_files=$max_files"

df -h . || true
df -h /tmp || true
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"

FAST5_S3_PREFIX="''' + prefix + '''"

s3_no_scheme=$(printf "%s" "$FAST5_S3_PREFIX" | sed 's#^s3://##')
bucket=$(printf "%s" "$s3_no_scheme" | cut -d/ -f1)
prefix=$(printf "%s" "$s3_no_scheme" | cut -d/ -f2-)

mkdir -p fast5_subset

aws s3 ls "s3://$bucket/$prefix" --recursive \
  | awk '{print $4}' \
  | grep -F ".fast5" \
  | sort > fast5_keys.txt

total=$(wc -l < fast5_keys.txt || echo 0)
echo "[CONVERT_POD5] total_fast5_keys=$total"
if [ "$total" -eq 0 ]; then
  echo "[CONVERT_POD5] ERROR: no .fast5 files found under s3://$bucket/$prefix" >&2
  exit 2
fi

if [ "$max_files" -gt 0 ]; then
  while read -r k; do
    h=$(printf "%s" "${seed}${k}" | md5sum | awk '{print $1}')
    printf "%s\t%s\n" "$h" "$k"
  done < fast5_keys.txt \
    | sort \
    | awk -v n="$max_files" 'NR<=n {print $2}' > fast5_pick.txt
else
  cp fast5_keys.txt fast5_pick.txt
fi

picked=$(wc -l < fast5_pick.txt || echo 0)
echo "[CONVERT_POD5] picked_fast5_keys=$picked"

while read -r key; do
  aws s3 cp "s3://$bucket/$key" fast5_subset/
done < fast5_pick.txt

echo "[CONVERT_POD5] downloaded_files=$(ls -1 fast5_subset | wc -l)"

pod5 convert fast5 fast5_subset --output "''' + sample_id + '''.pod5"
'''
}