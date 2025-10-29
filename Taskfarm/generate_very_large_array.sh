#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=generate_very_large_array
#SBATCH --output=../job_outputs/bootstrap/%x_%A_%a.out
#SBATCH --error=../job_logs/bootstrap/%x_%A_%a.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load GCC/14.3.0 SciPy-bundle/2025.07

basedir="$(cd "$SLURM_SUBMIT_DIR/.."&&pwd)"
codedir="$basedir/code"
jobsdir="$basedir/jobs"
logsdir="$basedir/job_logs"
outputdir="$basedir/job_outputs"

jobkey="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
task="${SLURM_ARRAY_TASK_ID:-0}"
joblogdir="$logsdir/$jobkey"
joboutdir="$outputdir/$jobkey"
mkdir -p "$joblogdir" "$joboutdir" "$joboutdir/output" "$joboutdir/data"

exec > >(tee -a "$joboutdir/output/task_${task}.out")
exec 2> >(tee -a "$joblogdir/task_${task}.err" >&2)

N=100000000
batches=10
base=$((N/batches))
remaining=$(N - base*batches)
echo "Total N: $N"
echo "Batches: $batches"
echo "Amount per batch: $base, remainder: $remaining"

for i in $(seq 1 $batches); do
    subN=$base
    if [[ $i -eq $batches]]; then
        subN=$((base + remaining))
    fi

    echo "[$(date '+%F %T')] base $i/$base -> subN=$subN"

    python "$codedir/generate_arrays.py" "$subN" "$joboutdir"

    src="$joboutdir/data/t_prime_${subN}_samples"
    destination="$joboutdir/data/t_prime_${N}_part_${i}.txt"

    if [[ -f "$src" ]]; then
        mv "$src" "$destination"
        echo "Saved: $destination"
    else
        echo "ERROR: expected output not found: $src"
        exit 1
    fi

done

combined_dir="$joboutdir/data/t_prime_${N}_combined.txt"
echo "Combining parts into: $combined_dir"
cat "$joboutdir"/data/t_prime_${N}_part_*.txt > "$combined_dir"

echo "Done."
