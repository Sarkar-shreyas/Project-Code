#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-6
#SBATCH --time=08:00:00
#SBATCH --job-name=generate_smaller_arrays
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

array_size_file="$jobsdir/array_sizes.txt"
N=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$array_size_file")

echo "Task $SLURM_ARRAY_TASK_ID -> N=$N"
python "$codedir/generate_arrays.py" "$N" "$joboutdir"
