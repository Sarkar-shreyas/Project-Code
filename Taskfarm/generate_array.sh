#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=3988
#SBATCH --cpus-per-task=1
#SBATCH --array=0-8
#SBATCH --time=08:00:00
#SBATCH --job-name=generate_arrays
#SBATCH --output=../job_outputs/%x_%j.out
#SBATCH --error=../job_logs/%x_%j.err

codedir="../code"
jobsdir="../jobs"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load GCC/14.3.0 SciPy-bundle/2025.07

array_size_file="$jobsdir/array_sizes.txt"
N=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$array_size_file")

echo "Task $SLURM_ARRAY_TASK_ID -> N=$N"
python "$codedir/generate_arrays.py" "$N"
