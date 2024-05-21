#!/bin/bash
#SBATCH --job-name=fp
#SBATCH -p joonholee,unrestricted,intermediate
#SBATCH -t 24:00:00
#SBATCH -n 5
#SBATCH --mem-per-cpu=1000
#SBATCH -o test_3000.out
#SBATCH -e test_3000.err

source ~/.bashrc
ulimit -s unlimited
cd $SLURM_SUBMIT_DIR
conda activate ipie_dev
#export PYTHONPATH=/n/home01/mbaumgarten/packages/ipie_devl/ipie:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n $SLURM_NTASKS --mpi=pmi2 python -u run_holstein_fp.py > output_3000walkers.txt
