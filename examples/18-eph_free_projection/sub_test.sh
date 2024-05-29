#!/bin/bash
#SBATCH --job-name=band
#SBATCH -p joonholee,unrestricted,intermediate
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --mem-per-cpu=1000
#SBATCH -o test.out
#SBATCH -e test.err

source ~/.bashrc
ulimit -s unlimited
cd $SLURM_SUBMIT_DIR
conda activate ipie_dev
#export PYTHONPATH=/n/home01/mbaumgarten/packages/ipie_devl/ipie:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python -u run_holstein_fp.py > output2.txt
