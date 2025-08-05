#!/bin/bash
#SBATCH --job-name=split_MaHi_Ca
#SBATCH --output=_jobs/split_MaHi_Ca.out
#SBATCH --error=_jobs/split_MaHi_Ca.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='v100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Ensure conda is sourced correctly
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate tf_env

# Change directory (use bash, not srun)
cd /home/ucapmgb/CO2_QN_NN_Herz || exit

# Run Python script
srun python Scripts/split_MaHi_Ca.py Data/
