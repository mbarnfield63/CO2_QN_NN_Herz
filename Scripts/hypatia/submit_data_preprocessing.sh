#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --output=_jobs/data_preprocessing.out
#SBATCH --error=_jobs/data_preprocessing.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='v100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Ensure conda is sourced correctly
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate tf_env

# Run Python script
srun python Scripts/data_preprocessing.py Data/Processed/
