#!/bin/bash
#SBATCH --job-name=CO2_NN
#SBATCH --output=_jobs/CO2_NN.out
#SBATCH --error=_jobs/CO2_NN.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='v100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# Ensure conda is sourced correctly
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate CO2

# Run Python script
srun python Scripts/main.py
