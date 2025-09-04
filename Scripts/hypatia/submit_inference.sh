#!/bin/bash
#SBATCH --job-name=CO2_inference
#SBATCH --output=_jobs/CO2_inference.out
#SBATCH --error=_jobs/CO2_inference.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=ucapmgb@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

# Ensure conda is sourced correctly
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate CO2

# Run Python script
srun python Scripts/inference.py
srun python Scripts/plot_inference_uncertainty.py
