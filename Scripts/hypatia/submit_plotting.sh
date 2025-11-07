#!/bin/bash
#SBATCH --job-name=CO2_plotting
#SBATCH --output=_jobs/CO2_plotting.out
#SBATCH --error=_jobs/CO2_plotting.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Ensure conda is sourced correctly
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate CO2

# Run Python script
srun python Scripts/plot_inference_uncertainty.py
