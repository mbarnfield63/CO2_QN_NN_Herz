#!/bin/bash
#SBATCH --job-name=CO2_fulltrain
#SBATCH --array=0-8
#SBATCH --output=_jobs/Calibration/CO2_fulltrain_%a.out
#SBATCH --error=_jobs/Calibration/CO2_fulltrain_%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=ucapmgb@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

# --- Job Array Setup ---
p_dropouts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

#    SLURM_ARRAY_TASK_ID will be 0 for the first job, 1 for the second, etc.
current_p_dropout=${p_dropouts[$SLURM_ARRAY_TASK_ID]}

# --- Environment and Execution ---
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate CO2

# Echo the parameters for this specific job to the output file for easy tracking
echo "Running job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "p_dropout value: $current_p_dropout"

# Run Python script with the selected p_dropout value
srun python Scripts/full_train.py --p_dropout $current_p_dropout