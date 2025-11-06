#!/bin/bash
#SBATCH --job-name=CO2_postprocess
#SBATCH --array=0-11
#SBATCH --output=_jobs/Post/CO2_postprocess%a.out
#SBATCH --error=_jobs/Post/CO2_postprocess%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=ucapmgb@ucl.ac.uk
#SBATCH --mail-type=END,FAIL

# --- Job Array Setup ---
isotopologues=(626 627 628 636 637 638 727 728 737 738 828 838)

#    SLURM_ARRAY_TASK_ID will be 0 for the first job, 1 for the second, etc.
iso=${isotopologues[$SLURM_ARRAY_TASK_ID]}

# --- Environment and Execution ---
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate CO2

# Echo the parameters for this specific job to the output file for easy tracking
echo "Running job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "Processing $iso"

# Run Python script with the selected p_dropout value
srun python Scripts/post_processing.py $iso