#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time 2-0:00:00
#SBATCH -p GPU
#SBATCH -N1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --output='./_jobs/-%j.%x.out'
#SBATCH --error='./_jobs/-%j.%x.out'

XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "To connect:
ssh -N -L ${port}:${node}:${port} ${user}@hypatia-login.hpc.phys.ucl.ac.uk

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)

Remember to scancel job when done. Check output below for access token if
you need it.
"

source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate tf_env
srun -n1 jupyter-lab --no-browser --port=${port} --ip=${node}