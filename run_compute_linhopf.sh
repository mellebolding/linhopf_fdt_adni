#!/bin/bash
#SBATCH --job-name=compute_linhopf         # Job name
#SBATCH --output=compute_linhopf.out       # Standard output log
#SBATCH --error=compute_linhopf.err        # Standard error log
#SBATCH --time=300:00:00                 # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2G                        # Memory per node (adjust as needed)
#SBATCH --mail-type=END,FAIL            # Notifications for job done or fail (optional)

# Activate your virtual environment
source /home/cluster/melle/FDT_ADNI/venv/bin/activate

export MPLCONFIGDIR=/tmp/mplconfig_$SLURM_JOB_ID
mkdir -p $MPLCONFIGDIR

# Go to your script directory (optional but clean)
cd /home/cluster/melle/LINHOPF_FDT_ADNI

# Run your Python script
python compute_linhopf.py
