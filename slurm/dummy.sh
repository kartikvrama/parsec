#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

export PYTHONUNBUFFERED=TRUE
source /nethome/kvr6/.bashrc
cd /srv/rail-lab/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter
echo Done
