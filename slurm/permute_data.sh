#!/bin/bash
#SBATCH --job-name=permute_data
#SBATCH --partition=rail-lab
#SBATCH --account=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err


export PYTHONUNBUFFERED=TRUE
cd /srv/rail-lab/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter/data_permutation

srun -u python -u permute_user_data.py \
    --data /srv/rail-lab/flash5/kvr6/data/declutter_user_data/arrangements_json/ \
    --destination /srv/rail-lab/flash5/kvr6/data/declutter_user_data/permuted_examples_mar26 \
    --nodryrun --verbose
echo Done
