#!/bin/bash
#SBATCH --job-name=viz_folds
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
source /nethome/kvr6/.bashrc
activatepy
conda activate declutter
cd dev/robo_declutter/data_split

srun -u python -u visualize_folds.py \
    --data_dir $DATA_HOME/permuted_examples_mar26/ \
    --fold_file $DATA_HOME/folds_mar27/out_distribution_prefs.json
echo Done
