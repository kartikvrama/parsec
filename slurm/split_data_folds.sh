#!/bin/bash
#SBATCH --job-name=split-data-folds
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=nestor
#SBATCH --exclude=conroy
#SBATCH --exclude=voltron

DEST_FOLDER=folds-2024-07-20

source /nethome/kvr6/.bashrc
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter/data_split
export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python

srun -u $PYTHONPATH -u split_data.py \
    --user_data_dir $DATA_HOME/arrangements_json/ \
    --dataset $DATA_HOME/permuted_examples_mar26 \
    --user_list $PROJ_HOME/labels/eligible_users.txt \
    --destination $DATA_HOME/$DEST_FOLDER
echo Done