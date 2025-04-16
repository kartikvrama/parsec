#!/bin/bash
#SBATCH --job-name=save-fold-pkl
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --qos=long
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

FOLD_FOLDER=folds-2024-09-10
DEST_FOLDER=batch-2024-09-10

source /nethome/kvr6/.bashrc
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter/data_split
export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python

for FOLD in $(ls $DATA_HOME/$FOLD_FOLDER); do
    echo $FOLD
    srun -u $PYTHONPATH -u save_fold_pkl.py \
        --dataset $DATA_HOME/permuted_examples_mar26 \
        --fold $DATA_HOME/$FOLD_FOLDER/$FOLD \
        --destination $DATA_HOME/$DEST_FOLDER
done
