#!/bin/bash
#SBATCH --job-name=save-llm-prompts
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

FOLD_FOLDER=folds-2024-07-20
DEST_FOLDER=prompts-2024-07-20

source /nethome/kvr6/.bashrc
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter/data_transform

for FOLD in $(ls $DATA_HOME/$FOLD_FOLDER); do
    FOLD_NAME=$(echo $FOLD | cut -d'.' -f 1)
    echo $FOLD_NAME
    srun -u python -u save_llm_prompts.py \
        --dataset $DATA_HOME/permuted_examples_mar26 \
        --fold $DATA_HOME/$FOLD_FOLDER/$FOLD \
        --destination_folder $DATA_HOME/$DEST_FOLDER/$FOLD_NAME
done
