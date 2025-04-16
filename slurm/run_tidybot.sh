#!/bin/bash
#SBATCH --job-name=run-tidybot
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

export PYTHONUNBUFFERED=TRUE
source /nethome/kvr6/.bashrc
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter
export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python

FOLD_NAME=out_distribution
PROMPT_FOLDER=$DATA_HOME/prompts-2024-08-15
DESTINATION_FOLDER=$PROJ_HOME/results/tidybot_responses-gpt4

CMD="srun -u $PYTHONPATH -u run_tidybot.py \
    --prompt_directory $PROMPT_FOLDER/$FOLD_NAME \
    --destination_folder $DESTINATION_FOLDER"
echo "Running command: $CMD"
eval $CMD
echo Done