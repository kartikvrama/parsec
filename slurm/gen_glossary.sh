#!/bin/bash
#SBATCH --job-name=gen-glossary
#SBATCH --partition=rail-lab
#SBATCH --account=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err


export PYTHONUNBUFFERED=TRUE
source /nethome/kvr6/.bashrc
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter/data_permutation

srun -u python -u generate_glossary.py \
    --dataset $DATA_HOME/permuted_examples_mar26
echo Done
