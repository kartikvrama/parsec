#!/bin/bash
#SBATCH --job-name=run-apricot
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

export PYTHONUNBUFFERED=TRUE
source /nethome/kvr6/.bash_aliases
cd /coc/flash5/kvr6/
source setup_conda.sh
conda activate declutter
cd dev/robo_declutter
export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python

CMD="srun -u $PYTHONPATH -u run_apricot_noquery.py"
echo "Running command: $CMD"
eval $CMD
echo Done