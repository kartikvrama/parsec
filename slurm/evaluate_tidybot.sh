#!/bin/bash
#SBATCH --job-name=eval-tidybot
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err

DEVICE="skynet"
PARTITION="rail-lab"
DATASET="None"
FOLD="None"
RESPONSES="None"
MODEL_TAG="None"

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
        DEVICE="$2"
        shift # past argument
        shift # past value
        ;;
    --partition)
        PARTITION="$2"
        shift # past argument
        shift # past value
        ;;
    --fold)
        FOLD="$2"
        shift # past argument
        shift # past value
        ;;
    --responses)
        RESPONSES="$2"
        shift # past argument
        shift # past value
        ;;
    --dataset)
        DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --model_tag)
        MODEL_TAG="$2"
        shift # past argument
        shift # past value
        ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [ $DEVICE = "skynet" ]
then
    if [ $PARTITION = "rail-lab" ]
    then
        echo "Using rail-lab partition"
    elif [ $PARTITION = "overcap" ]
    then
        echo "Using overcap partition"
    else
        echo "Unknown partition"
        exit 1
    fi
    #SBATCH --partition=$PARTITION

	export PYTHONUNBUFFERED=TRUE
	source /nethome/kvr6/.bashrc
	cd /coc/flash5/kvr6/
	source setup_conda.sh
	conda activate declutter
	cd dev/robo_declutter

    export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python
    srun -u $PYTHONPATH -u evaluate_tidybot.py \
        --dataset $DATASET \
        --fold $FOLD \
        --model_tag $MODEL_TAG \
        --responses $RESPONSES
else
    echo "Device not supported"
    exit 1
fi
echo Done
