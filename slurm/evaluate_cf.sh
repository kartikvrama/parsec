#!/bin/bash
#SBATCH --job-name=eval-cf
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=nestor
#SBATCH --exclude=conroy
#SBATCH --exclude=voltron

DEVICE="skynet"
PARTITION="rail-lab"
DATASET="None"
FOLD="None"
MODEL_TAG="None"
CHECKPOINT_FOLDER="None"

# Parse keyword arguments.
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
    --dataset)
        DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --fold)
        FOLD="$2"
        shift # past argument
        shift # past value
        ;;
    --model_tag)
        MODEL_TAG="$2"
        shift # past argument
        shift # past value
        ;;
    --checkpoint_folder)
        CHECKPOINT_FOLDER="$2"
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
    export CUDA_LAUNCH_BLOCKING=1

    export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python
    CMD="srun -u $PYTHONPATH evaluate_CF.py \
        --dataset $DATASET \
        --fold $FOLD \
        --model_tag $MODEL_TAG \
        --checkpoint_folder $CHECKPOINT_FOLDER"
elif [ $DEVICE = "hopper" ]
  then
    source ~/.bashrc
    cd /media/kartik/disk-2/robo_declutter
    export PYTHONPATH=/home/kartik/miniconda3/envs/declutter/bin/python
    export DATA_HOME=/media/kartik/disk-2/Data/declutter_user_data
    CMD="$PYTHONPATH -u evaluate_CF.py \
        --dataset $DATASET \
        --fold $FOLD \
        --model_tag $MODEL_TAG \
        --checkpoint_folder $CHECKPOINT_FOLDER"
else
	echo "Invalid argument for keyword arg 'device', must be either 'skynet' or 'hopper'"
fi
echo "Running command: $CMD"
eval $CMD
echo Done
