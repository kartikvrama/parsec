#!/bin/bash
#SBATCH --job-name=eval-neatnet
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=a40:1
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=nestor
#SBATCH --exclude=conroy

# Default Values.
DEVICE="skynet"
PARTITION="rail-lab"
ENVIRONMENT_CAT="None"
ENVIRONMENT_VAR="None"
FOLD="None"
DATASET="None"
MODEL_TAG="None"
CHECKPOINT_FOLDER="None"
STOPPING_METRIC="None"

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
    --environment_cat)
        ENVIRONMENT_CAT="$2"
        shift # past argument
        shift # past value
        ;;
    --environment_var)
        ENVIRONMENT_VAR="$2"
        shift # past argument
        shift # past value
        ;;
    --fold)
        FOLD="$2"
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
    --checkpoint_folder)
        CHECKPOINT_FOLDER="$2"
        shift # past argument
        shift # past value
        ;;
    --stopping_metric)
        STOPPING_METRIC="$2"
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
    srun -u $PYTHONPATH -u evaluate_neatnet.py \
        --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
        --dataset $DATASET \
        --fold $FOLD \
        --environment_cat $ENVIRONMENT_CAT \
        --environment_var $ENVIRONMENT_VAR \
        --model_tag $MODEL_TAG \
        --checkpoint_folder $CHECKPOINT_FOLDER \
        --stopping_metric $STOPPING_METRIC
fi
echo "Done"
