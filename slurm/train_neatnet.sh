#!/bin/bash
#SBATCH --job-name=train-neatnet
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a40:1
#SBATCH --qos=short
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=nestor
#SBATCH --exclude=conroy

# Default Values.
DEVICE="skynet"
PARTITION="rail-lab"
BATCH_SIZE=2
USER_DATA_DIR="None"
ENVIRONMENT_CAT="None"
ENVIRONMENT_VAR="None"
FOLD="None"
DATASET="None"
SAVE_TAG="None"
WANDB=false
NUM_EPOCHS=1000
GRAPH_DIM=64
USER_DIM=2
RELU_LEAK=0.2
POS_DIM=2
SEMANTIC_DIM=384
ENCODER_H_DIM=64
PREDICTOR_H_DIM=32
INIT_LR=1e-3
SCH_PATIENCE=200
SCH_COOLDOWN=100
SCH_FACTOR=0.5
NOISE_SCALE=0.02
VAE_BETA=0.01

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
    --batch_size)
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --user_data_dir)
      USER_DATA_DIR="$2"
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
    --num_epochs)
      NUM_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --graph_dim)
      GRAPH_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --user_dim)
      USER_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --relu_leak)
      RELU_LEAK="$2"
      shift # past argument
      shift # past value
      ;;
    --pos_dim)
      POS_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --semantic_dim)
      SEMANTIC_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --encoder_h_dim)
      ENCODER_H_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --predictor_h_dim)
      PREDICTOR_H_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --init_lr)
      INIT_LR="$2"
      shift # past argument
      shift # past value
      ;;
    --sch_patience)
      SCH_PATIENCE="$2"
      shift # past argument
      shift # past value
      ;;
    --sch_cooldown)
      SCH_COOLDOWN="$2"
      shift # past argument
      shift # past value
      ;;
    --sch_factor)
      SCH_FACTOR="$2"
      shift # past argument
      shift # past value
      ;;
    --noise_scale)
      NOISE_SCALE="$2"
      shift # past argument
      shift # past value
      ;;
    --vae_beta)
      VAE_BETA="$2"
      shift # past argument
      shift # past value
      ;;
    --save_tag)
      SAVE_TAG="$2"
      shift # past argument
      shift # past value
      ;;
    --wandb)
      WANDB=true
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
    CMD="srun -u $PYTHONPATH -u train_neatnet.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --batch_size $BATCH_SIZE \
      --user_data_dir $USER_DATA_DIR \
      --dataset $DATASET \
      --fold $FOLD \
      --environment_cat $ENVIRONMENT_CAT \
      --environment_var $ENVIRONMENT_VAR \
      --num_epochs $NUM_EPOCHS \
      --graph_dim $GRAPH_DIM \
      --user_dim $USER_DIM \
      --relu_leak $RELU_LEAK \
      --pos_dim $POS_DIM \
      --semantic_dim $SEMANTIC_DIM \
      --encoder_h_dim $ENCODER_H_DIM \
      --predictor_h_dim $PREDICTOR_H_DIM \
      --init_lr $INIT_LR \
      --num_epochs $NUM_EPOCHS \
      --sch_patience $SCH_PATIENCE \
      --sch_cooldown $SCH_COOLDOWN \
      --sch_factor $SCH_FACTOR \
      --noise_scale $NOISE_SCALE \
      --vae_beta $VAE_BETA \
      --save_tag $SAVE_TAG"
elif [ $DEVICE = "hopper" ]
  then
    source ~/.bashrc
    cd /media/kartik/disk-2/robo_declutter
    export PYTHONPATH=/home/kartik/miniconda3/envs/declutter/bin/python
    export DATA_HOME=/media/kartik/disk-2/Data/declutter_user_data
    CMD="$PYTHONPATH train_neatnet_mp.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --batch_size $BATCH_SIZE \
      --user_data_dir $USER_DATA_DIR \
      --dataset $DATASET \
      --fold $FOLD \
      --environment_cat $ENVIRONMENT_CAT \
      --environment_var $ENVIRONMENT_VAR \
      --num_epochs $NUM_EPOCHS \
      --graph_dim $GRAPH_DIM \
      --user_dim $USER_DIM \
      --relu_leak $RELU_LEAK \
      --pos_dim $POS_DIM \
      --semantic_dim $SEMANTIC_DIM \
      --encoder_h_dim $ENCODER_H_DIM \
      --predictor_h_dim $PREDICTOR_H_DIM \
      --init_lr $INIT_LR \
      --num_epochs $NUM_EPOCHS \
      --sch_patience $SCH_PATIENCE \
      --sch_cooldown $SCH_COOLDOWN \
      --sch_factor $SCH_FACTOR \
      --noise_scale $NOISE_SCALE \
      --vae_beta $VAE_BETA \
      --save_tag $SAVE_TAG"
else
	echo "Invalid argument for keyword arg 'device', must be either 'skynet' or 'hopper'"
fi

if [ "$WANDB" == true ]; then
  CMD="$CMD --wandb"
fi
eval $CMD
echo Done
