#!/bin/bash
#SBATCH --job-name=train-declutter
#SBATCH --partition=rail-lab
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a40:1
#SBATCH --qos=long
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --exclude=nestor
#SBATCH --exclude=conroy
#SBATCH --exclude=voltron

DEVICE="skynet"
FOLD="None"
SAVE_TAG="None"
DATASET="None"
NUM_EPOCHS=50
BATCH_SIZE=1
LRATE=1e-5
WT_DECAY=1e-20
ALPHA=1.0
BETA=0.0
TRIPLET_MARGIN_MAIN=0.75
TRIPLET_MARGIN_AUX=0.75
LR_SCHEDULER_TMAX=-1
NUM_HEADS=1
NUM_LAYERS=1
HIDDEN_LAYER_SIZE=512
DROPOUT=0.5
INSTANCE_ENCODER_DIM=64
TYPE_EMBEDDING_DIM=1
OBJECT_DIMENSION=384
NUM_CONTAINER_TYPES=5
NUM_SURFACE_TYPES=6
SURFACE_GRID_DIMENSION=32
WANDB=false

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device)
      DEVICE="$2"
      shift # past argument
      shift # past value
      ;;
    --fold)
      FOLD="$2"
      shift # past argument
      shift # past value
      ;;
    --save_tag)
      SAVE_TAG="$2"
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
    --batch_size)
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --lrate)
      LRATE="$2"
      shift # past argument
      shift # past value
      ;;
    --wt_decay)
      WT_DECAY="$2"
      shift # past argument
      shift # past value
      ;;
    --alpha)
      ALPHA="$2"
      shift # past argument
      shift # past value
      ;;
    --beta)
      BETA="$2"
      shift # past argument
      shift # past value
      ;;
    --triplet_margin_main)
      TRIPLET_MARGIN_MAIN="$2"
      shift # past argument
      shift # past value
      ;;
    --triplet_margin_aux)
      TRIPLET_MARGIN_AUX="$2"
      shift # past argument
      shift # past value
      ;;
    --lr_scheduler_tmax)
      LR_SCHEDULER_TMAX="$2"
      shift # past argument
      shift # past value
      ;;
    --num_heads)
      NUM_HEADS="$2"
      shift # past argument
      shift # past value
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift # past argument
      shift # past value
      ;;
    --hidden_layer_size)
      HIDDEN_LAYER_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --dropout)
      DROPOUT="$2"
      shift # past argument
      shift # past value
      ;;
    --instance_encoder_dim)
      INSTANCE_ENCODER_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --type_embedding_dim)
      TYPE_EMBEDDING_DIM="$2"
      shift # past argument
      shift # past value
      ;;
    --object_dimension)
      OBJECT_DIMENSION="$2"
      shift # past argument
      shift # past value
      ;;
    --num_container_types)
      NUM_CONTAINER_TYPES="$2"
      shift # past argument
      shift # past value
      ;;
    --num_surface_types)
      NUM_SURFACE_TYPES="$2"
      shift # past argument
      shift # past value
      ;;
    --surface_grid_dimension)
      SURFACE_GRID_DIMENSION="$2"
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
    export PYTHONUNBUFFERED=TRUE
    source /nethome/kvr6/.bashrc
    cd /coc/flash5/kvr6/
    source setup_conda.sh
    conda activate declutter
    cd dev/robo_declutter
    export PYTHONPATH=/coc/flash5/kvr6/anaconda3/envs/declutter/bin/python
    CMD="srun -u $PYTHONPATH -u train_declutter.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --dataset $DATASET \
      --fold $FOLD \
      --save_tag $SAVE_TAG \
      --num_epochs $NUM_EPOCHS \
      --batch_size $BATCH_SIZE \
      --lrate $LRATE \
      --wt_decay $WT_DECAY \
      --alpha $ALPHA \
      --beta $BETA \
      --triplet_margin_main $TRIPLET_MARGIN_MAIN \
      --triplet_margin_aux $TRIPLET_MARGIN_AUX \
      --lr_scheduler_tmax $LR_SCHEDULER_TMAX \
      --num_heads $NUM_HEADS \
      --num_layers $NUM_LAYERS \
      --hidden_layer_size $HIDDEN_LAYER_SIZE \
      --dropout $DROPOUT \
      --instance_encoder_dim $INSTANCE_ENCODER_DIM \
      --type_embedding_dim $TYPE_EMBEDDING_DIM \
      --object_dimension $OBJECT_DIMENSION \
      --num_container_types $NUM_CONTAINER_TYPES \
      --num_surface_types $NUM_SURFACE_TYPES \
      --surface_grid_dimension $SURFACE_GRID_DIMENSION"
elif [ $DEVICE = "hopper" ]
  then
    source ~/.bashrc
    cd /media/kartik/disk-2/robo_declutter
    export PYTHONPATH=/home/kartik/miniconda3/envs/declutter/bin/python
    export DATA_HOME=/media/kartik/disk-2/Data/declutter_user_data
    CMD="$PYTHONPATH -u train_declutter.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --dataset $DATASET \
      --fold $FOLD \
      --save_tag $SAVE_TAG \
      --num_epochs $NUM_EPOCHS \
      --batch_size $BATCH_SIZE \
      --lrate $LRATE \
      --wt_decay $WT_DECAY \
      --alpha $ALPHA \
      --beta $BETA \
      --triplet_margin_main $TRIPLET_MARGIN_MAIN \
      --triplet_margin_aux $TRIPLET_MARGIN_AUX \
      --lr_scheduler_tmax $LR_SCHEDULER_TMAX \
      --num_heads $NUM_HEADS \
      --num_layers $NUM_LAYERS \
      --hidden_layer_size $HIDDEN_LAYER_SIZE \
      --dropout $DROPOUT \
      --instance_encoder_dim $INSTANCE_ENCODER_DIM \
      --type_embedding_dim $TYPE_EMBEDDING_DIM \
      --object_dimension $OBJECT_DIMENSION \
      --num_container_types $NUM_CONTAINER_TYPES \
      --num_surface_types $NUM_SURFACE_TYPES \
      --surface_grid_dimension $SURFACE_GRID_DIMENSION"
else
    echo "Invalid argument for keyword arg 'device', must be either 'skynet' or 'hopper'"
fi
if [ "$WANDB" == true ]; then
  CMD="$CMD --wandb"
fi
echo "Running command: $CMD"
eval $CMD
echo Done