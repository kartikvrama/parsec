#!/bin/bash
#SBATCH --job-name=train-consor
#SBATCH --partition=rail-lab
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
FOLD="None"
DATASET="None"
SAVE_TAG="None"
WANDB=false
NUM_EPOCHS=500
SEMANTIC_DIM=384
HIDDEN_LAYER_SIZE=512
OUTPUT_DIMENSION=64
NUM_HEADS=2
NUM_LAYERS=3
DROPOUT=0.5
OBJECT_POS_ENCODING_DIM=32
SURFACE_POS_ENCODING_DIM=32
BATCH_SIZE=4
LRATE=1e-4
WT_DECAY=1e-20
LOSS_FN="triplet_margin"
TRIPLET_LOSS_MARGIN=0.75

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
    --dataset)
      DATASET="$2"
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
    --num_epochs)
      NUM_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --semantic_dim)
      SEMANTIC_DIM="$2"
      shift # past argument
      shift # past value
      ;;
      --hidden_layer_size)
      HIDDEN_LAYER_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
      --output_dimension)
      OUTPUT_DIMENSION="$2"
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
      --dropout)
      DROPOUT="$2"
      shift # past argument
      shift # past value
      ;;
      --object_pos_encoding_dim)
      OBJECT_POS_ENCODING_DIM="$2"
      shift # past argument
      shift # past value
      ;;
      --surface_pos_encoding_dim)
      SURFACE_POS_ENCODING_DIM="$2"
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
      --loss_fn)
      LOSS_FN="$2"
      shift # past argument
      shift # past value
      ;;
      --triplet_loss_margin)
      TRIPLET_LOSS_MARGIN="$2"
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
    CMD="srun -u $PYTHONPATH -u train_consor.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --dataset $DATASET \
      --fold $FOLD \
      --save_tag $SAVE_TAG \
      --num_epochs $NUM_EPOCHS \
      --semantic_embb_dim $SEMANTIC_DIM \
      --hidden_layer_size $HIDDEN_LAYER_SIZE \
      --output_dimension $OUTPUT_DIMENSION \
      --num_heads $NUM_HEADS \
      --num_layers $NUM_LAYERS \
      --dropout $DROPOUT \
      --object_pos_encoding_dim $OBJECT_POS_ENCODING_DIM \
      --surface_pos_encoding_dim $SURFACE_POS_ENCODING_DIM \
      --batch_size $BATCH_SIZE \
      --lrate $LRATE \
      --wt_decay $WT_DECAY \
      --loss_fn $LOSS_FN \
      --triplet_loss_margin $TRIPLET_LOSS_MARGIN"
elif [ $DEVICE = "hopper" ]
  then
    source ~/.bashrc
    cd /media/kartik/disk-2/robo_declutter
    export PYTHONPATH=/home/kartik/miniconda3/envs/declutter/bin/python
    export DATA_HOME=/media/kartik/disk-2/Data/declutter_user_data
    CMD="$PYTHONPATH -u train_consor.py \
      --embedding $DATA_HOME/object_embeddings/object_embeddings_all-MiniLM-L6-v2.pt \
      --dataset $DATASET \
      --fold $FOLD \
      --save_tag $SAVE_TAG \
      --num_epochs $NUM_EPOCHS \
      --semantic_embb_dim $SEMANTIC_DIM \
      --hidden_layer_size $HIDDEN_LAYER_SIZE \
      --output_dimension $OUTPUT_DIMENSION \
      --num_heads $NUM_HEADS \
      --num_layers $NUM_LAYERS \
      --dropout $DROPOUT \
      --object_pos_encoding_dim $OBJECT_POS_ENCODING_DIM \
      --surface_pos_encoding_dim $SURFACE_POS_ENCODING_DIM \
      --batch_size $BATCH_SIZE \
      --lrate $LRATE \
      --wt_decay $WT_DECAY \
      --loss_fn $LOSS_FN \
      --triplet_loss_margin $TRIPLET_LOSS_MARGIN"
else
	echo "Invalid argument for keyword arg 'device', must be either 'skynet' or 'hopper'"
fi

if [ "$WANDB" == true ]; then
  CMD="$CMD --wandb"
fi
echo "Running command: $CMD"
eval $CMD
echo Done
