#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=00:3:00:00
#$ -o outputs/llava/$JOB_ID.log
#$ -e outputs/llava/$JOB_ID.log
#$ -p -3

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# switch virtual env
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# training settings
PROMPT_VERSION=v1
CHECKPOINT_DIR=/gs/bs/tgh-24IDU/hf-checkpoints/vicuna-7b-v1.3
CHECKPOINT_SAVE_DIR=/gs/bs/tgh-24IDU/checkpoints/llava-7b/finetune

GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$(($GLOBAL_BATCH_SIZE / $NUM_GPUS / $MICRO_BATCH_SIZE))

TRAIN_EPOCHS=3
LR=2e-5
WEIGHT_DECAY=0.0
SEQ_LEN=2048

# huggingface cache
export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"
export HF_CACHE="/gs/bs/tge-gc24sp03/cache"
export HF_HOME="/gs/bs/tge-gc24sp03/cache"

# wandb
export WANDB_ENTITY="okoge"
export WANDB_PROJECT="llava"
export WANDB_NAME="LLaVA-7b-finetune-LR_${LR}-SEQ_${SEQ_LEN}-EPOCH_${TRAIN_EPOCHS}"

# training
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x LD_LIBRARY_PATH \
  -x USE_MPI=1 \
  -x PATH \
  -bind-to none \
  python llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $CHECKPOINT_DIR \
    --version $PROMPT_VERSION \
    --data_path /gs/bs/tgh-24IDU/datasets/vlm/llava/LLaVA-Instruct-150K/llava_instruct_150k.json \
    --image_folder /gs/bs/tgh-24IDU/datasets/vlm/llava/LLaVA-Instruct-150K/images/coco/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /gs/bs/tgh-24IDU/checkpoints/llava-7b/pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $CHECKPOINT_SAVE_DIR \
    --num_train_epochs $TRAIN_EPOCHS \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $SEQ_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
