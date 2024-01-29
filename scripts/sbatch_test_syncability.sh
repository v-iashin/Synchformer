#!/bin/bash

##SBATCH --account=project_2000936
##SBATCH --nodes=8
##SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:v100:1,nvme:500
##SBATCH --cpus-per-task=10
##SBATCH --mem-per-gpu=85G

#SBATCH --job-name=probe
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose

# exit when any command fails
set -e

## The following will assign a master port (picked randomly to avoid collision) and an address for ddp.
# We want names of master and slave nodes. Make sure this node (MASTER_ADDR) comes first
MASTER_ADDR=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
fi

# Get a random unused port on this host(MASTER_ADDR)
MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR
echo "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT "WORKERS" $WORKERS

# for AMD GPUs at LUMI
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
echo "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM:" $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM

# loading conda environment
source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
conda activate synchformer

SYNC_LOGS_PATH=$SCRATCH/vladimir/logs/sync/sync_models/

S2_CKPT_ID="24-01-04T16-39-21"
S3_CKPT_ID="24-01-22T20-34-52"

EPOCH="best"
ROOT_FOLDER=$SCRATCH

if [[ "$HOSTNAME" == *"nid"* ]]; then
    ROOT_FOLDER=${SCRATCH/\/scratch/\/flash}
    echo "Running on LUMI"
fi

srun python ./scripts/test_syncability.py \
    config_sync="$SYNC_LOGS_PATH/$S3_CKPT_ID/cfg-$S3_CKPT_ID.yaml" \
    config_off="$SYNC_LOGS_PATH/$S2_CKPT_ID/cfg-$S2_CKPT_ID.yaml" \
    ckpt_path_sync="$SYNC_LOGS_PATH/$S3_CKPT_ID/${S3_CKPT_ID}_$EPOCH.pt" \
    ckpt_path_off="$SYNC_LOGS_PATH/${S2_CKPT_ID}/${S2_CKPT_ID}_$EPOCH.pt" \
    data.dataset.target=dataset.vggsound.VGGSoundSparsePickedCleanTest \
    data.vids_path="$ROOT_FOLDER/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/" \
    data.n_segments=14 \
    logging.log_code_state=False \
    training.finetune=False \
    training.run_test_only=True \
    data.dataset.params.iter_times=25 \
    logging.use_wandb=False \
    data.dataset.params.load_fixed_offsets_on="[]"
    # training.base_batch_size=4 \

srun python main.py \
    config="$SYNC_LOGS_PATH/$S3_CKPT_ID/cfg-$S3_CKPT_ID.yaml" \
    data.dataset.target="dataset.vggsound.VGGSoundSparsePicked" \
    data.vids_path="$ROOT_FOLDER/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/" \
    logging.log_code_state=False \
    training.finetune="False" \
    training.run_test_only="True" \
    data.iter_times="25" \
    data.dataset.params.load_fixed_offsets_on="[]"

srun python main.py \
    config="$SYNC_LOGS_PATH/$S3_CKPT_ID/cfg-$S3_CKPT_ID.yaml" \
    data.dataset.target="dataset.vggsound.VGGSoundSparsePickedCleanTest" data.vids_path="$ROOT_FOLDER/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/" logging.log_code_state=False training.finetune="False" training.run_test_only="True" data.iter_times="25" data.dataset.params.load_fixed_offsets_on="[]"

# srun python main.py \
#     config="$SYNC_LOGS_PATH/$S3_CKPT_ID/cfg-$S3_CKPT_ID.yaml" \
#     data.dataset.target="dataset.lrs.LRS3" data.vids_path="$ROOT_FOLDER/vladimir/data/lrs3/h264_uncropped_25fps_256side_16000hz_aac/" logging.log_code_state=False training.finetune="False" training.run_test_only="True" data.iter_times="2" data.dataset.params.load_fixed_offsets_on="[]"
