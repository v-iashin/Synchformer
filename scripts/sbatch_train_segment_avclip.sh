#!/bin/bash

# Define the following from CLI
##SBATCH --job-name=ts
##SBATCH --account=project_2000936
##SBATCH --nodes=8
##SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:v100:1,nvme:500
##SBATCH --cpus-per-task=10
##SBATCH --mem-per-gpu=85G

#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose

# argparse. it is used by a submitting script (`./scripts/submit_job.sh`) and can be ignored
for i in "$@"; do
  case $i in
    -n=*|--now=*)
      NOW="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

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
echo "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM: " $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM

# loading conda environment
source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
conda activate synchformer

## select the dataset
# DATASET="dataset.lrs.LRS3" VIDS_PATH="/flash/project_462000293/vladimir/data/lrs3/h264_uncropped_25fps_256side_16000hz_aac/"
# DATASET="dataset.vggsound.VGGSound" VIDS_PATH="/flash/project_462000293/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/"
DATASET="dataset.audioset.AudioSet" VIDS_PATH="/scratch/project_462000293/vladimir/data/audioset/h264_video_25fps_256side_16000hz_aac/"

# # for LRS3
# srun python main.py \
#     start_time="$NOW" \
#     config="./configs/segment_avclip.yaml" \
#     data.vids_path="$VIDS_PATH" \
#     data.dataset.target="$DATASET" \
#     training.num_workers=7 \
#     logging.logdir="$SCRATCH/vladimir/logs/sync/avclip_models" \
#     logging.use_wandb=True \
#     data.sometimes_upscale_p=0.0 \
#     data.p_color_jitter=0.0 \
#     data.p_gray_scale=0.0 \
#     data.p_audio_aug=0.0 \
#     training.learning_rate=0.00005 \

# for VGGSound and AudioSet
srun python main.py \
    start_time="$NOW" \
    config="./configs/segment_avclip.yaml" \
    data.vids_path="$VIDS_PATH" \
    data.dataset.target="$DATASET" \
    training.num_workers=7 \
    logging.logdir="$SCRATCH/vladimir/logs/sync/avclip_models" \
    logging.use_wandb=True


# # for LRS3: ResNet18 and S3D features instead of AST and ViViT
# srun python main.py \
#     start_time="$NOW" \
#     config="./configs/segment_avclip.yaml" \
#     data.vids_path="$VIDS_PATH" \
#     data.dataset.target="$DATASET" \
#     training.num_workers=7 \
#     logging.logdir="$SCRATCH/vladimir/logs/sync/avclip_models" \
#     logging.use_wandb=True \
#     data.sometimes_upscale_p=0.0 \
#     data.p_color_jitter=0.0 \
#     data.p_gray_scale=0.0 \
#     data.p_audio_aug=0.0 \
#     training.learning_rate=0.00005 \
#     model.params.vfeat_extractor.target=model.modules.feat_extractors.visual.s3d.S3DVisualFeatures \
#     model.params.afeat_extractor.target=model.modules.feat_extractors.audio.resnet.ResNet18AudioFeatures \
#     model.params.vfeat_extractor.params.ckpt_path=./model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt \
#     model.params.afeat_extractor.params.ckpt_path=./model/modules/feat_extractors/audio/22-06-24T08-10-33/ResNetAudio-22-06-24T08-10-33.pt \
#     model.params.vproj.target=torch.nn.Linear model.params.vproj.params.in_features=1024 model.params.vproj.params.out_features=768 \
#     model.params.aproj.target=torch.nn.Linear model.params.aproj.params.in_features=512 model.params.aproj.params.out_features=768 \

# # for VGGSound and AudioSet: ResNet18 and S3D features instead of AST and ViViT
# srun python main.py \
#     start_time="$NOW" \
#     config="./configs/segment_avclip.yaml" \
#     data.vids_path="$VIDS_PATH" \
#     data.dataset.target="$DATASET" \
#     training.num_workers=7 \
#     logging.logdir="$SCRATCH/vladimir/logs/sync/avclip_models" \
#     logging.use_wandb=True \
#     model.params.vfeat_extractor.target=model.modules.feat_extractors.visual.s3d.S3DVisualFeatures \
#     model.params.afeat_extractor.target=model.modules.feat_extractors.audio.resnet.ResNet18AudioFeatures \
#     model.params.vfeat_extractor.params.ckpt_path=./model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt \
#     model.params.afeat_extractor.params.ckpt_path=./model/modules/feat_extractors/audio/22-06-24T08-10-33/ResNetAudio-22-06-24T08-10-33.pt \
#     model.params.vproj.target=torch.nn.Linear model.params.vproj.params.in_features=1024 model.params.vproj.params.out_features=768 \
#     model.params.aproj.target=torch.nn.Linear model.params.aproj.params.in_features=512 model.params.aproj.params.out_features=768 \
