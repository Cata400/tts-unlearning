#!/bin/bash
#SBATCH --job-name=catalin_unlearn
#SBATCH --time=99:00:00  # hh:mm:ss . It is usually a good idea to limit your job.
#SBATCH --output=./remote/logs/output_%A.log   # %x_%j_%N.log # %A is the job id
#SBATCH --error=./remote/logs/error_%A.log  # %x_%j_%N.log # %A is the job id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p supermicro


########### Arguments ###########
### $1: config
### $2: number of gpus (optional, default: 1)
#################################

EXPNAME=$1
CKPTSTEP=$2
echo "EXPNAME: $EXPNAME, CKPTSTEP: $CKPTSTEP"

if [ -z "$3" ]
then
    echo "Using default PROCESSED_LIBRITTS_DATASET_PATH: /alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/"
    PROCESSED_LIBRITTS_DATASET_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
else
    echo "PROCESSED_LIBRITTS_DATASET_PATH: $3"
    PROCESSED_LIBRITTS_DATASET_PATH=$3
fi

if [ -z "$4" ]
then
    echo "Using default N_GPUS: 1"
    N_GPUS=1
else
    echo "N_GPUS: $4"
    N_GPUS=$4
fi


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

if [ "$N_GPUS" -gt 1 ]
then
    accelerate launch ./eval/eval_libritts_infer_batch.py --expname $EXPNAME --ckptstep $CKPTSTEP --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH
else
    python3 ./eval/eval_libritts_infer_batch.py --expname $EXPNAME --ckptstep $CKPTSTEP --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH
fi