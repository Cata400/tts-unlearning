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
GEN_WAV_DIR=$1

if [ -z "$2" ]
then
    echo "Using default PROCESSED_LIBRITTS_PATH: /alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/"
    PROCESSED_LIBRITTS_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
else
    echo "PROCESSED_LIBRITTS_PATH: $2"
    PROCESSED_LIBRITTS_PATH=$2
fi

if [ -z "$3" ]
then
    echo "Using default CONFIG_NAME: F5TTS_v1_Base_unlearn"
    CONFIG_NAME=F5TTS_v1_Base_unlearn
else
    echo "CONFIG_NAME: $3"
    CONFIG_NAME=$3
fi

if [ -z "$4" ]
then
    echo "Using default GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL: /home/catalin.ciocirlan/phd/tts-unlearning/results/pretrained_unconditional"
    GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL=/home/catalin.ciocirlan/phd/tts-unlearning/results/pretrained_unconditional
else
    echo "GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL: $4"
    GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL=$4
fi


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

echo "Evaluating SIM"
python3 ./eval/eval_libritts.py --eval_task sim --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
echo "Evaluating WER"
python3 ./eval/eval_libritts.py --eval_task wer --config_name $CONFIG_NAME  --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
echo "Evaluating spk-ZRF"
python3 ./eval/eval_libritts.py --eval_task spk-ZRF --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --gen_wav_dir_pretrained_unconditional $GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL
