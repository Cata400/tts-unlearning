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

if [ -z "$1" ]
then
    echo "Using default GEN_WAV_DIR: ../../results/F5TTS_v1_Base_unlearn_last/train-clean-100_val_intra_speaker_split_0.2/seed42_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0/"
    GEN_WAV_DIR=../../results/F5TTS_v1_Base_unlearn_last/train-clean-100_val_intra_speaker_split_0.2/seed42_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0/
else
    echo "GEN_WAV_DIR: $1"
    GEN_WAV_DIR=$1
fi

if [ -z "$2" ]
then
    echo "Using default PROCESSED_LIBRITTS_PATH: /alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/"
    PROCESSED_LIBRITTS_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
else
    echo "PROCESSED_LIBRITTS_PATH: $2"
    PROCESSED_LIBRITTS_PATH=$2
fi


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

echo "Evaluating SIM"
python3 ./eval/eval_libritts.py --eval_task sim --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
echo "Evaluating WER"
python3 ./eval/eval_libritts.py --eval_task wer --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
