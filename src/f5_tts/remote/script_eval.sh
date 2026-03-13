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
### Named args (preferred):
###   --gen_wav_dir DIR
###   --processed_libritts_path PATH
###   --config_name NAME
###   --gen_wav_dir_pretrained_unconditional DIR
### Positional args (backwards compatible):
###   $1: gen_wav_dir
###   $2: processed_libritts_path (default below)
###   $3: config_name (default below)
###   $4: gen_wav_dir_pretrained_unconditional (default below)
#################################

usage() {
    echo "Usage: $0 --gen_wav_dir DIR [--processed_libritts_path PATH] [--config_name NAME] [--gen_wav_dir_pretrained_unconditional DIR]"
    echo "       $0 GEN_WAV_DIR [PROCESSED_PATH] [CONFIG_NAME] [GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL]"
}

GEN_WAV_DIR=""
PROCESSED_LIBRITTS_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
CONFIG_NAME=F5TTS_v1_Base_unlearn
GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL=/home/catalin.ciocirlan/phd/tts-unlearning/results/pretrained_unconditional

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gen_wav_dir|--gen_wavs|--gen_dir)
            GEN_WAV_DIR="$2"
            shift 2
            ;;
        --processed_libritts_path|--data_path)
            PROCESSED_LIBRITTS_PATH="$2"
            shift 2
            ;;
        --config_name|--config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --gen_wav_dir_pretrained_unconditional|--gen_wav_dir_pretrained)
            GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ -z "$GEN_WAV_DIR" ] && [ "${#POSITIONAL[@]}" -ge 1 ]; then
    GEN_WAV_DIR="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then
    PROCESSED_LIBRITTS_PATH="${POSITIONAL[1]}"
fi
if [ "${#POSITIONAL[@]}" -ge 3 ]; then
    CONFIG_NAME="${POSITIONAL[2]}"
fi
if [ "${#POSITIONAL[@]}" -ge 4 ]; then
    GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL="${POSITIONAL[3]}"
fi

if [ -z "$GEN_WAV_DIR" ]; then
    echo "Missing required --gen_wav_dir (or positional GEN_WAV_DIR)."
    usage
    exit 1
fi

echo "GEN_WAV_DIR: $GEN_WAV_DIR"
echo "PROCESSED_LIBRITTS_PATH: $PROCESSED_LIBRITTS_PATH"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL: $GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL"


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
