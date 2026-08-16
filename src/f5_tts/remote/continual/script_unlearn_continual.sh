#!/bin/bash
#SBATCH --job-name=catalin_unlearn
#SBATCH --time=999:00:00  # hh:mm:ss . It is usually a good idea to limit your job.
#SBATCH --output=./remote/logs/output_%A.log   # %x_%j_%N.log # %A is the job id
#SBATCH --error=./remote/logs/error_%A.log  # %x_%j_%N.log # %A is the job id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p supermicro

# Submit from src/f5_tts:  sbatch remote/continual/script_unlearn_continual.sh --config F5TTS_v1_Base_unlearn_continual
#
# Runs the WHOLE continual chain in one job: one unlearning step per speaker in
# unlearn.forget_speakers, each step starting from the previous step's checkpoint. With
# continual.resume=True (the config default) a requeued job picks up at the first step whose final
# checkpoint is missing, so hitting the time limit mid-chain is recoverable.

########### Arguments ###########
### Named args (preferred):
###   --config CONFIG
###   --n_gpus N
### Positional args (backwards compatible with remote/script_unlearn.sh):
###   $1: config
###   $2: n_gpus (default: 1)
#################################

usage() {
    echo "Usage: $0 --config CONFIG [--n_gpus N]"
    echo "       $0 CONFIG [N_GPUS]"
}

CONFIG=""
N_GPUS=1

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config|--config_name)
            CONFIG="$2"
            shift 2
            ;;
        --n_gpus|--ngpus)
            N_GPUS="$2"
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
            if [[ "$1" == *=* ]]; then
                echo "Hydra overrides are not accepted here: $1. Set it in the config instead."
                exit 1
            fi
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ] && [ "${#POSITIONAL[@]}" -ge 1 ]; then
    CONFIG="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then
    N_GPUS="${POSITIONAL[1]}"
fi

if [ -z "$CONFIG" ]; then
    echo "Missing required --config (or positional CONFIG)."
    usage
    exit 1
fi

echo "CONFIG: $CONFIG"
echo "N_GPUS: $N_GPUS"


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

if [ "$N_GPUS" -gt 1 ]
then
    accelerate launch ./train/unlearn_continual.py --config-name "$CONFIG"
else
    python3 ./train/unlearn_continual.py --config-name "$CONFIG"
fi
