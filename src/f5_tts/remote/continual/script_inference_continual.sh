#!/bin/bash
#SBATCH --job-name=catalin_unlearn
#SBATCH --time=99:00:00  # hh:mm:ss . It is usually a good idea to limit your job.
#SBATCH --output=./remote/logs/output_%A.log   # %x_%j_%N.log # %A is the job id
#SBATCH --error=./remote/logs/error_%A.log  # %x_%j_%N.log # %A is the job id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p supermicro

# Submit from src/f5_tts:  sbatch remote/continual/script_inference_continual.sh --expname F5TTS_v1_Base_unlearn_continual
#
# Generates the full test set once per continual step, into
#   results/{ckpt_dir_name}_step{i}_spk{id}/{testset}/seed..._nfe..._cfg..._speed...
# There is deliberately NO --svdiff / --svdiff_uv flag here: continual checkpoints are materialised
# (the SVD deltas are baked into plain .weight tensors), so they load with the stock loader.

########### Arguments ###########
### Named args (preferred):
###   --expname NAME                          (default: F5TTS_v1_Base_unlearn_continual)
###   --processed_libritts_dataset_path PATH
###   --seed SEED
###   --n_gpus N
###   --no_use_ema      generate from the online weights (the ones the chain is built from)
###   --skip_existing   skip a step whose output dir already holds a wav for every prompt
### Positional args (backwards compatible with remote/script_inference.sh):
###   $1: expname
###   $2: processed_libritts_dataset_path
###   $3: seed (default: 42)
###   $4: n_gpus (default: 1)
#################################

usage() {
    echo "Usage: $0 [--expname NAME] [--processed_libritts_dataset_path PATH] [--seed SEED] [--n_gpus N] [--no_use_ema] [--skip_existing]"
    echo "       $0 EXP_NAME [PROCESSED_PATH] [SEED] [N_GPUS]"
}

EXPNAME=F5TTS_v1_Base_unlearn_continual
PROCESSED_LIBRITTS_DATASET_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
SEED=42
N_GPUS=1
USE_EMA=1
SKIP_EXISTING=0

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --expname|--config|--config_name)
            EXPNAME="$2"
            shift 2
            ;;
        --processed_libritts_dataset_path|--processed_libritts_path|--data_path)
            PROCESSED_LIBRITTS_DATASET_PATH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --n_gpus|--ngpus)
            N_GPUS="$2"
            shift 2
            ;;
        --use_ema)
            USE_EMA=1
            shift
            ;;
        --no_use_ema|--no-use_ema)
            USE_EMA=0
            shift
            ;;
        --skip_existing)
            SKIP_EXISTING=1
            shift
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

if [ "${#POSITIONAL[@]}" -ge 1 ]; then
    EXPNAME="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then
    PROCESSED_LIBRITTS_DATASET_PATH="${POSITIONAL[1]}"
fi
if [ "${#POSITIONAL[@]}" -ge 3 ]; then
    SEED="${POSITIONAL[2]}"
fi
if [ "${#POSITIONAL[@]}" -ge 4 ]; then
    N_GPUS="${POSITIONAL[3]}"
fi

if [ -z "$EXPNAME" ]; then
    echo "Missing --expname (or positional EXP_NAME)."
    usage
    exit 1
fi

echo "EXPNAME: $EXPNAME"
echo "PROCESSED_LIBRITTS_DATASET_PATH: $PROCESSED_LIBRITTS_DATASET_PATH"
echo "SEED: $SEED"
echo "N_GPUS: $N_GPUS"
echo "USE_EMA: $USE_EMA"
echo "SKIP_EXISTING: $SKIP_EXISTING"


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

EXTRA_ARGS=()
if [ "$USE_EMA" -eq 0 ]; then
    EXTRA_ARGS+=(--no-use_ema)
fi
if [ "$SKIP_EXISTING" -eq 1 ]; then
    EXTRA_ARGS+=(--skip_existing)
fi

if [ "$N_GPUS" -gt 1 ]
then
    accelerate launch ./eval/eval_libritts_infer_batch_continual.py --expname $EXPNAME --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH --seed $SEED "${EXTRA_ARGS[@]}"
else
    python3 ./eval/eval_libritts_infer_batch_continual.py --expname $EXPNAME --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH --seed $SEED "${EXTRA_ARGS[@]}"
fi
