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
###   --expname NAME
###   --ckptstep STEP
###   --processed_libritts_dataset_path PATH
###   --seed SEED
###   --n_gpus N
### Positional args (backwards compatible):
###   $1: expname
###   $2: ckptstep (default: last)
###   $3: processed_libritts_dataset_path (default below)
###   $4: seed (default: 42)
###   $5: n_gpus (default: 1)
#################################

usage() {
    echo "Usage: $0 --expname NAME [--ckptstep STEP] [--processed_libritts_dataset_path PATH] [--seed SEED] [--n_gpus N]"
    echo "       $0 EXP_NAME [CKPTSTEP] [PROCESSED_PATH] [SEED] [N_GPUS]"
}

EXPNAME=""
CKPTSTEP=last
PROCESSED_LIBRITTS_DATASET_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
SEED=42
N_GPUS=1

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --expname|--config)
            EXPNAME="$2"
            shift 2
            ;;
        --ckptstep)
            CKPTSTEP="$2"
            shift 2
            ;;
        --processed_libritts_dataset_path|--data_path)
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

if [ -z "$EXPNAME" ] && [ "${#POSITIONAL[@]}" -ge 1 ]; then
    EXPNAME="${POSITIONAL[0]}"
fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then
    CKPTSTEP="${POSITIONAL[1]}"
fi
if [ "${#POSITIONAL[@]}" -ge 3 ]; then
    PROCESSED_LIBRITTS_DATASET_PATH="${POSITIONAL[2]}"
fi
if [ "${#POSITIONAL[@]}" -ge 4 ]; then
    SEED="${POSITIONAL[3]}"
fi
if [ "${#POSITIONAL[@]}" -ge 5 ]; then
    N_GPUS="${POSITIONAL[4]}"
fi

if [ -z "$EXPNAME" ]; then
    echo "Missing required --expname (or positional EXP_NAME)."
    usage
    exit 1
fi

echo "EXPNAME: $EXPNAME"
echo "CKPTSTEP: $CKPTSTEP"
echo "PROCESSED_LIBRITTS_DATASET_PATH: $PROCESSED_LIBRITTS_DATASET_PATH"
echo "SEED: $SEED"
echo "N_GPUS: $N_GPUS"


source /etc/profile.d/modules.sh
module load anaconda/3
module load cuda/12.6-9.5
conda activate f5-tts

if [ "$N_GPUS" -gt 1 ]
then
    accelerate launch ./eval/eval_libritts_infer_batch.py --expname $EXPNAME --ckptstep $CKPTSTEP --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH --seed $SEED
else
    python3 ./eval/eval_libritts_infer_batch.py --expname $EXPNAME --ckptstep $CKPTSTEP --processed_libritts_dataset_path $PROCESSED_LIBRITTS_DATASET_PATH --seed $SEED
fi
