#!/bin/bash
#SBATCH --job-name=catalin_unlearn
#SBATCH --time=99:00:00  # hh:mm:ss . It is usually a good idea to limit your job.
#SBATCH --output=./remote/logs/output_%A.log   # %x_%j_%N.log # %A is the job id
#SBATCH --error=./remote/logs/error_%A.log  # %x_%j_%N.log # %A is the job id
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -p supermicro

# Submit from src/f5_tts:  sbatch remote/continual/script_eval_continual.sh --expname F5TTS_v1_Base_unlearn_continual
#
# One python3 call evaluates every metric for ALL continual steps - the per-step generated-wav
# directories are reconstructed from the config, so no --gen_wav_dir is needed (unlike
# remote/script_eval.sh). A step with no generated wavs yet is reported and skipped. On top of the
# usual per-step results this writes a continual_avg_* block (current / past / future / cumulative
# forget + retain) and a chain-level step x forget-speaker summary matrix under
#   results/{ckpt_dir_name}/_{testset}_{task}_results[_{sim_model}]_continual_summary.json

########### Arguments ###########
### Named args (preferred):
###   --expname NAME                          (default: F5TTS_v1_Base_unlearn_continual)
###   --processed_libritts_path PATH
###   --eval_tasks "sim wer ..."                  (default: all of them)
###   --sim_model_type TYPE
###   --seed SEED / --nfestep N / --odemethod M / --swaysampling S / --cfg_strength C / --speed S
###   --gpu_nums N
###   --gen_wav_dir_pretrained DIR                (delta_sim)
###   --embeddings_dir_gt DIR                     (delta_sim)
###   --embeddings_dir_pretrained DIR             (delta_sim)
### Positional args are rejected: `$1` used to mean processed_libritts_path, so passing the config
### name first silently renamed the test set in every output path.
#################################

usage() {
    echo "Usage: $0 [--expname NAME] [--processed_libritts_path PATH] [--eval_tasks \"sim wer\"] [--sim_model_type TYPE] [...]"
}

EXPNAME=F5TTS_v1_Base_unlearn_continual
PROCESSED_LIBRITTS_PATH=/alpha/catalin.ciocirlan/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
EVAL_TASKS="sim sim_gt_matching delta_sim wer utmosv2"
SIM_MODEL_TYPE=speechbrain_ecapa
SEED=42
NFE_STEP=32
ODE_METHOD=euler
SWAY_SAMPLING=-1
CFG_STRENGTH=2.0
SPEED=1.0
GPU_NUMS=1
GEN_WAV_DIR_PRETRAINED=/home/catalin.ciocirlan/phd/tts-unlearning/results/pretrained/train-clean-100_val_intra_speaker_split_0.2/seed42_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0
EMBEDDINGS_DIR_GT=/home/catalin.ciocirlan/phd/tts-unlearning/results/embeddings_speechbrain_ecapa_train-clean-100_val_intra_speaker_split_0.2_gt
EMBEDDINGS_DIR_PRETRAINED=/home/catalin.ciocirlan/phd/tts-unlearning/results/embeddings_speechbrain_ecapa_train-clean-100_val_intra_speaker_split_0.2_pretrained

while [ $# -gt 0 ]; do
    case "$1" in
        --expname|--config_name|--config)
            EXPNAME="$2"
            shift 2
            ;;
        --processed_libritts_path|--processed_libritts_dataset_path|--data_path)
            PROCESSED_LIBRITTS_PATH="$2"
            shift 2
            ;;
        --eval_tasks|--eval_task)
            EVAL_TASKS="$2"
            shift 2
            ;;
        --sim_model_type)
            SIM_MODEL_TYPE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --nfestep|-nfe)
            NFE_STEP="$2"
            shift 2
            ;;
        --odemethod|-o)
            ODE_METHOD="$2"
            shift 2
            ;;
        --swaysampling|-ss)
            SWAY_SAMPLING="$2"
            shift 2
            ;;
        --cfg_strength)
            CFG_STRENGTH="$2"
            shift 2
            ;;
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --gpu_nums|--n_gpus|--ngpus)
            GPU_NUMS="$2"
            shift 2
            ;;
        --gen_wav_dir_pretrained)
            GEN_WAV_DIR_PRETRAINED="$2"
            shift 2
            ;;
        --embeddings_dir_gt)
            EMBEDDINGS_DIR_GT="$2"
            shift 2
            ;;
        --embeddings_dir_pretrained)
            EMBEDDINGS_DIR_PRETRAINED="$2"
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
            echo "Positional arguments are not accepted: $1. Use the named flags, e.g. --config_name."
            usage
            exit 1
            ;;
    esac
done

echo "EXPNAME: $EXPNAME"
echo "PROCESSED_LIBRITTS_PATH: $PROCESSED_LIBRITTS_PATH"
echo "EVAL_TASKS: $EVAL_TASKS"
echo "SIM_MODEL_TYPE: $SIM_MODEL_TYPE"
echo "SEED: $SEED"
echo "NFE_STEP: $NFE_STEP / ODE_METHOD: $ODE_METHOD / SWAY_SAMPLING: $SWAY_SAMPLING / CFG_STRENGTH: $CFG_STRENGTH / SPEED: $SPEED"
echo "GPU_NUMS: $GPU_NUMS"
echo "GEN_WAV_DIR_PRETRAINED (delta_sim): $GEN_WAV_DIR_PRETRAINED"
echo "EMBEDDINGS_DIR_GT (delta_sim): $EMBEDDINGS_DIR_GT"
echo "EMBEDDINGS_DIR_PRETRAINED (delta_sim): $EMBEDDINGS_DIR_PRETRAINED"


source /etc/profile.d/modules.sh
module load anaconda/3
# module load cuda/12.6-9.5
conda activate f5-tts

# arguments shared by every task: they select the config, the test set and the per-step wav dirs
COMMON_ARGS=(--config_name "$EXPNAME" --processed_libritts_path "$PROCESSED_LIBRITTS_PATH"
             --seed "$SEED" -nfe "$NFE_STEP" -o "$ODE_METHOD" -ss "$SWAY_SAMPLING"
             --cfg_strength "$CFG_STRENGTH" --speed "$SPEED" --gpu_nums "$GPU_NUMS")

SIM_ARGS=(--sim_model_type "$SIM_MODEL_TYPE")
DELTA_SIM_ARGS=(--gen_wav_dir_pretrained "$GEN_WAV_DIR_PRETRAINED"
                --embeddings_dir_gt "$EMBEDDINGS_DIR_GT"
                --embeddings_dir_pretrained "$EMBEDDINGS_DIR_PRETRAINED")

echo "Evaluating all steps for tasks: $EVAL_TASKS"
python3 ./eval/eval_libritts_continual.py --eval_tasks $EVAL_TASKS "${COMMON_ARGS[@]}" "${SIM_ARGS[@]}" "${DELTA_SIM_ARGS[@]}"
