########### Arguments ###########
### Named args (preferred):
###   --gen_wav_dir DIR
###   --processed_libritts_path PATH
###   --config_name NAME
###   --gen_wav_dir_pretrained_unconditional DIR
###   --gen_wav_dir_pretrained DIR
###   --embeddings_dir_gt DIR
###   --embeddings_dir_pretrained DIR
###   --sim_model_type TYPE
### Positional args (backwards compatible):
###   $1: gen_wav_dir
###   $2: processed_libritts_path (default below)
###   $3: config_name (default below)
###   $4: gen_wav_dir_pretrained_unconditional (default below)
###   $5: sim_model_type (default: speechbrain_ecapa)
###   $6: gen_wav_dir_pretrained (delta_sim)
###   $7: embeddings_dir_gt (delta_sim)
###   $8: embeddings_dir_pretrained (delta_sim)
#################################

usage() {
    echo "Usage: $0 --gen_wav_dir DIR [--processed_libritts_path PATH] [--config_name NAME] [--gen_wav_dir_pretrained_unconditional DIR] [--sim_model_type TYPE] [--gen_wav_dir_pretrained DIR] [--embeddings_dir_gt DIR] [--embeddings_dir_pretrained DIR]"
    echo "       $0 GEN_WAV_DIR [PROCESSED_PATH] [CONFIG_NAME] [GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL] [SIM_MODEL_TYPE] [GEN_WAV_DIR_PRETRAINED] [EMBEDDINGS_DIR_GT] [EMBEDDINGS_DIR_PRETRAINED]"
}

GEN_WAV_DIR=""
PROCESSED_LIBRITTS_PATH=/home/catalin/Desktop/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2/
CONFIG_NAME=F5TTS_v1_Base_unlearn
GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL=/home/catalin/Desktop/Projects/tts-unlearning/results/pretrained_unconditional/
SIM_MODEL_TYPE=speechbrain_ecapa
GEN_WAV_DIR_PRETRAINED=/home/catalin/Desktop/Projects/tts-unlearning/results/F5TTS_v1_Base_unlearn_last/train-clean-100_val_intra_speaker_split_0.2/server_experiments/F5TTS_v1_Base_last/train-clean-100_val_intra_speaker_split_0.2/seed42_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0
EMBEDDINGS_DIR_GT=/home/catalin/Desktop/Projects/tts-unlearning/results/embeddings_speechbrain_ecapa_train-clean-100_val_intra_speaker_split_0.2_gt
EMBEDDINGS_DIR_PRETRAINED=/home/catalin/Desktop/Projects/tts-unlearning/results/embeddings_speechbrain_ecapa_train-clean-100_val_intra_speaker_split_0.2_pretrained

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
        --gen_wav_dir_pretrained_unconditional)
            GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL="$2"
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
        --sim_model_type)
            SIM_MODEL_TYPE="$2"
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
if [ "${#POSITIONAL[@]}" -ge 5 ]; then
    SIM_MODEL_TYPE="${POSITIONAL[4]}"
fi
if [ "${#POSITIONAL[@]}" -ge 6 ]; then
    GEN_WAV_DIR_PRETRAINED="${POSITIONAL[5]}"
fi
if [ "${#POSITIONAL[@]}" -ge 7 ]; then
    EMBEDDINGS_DIR_GT="${POSITIONAL[6]}"
fi
if [ "${#POSITIONAL[@]}" -ge 8 ]; then
    EMBEDDINGS_DIR_PRETRAINED="${POSITIONAL[7]}"
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
echo "SIM_MODEL_TYPE: $SIM_MODEL_TYPE"
echo "GEN_WAV_DIR_PRETRAINED (delta_sim): $GEN_WAV_DIR_PRETRAINED"
echo "EMBEDDINGS_DIR_GT (delta_sim): $EMBEDDINGS_DIR_GT"
echo "EMBEDDINGS_DIR_PRETRAINED (delta_sim): $EMBEDDINGS_DIR_PRETRAINED"


echo "Evaluating SIM"
python3 ./eval/eval_libritts.py --eval_task sim --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --sim_model_type $SIM_MODEL_TYPE
echo "Evaluating SIM_GT_MATCHING"
python3 ./eval/eval_libritts.py --eval_task sim_gt_matching --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --sim_model_type $SIM_MODEL_TYPE
echo "Evaluating WER"
python3 ./eval/eval_libritts.py --eval_task wer --config_name $CONFIG_NAME  --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
echo "Evaluating spk-ZRF"
python3 ./eval/eval_libritts.py --eval_task spk-ZRF --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --gen_wav_dir_pretrained_unconditional $GEN_WAV_DIR_PRETRAINED_UNCONDITIONAL --sim_model_type $SIM_MODEL_TYPE
echo "Evaluating DIVERSITY"
python3 ./eval/eval_libritts.py --eval_task diversity --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --sim_model_type $SIM_MODEL_TYPE
echo "Evaluating UTMOSv2"
python3 ./eval/eval_libritts.py --eval_task utmosv2 --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH
echo "Evaluating DELTA_SIM"
python3 ./eval/eval_libritts.py --eval_task delta_sim --config_name $CONFIG_NAME --gen_wav_dir $GEN_WAV_DIR --processed_libritts_path $PROCESSED_LIBRITTS_PATH --sim_model_type $SIM_MODEL_TYPE --gen_wav_dir_pretrained $GEN_WAV_DIR_PRETRAINED --embeddings_dir_gt $EMBEDDINGS_DIR_GT --embeddings_dir_pretrained $EMBEDDINGS_DIR_PRETRAINED
