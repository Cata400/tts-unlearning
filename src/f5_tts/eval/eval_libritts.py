# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import ast
import json
import os
import random
import sys
import warnings
from importlib.resources import files

import numpy as np
import torch
from omegaconf import OmegaConf

from f5_tts.eval.utils_eval import (
    get_libritts_test,
    run_asr_wer,
    run_delta_sim,
    run_diversity,
    run_sim_gt_matching,
    run_sim_v2,
    run_spk_ZRF_pipeline_libritts,
    run_utmosv2,
)

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

rel_path = str(files("f5_tts").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument(
        "-e",
        "--eval_task",
        type=str,
        default="wer",
        choices=["sim", "wer", "spk-ZRF", "diversity", "utmosv2", "delta_sim", "sim_gt_matching"],
    )
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-p", "--processed_libritts_path", type=str, required=True)
    parser.add_argument(
        "-n", "--gpu_nums", type=str, default="1", help="Number of GPUs to use (e.g., 8) or GPU list (e.g., [0,1,2,3])"
    )
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    parser.add_argument(
        "--sim_model_type",
        type=str,
        default="speechbrain_ecapa",
        choices=["wavlm_large_finetune", "wavlm_base_plus_sv", "speechbrain_ecapa", "resemblyzer"],
    )
    parser.add_argument("--config_name", type=str, default="F5TTS_v1_Base_unlearn")
    # spk-ZRF args
    parser.add_argument(
        "--gen_wav_dir_pretrained_unconditional",
        type=str,
        default=None,
        help="Generated wav dir for θ⁻ (text + speaker prompt)",
    )
    parser.add_argument("--gen_name_tpl", type=str, default="{id}.wav")
    parser.add_argument("--skip_missing_gen", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    # delta_sim args
    parser.add_argument(
        "--embeddings_dir_gt",
        type=str,
        default=None,
        help="Directory containing ground truth embeddings",
    )
    parser.add_argument(
        "--gen_wav_dir_pretrained",
        type=str,
        default=None,
        help="Generated wav dir for the pretrained model",
    )
    parser.add_argument(
        "--embeddings_dir_pretrained",
        type=str,
        default=None,
        help="Directory containing pretrained model embeddings",
    )
    return parser.parse_args()


def parse_gpu_nums(gpu_nums_str):
    try:
        if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
            gpu_list = ast.literal_eval(gpu_nums_str)
            if isinstance(gpu_list, list):
                return gpu_list
        return list(range(int(gpu_nums_str)))
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            f"Invalid GPU specification: {gpu_nums_str}. Use a number (e.g., 8) or a list (e.g., [0,1,2,3])"
        )


def get_speaker_avg_results(full_results, metric_key):
    speaker_scores = {}
    for result in full_results:
        wav_name = result["wav"]
        speaker = wav_name.split("_", 1)[0]
        speaker_scores.setdefault(speaker, []).append(result[metric_key])

    speaker_avg_results = []
    for speaker, scores in speaker_scores.items():
        speaker_avg_results.append({"speaker": speaker, metric_key: round(float(np.mean(scores)), 5)})

    speaker_avg_results.sort(key=lambda x: int(x["speaker"]))
    return speaker_avg_results


def get_retain_forget_avg_results(speaker_avg_results, metric_key, forget_speakers):
    retain_results, forget_results = [], []
    for speaker_results in speaker_avg_results:
        if int(speaker_results["speaker"]) in forget_speakers:
            forget_results.append(speaker_results[metric_key])
        else:
            retain_results.append(speaker_results[metric_key])

    forget_results = torch.tensor(forget_results)
    retain_results = torch.tensor(retain_results)

    forget_avg = round(forget_results.mean().item(), 5)
    retain_avg = round(retain_results.mean().item(), 5)

    return {"retain_avg": retain_avg, "forget_avg": forget_avg}


def main():
    args = get_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_task = args.eval_task
    lang = args.lang
    processed_libritts_path = args.processed_libritts_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir
    sim_model_type = args.sim_model_type
    config_name = args.config_name
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{config_name}.yaml")))
    forget_speakers = model_cfg.unlearn.forget_speakers

    gpus = parse_gpu_nums(args.gpu_nums)
    if eval_task in ["sim", "wer", "diversity", "utmosv2", "delta_sim", "sim_gt_matching"]:
        print("Loading test set...")
        test_set = get_libritts_test(gen_wav_dir, gpus, processed_libritts_path)
    if eval_task == "delta_sim":
        assert args.embeddings_dir_gt is not None, "embeddings_dir_gt is required for delta_sim evaluation"
        assert (
            args.embeddings_dir_pretrained is not None
        ), "embeddings_dir_pretrained is required for delta_sim evaluation"

        if not os.path.exists(args.embeddings_dir_pretrained):
            os.makedirs(args.embeddings_dir_pretrained, exist_ok=True)

        if not os.path.exists(args.embeddings_dir_gt):
            os.makedirs(args.embeddings_dir_gt, exist_ok=True)

        assert (
            args.gen_wav_dir_pretrained is not None if not os.path.exists(args.embeddings_dir_pretrained) else True
        ), "gen_wav_dir_pretrained is required if embeddings_dir_pretrained does not exist"

        print("Loading pretrained test set for delta sim evaluation...")
        pretrained_test_set = get_libritts_test(args.gen_wav_dir_pretrained, gpus, processed_libritts_path)
        assert len(test_set) == len(
            pretrained_test_set
        ), "The number of samples in gen_wav_dir and gen_wav_dir_pretrained must be the same for delta_sim evaluation"

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir

    if eval_task in ["sim", "spk-ZRF", "diversity", "delta_sim", "sim_gt_matching"]:
        if sim_model_type == "wavlm_large_finetune":
            wavlm_ckpt_dir = os.path.join(rel_path, "ckpts", "UniSpeech", "wavlm_large_finetune.pth")
        elif sim_model_type == "wavlm_base_plus_sv":
            wavlm_ckpt_dir = "microsoft/wavlm-base-plus-sv"
        elif sim_model_type == "speechbrain_ecapa":
            wavlm_ckpt_dir = "speechbrain/spkrec-ecapa-voxceleb"
        elif sim_model_type == "wavlm_large":
            wavlm_ckpt_dir = "microsoft/wavlm-large"
        elif sim_model_type == "resemblyzer":
            wavlm_ckpt_dir = "resemblyzer"
        else:
            raise ValueError(f"Similarity model type {sim_model_type} is not available")
    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        full_results = run_asr_wer((test_set[0][0], lang, test_set[0][1], asr_ckpt_dir))
    elif eval_task == "sim":
        # full_results = run_sim((test_set[0][0], test_set[0][1], wavlm_ckpt_dir))
        full_results = run_sim_v2((test_set[0][0], test_set[0][1], wavlm_ckpt_dir, sim_model_type))
    elif eval_task == "spk-ZRF":
        full_results, _ = run_spk_ZRF_pipeline_libritts(args, wavlm_ckpt_dir)
    elif eval_task == "diversity":
        full_results = run_diversity((test_set[0][0], test_set[0][1], wavlm_ckpt_dir, sim_model_type))
    elif eval_task == "utmosv2":
        full_results = run_utmosv2(test_set[0][1])
    elif eval_task == "delta_sim":
        full_results = []
        speaker_avg_results = run_delta_sim(
            test_set[0][1],
            pretrained_test_set[0][1],
            wavlm_ckpt_dir,
            sim_model_type,
            args.embeddings_dir_gt,
            args.embeddings_dir_pretrained,
        )
    elif eval_task == "sim_gt_matching":
        full_results = run_sim_gt_matching(test_set[0][1], wavlm_ckpt_dir, sim_model_type)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    if (
        eval_task != "delta_sim"
    ):  # for delta_sim, we will compute speaker_avg_results inside run_delta_sim and directly use it for saving and averaging, since delta_sim is a speaker-level metric
        speaker_avg_results = get_speaker_avg_results(full_results, eval_task)
    else:
        speaker_avg_results.sort(key=lambda x: int(x["speaker"]))

    unlearning_avg_results = get_retain_forget_avg_results(speaker_avg_results, eval_task, forget_speakers)

    if eval_task == "delta_sim":
        full_results = speaker_avg_results

    for line in full_results:
        metrics.append(line[eval_task])
    metric = round(np.mean(metrics), 5)

    all_results = {
        f"avg_{eval_task}_results": metric,
        f"unlearning_avg_{eval_task}_results": unlearning_avg_results,
        f"speaker_avg_{eval_task}_results": speaker_avg_results,
        "all_results": full_results,
    }

    if eval_task == "delta_sim":
        unlearning_avg_results_sim_unlearned_gt_emb_avg = get_retain_forget_avg_results(
            speaker_avg_results, "sim_unlearned_gt_emb_avg", forget_speakers
        )
        unlearning_avg_results_sim_pretrained_gt_emb_avg = get_retain_forget_avg_results(
            speaker_avg_results, "sim_pretrained_gt_emb_avg", forget_speakers
        )

        metrics_sim_unlearned_gt_emb_avg = [line["sim_unlearned_gt_emb_avg"] for line in full_results]
        metric_sim_unlearned_gt_emb_avg = round(np.mean(metrics_sim_unlearned_gt_emb_avg), 5)
        metrics_sim_pretrained_gt_emb_avg = [line["sim_pretrained_gt_emb_avg"] for line in full_results]
        metric_sim_pretrained_gt_emb_avg = round(np.mean(metrics_sim_pretrained_gt_emb_avg), 5)

        all_results["avg_results_sim_unlearned_gt_emb_avg"] = metric_sim_unlearned_gt_emb_avg
        all_results["avg_results_sim_pretrained_gt_emb_avg"] = metric_sim_pretrained_gt_emb_avg
        all_results["unlearning_avg_results_sim_unlearned_gt_emb_avg"] = unlearning_avg_results_sim_unlearned_gt_emb_avg
        all_results["unlearning_avg_results_sim_pretrained_gt_emb_avg"] = (
            unlearning_avg_results_sim_pretrained_gt_emb_avg
        )

    result_path = f"{gen_wav_dir}/_{eval_task}_results{'_' + sim_model_type if eval_task in ['sim', 'spk-ZRF', 'diversity', 'delta_sim'] else ''}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
