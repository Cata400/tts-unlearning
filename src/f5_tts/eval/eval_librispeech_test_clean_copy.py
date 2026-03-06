# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import ast
import json
import os
import random
import sys

import torch

sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from f5_tts.eval.utils_eval import get_librispeech_test_copy, run_asr_wer, run_sim_v2

rel_path = str(files("f5_tts").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, required=True)
    parser.add_argument(
        "-n", "--gpu_nums", type=str, default="1", help="Number of GPUs to use (e.g., 8) or GPU list (e.g., [0,1,2,3])"
    )
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    parser.add_argument(
        "--sim_model_type",
        type=str,
        default="speechbrain_ecapa",
        choices=["wavlm_large_finetune", "wavlm_base_plus_sv", "speechbrain_ecapa"],
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


def main():
    args = get_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_task = args.eval_task
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir
    metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
    sim_model_type = args.sim_model_type

    gpus = parse_gpu_nums(args.gpu_nums)
    test_set = get_librispeech_test_copy(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

    ## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
    ## leading to a low similarity for the ground truth in some cases.
    # test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir

    if eval_task == "sim":
        if sim_model_type == "wavlm_large_finetune":
            wavlm_ckpt_dir = os.path.join(rel_path, "ckpts", "UniSpeech", "wavlm_large_finetune.pth")
        elif sim_model_type == "wavlm_base_plus_sv":
            wavlm_ckpt_dir = "microsoft/wavlm-base-plus-sv"
        elif sim_model_type == "speechbrain_ecapa":
            wavlm_ckpt_dir = "speechbrain/spkrec-ecapa-voxceleb"
        elif sim_model_type == "wavlm_large":
            wavlm_ckpt_dir = "microsoft/wavlm-large"
        else:
            raise ValueError(f"Similarity model ty[e {sim_model_type} is not available")

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if len(gpus) > 1:
        ctx = mp.get_context("spawn")

        if eval_task == "wer":
            with ctx.Pool(processes=len(gpus)) as pool:
                args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
                results = pool.map(run_asr_wer, args)
                for r in results:
                    full_results.extend(r)
        elif eval_task == "sim":
            with ctx.Pool(processes=len(gpus)) as pool:
                args = [(rank, sub_test_set, wavlm_ckpt_dir, sim_model_type) for (rank, sub_test_set) in test_set]
                results = pool.map(run_sim_v2, args)
                for r in results:
                    full_results.extend(r)
        else:
            raise ValueError(f"Unknown metric type: {eval_task}")

        result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line[eval_task])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")
    else:
        if eval_task == "wer":
            full_results = run_asr_wer((test_set[0][0], lang, test_set[0][1], asr_ckpt_dir))
        elif eval_task == "sim":
            # full_results = run_sim((test_set[0][0], test_set[0][1], wavlm_ckpt_dir))
            full_results = run_sim_v2((test_set[0][0], test_set[0][1], wavlm_ckpt_dir, sim_model_type))
        else:
            raise ValueError(f"Unknown metric type: {eval_task}")

        result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line[eval_task])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
