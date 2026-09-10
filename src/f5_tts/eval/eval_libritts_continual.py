"""Evaluate every step of a continual unlearning run in a single call.

For each step this writes the usual `_{task}_results[_{sim_model}].json` next to that step's
generated wavs, extended with a `continual_avg_{task}_results` block that splits the speakers by
their position in the unlearning order:

| group               | speakers                                              |
|---------------------|-------------------------------------------------------|
| `current_forget`    | the speaker unlearned at this step                     |
| `past_forget`       | speakers unlearned at earlier steps (+ per speaker)    |
| `cumulative_forget` | past + current                                         |
| `future_forget`     | speakers not requested yet at this step                |
| `retain`            | every speaker outside `unlearn.forget_speakers`        |

and a chain-level summary holding a step x forget-speaker matrix, so "forgotten at step i, drifted
back by step j" is a direct row/column read.
"""

import argparse
import json
import os
import sys
import warnings
from importlib.resources import files
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from f5_tts.eval.eval_libritts import (
    get_retain_forget_avg_results,
    get_speaker_avg_results,
    parse_gpu_nums,
)
from f5_tts.eval.utils_eval import (
    get_libritts_test,
    run_asr_wer,
    run_delta_sim,
    run_sim_gt_matching,
    run_sim_v2,
    run_utmosv2,
)
from f5_tts.model.continual_utils import (
    continual_results_subdir,
    continual_steps,
    continual_summary_subdir,
)
from f5_tts.model.utils import seed_everything

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

rel_path = str(files("f5_tts").joinpath("../../"))

# Evaluation order, and the default: one call covers every metric for every step.
EVAL_TASKS = ["sim", "sim_gt_matching", "delta_sim", "wer", "utmosv2"]
SIM_MODEL_TASKS = ["sim", "sim_gt_matching", "delta_sim"]  # metrics whose result file is suffixed with the sim model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument(
        "-e",
        "--eval_tasks",
        type=str,
        nargs="+",
        default=EVAL_TASKS,
        choices=EVAL_TASKS,
        help="One or more evaluation tasks, run for every continual step.",
    )
    parser.add_argument("-l", "--lang", type=str, default="en")
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
    parser.add_argument("--config_name", type=str, default="F5TTS_v1_Base_unlearn_continual")
    # generated-wav directory reconstruction (must match eval_libritts_infer_batch_continual.py)
    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)
    parser.add_argument("--cfg_strength", default=2.0, type=float)
    parser.add_argument("--speed", default=1.0, type=float)
    parser.add_argument(
        "--gen_wav_dir_template",
        type=str,
        default=None,
        help="Escape hatch for non-standard layouts; must contain '{step_tag}'.",
    )
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


def resolve_sim_ckpt_dir(sim_model_type):
    if sim_model_type == "wavlm_large_finetune":
        return os.path.join(rel_path, "ckpts", "UniSpeech", "wavlm_large_finetune.pth")
    if sim_model_type == "wavlm_base_plus_sv":
        return "microsoft/wavlm-base-plus-sv"
    if sim_model_type == "speechbrain_ecapa":
        return "speechbrain/spkrec-ecapa-voxceleb"
    if sim_model_type == "wavlm_large":
        return "microsoft/wavlm-large"
    if sim_model_type == "resemblyzer":
        return "resemblyzer"
    raise ValueError(f"Similarity model type {sim_model_type} is not available")


def get_continual_group_results(speaker_avg_results, metric_key, step, forget_speakers):
    """Split per-speaker scores by each speaker's position in the unlearning order."""
    current = step["speaker"]
    past = set(step["past"])
    future = set(step["future"])

    groups = {"current_forget": [], "past_forget": [], "future_forget": [], "retain": []}
    forget_per_speaker = {}

    for speaker_results in speaker_avg_results:
        speaker = int(speaker_results["speaker"])
        value = speaker_results[metric_key]
        if speaker in set(forget_speakers):
            forget_per_speaker[speaker] = value
        if speaker == current:
            groups["current_forget"].append(value)
        elif speaker in past:
            groups["past_forget"].append(value)
        elif speaker in future:
            groups["future_forget"].append(value)
        else:
            groups["retain"].append(value)

    def average(values):
        return round(float(np.mean(values)), 5) if values else None

    results = {f"{name}_avg": average(values) for name, values in groups.items()}
    results["cumulative_forget_avg"] = average(groups["current_forget"] + groups["past_forget"])
    results["current_forget_speaker"] = current
    results["past_forget_speakers"] = list(step["past"])
    results["future_forget_speakers"] = list(step["future"])
    results["past_forget_per_speaker"] = {
        str(speaker): forget_per_speaker[speaker] for speaker in step["past"] if speaker in forget_per_speaker
    }
    results["forget_per_speaker"] = {str(speaker): forget_per_speaker.get(speaker) for speaker in forget_speakers}
    return results


def run_eval_task(args, eval_task, gen_wav_dir, gpus, asr_ckpt_dir, wavlm_ckpt_dir):
    """Run one metric over one directory. Mirrors the task dispatch of `eval_libritts.main`."""
    test_set = get_libritts_test(gen_wav_dir, gpus, args.processed_libritts_path, seed=args.seed)

    if eval_task == "wer":
        full_results = run_asr_wer((test_set[0][0], args.lang, test_set[0][1], asr_ckpt_dir))
    elif eval_task == "sim":
        full_results = run_sim_v2((test_set[0][0], test_set[0][1], wavlm_ckpt_dir, args.sim_model_type))
    elif eval_task == "utmosv2":
        full_results = run_utmosv2(test_set[0][1])
    elif eval_task == "sim_gt_matching":
        full_results = run_sim_gt_matching(test_set[0][1], wavlm_ckpt_dir, args.sim_model_type)
    elif eval_task == "delta_sim":
        pretrained_test_set = get_libritts_test(
            args.gen_wav_dir_pretrained, gpus, args.processed_libritts_path, seed=args.seed
        )
        assert len(test_set) == len(
            pretrained_test_set
        ), "The number of samples in gen_wav_dir and gen_wav_dir_pretrained must be the same for delta_sim evaluation"
        speaker_avg_results = run_delta_sim(
            test_set[0][1],
            pretrained_test_set[0][1],
            wavlm_ckpt_dir,
            args.sim_model_type,
            args.embeddings_dir_gt,
            args.embeddings_dir_pretrained,
        )
        speaker_avg_results.sort(key=lambda x: int(x["speaker"]))
        # delta_sim is a speaker-level metric; there is no per-utterance level to average from.
        return speaker_avg_results, speaker_avg_results
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    return full_results, get_speaker_avg_results(full_results, eval_task)


def result_path_for(gen_wav_dir, eval_task, sim_model_type):
    suffix = f"_{sim_model_type}" if eval_task in SIM_MODEL_TASKS else ""
    return f"{gen_wav_dir}/_{eval_task}_results{suffix}.json"


def main():
    args = get_args()
    seed_everything(args.seed)
    eval_tasks = [task for task in EVAL_TASKS if task in args.eval_tasks]

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{args.config_name}.yaml")))
    forget_speakers = [int(speaker) for speaker in model_cfg.unlearn.forget_speakers]
    ckpt_dir_name = model_cfg.ckpts.save_dir.split("/")[-1]
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    if not os.path.isdir(args.processed_libritts_path):
        raise ValueError(
            f"--processed_libritts_path is not a directory: {args.processed_libritts_path}. Its basename names "
            "the test set in every output path, so a wrong value silently points the eval at empty directories."
        )
    testset = Path(args.processed_libritts_path).name

    steps = continual_steps(forget_speakers)

    gpus = parse_gpu_nums(args.gpu_nums)
    asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3" if args.local else ""
    wavlm_ckpt_dir = resolve_sim_ckpt_dir(args.sim_model_type)

    if "delta_sim" in eval_tasks:
        assert args.embeddings_dir_gt is not None, "embeddings_dir_gt is required for delta_sim evaluation"
        assert (
            args.embeddings_dir_pretrained is not None
        ), "embeddings_dir_pretrained is required for delta_sim evaluation"
        assert (
            args.gen_wav_dir_pretrained is not None if not os.path.exists(args.embeddings_dir_pretrained) else True
        ), "gen_wav_dir_pretrained is required if embeddings_dir_pretrained does not exist"
        os.makedirs(args.embeddings_dir_pretrained, exist_ok=True)
        os.makedirs(args.embeddings_dir_gt, exist_ok=True)

    def gen_wav_dir_for(step):
        if args.gen_wav_dir_template:
            return args.gen_wav_dir_template.format(step_tag=step["tag"])
        return f"{rel_path}/" + continual_results_subdir(
            ckpt_dir_name,
            step["tag"],
            testset,
            seed=args.seed,
            ode_method=args.odemethod,
            nfe_step=args.nfestep,
            mel_spec_type=mel_spec_type,
            sway_sampling_coef=args.swaysampling,
            cfg_strength=args.cfg_strength,
            speed=args.speed,
        )

    summary = {task: {"steps": [], "forget_speakers": forget_speakers} for task in eval_tasks}

    for step in steps:
        gen_wav_dir = gen_wav_dir_for(step)
        if not os.path.isdir(gen_wav_dir):
            print(
                f"\n=== Step {step['index']}/{len(steps)} | forget speaker {step['speaker']}: no generated wavs at "
                f"{gen_wav_dir}, skipping. Run eval_libritts_infer_batch_continual.py for this step first. ==="
            )
            continue
        print(f"\n=== Step {step['index']}/{len(steps)} | forget speaker {step['speaker']} | {gen_wav_dir} ===")

        for eval_task in eval_tasks:
            print(f"\n--- {eval_task} @ {step['tag']} ---")
            full_results, speaker_avg_results = run_eval_task(
                args, eval_task, gen_wav_dir, gpus, asr_ckpt_dir, wavlm_ckpt_dir
            )

            unlearning_avg_results = get_retain_forget_avg_results(speaker_avg_results, eval_task, forget_speakers)
            continual_avg_results = get_continual_group_results(speaker_avg_results, eval_task, step, forget_speakers)

            metrics = [line[eval_task] for line in full_results]
            metric = round(float(np.mean(metrics)), 5)

            all_results = {
                f"avg_{eval_task}_results": metric,
                f"unlearning_avg_{eval_task}_results": unlearning_avg_results,
                f"continual_avg_{eval_task}_results": continual_avg_results,
                f"speaker_avg_{eval_task}_results": speaker_avg_results,
                "all_results": full_results,
                "continual": {
                    "step_index": step["index"],
                    "num_steps": len(steps),
                    "forget_speaker": step["speaker"],
                    "past_forget_speakers": list(step["past"]),
                    "future_forget_speakers": list(step["future"]),
                },
            }

            if eval_task == "delta_sim":
                for extra_key in ("sim_unlearned_gt_emb_avg", "sim_pretrained_gt_emb_avg"):
                    extra_metrics = [line[extra_key] for line in full_results]
                    all_results[f"avg_results_{extra_key}"] = round(float(np.mean(extra_metrics)), 5)
                    all_results[f"unlearning_avg_results_{extra_key}"] = get_retain_forget_avg_results(
                        speaker_avg_results, extra_key, forget_speakers
                    )
                    all_results[f"continual_avg_results_{extra_key}"] = get_continual_group_results(
                        speaker_avg_results, extra_key, step, forget_speakers
                    )

            result_path = result_path_for(gen_wav_dir, eval_task, args.sim_model_type)
            with open(result_path, "w") as f:
                json.dump(all_results, f, indent=4)

            summary[eval_task]["steps"].append(
                {
                    "step_index": step["index"],
                    "step_tag": step["tag"],
                    "forget_speaker": step["speaker"],
                    "gen_wav_dir": gen_wav_dir,
                    f"avg_{eval_task}": metric,
                    "retain_avg": unlearning_avg_results["retain_avg"],
                    "current_forget_avg": continual_avg_results["current_forget_avg"],
                    "past_forget_avg": continual_avg_results["past_forget_avg"],
                    "cumulative_forget_avg": continual_avg_results["cumulative_forget_avg"],
                    "future_forget_avg": continual_avg_results["future_forget_avg"],
                    "forget_per_speaker": continual_avg_results["forget_per_speaker"],
                }
            )

            print(f"\nTotal {len(metrics)} samples")
            print(f"{eval_task.upper()}: {metric}")
            print(
                f"{eval_task.upper()} retain={unlearning_avg_results['retain_avg']} "
                f"current_forget={continual_avg_results['current_forget_avg']} "
                f"past_forget={continual_avg_results['past_forget_avg']} "
                f"future_forget={continual_avg_results['future_forget_avg']}"
            )
            print(f"{eval_task.upper()} results saved to {result_path}")

    summary_dir = f"{rel_path}/" + continual_summary_subdir(ckpt_dir_name)
    os.makedirs(summary_dir, exist_ok=True)
    for eval_task in eval_tasks:
        suffix = f"_{args.sim_model_type}" if eval_task in SIM_MODEL_TASKS else ""
        summary_path = f"{summary_dir}/_{testset}_{eval_task}_results{suffix}_continual_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary[eval_task], f, indent=4)
        print(f"\n{eval_task.upper()} continual summary saved to {summary_path}")

        print(f"\n{eval_task.upper()} step x forget-speaker matrix (rows = steps):")
        header = "step  " + "  ".join(f"spk{speaker:>6}" for speaker in forget_speakers) + "   retain"
        print(header)
        for row in summary[eval_task]["steps"]:
            cells = "  ".join(
                (
                    f"{row['forget_per_speaker'].get(str(speaker)):>9}"
                    if row["forget_per_speaker"].get(str(speaker)) is not None
                    else f"{'-':>9}"
                )
                for speaker in forget_speakers
            )
            print(f"{row['step_index']:>4}  {cells}   {row['retain_avg']}")


if __name__ == "__main__":
    main()
