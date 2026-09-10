"""Batch inference over every checkpoint of a continual unlearning run, in a single call.

One `unlearned_model_step{i}_spk{id}.pt` per continual step is generated into its own results
directory, so the forget speaker of each step is visible in the path. The heavy setup (vocoder,
tokenizer, prompt preparation, model skeleton) happens once and is reused across steps.

Continual checkpoints are materialised, so they load with the stock loader - there are deliberately
no `--svdiff` / `--svdiff_uv` flags here, unlike `eval_libritts_infer_batch.py`.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_processed_libritts_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.continual_utils import (
    continual_checkpoint_name,
    continual_results_subdir,
    continual_steps,
)
from f5_tts.model.utils import get_tokenizer, seed_everything

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


target_rms = 0.1


rel_path = str(files("f5_tts").joinpath("../../"))


def parse_step_selection(steps_arg: str | None, steps: list[dict]) -> list[dict]:
    if not steps_arg:
        return steps
    wanted = {int(token.strip()) for token in steps_arg.split(",") if token.strip()}
    unknown = wanted - {step["index"] for step in steps}
    if unknown:
        raise ValueError(f"--steps refers to unknown step indices {sorted(unknown)} (1-based).")
    return [step for step in steps if step["index"] in wanted]


def resolve_checkpoint_path(exp_name: str, save_dir: str, step_tag: str) -> str:
    filename = continual_checkpoint_name(step_tag)
    candidates = [
        f"{rel_path}/ckpts/{exp_name}/{filename}",
        f"{rel_path}/{save_dir}/{filename}",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise ValueError(f"No checkpoint for step {step_tag}; looked in {candidates}")


def main():
    parser = argparse.ArgumentParser(description="batch inference over a continual unlearning chain")

    parser.add_argument("-s", "--seed", default=42, type=int)
    parser.add_argument("-n", "--expname", required=True, help="Config name, e.g. F5TTS_v1_Base_unlearn_continual")

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument(
        "-p", "--processed_libritts_dataset_path", default=f"{rel_path}/data/LibriTTS/test-clean", type=str
    )

    parser.add_argument("--local", action="store_true", help="Use local vocoder checkpoint directory")
    parser.add_argument(
        "--steps",
        default=None,
        type=str,
        help="Comma-separated 1-based step indices to run (default: every step in unlearn.forget_speakers).",
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate from the EMA weights, as eval_libritts_infer_batch.py does (default). --no-use_ema "
        "uses the online weights, i.e. exactly the weights the continual chain is built from.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a step whose output directory already holds a wav for every prompt.",
    )

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    use_ema = args.use_ema

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    seed_everything(seed)

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_dir_name = model_cfg.ckpts.save_dir.split("/")[-1]

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    steps = continual_steps(list(model_cfg.unlearn.forget_speakers))
    selected_steps = parse_step_selection(args.steps, steps)
    print(f"Continual chain: {[step['tag'] for step in steps]}")
    print(f"Running inference for: {[step['tag'] for step in selected_steps]}")

    processed_libritts_test_path = args.processed_libritts_dataset_path
    metainfo = get_processed_libritts_metainfo(processed_libritts_test_path, seed=seed)
    testset = Path(processed_libritts_test_path).name
    print(
        f"Loaded {len(metainfo)} samples from {testset} for inference. Some samples may be further skipped due to length limits."
    )

    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )
    print(f"Total {len(prompts_all)} prompts for batch inference.")
    num_expected_wavs = sum(len(prompt[0]) for prompt in prompts_all)

    # Vocoder model
    local = args.local
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model skeleton, reused across steps: only the weights change.
    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
            diffit=model_cfg["model"].get("finetune", {}).get("diffit", {}).get("use", False),
            diffit_blocks=model_cfg["model"].get("finetune", {}).get("diffit", {}).get("diffit_blocks", []),
        ),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None

    for step in selected_steps:
        step_tag = step["tag"]
        output_dir = f"{rel_path}/" + continual_results_subdir(
            ckpt_dir_name,
            step_tag,
            testset,
            seed=seed,
            ode_method=ode_method,
            nfe_step=nfe_step,
            mel_spec_type=mel_spec_type,
            sway_sampling_coef=sway_sampling_coef,
            cfg_strength=cfg_strength,
            speed=speed,
            use_truth_duration=use_truth_duration,
            no_ref_audio=no_ref_audio,
        )

        if args.skip_existing and os.path.isdir(output_dir):
            num_existing = len([name for name in os.listdir(output_dir) if name.endswith(".wav")])
            if num_existing >= num_expected_wavs:
                print(f"\n=== {step_tag}: {num_existing} wavs already in {output_dir}, skipping. ===")
                continue

        ckpt_path = resolve_checkpoint_path(exp_name, model_cfg.ckpts.save_dir, step_tag)
        print(f"\n=== Step {step['index']}/{len(steps)} | forget speaker {step['speaker']} ===")
        print(f"Loading model checkpoint from {ckpt_path}")
        load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

        if not os.path.exists(output_dir) and accelerator.is_main_process:
            os.makedirs(output_dir)

        # write metainfo to output dir for debugging and reference
        if accelerator.is_main_process:
            with open(f"{output_dir}/metainfo.txt", "w") as f:
                for line in metainfo:
                    gen_utt, ref_text, ref_wav, gen_text, gen_wav = line
                    f.write(f"{gen_utt}\t{ref_text}\t{ref_wav}\t{gen_text}\t{gen_wav}\n")

        # start batch inference
        seed_everything(seed)
        accelerator.wait_for_everyone()
        start = time.time()

        with accelerator.split_between_processes(prompts_all) as prompts:
            for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
                utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
                ref_mels = ref_mels.to(device)
                ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
                total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

                # Inference
                with torch.inference_mode():
                    generated, _ = model.sample(
                        cond=ref_mels,
                        text=final_text_list,
                        duration=total_mel_lens,
                        lens=ref_mel_lens,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        no_ref_audio=no_ref_audio,
                        seed=seed,
                    )
                    # Final result
                    for i, gen in enumerate(generated):
                        gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                        gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                        if mel_spec_type == "vocos":
                            generated_wave = vocoder.decode(gen_mel_spec).cpu()
                        elif mel_spec_type == "bigvgan":
                            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                        if ref_rms_list[i] < target_rms:
                            generated_wave = generated_wave * ref_rms_list[i] / target_rms
                        torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            timediff = time.time() - start
            print(f"Done batch inference for {step_tag} in {timediff / 60:.2f} minutes -> {output_dir}")


if __name__ == "__main__":
    main()
