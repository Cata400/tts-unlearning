# training script.

import os

# supress warnings
import warnings
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, TrainerUnlearn
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer, seed_everything

warnings.filterwarnings("ignore", category=UserWarning)

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    seed_everything(model_cfg.unlearn.random_seed)

    # Check unlearn methods
    unlearn_methods_use = [params.use for _, params in model_cfg.unlearn.unlearn_methods.items()]
    assert sum(unlearn_methods_use) == 1, "A single unlearning method can be in use at a time"

    for method, params in model_cfg.unlearn.unlearn_methods.items():
        if params.use:
            print(f"Using unlearn method {method} with params: {params}")
            unlearn_params = params
            unlearn_method = method
            break

    svdiff_methods = {"svdiff", "svdiff_uv"}
    svdiff_methods_use = [params.use for method, params in model_cfg.model.finetune.items() if method in svdiff_methods]
    assert sum(svdiff_methods_use) <= 1, "Only one SVDiff variant can be used at the same time"

    finetune_methods_use = [
        params.use for method, params in model_cfg.model.finetune.items() if method not in svdiff_methods
    ]  # allow SVDiff variants to be used with finetuning methods
    assert sum(finetune_methods_use) <= 1, "Multiple finetune methods cannot be used at the same time"

    if sum(finetune_methods_use) == 0:
        print("The entire model will be finetuned without any parameter freezing.")
    else:
        for method, params in model_cfg.model.finetune.items():
            if params.use:
                print(f"Using finetune method {method} with params: {params}")
                break

    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = model_cfg.ckpts.save_dir.split("/")[-1]
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set models
    model = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
            diffit=model_cfg["model"].get("finetune", {}).get("diffit", {}).get("use", False),
            diffit_blocks=model_cfg["model"].get("finetune", {}).get("diffit", {}).get("diffit_blocks", []),
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    teacher = CFM(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size,
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
            diffit=False,
            diffit_blocks=[],
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # init trainer
    trainer = TrainerUnlearn(
        model,
        teacher,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        teacher_path=model_cfg.ckpts.pretrained_path,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="F5-TTS-Unlearning",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        unlearn_params=unlearn_params,
        forget_speakers=model_cfg.unlearn.forget_speakers,
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        tokenizer,
        dataset_type="CustomUnlearningDataset",
        mel_spec_kwargs=model_cfg.model.mel_spec,
        forget_speakers=model_cfg.unlearn.forget_speakers,
    )

    if unlearn_method == "TGU":
        trainer.train_TGU(
            train_dataset,
            num_workers=model_cfg.datasets.num_workers,
            resumable_with_seed=model_cfg.unlearn.random_seed,  # seed for shuffling dataset
        )
    elif unlearn_method == "SGU":
        trainer.train_SGU(
            train_dataset,
            num_workers=model_cfg.datasets.num_workers,
            resumable_with_seed=model_cfg.unlearn.random_seed,  # seed for shuffling dataset
        )


if __name__ == "__main__":
    main()
