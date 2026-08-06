from __future__ import annotations

import gc
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
from torchjd import autojac
from torchjd.aggregation import UPGrad
from torchjd.autojac import jac_to_grad
from tqdm import tqdm

import wandb
from f5_tts.infer.utils_infer import (
    cfg_strength,
    nfe_step,
    sway_sampling_coef,
)
from f5_tts.model import CFM
from f5_tts.model.dataset import (
    BalancedUnlearningSampleBatchSampler,
    DynamicBatchSampler,
    DynamicUnlearningBatchSampler,
    build_unlearning_sample_weights,
    collate_fn_unlearning,
    get_dataset_num_speakers,
)
from f5_tts.model.finetune_strategies import build_finetune_strategy
from f5_tts.model.utils import default, exists


class TrainerUnlearn:  # TODO add info logger
    def __init__(
        self,
        model: CFM,
        teacher: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        teacher_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "tensorboard",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        unlearn_params: dict = dict(),  # unlearning params
        forget_speakers: list = [],  # list of speakers to forget
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.finetune_strategy = build_finetune_strategy(model_cfg_dict)

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

            if self.accelerator.is_local_main_process:
                self._define_wandb_metrics()

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model_cfg_dict = model_cfg_dict

        self.model = model
        self.teacher = teacher
        self.ema_kwargs = ema_kwargs

        if self.is_main:
            self.ema_model = EMA(self.model, include_online_model=False, **self.ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")
        self.teacher_path = teacher_path

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor
        self.bnb_optimizer = bnb_optimizer

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.teacher, self.model, self.optimizer = self.accelerator.prepare(self.teacher, self.model, self.optimizer)

        self.unlearn_params = unlearn_params
        self.forget_speakers = forget_speakers

        if not self.unlearn_params.get("always_retain_and_forget", False) and self.unlearn_params.get(
            "use_torchjd", False
        ):
            raise ValueError(
                "Unlearning configuration error: 'always_retain_and_forget' must be True when 'use_torchjd' is enabled."
            )

        if self.unlearn_params.get("always_retain_and_forget", False) and self.unlearn_params.get("use_torchjd", False):
            self.aggregator = UPGrad()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def _define_wandb_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("loss", step_metric="train_step")
        wandb.define_metric("lr", step_metric="train_step")
        wandb.define_metric("weight_stats_step")
        wandb.define_metric("weight_stats/*", step_metric="weight_stats_step")
        self.finetune_strategy.register_wandb_metrics(self.logger)

    @staticmethod
    def _weight_stat_metric_prefix(name: str) -> str:
        return f"weight_stats/{name.replace('.', '/')}"

    def _get_lr_schedule_updates(self, train_dataloader: DataLoader):
        # accelerator.prepare() dispatches batches to devices;
        # which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        return warmup_updates, decay_updates

    def _build_lr_scheduler(self, warmup_updates: int, decay_updates: int):
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        return SequentialLR(self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates])

    def reset_optimizer_and_scheduler_for_trainable_params(
        self, warmup_updates: int, decay_updates: int, context: str = "finetuning"
    ):
        trainable_params = [p for p in self.accelerator.unwrap_model(self.model).parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(f"{context} reset found no trainable parameters for the optimizer.")

        if self.bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(trainable_params, lr=self.learning_rate)
        else:
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate)

        if self.is_main:
            self.ema_model = EMA(self.model, include_online_model=False, **self.ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.scheduler = self._build_lr_scheduler(warmup_updates, decay_updates)
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/unlearned_model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/unlearned_model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("unlearned_model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "unlearned_model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            print(f"No checkpoint found in {self.checkpoint_path}, training from teacher weights")
            return 0

        self.accelerator.wait_for_everyone()
        if "unlearned_model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "unlearned_model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("unlearned_model_") or f.startswith("pretrained_"))
                and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [
                f for f in all_checkpoints if f.startswith("unlearned_model_") and f != "unlearned_model_last.pt"
            ]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.startswith("pretrained_"):
            print(f"Only pretrained checkpoint found: {latest_checkpoint}, skipping loading student weights.")
            return 0

        print(f"Loading checkpoint {latest_checkpoint} from {self.checkpoint_path}")
        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def load_pretrained_checkpoint(self, model):
        if (
            not exists(self.teacher_path)
            or not os.path.exists(self.teacher_path)
            or not self.teacher_path.endswith((".pt", ".safetensors"))
        ):
            raise ValueError(f"No pretrained checkpoint found in {self.teacher_path}, aborting")

        self.accelerator.wait_for_everyone()

        print(f"Loading pretrained checkpoint {self.teacher_path}")
        if self.teacher_path.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(self.teacher_path, device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif self.teacher_path.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state
            checkpoint = torch.load(self.teacher_path, weights_only=True, map_location="cpu")

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        print(f"strict = {not model.transformer.diffit}")
        print(
            f'EMA strict = {not self.model_cfg_dict["model"].get("finetune", {}).get("diffit", {}).get("use", False)}'
        )
        if self.is_main:
            self.ema_model.load_state_dict(
                checkpoint["ema_model_state_dict"],
                strict=not self.model_cfg_dict["model"].get("finetune", {}).get("diffit", {}).get("use", False),
            )

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(model).load_state_dict(
                checkpoint["model_state_dict"], strict=not model.transformer.diffit
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = 0
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(model).load_state_dict(
                checkpoint["model_state_dict"], strict=not model.transformer.diffit
            )
            update = 0

        del checkpoint
        gc.collect()
        return update

    def create_dataloader(self, train_dataset, num_workers=16, resumable_with_seed: int = None):
        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        def worker_init_fn(worker_id):
            if resumable_with_seed is not None:
                worker_seed = resumable_with_seed + worker_id
                import random

                import numpy as np

                random.seed(worker_seed)
                np.random.seed(worker_seed)
                torch.manual_seed(worker_seed)

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_unlearning,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )
        elif (
            self.batch_size_type == "unlearn_sample"
        ):  # oversample forget samples to have equal number of forget and retain samples in training
            num_speakers = get_dataset_num_speakers(train_dataset)
            unlearning_class_weights = {
                1: num_speakers / (num_speakers - len(self.forget_speakers)),
                -1: num_speakers / len(self.forget_speakers),
            }
            unlearning_sample_weights = build_unlearning_sample_weights(
                train_dataset,
                unlearning_class_weights,
            )
            sampler = WeightedRandomSampler(
                unlearning_sample_weights,
                num_samples=len(unlearning_sample_weights),
                replacement=True,
                generator=generator,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_unlearning,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                sampler=sampler,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )
        elif self.batch_size_type == "balanced_unlearn_sample":
            batch_sampler = BalancedUnlearningSampleBatchSampler(
                train_dataset,
                batch_size=self.batch_size_per_gpu,
                random_seed=resumable_with_seed,
                oversample_forget=self.unlearn_params.get("balanced_unlearn_sample_oversample_forget", False),
                retain_duration_budget_ratio=self.unlearn_params.get("retain_duration_budget_ratio"),
                retain_duration_budget_reset_per_epoch=self.unlearn_params.get(
                    "retain_duration_budget_reset_per_epoch", False
                ),
                forget_duration_budget_ratio=self.unlearn_params.get("forget_duration_budget_ratio", None),
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_unlearning,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
                worker_init_fn=worker_init_fn,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_unlearning,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
                worker_init_fn=worker_init_fn,
            )
            print(f"Total {len(train_dataloader)} batches per epoch with frame-based batch size.")
        elif self.batch_size_type == "unlearn_frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicUnlearningBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_unlearning,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
                worker_init_fn=worker_init_fn,
            )
            print(f"Total {len(train_dataloader)} batches per epoch with frame-based batch size.")
        else:
            raise ValueError(
                "batch_size_type must be one of "
                "'sample', 'unlearn_sample', 'balanced_unlearn_sample', 'frame', or 'unlearn_frame', "
                f"but received {self.batch_size_type}"
            )

        return train_dataloader

    def _compute_pre_grad_losses(self, batch, unlearn_method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (retain_loss, forget_loss) for one profiling batch (used by fine-tune pre-grad hooks)."""
        text_inputs_retain = batch["text_retain"]
        text_inputs_forget = batch["text_forget"]
        mel_spec_retain = batch["mel_retain"].permute(0, 2, 1)
        mel_spec_forget = batch["mel_forget"].permute(0, 2, 1)
        mel_lengths_retain = batch["mel_lengths_retain"]
        mel_lengths_forget = batch["mel_lengths_forget"]

        if unlearn_method == "TGU":
            if mel_spec_retain.numel() > 0:
                retain_loss, _, _ = self.model.forward_unlearn(
                    mel_spec_retain,
                    text=text_inputs_retain,
                    lens=mel_lengths_retain,
                    noise_scheduler=self.noise_scheduler,
                    forget=False,
                )
            else:
                retain_loss = torch.tensor(0.0, device=self.accelerator.device)

            if mel_spec_forget.numel() > 0:
                infer_texts_forget = [
                    text_inputs_forget[i]
                    + ([" "] if isinstance(text_inputs_forget[i], list) else " ")
                    + text_inputs_forget[i]
                    for i in range(len(text_inputs_forget))
                ]
                with torch.inference_mode():
                    unconditioned_mel_spec_forget, _ = self.accelerator.unwrap_model(self.teacher).sample(
                        cond=torch.zeros_like(mel_spec_forget),
                        text=infer_texts_forget,
                        duration=mel_spec_forget.size(1) * 2,
                        max_duration=mel_spec_forget.size(1) * 2,
                        no_ref_audio=True,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )
                    unconditioned_mel_spec_forget = unconditioned_mel_spec_forget.to(torch.float32)
                    unconditioned_mel_spec_forget = unconditioned_mel_spec_forget[:, mel_spec_forget.size(1) :].to(
                        self.accelerator.device
                    )

                forget_loss, _, _ = self.model.forward_unlearn(
                    mel_spec_forget,
                    text=text_inputs_forget,
                    lens=mel_lengths_forget,
                    noise_scheduler=self.noise_scheduler,
                    forget=True,
                    flow_inp=unconditioned_mel_spec_forget,
                )
            else:
                forget_loss = torch.tensor(0.0, device=self.accelerator.device)

        elif unlearn_method == "SGU":
            if mel_spec_forget.numel() > 0:
                mel_spec_concat = torch.cat([mel_spec_forget, mel_spec_retain], dim=1)
                text_inputs_concat = [
                    text_inputs_forget[i] + " " + text_inputs_retain[i] for i in range(len(text_inputs_retain))
                ]
                mel_lengths_concat = mel_lengths_forget + mel_lengths_retain

                forget_loss, _, _ = self.model.forward_unlearn_SGU(
                    mel_spec_concat,
                    text=text_inputs_concat,
                    lens=mel_lengths_concat,
                    retain_lens=mel_lengths_retain,
                    noise_scheduler=self.noise_scheduler,
                )
            else:
                forget_loss = torch.tensor(0.0, device=self.accelerator.device)

            retain_loss, _, _ = self.model.forward_unlearn(
                mel_spec_retain,
                text=text_inputs_retain,
                lens=mel_lengths_retain,
                noise_scheduler=self.noise_scheduler,
                forget=False,
            )

        else:
            raise ValueError(f"Unknown unlearning method for pre-grad losses: {unlearn_method}")

        return retain_loss, forget_loss

    def train_TGU(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            # vocoder = load_vocoder(
            #     vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            # )
            self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        train_dataloader = self.create_dataloader(train_dataset, num_workers, resumable_with_seed)

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates, decay_updates = self._get_lr_schedule_updates(train_dataloader)
        self.scheduler = self._build_lr_scheduler(warmup_updates, decay_updates)
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        print("Load teacher")
        self.load_pretrained_checkpoint(self.teacher)
        print("Load student")
        self.load_pretrained_checkpoint(self.model)
        start_update = self.load_checkpoint()
        global_update = start_update

        # Stashed so strategies can build their own profiling dataloader from apply().
        self._pretrain_dataset = train_dataset
        self._pretrain_num_workers = num_workers
        self._pretrain_resumable_with_seed = resumable_with_seed
        self._pretrain_unlearn_method = "TGU"
        self.finetune_strategy.apply(self.accelerator.unwrap_model(self.model), trainer=self)
        if self.finetune_strategy.requires_optimizer_reset:
            self.reset_optimizer_and_scheduler_for_trainable_params(
                warmup_updates, decay_updates, context=self.finetune_strategy.name
            )
        self.finetune_strategy.run_pre_training_hook(
            self,
            unlearn_method="TGU",
            train_dataset=train_dataset,
            num_workers=num_workers,
            resumable_with_seed=resumable_with_seed,
        )

        num_trainable_params = sum(
            p.numel() for p in self.accelerator.unwrap_model(self.model).parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters in student model: {num_trainable_params / 1e6:.3f}M")

        # set teacher to eval and no grad
        self.accelerator.unwrap_model(self.teacher).eval()
        for param in self.accelerator.unwrap_model(self.teacher).parameters():
            param.requires_grad = False

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs_retain = batch["text_retain"]
                    text_inputs_forget = batch["text_forget"]

                    mel_spec_retain = batch["mel_retain"].permute(0, 2, 1)
                    mel_spec_forget = batch["mel_forget"].permute(0, 2, 1)

                    mel_lengths_retain = batch["mel_lengths_retain"]
                    mel_lengths_forget = batch["mel_lengths_forget"]

                    if mel_spec_retain.numel() > 0:
                        retain_loss, retain_cond, retain_pred = self.model.forward_unlearn(
                            mel_spec_retain,
                            text=text_inputs_retain,
                            lens=mel_lengths_retain,
                            noise_scheduler=self.noise_scheduler,
                            forget=False,
                        )
                    else:
                        retain_loss = torch.tensor(0.0, device=self.accelerator.device)

                    if mel_spec_forget.numel() > 0:  # if there is at least one forget sample
                        infer_texts_forget = [
                            text_inputs_forget[i]
                            + ([" "] if isinstance(text_inputs_forget[i], list) else " ")
                            + text_inputs_forget[i]
                            for i in range(len(text_inputs_forget))
                        ]
                        with torch.inference_mode():
                            unconditioned_mel_spec_forget, _ = self.accelerator.unwrap_model(self.teacher).sample(
                                cond=torch.zeros_like(mel_spec_forget),  # This is only for the shape
                                text=infer_texts_forget,
                                duration=mel_spec_forget.size(1) * 2,
                                max_duration=mel_spec_forget.size(1) * 2,
                                no_ref_audio=True,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            unconditioned_mel_spec_forget = unconditioned_mel_spec_forget.to(torch.float32)
                            unconditioned_mel_spec_forget = unconditioned_mel_spec_forget[
                                :, mel_spec_forget.size(1) :
                            ].to(self.accelerator.device)

                        forget_loss, forget_cond, forget_pred = self.model.forward_unlearn(
                            mel_spec_forget,
                            text=text_inputs_forget,
                            lens=mel_lengths_forget,
                            noise_scheduler=self.noise_scheduler,
                            forget=True,
                            flow_inp=unconditioned_mel_spec_forget,
                        )

                    else:
                        forget_loss = torch.tensor(0.0, device=self.accelerator.device)

                    loss = (
                        self.unlearn_params["lambda"] * retain_loss + (1 - self.unlearn_params["lambda"]) * forget_loss
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"train_step": global_update, "loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                        step=global_update,
                    )
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:  # TODO change this at some point
                        pass
                        # ref_audio_len = mel_lengths_retain[0]
                        # infer_text = [
                        #     text_inputs_retain[0]
                        #     + ([" "] if isinstance(text_inputs_retain[0], list) else " ")
                        #     + text_inputs_retain[0]
                        # ]
                        # with torch.inference_mode():
                        #     generated, _ = self.accelerator.unwrap_model(self.model).sample(
                        #         cond=mel_spec_retain[0][:ref_audio_len].unsqueeze(0),
                        #         text=infer_text,
                        #         duration=ref_audio_len * 2,
                        #         steps=nfe_step,
                        #         cfg_strength=cfg_strength,
                        #         sway_sampling_coef=sway_sampling_coef,
                        #     )
                        #     generated = generated.to(torch.float32)
                        #     gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                        #     ref_mel_spec = batch["mel"][0].unsqueeze(0)
                        #     if self.vocoder_name == "vocos":
                        #         gen_audio = vocoder.decode(gen_mel_spec).cpu()
                        #         ref_audio = vocoder.decode(ref_mel_spec).cpu()
                        #     elif self.vocoder_name == "bigvgan":
                        #         gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                        #         ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        # torchaudio.save(
                        #     f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        # )
                        # torchaudio.save(
                        #     f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        # )
                        # self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()

    def train_SGU(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            # vocoder = load_vocoder(
            #     vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            # )
            self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        assert (
            self.batch_size_type == "balanced_unlearn_sample"
        ), "Batches should be balance, this method requires an equal number of forget and retain samples per batch. Use batch_size_type == 'balanced_unlearn_sample'"
        assert self.batch_size_per_gpu % 2 == 0
        train_dataloader = self.create_dataloader(train_dataset, num_workers, resumable_with_seed)

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates, decay_updates = self._get_lr_schedule_updates(train_dataloader)
        self.scheduler = self._build_lr_scheduler(warmup_updates, decay_updates)
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        print("Load teacher")
        self.load_pretrained_checkpoint(self.teacher)
        print("Load student")
        self.load_pretrained_checkpoint(self.model)
        start_update = self.load_checkpoint()
        global_update = start_update

        # Stashed so strategies can build their own profiling dataloader from apply().
        self._pretrain_dataset = train_dataset
        self._pretrain_num_workers = num_workers
        self._pretrain_resumable_with_seed = resumable_with_seed
        self._pretrain_unlearn_method = "SGU"
        self.finetune_strategy.apply(self.accelerator.unwrap_model(self.model), trainer=self)
        if self.finetune_strategy.requires_optimizer_reset:
            self.reset_optimizer_and_scheduler_for_trainable_params(
                warmup_updates, decay_updates, context=self.finetune_strategy.name
            )
        self.finetune_strategy.run_pre_training_hook(
            self,
            unlearn_method="SGU",
            train_dataset=train_dataset,
            num_workers=num_workers,
            resumable_with_seed=resumable_with_seed,
        )

        num_trainable_params = sum(
            p.numel() for p in self.accelerator.unwrap_model(self.model).parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters in student model: {num_trainable_params / 1e6:.3f}M")

        # set teacher to eval and no grad
        self.accelerator.unwrap_model(self.teacher).eval()
        for param in self.accelerator.unwrap_model(self.teacher).parameters():
            param.requires_grad = False

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        weight_stats_update_interval = max(1, math.ceil(len(train_dataloader) / (4 * self.grad_accumulation_steps)))
        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for i, batch in enumerate(current_dataloader):
                with self.accelerator.accumulate(self.model):
                    text_inputs_retain = batch["text_retain"]
                    text_inputs_forget = batch["text_forget"]

                    mel_spec_retain = batch["mel_retain"].permute(0, 2, 1)
                    mel_spec_forget = batch["mel_forget"].permute(0, 2, 1)

                    mel_lengths_retain = batch["mel_lengths_retain"]
                    mel_lengths_forget = batch["mel_lengths_forget"]

                    if self.unlearn_params.get("always_retain_and_forget", False):
                        if mel_spec_forget.numel() > 0:  # if there is at least one forget sample
                            mel_spec_concat = torch.cat([mel_spec_forget, mel_spec_retain], dim=1)
                            text_inputs_concat = [
                                text_inputs_forget[i] + " " + text_inputs_retain[i]
                                for i in range(len(text_inputs_retain))
                            ]
                            mel_lengths_concat = mel_lengths_forget + mel_lengths_retain

                            forget_loss, forget_cond, forget_pred = self.model.forward_unlearn_SGU(
                                mel_spec_concat,
                                text=text_inputs_concat,
                                lens=mel_lengths_concat,
                                retain_lens=mel_lengths_retain,
                                noise_scheduler=self.noise_scheduler,
                            )
                        else:
                            forget_loss = torch.tensor(0.0, device=self.accelerator.device)

                        retain_loss, retain_cond, retain_pred = self.model.forward_unlearn(
                            mel_spec_retain,
                            text=text_inputs_retain,
                            lens=mel_lengths_retain,
                            noise_scheduler=self.noise_scheduler,
                            forget=False,
                        )

                        if not self.unlearn_params.get("use_torchjd", False):
                            # type 2
                            loss = retain_loss + self.unlearn_params["forget_ratio"] * forget_loss
                    else:
                        do_forget = torch.rand((1,)).item() < self.unlearn_params["forget_ratio"]
                        if do_forget and mel_spec_forget.numel() > 0:  # if there is at least one forget sample
                            mel_spec_concat = torch.cat([mel_spec_forget, mel_spec_retain], dim=1)
                            text_inputs_concat = [
                                text_inputs_forget[i] + " " + text_inputs_retain[i]
                                for i in range(len(text_inputs_retain))
                            ]
                            mel_lengths_concat = mel_lengths_forget + mel_lengths_retain

                            loss, cond, pred = self.model.forward_unlearn_SGU(
                                mel_spec_concat,
                                text=text_inputs_concat,
                                lens=mel_lengths_concat,
                                retain_lens=mel_lengths_retain,
                                noise_scheduler=self.noise_scheduler,
                            )
                        else:
                            loss, cond, pred = self.model.forward_unlearn(
                                mel_spec_retain,
                                text=text_inputs_retain,
                                lens=mel_lengths_retain,
                                noise_scheduler=self.noise_scheduler,
                                forget=False,
                            )

                    if not self.unlearn_params.get("use_torchjd", False):
                        self.accelerator.backward(loss)
                    else:
                        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                        autojac.backward([retain_loss, forget_loss], inputs=trainable_params)
                        jac_to_grad(trainable_params, self.aggregator)
                        loss = retain_loss + forget_loss  # for logging purposes

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"train_step": global_update, "loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]},
                        step=global_update,
                    )
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                    if self.accelerator.sync_gradients and (
                        global_update % weight_stats_update_interval == 0 or i == 0
                    ):
                        weight_stats = {"weight_stats_step": global_update // weight_stats_update_interval}
                        for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                            if param.requires_grad and ".weight" in name:
                                l2_norm = torch.norm(param.data, p=2).item()
                                frobenius_norm = torch.norm(param.data, p="fro").item()
                                soft_sparsity = (param.data.abs() < 1e-3).float().mean().item()
                                metric_prefix = self._weight_stat_metric_prefix(name)
                                weight_stats[f"{metric_prefix}/l2_norm"] = l2_norm
                                weight_stats[f"{metric_prefix}/frobenius_norm"] = frobenius_norm
                                weight_stats[f"{metric_prefix}/soft_sparsity"] = soft_sparsity
                        self.accelerator.log(weight_stats, step=global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:  # TODO change this at some point
                        pass
                        # ref_audio_len = mel_lengths_retain[0]
                        # infer_text = [
                        #     text_inputs_retain[0]
                        #     + ([" "] if isinstance(text_inputs_retain[0], list) else " ")
                        #     + text_inputs_retain[0]
                        # ]
                        # with torch.inference_mode():
                        #     generated, _ = self.accelerator.unwrap_model(self.model).sample(
                        #         cond=mel_spec_retain[0][:ref_audio_len].unsqueeze(0),
                        #         text=infer_text,
                        #         duration=ref_audio_len * 2,
                        #         steps=nfe_step,continue
                        #         cfg_strength=cfg_strength,
                        #         sway_sampling_coef=sway_sampling_coef,
                        #     )
                        #     generated = generated.to(torch.float32)
                        #     gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                        #     ref_mel_spec = batch["mel"][0].unsqueeze(0)
                        #     if self.vocoder_name == "vocos":
                        #         gen_audio = vocoder.decode(gen_mel_spec).cpu()
                        #         ref_audio = vocoder.decode(ref_mel_spec).cpu()
                        #     elif self.vocoder_name == "bigvgan":
                        #         gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                        #         ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        # torchaudio.save(
                        #     f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        # )
                        # torchaudio.save(
                        #     f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        # )
                        # self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
