from __future__ import annotations

import gc
import math
import os

import torch
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchjd import autojac
from torchjd.autojac import jac_to_grad
from tqdm import tqdm

import wandb
from f5_tts.infer.utils_infer import (
    cfg_strength,
    nfe_step,
    sway_sampling_coef,
)
from f5_tts.model.continual_utils import (
    VALID_FISHER_MODES,
    EWCAnchor,
    build_step_dataset,
    compute_trainable_fisher,
    continual_checkpoint_name,
    continual_steps,
    count_step_samples,
    materialized_ema_state_dict,
    materialized_state_dict,
    strip_parametrizations,
)
from f5_tts.model.finetune_strategies import build_finetune_strategy
from f5_tts.model.trainer_unlearn import TrainerUnlearn
from f5_tts.model.utils import exists


class TrainerUnlearnContinual(TrainerUnlearn):
    """Sequential speaker unlearning: one forget speaker per step, chained through the online weights.

    Each step re-runs the *whole* single-shot algorithm of `TrainerUnlearn` on a model initialised
    from the previous step's result, with one forget speaker and its own optimiser / EMA / LR
    schedule. Two properties make the chain work with the SVD-based fine-tuning strategies:

    * the model is de-parametrised (`strip_parametrizations`) at the start of every step, so unit
      selection and the SVD basis are recomputed from the *current* weights, and
    * every checkpoint is materialised (`materialized_state_dict`), so it loads into a plain model
      with the stock loaders - no `--svdiff` / `--svdiff_uv` flags anywhere downstream.

    The optional `continual.lambda` adds an EWC-style anchor term pulling this step's trainable
    parameters back towards their step-start values, i.e. towards the previous step's model.

    Nothing in `TrainerUnlearn` is modified: single-shot runs are bit-identical to before, and the
    first continual step follows exactly the same code path as a single-speaker single-shot run.
    """

    def __init__(self, *args, continual_params: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        continual_params = continual_params if continual_params is not None else {}
        self.continual_params = continual_params

        ewc_lambda = continual_params.get("lambda", None)
        self.ewc_lambda = float(ewc_lambda) if ewc_lambda is not None else None

        self.fisher_mode = str(continual_params.get("fisher", "none")).lower()
        if self.fisher_mode not in VALID_FISHER_MODES:
            raise ValueError(f"continual.fisher must be one of {VALID_FISHER_MODES}, got {self.fisher_mode!r}")

        self.fisher_profile_steps = int(continual_params.get("fisher_profile_steps", 200))
        self.fisher_loss_source = str(continual_params.get("fisher_loss_source", "retain"))
        self.include_future_forget_speakers_in_retain = bool(
            continual_params.get("include_future_forget_speakers_in_retain", False)
        )
        self.continual_resume = bool(continual_params.get("resume", True))

        # Chain bookkeeping. `_global_update` is monotonic across the whole chain so the wandb x-axis
        # never goes backwards; per-step save cadence uses a step-local counter instead.
        self._global_update = 0
        self._weight_stats_step = 0
        self._current_step_tag = None
        self._current_step = None
        self._teacher_loaded = False

    # ------------------------------------------------------------------ #
    # logging
    # ------------------------------------------------------------------ #

    def _define_wandb_metrics(self):
        super()._define_wandb_metrics()
        wandb.define_metric("task_loss", step_metric="train_step")
        wandb.define_metric("ewc_penalty", step_metric="train_step")
        wandb.define_metric("ewc_loss", step_metric="train_step")
        wandb.define_metric("continual/*", step_metric="train_step")

    def _log_scalars(self, values: dict):
        if not self.accelerator.is_local_main_process:
            return
        payload = {"train_step": self._global_update, **values}
        self.accelerator.log(payload, step=self._global_update)
        if self.logger == "tensorboard":
            for key, value in values.items():
                self.writer.add_scalar(key, value, self._global_update)

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #

    def load_checkpoint(self):
        """Disabled: resume in the continual stack is per step, driven by `train_continual`.

        The inherited implementation scans `checkpoint_path` and would happily adopt another step's
        checkpoint as this step's starting point.
        """
        return 0

    def _continual_metadata(self, step: dict) -> dict:
        return {
            "step_index": int(step["index"]),
            "num_steps": int(step["num_steps"]),
            "forget_speaker": step["speaker"],
            "forget_speakers_done": list(step["past"]) + [step["speaker"]],
            "forget_speakers_pending": list(step["future"]),
            "forget_speakers_all": list(step["past"]) + [step["speaker"]] + list(step["future"]),
            "ewc_lambda": self.ewc_lambda if self.ewc_lambda is not None else -1.0,
            "fisher": self.fisher_mode,
            "materialized": 1,
        }

    def save_checkpoint(self, update, last=False, *, final=False, step=None):
        """Save a *materialised* checkpoint for the current continual step.

        `final=True` writes `unlearned_model_{step_tag}.pt`, the artefact the next step starts from
        and the one inference/evaluation consume; it also drops the step's `..._last.pt`, which only
        exists for crash recovery while the step is running. Intermediate saves are
        `unlearned_model_{step_tag}_{update}.pt` / `..._last.pt` and are rotated within the step
        only, so steps can never delete each other's files.
        """
        step = step if step is not None else self._current_step
        step_tag = step["tag"] if step is not None else self._current_step_tag
        if step_tag is None:
            raise RuntimeError("save_checkpoint was called outside of a continual step.")

        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=materialized_state_dict(self.accelerator.unwrap_model(self.model)),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=materialized_ema_state_dict(self.ema_model),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if step is not None:
                checkpoint["continual"] = self._continual_metadata(step)

            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            if final:
                filename = continual_checkpoint_name(step_tag)
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/{filename}")
                print(f"Saved final continual checkpoint {filename} at update {update}")
                last_filename = f"unlearned_model_{step_tag}_last.pt"
                if os.path.exists(f"{self.checkpoint_path}/{last_filename}"):
                    os.remove(f"{self.checkpoint_path}/{last_filename}")
                    print(f"Removed mid-step checkpoint: {last_filename}")
            elif last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/unlearned_model_{step_tag}_last.pt")
                print(f"Saved last checkpoint for {step_tag} at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/unlearned_model_{step_tag}_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    self._rotate_step_checkpoints(step_tag)

            del checkpoint
            gc.collect()
        self.accelerator.wait_for_everyone()

    def _rotate_step_checkpoints(self, step_tag: str):
        prefix = f"unlearned_model_{step_tag}_"
        candidates = []
        for filename in os.listdir(self.checkpoint_path):
            if not filename.startswith(prefix) or not filename.endswith(".pt"):
                continue
            suffix = filename[len(prefix) : -len(".pt")]
            if not suffix.isdigit():  # skips `_last.pt` and anything hand-named
                continue
            candidates.append((int(suffix), filename))

        candidates.sort()
        while len(candidates) > self.keep_last_n_checkpoints:
            _, oldest_checkpoint = candidates.pop(0)
            os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
            print(f"Removed old checkpoint: {oldest_checkpoint}")

    def step_checkpoint_path(self, step_tag: str) -> str:
        return os.path.join(self.checkpoint_path, continual_checkpoint_name(step_tag))

    # ------------------------------------------------------------------ #
    # per-step setup
    # ------------------------------------------------------------------ #

    def _rebuild_base_optimizer_and_ema(self):
        """Fresh optimiser + EMA over the model's *current* parameter objects.

        `remove_parametrizations` installs brand-new `nn.Parameter`s, so any optimiser built before
        the strip holds dead references and would silently stop updating the model. This mirrors
        what `TrainerUnlearn.__init__` does, and runs before the strategy's own (optional) reset.
        """
        if self.bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        if self.is_main:
            self.ema_model = EMA(self.model, include_online_model=False, **self.ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.optimizer = self.accelerator.prepare(self.optimizer)

    def _read_plain_state_dicts(self, checkpoint_path: str) -> tuple[dict, dict]:
        """`(model_state_dict, ema_weights)` from a checkpoint, both with plain (materialised) keys.

        Read on every rank so multi-GPU runs stay in sync without broadcasting tensors. `ema_weights`
        has the `ema_model.` prefix stripped.
        """
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = {"ema_model_state_dict": load_file(checkpoint_path, device="cpu")}
        else:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        ema_state_dict = checkpoint.get("ema_model_state_dict", {})
        ema_weights = {
            key.replace("ema_model.", ""): value
            for key, value in ema_state_dict.items()
            if key not in ("initted", "update", "step")
        }
        model_state_dict = checkpoint.get("model_state_dict", ema_weights)
        return model_state_dict, ema_weights

    def _prepare_step_model(self, step: dict, previous_checkpoint_path: str | None) -> None:
        """Load this step's starting weights.

        Step 1 goes through `load_pretrained_checkpoint` exactly like a single-shot run. Later steps
        load the previous step's materialised checkpoint (online weights into the model, EMA payload
        into the EMA, mirroring the side-effect of `load_pretrained_checkpoint`).
        """
        if not self._teacher_loaded:
            print("Load teacher")
            self.load_pretrained_checkpoint(self.teacher)
            self._teacher_loaded = True

        if previous_checkpoint_path is None:
            print("Load student")
            self.load_pretrained_checkpoint(self.model)
            return

        print(f"Step {step['index']}: initialising from {os.path.basename(previous_checkpoint_path)}")
        model_state_dict, _ = self._read_plain_state_dicts(previous_checkpoint_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(model_state_dict, strict=True)

        if self.is_main:
            checkpoint = torch.load(previous_checkpoint_path, weights_only=True, map_location="cpu")
            if "ema_model_state_dict" in checkpoint:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            del checkpoint

        del model_state_dict
        gc.collect()

    def _maybe_compute_fisher(self, step: dict) -> dict | None:
        """Diagonal Fisher weighting the anchor, estimated at this step's starting weights.

        Classic EWC scores the previous task, but the previous step's data is gone by now and it
        also holds the current forget speaker as retain material - which would raise `F` on exactly
        the parameters the forget loss must move. This step's own data with `fisher_loss_source:
        retain` is the accessible, conflict-free stand-in.
        """
        if self.ewc_lambda is None or self.fisher_mode == "none":
            return None

        print(
            f"[continual-EWC {step['tag']}] estimating the Fisher on this step's data "
            f"(loss_source={self.fisher_loss_source})."
        )

        fisher = compute_trainable_fisher(
            self,
            profile_steps=self.fisher_profile_steps,
            loss_source=self.fisher_loss_source,
            log_prefix=f"continual-EWC {step['tag']}",
        )
        gc.collect()
        return fisher

    # ------------------------------------------------------------------ #
    # driver
    # ------------------------------------------------------------------ #

    def train_continual(
        self,
        train_dataset: Dataset,
        unlearn_method: str,
        num_workers: int = 16,
        resumable_with_seed: int | None = None,
    ):
        if unlearn_method not in ("TGU", "SGU"):
            raise ValueError(f"Unknown unlearning method for continual training: {unlearn_method}")
        if unlearn_method == "SGU":
            assert self.batch_size_type == "balanced_unlearn_sample", (
                "Batches should be balanced, this method requires an equal number of forget and retain "
                "samples per batch. Use batch_size_type == 'balanced_unlearn_sample'"
            )
            assert self.batch_size_per_gpu % 2 == 0

        if self.log_samples:
            self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            os.makedirs(f"{self.checkpoint_path}/samples", exist_ok=True)

        all_forget_speakers = list(self.forget_speakers)
        if len(set(all_forget_speakers)) != len(all_forget_speakers):
            raise ValueError(f"unlearn.forget_speakers must not repeat a speaker: {all_forget_speakers}")
        steps = continual_steps(all_forget_speakers)

        print(
            f"\n=== Continual unlearning: {len(steps)} steps, method={unlearn_method}, "
            f"ewc_lambda={self.ewc_lambda}, fisher={self.fisher_mode}, "
            f"future_speakers_in_retain={self.include_future_forget_speakers_in_retain} ===\n"
        )

        previous_checkpoint_path = None

        for step in steps:
            step_tag = step["tag"]
            checkpoint_path = self.step_checkpoint_path(step_tag)

            excluded = set(step["past"])
            if not self.include_future_forget_speakers_in_retain:
                excluded |= set(step["future"])
            step_dataset = build_step_dataset(train_dataset, step["speaker"], excluded)

            if self.continual_resume and os.path.exists(checkpoint_path):
                print(f"[{step_tag}] final checkpoint already exists, skipping this step (continual.resume=True).")
                previous_checkpoint_path = checkpoint_path
                continue

            num_retain, num_forget = count_step_samples(step_dataset, step["speaker"])
            print(
                f"\n--- Step {step['index']}/{len(steps)} | forget speaker {step['speaker']} | "
                f"excluded {sorted(excluded)} | retain samples {num_retain} | forget samples {num_forget} ---"
            )

            self._run_continual_step(
                step,
                step_dataset,
                previous_checkpoint_path,
                unlearn_method=unlearn_method,
                num_workers=num_workers,
                resumable_with_seed=resumable_with_seed,
                num_retain=num_retain,
                num_forget=num_forget,
            )

            previous_checkpoint_path = checkpoint_path

        self.forget_speakers = all_forget_speakers
        self.accelerator.end_training()

    def _run_continual_step(
        self,
        step: dict,
        step_dataset,
        previous_checkpoint_path: str | None,
        *,
        unlearn_method: str,
        num_workers: int,
        resumable_with_seed: int | None,
        num_retain: int,
        num_forget: int,
    ):
        self._current_step = step
        self._current_step_tag = step["tag"]
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # 1. clean slate: bake in the previous step's deltas, then unfreeze everything so the
        #    freezer / selector strategies start from the same state they see in a single-shot run.
        removed = strip_parametrizations(unwrapped_model)
        if removed:
            print(f"[{step['tag']}] materialised {removed} parametrised tensors from the previous step.")
        for param in unwrapped_model.parameters():
            param.requires_grad = True

        # 2. fresh optimiser + EMA bound to the current parameter objects.
        self._rebuild_base_optimizer_and_ema()

        # 3. this step's data (single forget speaker, already-unlearned speakers dropped entirely).
        self.forget_speakers = [step["speaker"]]
        train_dataloader = self.create_dataloader(step_dataset, num_workers, resumable_with_seed)

        warmup_updates, decay_updates = self._get_lr_schedule_updates(train_dataloader)
        if decay_updates <= 0:
            print(
                f"F5-TTS WARNING: [{step['tag']}] num_warmup_updates={self.num_warmup_updates} covers the whole step "
                f"({math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs} updates); "
                "the learning rate will never leave warmup. Lower optim.num_warmup_updates - it is per step here."
            )
        self.scheduler = self._build_lr_scheduler(warmup_updates, decay_updates)
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)

        # 4. starting weights for this step.
        self._prepare_step_model(step, previous_checkpoint_path)

        # Stashed so strategies can build their own profiling dataloader from apply().
        self._pretrain_dataset = step_dataset
        self._pretrain_num_workers = num_workers
        self._pretrain_resumable_with_seed = resumable_with_seed
        self._pretrain_unlearn_method = unlearn_method

        # 5. fresh strategy: re-select units and recompute the SVD basis from the current weights.
        self.finetune_strategy = build_finetune_strategy(self.model_cfg_dict)
        self.finetune_strategy.apply(unwrapped_model, trainer=self)
        if self.finetune_strategy.requires_optimizer_reset:
            self.reset_optimizer_and_scheduler_for_trainable_params(
                warmup_updates, decay_updates, context=self.finetune_strategy.name
            )
        self.finetune_strategy.run_pre_training_hook(
            self,
            unlearn_method=unlearn_method,
            train_dataset=step_dataset,
            num_workers=num_workers,
            resumable_with_seed=resumable_with_seed,
        )

        num_trainable_params = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in student model: {num_trainable_params / 1e6:.3f}M")

        # 6. Fisher and anchor come after the strategy, so both are keyed by the parameters this
        #    step actually trains. Neither profiling pass steps the optimiser, so the deltas are
        #    still zero here and the penalty opens at exactly 0.
        anchor = None
        if self.ewc_lambda is not None:
            fisher = self._maybe_compute_fisher(step)
            anchor = EWCAnchor.build(unwrapped_model, fisher=fisher)
            print(
                f"[{step['tag']}] EWC anchor over {len(anchor)} tensors "
                f"({anchor.num_elements / 1e6:.1f}M elements, lambda={self.ewc_lambda}, "
                f"fisher_weighted={anchor.fisher_weighted})"
            )
            del fisher
            gc.collect()

        # set teacher to eval and no grad
        self.accelerator.unwrap_model(self.teacher).eval()
        for param in self.accelerator.unwrap_model(self.teacher).parameters():
            param.requires_grad = False

        self._log_scalars(
            {
                "continual/step_index": step["index"],
                "continual/forget_speaker": step["speaker"],
                "continual/num_retain_samples": num_retain,
                "continual/num_forget_samples": num_forget,
                "continual/num_trainable_params": num_trainable_params,
            }
        )

        # 7. train.
        if unlearn_method == "TGU":
            local_update = self._train_step_TGU(
                train_dataloader, step=step, anchor=anchor, resumable_with_seed=resumable_with_seed
            )
        else:
            local_update = self._train_step_SGU(
                train_dataloader, step=step, anchor=anchor, resumable_with_seed=resumable_with_seed
            )

        # 8. the artefact the next step (and inference/evaluation) starts from.
        self.save_checkpoint(local_update, final=True, step=step)

        # 9. tear the step down: accelerate keeps every object it prepared, so without this each
        #    step leaks its dataloaders' persistent workers and its optimizer state into the next.
        self.release_dataloader(train_dataloader)
        self.release_stale_prepared_optimizers()

        del anchor, train_dataloader
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------ #
    # per-step training loops
    # ------------------------------------------------------------------ #

    def _ewc_terms(self, anchor: EWCAnchor | None):
        """`(penalty, weighted_term)` for this update, or `(None, None)` when EWC is off."""
        if anchor is None:
            return None, None
        penalty = anchor.penalty()
        return penalty, self.ewc_lambda * penalty

    def _train_step_TGU(self, train_dataloader, *, step, anchor, resumable_with_seed) -> int:
        """One continual step of TGU. Mirrors `TrainerUnlearn.train_TGU`'s epoch loop."""
        start_update = 0
        local_update = start_update

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
                desc=f"[{step['tag']}] Epoch {epoch + 1}/{self.epochs}",
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

                    task_loss = (
                        self.unlearn_params["lambda"] * retain_loss + (1 - self.unlearn_params["lambda"]) * forget_loss
                    )

                    ewc_penalty, ewc_term = self._ewc_terms(anchor)
                    loss = task_loss if ewc_term is None else task_loss + ewc_term

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    local_update += 1
                    self._global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(local_update), loss=loss.item())

                scalars = {
                    "loss": loss.item(),
                    "task_loss": task_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if ewc_term is not None:
                    scalars["ewc_penalty"] = ewc_penalty.item()
                    scalars["ewc_loss"] = ewc_term.item()
                self._log_scalars(scalars)

                if local_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(local_update, last=True, step=step)

                if local_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(local_update, step=step)

        return local_update  # the caller immediately writes the step's final checkpoint

    def _train_step_SGU(self, train_dataloader, *, step, anchor, resumable_with_seed) -> int:
        """One continual step of SGU. Mirrors `TrainerUnlearn.train_SGU`'s epoch loop."""
        start_update = 0
        local_update = start_update

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
                desc=f"[{step['tag']}] Epoch {epoch + 1}/{self.epochs}",
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
                            task_loss = retain_loss + self.unlearn_params["forget_ratio"] * forget_loss
                    else:
                        do_forget = torch.rand((1,)).item() < self.unlearn_params["forget_ratio"]
                        if do_forget and mel_spec_forget.numel() > 0:  # if there is at least one forget sample
                            mel_spec_concat = torch.cat([mel_spec_forget, mel_spec_retain], dim=1)
                            text_inputs_concat = [
                                text_inputs_forget[i] + " " + text_inputs_retain[i]
                                for i in range(len(text_inputs_retain))
                            ]
                            mel_lengths_concat = mel_lengths_forget + mel_lengths_retain

                            task_loss, cond, pred = self.model.forward_unlearn_SGU(
                                mel_spec_concat,
                                text=text_inputs_concat,
                                lens=mel_lengths_concat,
                                retain_lens=mel_lengths_retain,
                                noise_scheduler=self.noise_scheduler,
                            )
                        else:
                            task_loss, cond, pred = self.model.forward_unlearn(
                                mel_spec_retain,
                                text=text_inputs_retain,
                                lens=mel_lengths_retain,
                                noise_scheduler=self.noise_scheduler,
                                forget=False,
                            )

                    ewc_penalty, ewc_term = self._ewc_terms(anchor)

                    if not self.unlearn_params.get("use_torchjd", False):
                        loss = task_loss if ewc_term is None else task_loss + ewc_term
                        self.accelerator.backward(loss)
                    else:
                        # The anchor rides with the retain objective: it is a stability constraint,
                        # so putting it in the forget row would have UPGrad reconcile "stay near the
                        # anchor" against "move away from the speaker" as one direction.
                        task_loss = retain_loss + forget_loss  # for logging purposes
                        retain_objective = retain_loss if ewc_term is None else retain_loss + ewc_term
                        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                        autojac.backward([retain_objective, forget_loss], inputs=trainable_params)
                        jac_to_grad(trainable_params, self.aggregator)
                        loss = task_loss if ewc_term is None else task_loss + ewc_term

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    local_update += 1
                    self._global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(local_update), loss=loss.item())

                scalars = {
                    "loss": loss.item(),
                    "task_loss": task_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if ewc_term is not None:
                    scalars["ewc_penalty"] = ewc_penalty.item()
                    scalars["ewc_loss"] = ewc_term.item()
                self._log_scalars(scalars)

                if self.accelerator.is_local_main_process:
                    if self.accelerator.sync_gradients and (local_update % weight_stats_update_interval == 0 or i == 0):
                        self._weight_stats_step += 1
                        weight_stats = {"weight_stats_step": self._weight_stats_step}
                        for name, param in self.accelerator.unwrap_model(self.model).named_parameters():
                            if param.requires_grad and ".weight" in name:
                                l2_norm = torch.norm(param.data, p=2).item()
                                frobenius_norm = torch.norm(param.data, p="fro").item()
                                soft_sparsity = (param.data.abs() < 1e-3).float().mean().item()
                                metric_prefix = self._weight_stat_metric_prefix(name)
                                weight_stats[f"{metric_prefix}/l2_norm"] = l2_norm
                                weight_stats[f"{metric_prefix}/frobenius_norm"] = frobenius_norm
                                weight_stats[f"{metric_prefix}/soft_sparsity"] = soft_sparsity
                        self.accelerator.log(weight_stats, step=self._global_update)

                if local_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(local_update, last=True, step=step)

                if local_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(local_update, step=step)

        return local_update  # the caller immediately writes the step's final checkpoint
