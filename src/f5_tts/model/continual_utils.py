"""Helpers shared by the continual (sequential) speaker-unlearning stack.

The continual framework unlearns one speaker per step, each step starting from the model produced
by the previous one. Two invariants make that work with the SVD-based fine-tuning strategies:

1. Every checkpoint written by the continual trainer is *materialized*: the SVD deltas are baked
   back into plain `.weight` tensors (`materialized_state_dict`). Checkpoints therefore load into a
   plain model with the stock loaders, with no `--svdiff` / `--svdiff_uv` flags.
2. At the beginning of each step the live model is de-parametrized in place
   (`strip_parametrizations`), so unit selection and the SVD basis are recomputed from the
   *current* weights rather than from the original pretrained ones.

The optional EWC term (`continual.lambda`) is written against the step's *trainable* parameters -
the SVD deltas when a svdiff strategy is active - and is anchored on their step-start values.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from f5_tts.model.finetune_strategies._fisher_profiling import (
    compute_fisher,
    select_scored_param_names,
)

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


VALID_FISHER_MODES = ("none", "current_retain")


# --------------------------------------------------------------------------------------------- #
# parametrization <-> plain weights
# --------------------------------------------------------------------------------------------- #


def materialized_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return `model.state_dict()` with every parametrized tensor replaced by its effective value.

    A parametrized `nn.Linear` normally serializes as
    `...parametrizations.weight.original` + `...parametrizations.weight.0.{U,S,Vh,delta_U}`.
    Those keys are dropped and replaced by a single `...weight` holding `(U + delta_U) S Vh`, so the
    result is key-for-key identical to the state dict of a plain (never-parametrized) model.

    The model itself is left untouched; use `strip_parametrizations` to de-parametrize in place.
    """
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if not parametrize.is_parametrized(module):
                continue
            module_prefix = f"{module_name}." if module_name else ""
            for tensor_name in list(module.parametrizations.keys()):
                effective = getattr(module, tensor_name).detach().cpu().clone()
                stale_prefix = f"{module_prefix}parametrizations.{tensor_name}."
                for key in [key for key in state_dict if key.startswith(stale_prefix)]:
                    del state_dict[key]
                state_dict[f"{module_prefix}{tensor_name}"] = effective

    return state_dict


def materialized_ema_state_dict(ema_model: nn.Module) -> dict[str, torch.Tensor]:
    """Materialized `EMA.state_dict()` (the inner `ema_model` is parametrized too)."""
    state_dict = {f"ema_model.{key}": value for key, value in materialized_state_dict(ema_model.ema_model).items()}
    full_state_dict = ema_model.state_dict()
    for key in ("initted", "step"):
        if key in full_state_dict:
            state_dict[key] = full_state_dict[key].detach().cpu().clone()
    return state_dict


def strip_parametrizations(model: nn.Module) -> int:
    """De-parametrize `model` in place, keeping the effective weights. Returns how many were removed.

    After this the module holds plain `nn.Parameter` weights again (byte-identical forward output),
    so a fresh strategy can re-select units and recompute the SVD basis from the current values.
    """
    removed = 0
    for _, module in list(model.named_modules()):
        if not parametrize.is_parametrized(module):
            continue
        for tensor_name in list(module.parametrizations.keys()):
            parametrize.remove_parametrizations(module, tensor_name, leave_parametrized=True)
            removed += 1
    return removed


def trainable_parameter_entries(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """`(name, param)` for every parameter this step trains: SVD deltas, plain weights and biases."""
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


# --------------------------------------------------------------------------------------------- #
# per-step data
# --------------------------------------------------------------------------------------------- #


def dataset_speaker_ids(dataset) -> list:
    """Speaker id per row, read as a single columnar access instead of one `__getitem__` per row."""
    if "speaker_id" not in dataset.data.column_names:
        raise ValueError(
            "Continual unlearning requires a `speaker_id` column in the dataset, "
            f"but found columns: {dataset.data.column_names}"
        )
    return list(dataset.data["speaker_id"])


def build_step_dataset(base_dataset, current_speaker, excluded_speakers: Iterable):
    """A view of `base_dataset` for one continual step.

    `current_speaker` becomes the only forget speaker; every speaker in `excluded_speakers` is
    dropped from the step entirely (neither retain nor forget). The returned object is a shallow
    copy sharing the mel-spectrogram module, with `.data` narrowed via `Dataset.select` so no audio
    is re-read. `BalancedUnlearningSampleBatchSampler` recomputes its retain/forget index lists from
    `.data` / `.forget_speakers` at construction, so a fresh dataloader picks the change up.
    """
    excluded = set(excluded_speakers)
    if current_speaker in excluded:
        raise ValueError(f"The current forget speaker {current_speaker} cannot also be excluded.")

    speaker_ids = dataset_speaker_ids(base_dataset)
    keep_indices = [index for index, speaker in enumerate(speaker_ids) if speaker not in excluded]
    if not keep_indices:
        raise ValueError(f"Excluding speakers {sorted(excluded)} removed every sample from the dataset.")
    if not any(speaker_ids[index] == current_speaker for index in keep_indices):
        raise ValueError(f"Forget speaker {current_speaker} has no samples in the dataset.")

    step_dataset = copy.copy(base_dataset)
    step_dataset.data = base_dataset.data.select(keep_indices)
    if getattr(base_dataset, "durations", None) is not None:
        step_dataset.durations = [base_dataset.durations[index] for index in keep_indices]
    step_dataset.forget_speakers = [current_speaker]
    return step_dataset


def count_step_samples(step_dataset, current_speaker) -> tuple[int, int]:
    """`(num_retain_samples, num_forget_samples)` for a step dataset."""
    speaker_ids = dataset_speaker_ids(step_dataset)
    num_forget = sum(1 for speaker in speaker_ids if speaker == current_speaker)
    return len(speaker_ids) - num_forget, num_forget


# --------------------------------------------------------------------------------------------- #
# EWC anchoring
# --------------------------------------------------------------------------------------------- #


class EWCAnchor:
    """Elastic-Weight-Consolidation anchor over this step's trainable parameters: `0.5 * sum_i F_i * (p_i - p*_i)^2`.

    The penalty is written against the parameters the optimizer actually updates - the SVD deltas
    when a svdiff strategy is active, plain weights/biases otherwise - so gradients reach them
    directly, with no extra parametrization forward per update. For a delta the anchor is its
    step-start value (zero), and since `W = (U + delta_U) S Vh` with `U`, `S`, `Vh` frozen, that is
    identical to anchoring `U + delta_U` on the step's starting weights.

    `F_i` is the optional diagonal Fisher; `None` means identity weighting (plain L2 anchoring).
    """

    def __init__(
        self,
        entries: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor | None]],
        device,
        fisher_weighted: bool,
    ):
        self.entries = entries
        self.device = device
        self.fisher_weighted = fisher_weighted

    @classmethod
    def build(
        cls,
        unwrapped_model: nn.Module,
        *,
        fisher: dict[str, torch.Tensor] | None = None,
    ) -> "EWCAnchor":
        """Snapshot the anchor over the parameters this step trains.

        Call this after `apply()` / `run_pre_training_hook()`, so the trainable set is final and the
        deltas are still zero - the penalty then starts at exactly 0.
        """
        entries = []
        device = next(unwrapped_model.parameters()).device
        unscored: list[str] = []

        for name, param in trainable_parameter_entries(unwrapped_model):
            fisher_tensor = None
            if fisher is not None:
                if name not in fisher:
                    # No measured importance, so no constraint - the EWC reading of F_i = 0.
                    unscored.append(name)
                    continue
                fisher_tensor = fisher[name].to(device=param.device, dtype=param.dtype)

            entries.append((name, param, param.detach().clone(), fisher_tensor))

        if unscored:
            print(
                f"[continual-EWC] {len(unscored)} trainable tensors never received a gradient during Fisher "
                f"profiling and are left unconstrained (e.g. {unscored[0]})."
            )

        return cls(entries, device, fisher is not None)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_elements(self) -> int:
        return sum(anchor.numel() for _, _, anchor, _ in self.entries)

    def penalty(self) -> torch.Tensor:
        total = None
        for _, param, anchor, fisher in self.entries:
            squared_diff = (param - anchor).pow(2)
            # Entries are homogeneous: `build` drops every tensor missing from the fisher dict, so a
            # `None` here means the whole anchor is unweighted, not that this tensor scored zero.
            if fisher is not None:
                squared_diff = squared_diff * fisher
            summed = squared_diff.sum()
            total = summed if total is None else total + summed
        if total is None:
            return torch.zeros((), device=self.device)
        return 0.5 * total


def compute_trainable_fisher(
    trainer: "TrainerUnlearn",
    *,
    profile_steps: int,
    loss_source: str,
    log_prefix: str = "continual-EWC",
) -> dict[str, torch.Tensor] | None:
    """Diagonal empirical Fisher over the step's trainable parameters, keyed by parameter name.

    Estimated at the step-start weights on the step's *own* data - the only data available once a
    speaker has been unlearned - through the profiling context the trainer already stashed
    (`_pretrain_dataset`, `forget_speakers`). Run this after the fine-tuning strategy has been
    applied, so the scores land in the same coordinates the optimizer updates (`delta_U` under
    svdiff, plain weights/biases otherwise) and can weight an anchor written against those
    parameters.

    With `loss_source="retain"` the current forget speaker contributes no gradient, so the anchor
    cannot end up protecting the parameters this step's forget loss has to move.
    """
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    scored_names = select_scored_param_names(unwrapped_model, "all")
    if not scored_names:
        print(f"[{log_prefix}] no trainable tensors to score; skipping Fisher profiling.")
        return None

    fisher = compute_fisher(
        trainer,
        unwrapped_model,
        scored_names,
        profile_steps=profile_steps,
        loss_source=loss_source,
        log_prefix=log_prefix,
    )

    if fisher is None:
        print(f"[{log_prefix}] Fisher profiling collected zero effective steps; falling back to identity weighting.")
    return fisher


# --------------------------------------------------------------------------------------------- #
# naming, shared by the trainer, the inference script and the evaluation script
# --------------------------------------------------------------------------------------------- #


def continual_step_tag(step_index: int, forget_speaker) -> str:
    """`step03_spk40` - 1-based index so checkpoints sort in unlearning order."""
    return f"step{step_index:02d}_spk{forget_speaker}"


def continual_steps(forget_speakers: Sequence) -> list[dict[str, Any]]:
    """Expand `unlearn.forget_speakers` into per-step descriptors (index, speaker, tag, past, future)."""
    forget_speakers = list(forget_speakers)
    steps = []
    for position, speaker in enumerate(forget_speakers):
        step_index = position + 1
        steps.append(
            {
                "index": step_index,
                "num_steps": len(forget_speakers),
                "speaker": speaker,
                "tag": continual_step_tag(step_index, speaker),
                "past": forget_speakers[:position],
                "future": forget_speakers[position + 1 :],
            }
        )
    return steps


def continual_checkpoint_name(step_tag: str) -> str:
    """Final checkpoint of a step (materialized, loads into a plain model)."""
    return f"unlearned_model_{step_tag}.pt"


def continual_results_subdir(
    ckpt_dir_name: str,
    step_tag: str,
    testset: str,
    *,
    seed: int,
    ode_method: str,
    nfe_step: int,
    mel_spec_type: str,
    sway_sampling_coef: float,
    cfg_strength: float,
    speed: float,
    use_truth_duration: bool = False,
    no_ref_audio: bool = False,
) -> str:
    """Path under `results/` for one step's generated wavs; mirrors `eval_libritts_infer_batch.py`."""
    return (
        f"results/{ckpt_dir_name}/{step_tag}_{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )


def continual_summary_subdir(ckpt_dir_name: str, testset: str) -> str:
    """Path under `results/` for the chain-level evaluation summary."""
    return f"results/{ckpt_dir_name}_continual_summary/{testset}"
