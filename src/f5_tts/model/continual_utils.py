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

import contextlib
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
VALID_ANCHOR_MODES = ("step_start", "pretrained")
VALID_PENALTY_REDUCTIONS = ("sum", "mean")
VALID_FISHER_NORMALIZATIONS = ("none", "mean", "max")

_PARAMETRIZATION_MARKER = ".parametrizations."


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


def _split_parametrized_name(name: str) -> tuple[str, str, str] | None:
    """`(module_path, tensor_name, delta_name)` for an SVD delta, or None for a plain parameter."""
    if _PARAMETRIZATION_MARKER not in name:
        return None
    module_path, rest = name.split(_PARAMETRIZATION_MARKER, 1)
    parts = rest.split(".")
    if len(parts) != 3:
        return None
    tensor_name, _index, delta_name = parts
    return module_path, tensor_name, delta_name


def _pretrained_anchor_entry(
    unwrapped_model: nn.Module,
    name: str,
    param: nn.Parameter,
    pretrained_weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """`(target, scale)` such that `scale * param - target` is the residual against the pretrained weight.

    For a plain parameter that residual is simply `p - p_pretrained`. For an SVD delta it is written
    in *weight* space, which keeps the closed form free of any division by a singular value:

      delta_U: W(D) = (U + D) diag(S) Vh, and `Vh` has orthonormal rows, so
               ||W(D) - W0||_F^2 = ||D * S - (W0 Vh^T - U * S)||_F^2 + const.
      delta_V: W(D) = U diag(S) (Vh + D^T), and `U` has orthonormal columns, so
               ||W(D) - W0||_F^2 = ||D * S - (U^T W0 - S * Vh)^T||_F^2 + const.
      delta_S: W(dS) = U diag(S + dS) Vh, so the dS-dependent part is
               ||dS - (diag(U^T W0 Vh^T) - S)||^2.

    The dropped constants are the components of `W0` outside the frozen SVD subspace; they carry no
    gradient. Returns None when the pretrained weight for this parameter cannot be resolved.
    """
    split = _split_parametrized_name(name)

    if split is None:
        target = pretrained_weights.get(name)
        if target is None:
            return None
        return target.to(device=param.device, dtype=param.dtype), None

    module_path, tensor_name, delta_name = split
    plain_key = f"{module_path}.{tensor_name}" if module_path else tensor_name
    w0 = pretrained_weights.get(plain_key)
    if w0 is None:
        return None

    try:
        parametrization = unwrapped_model.get_submodule(module_path).parametrizations[tensor_name][0]
    except (AttributeError, KeyError, IndexError):
        return None

    u, s, vh = parametrization.U, parametrization.S, parametrization.Vh
    w0 = w0.to(device=u.device, dtype=u.dtype)

    if delta_name == "delta_U":
        target = w0 @ vh.T - u * s
        scale = s.unsqueeze(0)
    elif delta_name == "delta_V":
        target = (u.T @ w0 - s.unsqueeze(-1) * vh).T
        scale = s.unsqueeze(0)
    elif delta_name == "delta_S":
        target = ((u.T @ w0) * vh).sum(-1) - s
        scale = None
    else:
        return None

    target = target.to(device=param.device, dtype=param.dtype)
    scale = scale.to(device=param.device, dtype=param.dtype) if scale is not None else None
    return target, scale


def _undo_singular_value_factor(fisher: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Divide a delta-space Fisher by `S^2`, leaving the Fisher of the effective weight.

    A gradient w.r.t. `delta_U` is `S_j * [G Vh^T]_ij` (and `S_j * [G^T U]_qj` for `delta_V`), so the
    measured `F = S^2 * Ghat^2`. `anchor: pretrained` measures its residual in weight space, so
    without this the penalty would carry an extra `S_j^2` and protect the dominant singular
    directions far more strongly than `anchor: step_start` does. Dividing it out leaves both modes
    weighting a weight-space displacement by the same `Ghat^2`, so they differ only in anchor point.

    Columns with a negligible singular value are floored rather than divided; `F` is proportional to
    `S^2` there anyway, so they carry no measurable importance either way.
    """
    scale_sq = scale.pow(2)
    floor = max(float(scale_sq.max()) * 1e-12, torch.finfo(scale.dtype).tiny)
    return fisher / scale_sq.clamp_min(floor)


def normalize_fisher(fisher: dict[str, torch.Tensor] | None, mode: str) -> dict[str, torch.Tensor] | None:
    """Rescale the Fisher so one `continual.lambda` range works across `continual.fisher` modes.

    Raw squared gradients sit around 1e-10, so an unnormalised Fisher makes the anchor a no-op while
    identity weighting (`fisher: none`) makes the same penalty dominate the task loss. Dividing by
    the global mean or max puts both on a comparable scale.
    """
    if mode not in VALID_FISHER_NORMALIZATIONS:
        raise ValueError(f"continual.fisher_normalize must be one of {VALID_FISHER_NORMALIZATIONS}, got {mode!r}")
    if fisher is None or mode == "none" or not fisher:
        return fisher

    if mode == "mean":
        total = sum(tensor.sum() for tensor in fisher.values())
        count = sum(tensor.numel() for tensor in fisher.values())
        scale = total / count if count else None
    else:
        scale = max(tensor.max() for tensor in fisher.values())

    if scale is None or not torch.isfinite(torch.as_tensor(scale)) or float(scale) <= 0.0:
        print(f"[continual-EWC] fisher_normalize={mode} produced a non-positive scale; leaving the Fisher as is.")
        return fisher

    print(f"[continual-EWC] fisher_normalize={mode}: dividing the Fisher by {float(scale):.3e}")
    return {name: tensor / scale for name, tensor in fisher.items()}


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
    """Elastic-Weight-Consolidation anchor over this step's trainable parameters.

    `penalty = 0.5 * sum_i F_i * (scale_i * p_i - target_i)^2`, optionally divided by the element
    count (`reduction="mean"`).

    The penalty is written against the parameters the optimizer actually updates - the SVD deltas
    when a svdiff strategy is active, plain weights/biases otherwise - so gradients reach them
    directly, with no extra parametrization forward per update.

    `anchor_mode` picks what the term pulls back towards:

    * `step_start` - each parameter's value at the start of this step (`scale_i = 1`). For a delta
      that value is zero, and since `W = (U + delta_U) S Vh` with `U`, `S`, `Vh` frozen, this is
      identical to anchoring `U + delta_U` on the step's starting weights. The penalty opens at 0
      and only bounds *per-step* drift, so the chain's total drift stays unbounded.
    * `pretrained` - the original pretrained weights, in weight space (see
      `_pretrained_anchor_entry` for the closed form and the per-column `scale_i = S_j`). The
      penalty opens at the drift already accumulated by earlier steps, which is the point: it
      bounds drift across the whole chain.

    Both modes weight a *weight-space* displacement by the same quantity: `step_start` gets that for
    free (`F * delta^2 == Ghat^2 * (delta * S)^2`), and `pretrained` gets there via
    `_undo_singular_value_factor`. So the two differ only in where they anchor, never in the metric.

    `F_i` is the optional diagonal Fisher; `None` means identity weighting (plain L2 anchoring).
    """

    def __init__(
        self,
        entries: list[tuple[str, nn.Parameter, torch.Tensor, torch.Tensor | None, torch.Tensor | None]],
        device,
        fisher_weighted: bool,
        anchor_mode: str = "step_start",
        reduction: str = "sum",
    ):
        self.entries = entries
        self.device = device
        self.fisher_weighted = fisher_weighted
        self.anchor_mode = anchor_mode
        self.reduction = reduction
        self._num_elements = sum(param.numel() for _, param, _, _, _ in entries)

    @classmethod
    def build(
        cls,
        unwrapped_model: nn.Module,
        *,
        fisher: dict[str, torch.Tensor] | None = None,
        anchor_mode: str = "step_start",
        pretrained_weights: dict[str, torch.Tensor] | None = None,
        reduction: str = "sum",
    ) -> "EWCAnchor":
        """Snapshot the anchor over the parameters this step trains.

        Call this after `apply()` / `run_pre_training_hook()`, so the trainable set is final and the
        deltas are still zero.
        """
        if anchor_mode not in VALID_ANCHOR_MODES:
            raise ValueError(f"continual.anchor must be one of {VALID_ANCHOR_MODES}, got {anchor_mode!r}")
        if reduction not in VALID_PENALTY_REDUCTIONS:
            raise ValueError(
                f"continual.penalty_reduction must be one of {VALID_PENALTY_REDUCTIONS}, got {reduction!r}"
            )
        if anchor_mode == "pretrained" and not pretrained_weights:
            raise ValueError("continual.anchor='pretrained' needs the pretrained weights, but none were supplied.")

        entries = []
        device = next(unwrapped_model.parameters()).device
        unscored: list[str] = []
        unresolved: list[str] = []

        for name, param in trainable_parameter_entries(unwrapped_model):
            fisher_tensor = None
            if fisher is not None:
                if name not in fisher:
                    # No measured importance, so no constraint - the EWC reading of F_i = 0.
                    unscored.append(name)
                    continue
                fisher_tensor = fisher[name].to(device=param.device, dtype=param.dtype)

            if anchor_mode == "pretrained":
                resolved = _pretrained_anchor_entry(unwrapped_model, name, param, pretrained_weights)
                if resolved is None:
                    unresolved.append(name)
                    continue
                target, scale = resolved
                if fisher_tensor is not None and scale is not None:
                    fisher_tensor = _undo_singular_value_factor(fisher_tensor, scale)
            else:
                target, scale = param.detach().clone(), None

            entries.append((name, param, target, fisher_tensor, scale))

        if unscored:
            print(
                f"[continual-EWC] {len(unscored)} trainable tensors never received a gradient during Fisher "
                f"profiling and are left unconstrained (e.g. {unscored[0]})."
            )
        if unresolved:
            print(
                f"[continual-EWC] {len(unresolved)} trainable tensors have no resolvable pretrained weight and are "
                f"left unconstrained (e.g. {unresolved[0]})."
            )

        return cls(entries, device, fisher is not None, anchor_mode=anchor_mode, reduction=reduction)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_elements(self) -> int:
        return self._num_elements

    def penalty(self) -> torch.Tensor:
        total = None
        for _, param, target, fisher, scale in self.entries:
            residual = param if scale is None else param * scale
            squared_diff = (residual - target).pow(2)
            # Entries are homogeneous: `build` drops every tensor missing from the fisher dict, so a
            # `None` here means the whole anchor is unweighted, not that this tensor scored zero.
            if fisher is not None:
                squared_diff = squared_diff * fisher
            summed = squared_diff.sum()
            total = summed if total is None else total + summed
        if total is None:
            return torch.zeros((), device=self.device)
        penalty = 0.5 * total
        if self.reduction == "mean" and self._num_elements:
            penalty = penalty / self._num_elements
        return penalty


@contextlib.contextmanager
def parameters_at_pretrained_weights(
    unwrapped_model: nn.Module,
    pretrained_weights: dict[str, torch.Tensor],
    *,
    rtol: float = 1e-6,
    log_prefix: str = "continual-EWC",
):
    """Temporarily move every trainable parameter to the value that reproduces the pretrained weight.

    EWC's quadratic is only a valid expansion of the loss when the Fisher and the anchor sit at the
    same point, so `anchor: pretrained` wants curvature measured at the pretrained weights - while
    the gradients still have to arrive in *this* step's coordinates. That value is `target / scale`,
    the point where the anchor's own residual is zero: `W0` itself for a plain weight, and
    `D0 = target / S` for an SVD delta, whose effective weight `(U + D0) S Vh` is then `W0` and whose
    backward pass yields `dL/dD = S * [G(W0) Vh^T]`, exactly what the anchor is weighted by.

    A plain parameter is restored bit-exactly. An SVD delta is not: columns whose singular value is
    negligible are floored, and a weight with more columns than rows has a component of `W0` outside
    the frozen basis that no delta can reach. The worst relative error is printed, and it is the
    number to check if the anchor misbehaves.
    """
    saved: list[tuple[nn.Parameter, torch.Tensor]] = []

    try:
        with torch.no_grad():
            for name, param in trainable_parameter_entries(unwrapped_model):
                resolved = _pretrained_anchor_entry(unwrapped_model, name, param, pretrained_weights)
                if resolved is None:
                    continue
                target, scale = resolved
                delta0 = target if scale is None else target / scale.clamp_min(float(scale.max()) * rtol)
                saved.append((param, param.detach().clone()))
                param.copy_(delta0)

            worst, worst_name = 0.0, None
            for module_name, module in unwrapped_model.named_modules():
                if not parametrize.is_parametrized(module):
                    continue
                for tensor_name in list(module.parametrizations.keys()):
                    reference = pretrained_weights.get(f"{module_name}.{tensor_name}")
                    if reference is None:
                        continue
                    effective = getattr(module, tensor_name)
                    reference = reference.to(device=effective.device, dtype=effective.dtype)
                    denominator = reference.norm().clamp_min(torch.finfo(effective.dtype).tiny)
                    error = ((effective - reference).norm() / denominator).item()
                    if error > worst:
                        worst, worst_name = error, f"{module_name}.{tensor_name}"

        print(
            f"[{log_prefix}] profiling the Fisher at the pretrained weights over {len(saved)} trainable tensors "
            f"(worst relative reconstruction error {worst:.2e} on {worst_name})."
        )
        yield worst
    finally:
        with torch.no_grad():
            for param, original in saved:
                param.copy_(original)


def compute_trainable_fisher(
    trainer: "TrainerUnlearn",
    *,
    profile_steps: int,
    loss_source: str,
    normalize: str = "none",
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
    cannot end up protecting the parameters this step's forget loss has to move. `normalize`
    rescales the result so `continual.lambda` keeps a usable range - see `normalize_fisher`.
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
        return None

    return normalize_fisher(fisher, normalize)


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
    if float(sway_sampling_coef).is_integer():
        sway_sampling_coef = int(sway_sampling_coef)
    return (
        f"results/{ckpt_dir_name}/{step_tag}_{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )


def continual_summary_subdir(ckpt_dir_name: str) -> str:
    """Directory for the chain-level summaries: the per-step wav dirs' parent, so one run is one tree."""
    return f"results/{ckpt_dir_name}"
