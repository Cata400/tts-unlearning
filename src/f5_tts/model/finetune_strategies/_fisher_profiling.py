from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


VALID_LOSS_SOURCES = ("combined", "retain", "forget")
VALID_PARAM_FILTERS = ("all", "weights_only")


def select_scored_param_names(unwrapped_model: nn.Module, param_filter: str) -> list[str]:
    """Return trainable parameter names, optionally restricted to those containing 'weight'."""
    names: list[str] = []
    for name, param in unwrapped_model.named_parameters():
        if not param.requires_grad:
            continue
        if param_filter == "weights_only" and "weight" not in name:
            continue
        names.append(name)
    return names


def select_loss(
    trainer: "TrainerUnlearn",
    unlearn_method: str,
    retain_loss: torch.Tensor,
    forget_loss: torch.Tensor,
    loss_source: str,
) -> torch.Tensor:
    """Pick the scalar loss driving Fisher backprop for the current unlearning method."""
    if loss_source == "retain":
        return retain_loss
    if loss_source == "forget":
        return forget_loss
    if unlearn_method == "TGU":
        lam = trainer.unlearn_params["lambda"]
        return lam * retain_loss + (1 - lam) * forget_loss
    if unlearn_method == "SGU":
        return retain_loss + forget_loss
    raise ValueError(f"Unknown unlearning method for FIM profiling: {unlearn_method}")


def compute_fisher(
    trainer: "TrainerUnlearn",
    unwrapped_model: nn.Module,
    scored_names: list[str],
    *,
    profile_steps: int,
    loss_source: str,
    log_prefix: str,
) -> dict[str, torch.Tensor] | None:
    """Accumulate the diagonal empirical Fisher (mean of squared grads) over `profile_steps` batches.

    Returns a mapping from parameter name (restricted to `scored_names`) to a float64 tensor of
    the same shape holding the per-element averaged squared gradient. Returns None if the
    trainer lacks pretrain context or if no effective steps were collected.
    """
    unlearn_method = getattr(trainer, "_pretrain_unlearn_method", None)
    train_dataset = getattr(trainer, "_pretrain_dataset", None)
    if unlearn_method is None or train_dataset is None:
        print(f"[{log_prefix}] trainer is missing pretrain context; skipping.")
        return None

    num_workers = getattr(trainer, "_pretrain_num_workers", 0)
    resumable_with_seed = getattr(trainer, "_pretrain_resumable_with_seed", None)

    print(
        f"[{log_prefix}] profiling diagonal empirical Fisher for {profile_steps} steps "
        f"(method={unlearn_method}, loss_source={loss_source}, tracked_tensors={len(scored_names)})"
    )

    pre_dataloader = trainer.create_dataloader(
        train_dataset, num_workers=num_workers, resumable_with_seed=resumable_with_seed
    )
    pre_dataloader = trainer.accelerator.prepare(pre_dataloader)

    was_training = trainer.model.training
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    scored_set = set(scored_names)
    fisher_sums: dict[str, torch.Tensor] = {}
    effective_steps = 0

    try:
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        pre_iter = iter(pre_dataloader)

        for _ in range(profile_steps):
            try:
                batch = next(pre_iter)
            except StopIteration:
                break

            retain_loss, forget_loss = trainer._compute_pre_grad_losses(batch, unlearn_method)
            loss = select_loss(trainer, unlearn_method, retain_loss, forget_loss, loss_source)
            if not loss.requires_grad:
                continue

            trainer.accelerator.backward(loss)

            has_any_grad = False
            for name, param in unwrapped_model.named_parameters():
                if name not in scored_set or param.grad is None:
                    continue
                grad_sq = (param.grad.detach() ** 2).to(dtype=torch.float64)
                if name not in fisher_sums:
                    fisher_sums[name] = torch.zeros_like(grad_sq)
                fisher_sums[name] += grad_sq
                has_any_grad = True

            if has_any_grad:
                effective_steps += 1

            trainer.optimizer.zero_grad(set_to_none=True)
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            trainer.model.train()
        else:
            trainer.model.eval()

    if effective_steps == 0 or not fisher_sums:
        return None

    return {name: s / effective_steps for name, s in fisher_sums.items()}
