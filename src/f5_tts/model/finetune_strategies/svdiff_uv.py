from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torchjd import autojac
from torchjd.aggregation import UPGrad
from torchjd.autojac import jac_to_grad

import wandb
from f5_tts.model.finetune_strategies.base import FineTuningStrategy
from f5_tts.model.finetune_strategies.svdiff import register_svd_parametrization
from f5_tts.model.modules import SVDParametrizationU, SVDParametrizationV

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from f5_tts.model.trainer_unlearn import TrainerUnlearn


def _pre_grad_namespace(stream: str) -> str:
    if stream == "combined":
        return "svdiff_uv_pre_grad"
    if stream == "retain":
        return "svdiff_uv_pre_grad_retain"
    if stream == "forget":
        return "svdiff_uv_pre_grad_forget"
    raise ValueError(f"Unknown svdiff_uv pre-grad stream: {stream!r}")


def _pre_grad_metric_prefix(index: int, variant: str, stream: str = "combined") -> str:
    return f"{_pre_grad_namespace(stream)}/delta_{variant.lower()}_{index:03d}"


def _delta_param_name(variant: str) -> str:
    return "delta_U" if variant == "u" else "delta_V"


def _get_singular_values_by_param_name(unwrapped_model: nn.Module, variant: str) -> dict[str, torch.Tensor]:
    delta_attr = _delta_param_name(variant)
    singular_values_by_param: dict[str, torch.Tensor] = {}
    for module_name, module in unwrapped_model.named_modules():
        if not hasattr(module, "parametrizations") or "weight" not in module.parametrizations:
            continue

        parametrizations = module.parametrizations["weight"]
        if len(parametrizations) == 0:
            continue
        parametrization = parametrizations[0]
        if not (hasattr(parametrization, delta_attr) and hasattr(parametrization, "S")):
            continue

        param_name = (
            f"{module_name}.parametrizations.weight.0.{delta_attr}"
            if module_name
            else f"parametrizations.weight.0.{delta_attr}"
        )
        singular_values_by_param[param_name] = parametrization.S.detach().to(device="cpu", dtype=torch.float64)

    return singular_values_by_param


def _accumulate_delta_grads_from_loss(
    loss: torch.Tensor,
    delta_named_params: list[tuple[str, nn.Parameter]],
    sums_dict: dict[str, torch.Tensor],
) -> bool:
    """Grad `loss` wrt each delta param; accumulate per-column mean |grad|. Retains graph."""
    if not loss.requires_grad or not delta_named_params:
        return False
    params = [p for _, p in delta_named_params]
    try:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    except RuntimeError:
        return False

    has_any = False
    for (name, _), grad in zip(delta_named_params, grads):
        if grad is None:
            continue
        grad = grad.detach()
        if grad.ndim >= 2:
            reduce_dims = tuple(range(grad.ndim - 1))
            mean_abs_grad = grad.abs().mean(dim=reduce_dims)
        else:
            mean_abs_grad = grad.abs()
        mean_abs_grad = mean_abs_grad.to(device="cpu", dtype=torch.float64)
        if name not in sums_dict:
            sums_dict[name] = torch.zeros_like(mean_abs_grad)
        sums_dict[name] += mean_abs_grad
        has_any = True
    return has_any


class SVDiffUVStrategy(FineTuningStrategy):
    """Reparametrizes every currently-trainable `.weight` as an SVD with an additive delta on `U` or `Vh`.

    `U`, `S`, `Vh` are computed once from the initial weight and frozen as buffers; only the
    per-tensor delta is trained:
    - `type="u"`: weight becomes `(U + delta_U) @ diag(S) @ Vh`.
    - `type="v"`: weight becomes `U @ diag(S) @ (Vh + delta_V.T)`.

    Requires an optimizer reset after `apply` because the trainable parameter set changes.
    Composes with upstream freezers (e.g. `DitBlocksMlp`, `FIM` at `granularity="layer"`), which
    determine which weight tensors the parametrization is registered on.

    Optionally runs a pre-training gradient profiling pass over `delta_U` / `delta_V` to log
    per-column mean |grad| and, when `pre_grad_top_k` is set, install a column-wise grad mask
    that keeps only the top-k columns per delta trainable during the actual training run.
    """

    name = "SVDiff-UV"

    def __init__(self, config: dict):
        self.config = config
        variant = str(config.get("type", "u")).lower()
        if variant not in ("u", "v"):
            raise ValueError(f"svdiff_uv.type must be 'u' or 'v', got: {variant!r}")
        self.variant = variant
        self.name = f"SVDiff-{variant.upper()}"

    @property
    def requires_optimizer_reset(self) -> bool:
        return True

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        if self.variant == "u":
            register_svd_parametrization(unwrapped_model, SVDParametrizationU, "delta_U", "SVDiff-U")
        elif self.variant == "v":
            register_svd_parametrization(unwrapped_model, SVDParametrizationV, "delta_V", "SVDiff-V")
        else:
            raise ValueError(f"Invalid svdiff_uv variant: {self.variant!r} (expected 'u' or 'v').")

    def register_wandb_metrics(self, logger: str | None) -> None:
        if logger != "wandb":
            return
        wandb.define_metric("svdiff_uv_pre_grad_step")
        wandb.define_metric("svdiff_uv_pre_grad/*", step_metric="svdiff_uv_pre_grad_step")
        wandb.define_metric("svdiff_uv_pre_grad_retain/*", step_metric="svdiff_uv_pre_grad_step")
        wandb.define_metric("svdiff_uv_pre_grad_forget/*", step_metric="svdiff_uv_pre_grad_step")

    def run_pre_training_hook(
        self,
        trainer: TrainerUnlearn,
        *,
        unlearn_method: str,
        train_dataset: Dataset,
        num_workers: int,
        resumable_with_seed: int | None,
    ) -> None:
        grad_means = self._run_pre_grad_logging(
            trainer,
            unlearn_method=unlearn_method,
            train_dataset=train_dataset,
            num_workers=num_workers,
            resumable_with_seed=resumable_with_seed,
        )

        pre_grad_top_k = self.config.get("pre_grad_top_k", None)
        if grad_means is not None and pre_grad_top_k is not None:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            delta_param_name = _delta_param_name(self.variant)
            effective_delta_uv_elements = self._apply_top_k_column_mask(unwrapped, grad_means, int(pre_grad_top_k))
            # Effective trainable = non-delta trainable + effective delta elements
            non_delta_uv_trainable = sum(
                p.numel() for n, p in unwrapped.named_parameters() if p.requires_grad and delta_param_name not in n
            )
            effective_trainable = non_delta_uv_trainable + effective_delta_uv_elements
            print(f"Effective trainable parameters (after top-k masking): {effective_trainable / 1e6:.3f}M")

    def _run_pre_grad_logging(
        self,
        trainer: TrainerUnlearn,
        *,
        unlearn_method: str,
        train_dataset: Dataset,
        num_workers: int,
        resumable_with_seed: int | None,
    ) -> dict[str, torch.Tensor] | None:
        pre_grad_enabled = self.config.get("pre_grad_enabled", False)
        pre_grad_steps = int(self.config.get("pre_grad_steps", 0))
        if not pre_grad_enabled or pre_grad_steps <= 0:
            return None

        variant = self.variant
        delta_param_name = _delta_param_name(variant)
        label = f"SVDiff-{variant.upper()}"
        print(f"Running {label} pre-grad logging for {pre_grad_steps} steps (method={unlearn_method})")

        log_separately = self.config.get("pre_grad_log_forget_retain_separately", False)
        use_torchjd = self.config.get("use_torchjd", False)
        if log_separately and use_torchjd:
            print(
                f"{label} pre-grad per-loss logging is not supported when svdiff_uv.use_torchjd is True; "
                "ignoring pre_grad_log_forget_retain_separately."
            )
            log_separately = False

        pre_dataloader = trainer.create_dataloader(
            train_dataset, num_workers=num_workers, resumable_with_seed=resumable_with_seed
        )
        pre_dataloader = trainer.accelerator.prepare(pre_dataloader)

        was_training = trainer.model.training
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        grad_abs_sums: dict[str, torch.Tensor] = {}
        grad_abs_sums_retain: dict[str, torch.Tensor] = {}
        grad_abs_sums_forget: dict[str, torch.Tensor] = {}
        effective_steps = 0
        effective_steps_retain = 0
        effective_steps_forget = 0

        unwrapped = trainer.accelerator.unwrap_model(trainer.model)

        delta_named_params: list[tuple[str, nn.Parameter]] = []
        if log_separately:
            delta_named_params = [
                (n, p) for n, p in unwrapped.named_parameters() if delta_param_name in n and p.requires_grad
            ]
            if not delta_named_params:
                print(f"{label} pre-grad per-loss logging disabled: no trainable {delta_param_name} parameters found.")
                log_separately = False

        aggregator = UPGrad() if use_torchjd else None
        try:
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            pre_iter = iter(pre_dataloader)

            for _ in range(pre_grad_steps):
                try:
                    batch = next(pre_iter)
                except StopIteration:
                    break

                retain_loss, forget_loss = trainer._compute_pre_grad_losses(batch, unlearn_method)

                if unlearn_method == "TGU":
                    loss = (
                        trainer.unlearn_params["lambda"] * retain_loss
                        + (1 - trainer.unlearn_params["lambda"]) * forget_loss
                    )
                elif unlearn_method == "SGU":
                    if not use_torchjd:
                        loss = retain_loss + forget_loss
                else:
                    raise ValueError(f"Unknown unlearning method for pre-grad logging: {unlearn_method}")

                if log_separately:
                    if _accumulate_delta_grads_from_loss(retain_loss, delta_named_params, grad_abs_sums_retain):
                        effective_steps_retain += 1
                    if _accumulate_delta_grads_from_loss(forget_loss, delta_named_params, grad_abs_sums_forget):
                        effective_steps_forget += 1

                if not use_torchjd:
                    trainer.accelerator.backward(loss)
                else:
                    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
                    autojac.backward([retain_loss, forget_loss], inputs=trainable_params)
                    jac_to_grad(trainable_params, aggregator)

                has_delta_grad = False
                for name, param in unwrapped.named_parameters():
                    if delta_param_name not in name or param.grad is None:
                        continue

                    grad = param.grad.detach()
                    if grad.ndim >= 2:
                        reduce_dims = tuple(range(grad.ndim - 1))
                        mean_abs_grad = grad.abs().mean(dim=reduce_dims)
                    else:
                        mean_abs_grad = grad.abs()

                    mean_abs_grad = mean_abs_grad.to(device="cpu", dtype=torch.float64)
                    if name not in grad_abs_sums:
                        grad_abs_sums[name] = torch.zeros_like(mean_abs_grad)
                    grad_abs_sums[name] += mean_abs_grad
                    has_delta_grad = True

                if has_delta_grad:
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
            pre_iter = None
            trainer.release_dataloader(pre_dataloader)

        if effective_steps == 0:
            print(f"{label} pre-grad logging skipped: no {delta_param_name} gradients were collected.")
            return None

        if trainer.accelerator.is_local_main_process:
            metrics: dict = {
                "svdiff_uv_pre_grad_step": 1,
                "svdiff_uv_pre_grad/num_effective_steps": float(effective_steps),
                "svdiff_uv_pre_grad/variant": variant,
            }
            singular_values_by_param = _get_singular_values_by_param_name(unwrapped, variant)
            sorted_grad_items = sorted(grad_abs_sums.items(), key=lambda x: x[0])
            for index, (name, grad_abs_sum) in enumerate(sorted_grad_items):
                mean_grad = grad_abs_sum / effective_steps
                metric_prefix = _pre_grad_metric_prefix(index, variant)
                metrics[f"{metric_prefix}/mean"] = mean_grad.mean().item()
                if trainer.logger == "wandb":
                    x_vals = list(range(mean_grad.numel()))
                    y_vals = mean_grad.cpu().tolist()
                    metrics[f"{metric_prefix}/per_column_curve"] = wandb.plot.line_series(
                        xs=x_vals,
                        ys=[y_vals],
                        keys=["avg_grad_magnitude"],
                        title=f"{name} per-column average |grad|",
                        xname="column_index",
                    )

                    singular_values = singular_values_by_param.get(name)
                    if singular_values is not None:
                        n_cols = min(mean_grad.numel(), singular_values.numel())
                        normalized_grad = mean_grad[:n_cols] / singular_values[:n_cols].abs().clamp_min(1e-12)
                        metrics[f"{metric_prefix}/mean_normalized_by_s"] = normalized_grad.mean().item()
                        metrics[f"{metric_prefix}/per_column_curve_normalized_by_s"] = wandb.plot.line_series(
                            xs=list(range(n_cols)),
                            ys=[normalized_grad.cpu().tolist()],
                            keys=["avg_grad_over_s"],
                            title=f"{name} per-column average |grad| / (S + eps)",
                            xname="column_index",
                        )
                print(f"{label} metric mapping: delta_{variant}_{index:03d} -> {name}")

            trainer.accelerator.log(metrics, step=0)
            print(
                f"Logged {label} per-column mean |grad| as one graph per {delta_param_name} tensor "
                f"for {len(grad_abs_sums)} {delta_param_name} tensors "
                f"over {effective_steps} steps."
            )

            if log_separately:
                stream_metrics: dict = {"svdiff_uv_pre_grad_step": 1}
                self._emit_pre_grad_stream_metrics(
                    trainer,
                    stream_metrics,
                    grad_abs_sums_retain,
                    effective_steps_retain,
                    stream="retain",
                    singular_values_by_param=singular_values_by_param,
                    label=label,
                )
                self._emit_pre_grad_stream_metrics(
                    trainer,
                    stream_metrics,
                    grad_abs_sums_forget,
                    effective_steps_forget,
                    stream="forget",
                    singular_values_by_param=singular_values_by_param,
                    label=label,
                )
                if len(stream_metrics) > 1:
                    trainer.accelerator.log(stream_metrics, step=0)

        grad_means: dict[str, torch.Tensor] = {}
        for name, grad_abs_sum in grad_abs_sums.items():
            grad_means[name] = grad_abs_sum / effective_steps
        return grad_means

    def _emit_pre_grad_stream_metrics(
        self,
        trainer: TrainerUnlearn,
        metrics: dict,
        sums_dict: dict[str, torch.Tensor],
        steps: int,
        stream: str,
        singular_values_by_param: dict[str, torch.Tensor],
        label: str,
    ) -> None:
        """Populate `metrics` with per-column mean |grad| curves for a single stream."""
        if steps <= 0 or not sums_dict:
            return
        variant = self.variant
        namespace = _pre_grad_namespace(stream)
        metrics[f"{namespace}/num_effective_steps"] = float(steps)
        metrics[f"{namespace}/variant"] = variant
        sorted_items = sorted(sums_dict.items(), key=lambda x: x[0])
        for index, (name, grad_abs_sum) in enumerate(sorted_items):
            mean_grad = grad_abs_sum / steps
            metric_prefix = _pre_grad_metric_prefix(index, variant, stream)
            metrics[f"{metric_prefix}/mean"] = mean_grad.mean().item()
            if trainer.logger == "wandb":
                metrics[f"{metric_prefix}/per_column_curve"] = wandb.plot.line_series(
                    xs=list(range(mean_grad.numel())),
                    ys=[mean_grad.cpu().tolist()],
                    keys=["avg_grad_magnitude"],
                    title=f"{name} [{stream}] per-column average |grad|",
                    xname="column_index",
                )
                singular_values = singular_values_by_param.get(name)
                if singular_values is not None:
                    n_cols = min(mean_grad.numel(), singular_values.numel())
                    normalized_grad = mean_grad[:n_cols] / singular_values[:n_cols].abs().clamp_min(1e-12)
                    metrics[f"{metric_prefix}/mean_normalized_by_s"] = normalized_grad.mean().item()
                    metrics[f"{metric_prefix}/per_column_curve_normalized_by_s"] = wandb.plot.line_series(
                        xs=list(range(n_cols)),
                        ys=[normalized_grad.cpu().tolist()],
                        keys=["avg_grad_over_s"],
                        title=f"{name} [{stream}] per-column average |grad| / (S + eps)",
                        xname="column_index",
                    )
        print(
            f"{label} logged {stream} per-column mean |grad| for {len(sums_dict)} "
            f"{_delta_param_name(variant)} tensors over {steps} steps."
        )

    def _apply_top_k_column_mask(
        self,
        unwrapped_model: nn.Module,
        grad_means: dict[str, torch.Tensor],
        top_k: int,
    ) -> int:
        """Register gradient hooks on delta params to zero non-top-k columns; return effective element count."""
        variant = self.variant
        delta_param_name = _delta_param_name(variant)
        label = f"SVDiff-{variant.upper()}"
        print(f"Applying {label} top-k column mask: keeping {top_k} columns per {delta_param_name}")

        effective_delta_elements = 0
        total_delta_elements = 0

        for name, param in unwrapped_model.named_parameters():
            if delta_param_name not in name or name not in grad_means:
                continue

            mean_grad = grad_means[name]
            num_columns = mean_grad.numel()
            k = min(top_k, num_columns)

            _, top_indices = torch.topk(mean_grad, k=k)
            mask = torch.zeros(num_columns, device=param.device, dtype=param.dtype)
            mask[top_indices] = 1.0

            if param.ndim == 2:
                column_mask = mask.unsqueeze(0)
                rows = param.shape[0]
            else:
                column_mask = mask
                rows = 1

            param.register_hook(lambda grad, m=column_mask: grad * m)

            effective_delta_elements += rows * k
            total_delta_elements += param.numel()

            print(
                f"  {name}: kept {k}/{num_columns} columns "
                f"(indices: {sorted(top_indices.cpu().tolist())[:10]}{'...' if k > 10 else ''})"
            )

        print(
            f"  Effective {delta_param_name} elements: {effective_delta_elements:,} / {total_delta_elements:,} "
            f"({100 * effective_delta_elements / max(total_delta_elements, 1):.1f}%)"
        )
        return effective_delta_elements
