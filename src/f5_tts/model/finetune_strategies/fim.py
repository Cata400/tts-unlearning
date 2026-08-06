from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

import wandb
from f5_tts.model.finetune_strategies.base import FineTuningStrategy

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


_VALID_GRANULARITIES = ("layer", "weight")
_VALID_TOP_K_MODES = ("per_tensor", "global")
_VALID_LAYER_SCORES = ("mean", "sum")
_VALID_LAYER_UNITS = ("module", "parameter")
_VALID_LOSS_SOURCES = ("combined", "retain", "forget")
_VALID_PARAM_FILTERS = ("all", "weights_only")


class FIMStrategy(FineTuningStrategy):
    """Selects trainable parameters by diagonal empirical Fisher (mean of squared grads).

    Two granularities, mirroring the top-k column selection of `SVDiff-UV`:
    - `layer`: keep top-k parameter tensors trainable, freeze the rest.
    - `weight`: keep top-k elements per tensor (or globally) via a grad-mask hook.

    Composes with upstream freezers (only params currently `requires_grad=True` are scored)
    and with `svdiff` / `svdiff_uv` at `granularity="layer"` (the SVD parametrization is then
    only registered on FIM-selected weight tensors).
    """

    name = "FIM"

    def __init__(self, config: dict):
        self.config = config

        granularity = str(config.get("granularity", "layer")).lower()
        if granularity not in _VALID_GRANULARITIES:
            raise ValueError(f"fim.granularity must be one of {_VALID_GRANULARITIES}, got {granularity!r}")
        self.granularity = granularity

        top_k = config.get("top_k")
        if top_k is None or int(top_k) <= 0:
            raise ValueError(f"fim.top_k must be a positive integer, got {top_k!r}")
        self.top_k = int(top_k)

        top_k_mode = str(config.get("top_k_mode", "per_tensor")).lower()
        if top_k_mode not in _VALID_TOP_K_MODES:
            raise ValueError(f"fim.top_k_mode must be one of {_VALID_TOP_K_MODES}, got {top_k_mode!r}")
        self.top_k_mode = top_k_mode

        layer_score = str(config.get("layer_score", "mean")).lower()
        if layer_score not in _VALID_LAYER_SCORES:
            raise ValueError(f"fim.layer_score must be one of {_VALID_LAYER_SCORES}, got {layer_score!r}")
        self.layer_score = layer_score

        layer_unit = str(config.get("layer_unit", "module")).lower()
        if layer_unit not in _VALID_LAYER_UNITS:
            raise ValueError(f"fim.layer_unit must be one of {_VALID_LAYER_UNITS}, got {layer_unit!r}")
        self.layer_unit = layer_unit

        profile_steps = int(config.get("profile_steps", 0))
        if profile_steps <= 0:
            raise ValueError(f"fim.profile_steps must be a positive integer, got {profile_steps!r}")
        self.profile_steps = profile_steps

        loss_source = str(config.get("loss_source", "combined")).lower()
        if loss_source not in _VALID_LOSS_SOURCES:
            raise ValueError(f"fim.loss_source must be one of {_VALID_LOSS_SOURCES}, got {loss_source!r}")
        self.loss_source = loss_source

        param_filter = str(config.get("param_filter", "all")).lower()
        if param_filter not in _VALID_PARAM_FILTERS:
            raise ValueError(f"fim.param_filter must be one of {_VALID_PARAM_FILTERS}, got {param_filter!r}")
        self.param_filter = param_filter

        self.log_wandb = bool(config.get("log_wandb", False))

        self.name = f"FIM-{self.granularity}"

    def register_wandb_metrics(self, logger: str | None) -> None:
        if logger != "wandb" or not self.log_wandb:
            return
        wandb.define_metric("fim_step")
        wandb.define_metric("fim/*", step_metric="fim_step")

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        if trainer is None:
            print(f"[{self.name}] no trainer context provided; skipping FIM apply().")
            return

        scored_names = self._scored_param_names(unwrapped_model)
        if not scored_names:
            print(f"[{self.name}] no trainable parameters match the current filter; skipping.")
            return

        fisher = self._compute_fisher(trainer, unwrapped_model, scored_names)
        if fisher is None:
            print(f"[{self.name}] Fisher profiling collected zero effective steps; skipping.")
            return

        if self.granularity == "layer":
            selected = self._apply_layer_selection(unwrapped_model, fisher)
        else:
            selected = self._apply_weight_selection(unwrapped_model, fisher)

        num_trainable_elems = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        effective_elems = selected["effective_elements"]
        unit_label = selected["unit_label"]
        print(
            f"[{self.name}] trainable {unit_label}: {selected['num_selected_units']} "
            f"(elements kept by FIM: {effective_elems:,}; total requires_grad elems now: {num_trainable_elems:,} "
            f"= {num_trainable_elems / 1e6:.3f}M)"
        )

        if self.log_wandb and trainer.logger == "wandb" and trainer.accelerator.is_local_main_process:
            trainer.accelerator.log(
                {
                    "fim_step": 1,
                    "fim/granularity": self.granularity,
                    "fim/top_k": float(self.top_k),
                    "fim/top_k_mode": self.top_k_mode if self.granularity == "weight" else "n/a",
                    "fim/layer_score": self.layer_score if self.granularity == "layer" else "n/a",
                    "fim/layer_unit": self.layer_unit if self.granularity == "layer" else "n/a",
                    "fim/loss_source": self.loss_source,
                    "fim/param_filter": self.param_filter,
                    "fim/profile_steps": float(self.profile_steps),
                    "fim/num_selected_units": float(selected["num_selected_units"]),
                    "fim/effective_elements": float(effective_elems),
                },
                step=0,
            )

    def _scored_param_names(self, unwrapped_model: nn.Module) -> list[str]:
        names: list[str] = []
        for name, param in unwrapped_model.named_parameters():
            if not param.requires_grad:
                continue
            if self.param_filter == "weights_only" and "weight" not in name:
                continue
            names.append(name)
        return names

    def _compute_fisher(
        self,
        trainer: "TrainerUnlearn",
        unwrapped_model: nn.Module,
        scored_names: list[str],
    ) -> dict[str, torch.Tensor] | None:
        unlearn_method = getattr(trainer, "_pretrain_unlearn_method", None)
        train_dataset = getattr(trainer, "_pretrain_dataset", None)
        if unlearn_method is None or train_dataset is None:
            print(f"[{self.name}] trainer is missing pretrain context; skipping.")
            return None

        num_workers = getattr(trainer, "_pretrain_num_workers", 0)
        resumable_with_seed = getattr(trainer, "_pretrain_resumable_with_seed", None)

        print(
            f"[{self.name}] profiling diagonal empirical Fisher for {self.profile_steps} steps "
            f"(method={unlearn_method}, loss_source={self.loss_source}, tracked_tensors={len(scored_names)})"
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

            for _ in range(self.profile_steps):
                try:
                    batch = next(pre_iter)
                except StopIteration:
                    break

                retain_loss, forget_loss = trainer._compute_pre_grad_losses(batch, unlearn_method)
                loss = self._select_loss(trainer, unlearn_method, retain_loss, forget_loss)
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

    def _select_loss(
        self,
        trainer: "TrainerUnlearn",
        unlearn_method: str,
        retain_loss: torch.Tensor,
        forget_loss: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_source == "retain":
            return retain_loss
        if self.loss_source == "forget":
            return forget_loss
        if unlearn_method == "TGU":
            lam = trainer.unlearn_params["lambda"]
            return lam * retain_loss + (1 - lam) * forget_loss
        if unlearn_method == "SGU":
            return retain_loss + forget_loss
        raise ValueError(f"Unknown unlearning method for FIM profiling: {unlearn_method}")

    def _apply_layer_selection(
        self,
        unwrapped_model: nn.Module,
        fisher: dict[str, torch.Tensor],
    ) -> dict:
        if self.layer_unit == "module":
            return self._apply_layer_selection_by_module(unwrapped_model, fisher)
        return self._apply_layer_selection_by_parameter(unwrapped_model, fisher)

    def _apply_layer_selection_by_parameter(
        self,
        unwrapped_model: nn.Module,
        fisher: dict[str, torch.Tensor],
    ) -> dict:
        scores: list[tuple[str, float, int]] = []
        for name, tensor in fisher.items():
            score = tensor.mean().item() if self.layer_score == "mean" else tensor.sum().item()
            scores.append((name, score, tensor.numel()))
        scores.sort(key=lambda x: x[1], reverse=True)

        k = min(self.top_k, len(scores))
        selected_names = {name for name, _, _ in scores[:k]}

        for name, param in unwrapped_model.named_parameters():
            if name in fisher and name not in selected_names:
                param.requires_grad = False

        effective_elements = sum(numel for name, _, numel in scores[:k])

        print(f"[{self.name}] parameter-tensor selection (score={self.layer_score}, kept {k}/{len(scores)}):")
        for name, score, numel in scores[:k]:
            print(f"  + {name}  score={score:.3e}  numel={numel}")
        for name, score, numel in scores[k:]:
            print(f"  - {name}  score={score:.3e}  numel={numel}")

        return {
            "num_selected_units": k,
            "effective_elements": effective_elements,
            "unit_label": "parameter tensors",
        }

    def _apply_layer_selection_by_module(
        self,
        unwrapped_model: nn.Module,
        fisher: dict[str, torch.Tensor],
    ) -> dict:
        # Map each direct Parameter to the qualified name of its immediate parent module.
        param_to_module: dict[str, str] = {}
        module_to_params: dict[str, list[tuple[str, nn.Parameter]]] = {}
        for module_name, module in unwrapped_model.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{pname}" if module_name else pname
                param_to_module[full_name] = module_name
                module_to_params.setdefault(module_name, []).append((full_name, param))

        # Aggregate Fisher over each module's scored direct params.
        module_agg: dict[str, dict] = {}
        for name, fisher_tensor in fisher.items():
            mname = param_to_module.get(name)
            if mname is None:
                continue
            entry = module_agg.setdefault(mname, {"sum": 0.0, "numel": 0, "scored_params": []})
            entry["sum"] += float(fisher_tensor.sum().item())
            entry["numel"] += int(fisher_tensor.numel())
            entry["scored_params"].append(name)

        ranked: list[tuple[str, float, int, int]] = []
        for mname, agg in module_agg.items():
            score = agg["sum"] / max(agg["numel"], 1) if self.layer_score == "mean" else agg["sum"]
            ranked.append((mname, score, agg["numel"], len(agg["scored_params"])))
        ranked.sort(key=lambda x: x[1], reverse=True)

        k = min(self.top_k, len(ranked))
        selected_modules = {mname for mname, _, _, _ in ranked[:k]}

        # Freeze every direct param of non-selected modules (including biases/norms that
        # weren't scored) so a module either fine-tunes as a whole or stays frozen.
        for mname, _, _, _ in ranked[k:]:
            for _, param in module_to_params.get(mname, []):
                param.requires_grad = False

        effective_elements = 0
        for mname in selected_modules:
            for _, param in module_to_params.get(mname, []):
                if param.requires_grad:
                    effective_elements += param.numel()

        print(f"[{self.name}] module selection (score={self.layer_score}, kept {k}/{len(ranked)}):")
        for mname, score, numel, n_params in ranked[:k]:
            print(f"  + {mname or '<root>'}  score={score:.3e}  scored_numel={numel}  scored_params={n_params}")
        for mname, score, numel, n_params in ranked[k:]:
            print(f"  - {mname or '<root>'}  score={score:.3e}  scored_numel={numel}  scored_params={n_params}")

        return {
            "num_selected_units": k,
            "effective_elements": effective_elements,
            "unit_label": "modules",
        }

    def _apply_weight_selection(
        self,
        unwrapped_model: nn.Module,
        fisher: dict[str, torch.Tensor],
    ) -> dict:
        params_by_name = dict(unwrapped_model.named_parameters())

        if self.top_k_mode == "per_tensor":
            masks: dict[str, torch.Tensor] = {}
            effective = 0
            for name, score_tensor in fisher.items():
                param = params_by_name.get(name)
                if param is None:
                    continue
                numel = score_tensor.numel()
                k = min(self.top_k, numel)
                flat_scores = score_tensor.reshape(-1)
                _, top_idx = torch.topk(flat_scores, k=k)
                mask = torch.zeros(numel, dtype=param.dtype, device=param.device)
                mask[top_idx.to(param.device)] = 1.0
                masks[name] = mask.reshape(param.shape)
                effective += k
        else:
            all_scores = torch.cat([t.reshape(-1) for t in fisher.values()])
            total = all_scores.numel()
            k_global = min(self.top_k, total)
            if k_global <= 0:
                threshold = float("inf")
            else:
                threshold = torch.topk(all_scores, k=k_global).values.min().item()
            masks = {}
            effective = 0
            for name, score_tensor in fisher.items():
                param = params_by_name.get(name)
                if param is None:
                    continue
                keep = (score_tensor >= threshold).to(dtype=param.dtype, device=param.device)
                masks[name] = keep.reshape(param.shape)
                effective += int(keep.sum().item())

        num_selected_tensors = 0
        for name, mask in masks.items():
            param = params_by_name[name]
            kept = int(mask.sum().item())
            if kept == 0:
                param.requires_grad = False
                continue
            num_selected_tensors += 1
            param.register_hook(lambda grad, m=mask: grad * m)

        print(
            f"[{self.name}] weight selection (mode={self.top_k_mode}, k={self.top_k}): "
            f"kept elements={effective:,} across {num_selected_tensors} tensors."
        )
        return {
            "num_selected_units": num_selected_tensors,
            "effective_elements": effective,
            "unit_label": "parameter tensors (per-element masked)",
        }
