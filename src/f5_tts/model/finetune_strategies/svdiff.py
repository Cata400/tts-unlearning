from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from f5_tts.model.finetune_strategies.base import FineTuningStrategy
from f5_tts.model.modules import SVDParametrization

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


def register_svd_parametrization(
    unwrapped_model: nn.Module,
    parametrization_cls: type[nn.Module],
    trainable_param_name: str,
    label: str,
) -> None:
    """Register `parametrization_cls` on every trainable `.weight` and freeze everything except deltas."""
    module_params_dict = {
        module: list(module.named_parameters(recurse=False)) for _, module in unwrapped_model.named_modules()
    }
    for name, module in unwrapped_model.named_modules():
        if module not in module_params_dict:
            continue
        for param_name, param in module_params_dict[module]:
            if param.requires_grad:
                full_param_name = f"{name}.{param_name}" if name else param_name
                if "weight" in full_param_name:
                    try:
                        parametrize.register_parametrization(module, param_name, parametrization_cls(param))
                    except (ValueError, RuntimeError):
                        continue

    trainable_names = [name for name, _ in unwrapped_model.named_parameters() if trainable_param_name in name]

    print(f"Trainable parameters for {label}:")
    trainable_names = sorted(list(set(trainable_names)))
    for name in trainable_names:
        print(f"  - {name}")

    for _, p in unwrapped_model.named_parameters():
        p.requires_grad = False

    for n, p in unwrapped_model.named_parameters():
        if n in trainable_names:
            p.requires_grad = True


class SVDiffStrategy(FineTuningStrategy):
    """Reparametrizes every currently-trainable `.weight` as `U @ diag(S + delta_S) @ Vh` (SVDiff).

    `U`, `S`, `Vh` are computed once from the initial weight and frozen as buffers; only the
    per-tensor `delta_S` vectors are trained. All other parameters are frozen.

    Requires an optimizer reset after `apply` because the trainable parameter set changes.
    Composes with upstream freezers (e.g. `DitBlocksMlp`, `FIM` at `granularity="layer"`), which
    determine which weight tensors the parametrization is registered on.
    """

    name = "SVDiff"

    def __init__(self, config: dict):
        self.config = config

    @property
    def requires_optimizer_reset(self) -> bool:
        return True

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        register_svd_parametrization(unwrapped_model, SVDParametrization, "delta_S", "SVDiff")
