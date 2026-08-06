from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from f5_tts.model.finetune_strategies.base import FineTuningStrategy

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


class DiffFitStrategy(FineTuningStrategy):
    """Freezes the DiT to a small set of bias/norm/embed parameters (DiffFit-style parameter-efficient tuning).

    Six variants control what stays trainable on top of the always-trainable AdaLN `gamma_*` scales
    and non-embedding `.bias` parameters:
    - `v1`: full `norm.*` (weight + bias) and the full `text_embed` module.
    - `v2`: `norm.bias` only (norm weights frozen) and the full `text_embed` module.
    - `v3`: `norm.bias` only and only `text_embed.*.bias`.
    - `v4`: like `v1`, plus the full `input_embed` module.
    - `v5`: like `v2`, plus the full `input_embed` module.
    - `v6`: like `v3`, plus only `input_embed.*.bias`.

    Freezes every other parameter. All `time_embed` parameters remain frozen in every variant.
    """

    name = "DiffFit"

    def __init__(self, config: dict):
        self.config = config
        version = config.get("version")
        if version not in ("v1", "v2", "v3", "v4", "v5", "v6"):
            raise ValueError(f"Unknown DiffFit version: {version}")
        self.version = version

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        version = self.version

        #### V1: norm.weight is trainable, norm.bias is trainable
        if version == "v1":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name]
            )

        #### V2: norm.weight is frozen, norm.bias is trainable
        elif version == "v2":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name and ".bias" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name]
            )

        ### V3: like V2 but only bias from text embed
        elif version == "v3":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name and ".bias" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name and ".bias" in name]
            )

        ### V4: like V1 but with input_embed also trainable
        elif version == "v4":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "input_embed" in name]
            )

        ### V5: Like V2 but with input_embed also trainable
        elif version == "v5":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name and ".bias" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "input_embed" in name]
            )

        ### V6: like V3 but input_embed.bias also trainable
        elif version == "v6":
            trainable_names = (
                [name for name, _ in unwrapped_model.named_parameters() if "gamma_" in name]
                + [
                    name
                    for name, _ in unwrapped_model.named_parameters()
                    if ".bias" in name
                    and "input_embed" not in name
                    and "time_embed" not in name
                    and "text_embed" not in name
                ]
                + [name for name, _ in unwrapped_model.named_parameters() if "norm" in name and ".bias" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "text_embed" in name and ".bias" in name]
                + [name for name, _ in unwrapped_model.named_parameters() if "input_embed" in name and ".bias" in name]
            )
        else:
            raise ValueError(f"Unknown DiffFit version: {version}")

        print("Trainable parameters for DiffFit:")
        trainable_names = sorted(list(set(trainable_names)))
        for name in trainable_names:
            print(f"  - {name}")

        for _, p in unwrapped_model.named_parameters():
            p.requires_grad = False

        for n, p in unwrapped_model.named_parameters():
            if n in trainable_names:
                p.requires_grad = True
