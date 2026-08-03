from __future__ import annotations

import torch.nn as nn

from f5_tts.model.finetune_strategies.base import FineTuningStrategy


class DiffITStrategy(FineTuningStrategy):
    name = "DiffIT"

    def __init__(self, config: dict):
        self.config = config
        version = config.get("version")
        if version not in ("v1", "v2", "v3", "v4", "v5", "v6"):
            raise ValueError(f"Unknown DiffIT version: {version}")
        self.version = version

    def apply(self, unwrapped_model: nn.Module) -> None:
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
            raise ValueError(f"Unknown DiffIT version: {version}")

        print("Trainable parameters for DiffIT:")
        trainable_names = sorted(list(set(trainable_names)))
        for name in trainable_names:
            print(f"  - {name}")

        for _, p in unwrapped_model.named_parameters():
            p.requires_grad = False

        for n, p in unwrapped_model.named_parameters():
            if n in trainable_names:
                p.requires_grad = True
