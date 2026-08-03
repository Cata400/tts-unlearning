from __future__ import annotations

import torch.nn as nn

from f5_tts.model.finetune_strategies.base import FineTuningStrategy


class DitBlocksMlpStrategy(FineTuningStrategy):
    name = "DitBlocksMlp"

    def __init__(self, config: dict):
        self.config = config
        version = config.get("version")
        if version not in ("v1", "v2"):
            raise ValueError(f"Unknown DIT blocks MLP version: {version}")
        self.version = version
        self.blocks = config.get("blocks", [])

    def apply(self, unwrapped_model: nn.Module) -> None:
        blocks = self.blocks
        trainable_names = [
            name for name, _ in unwrapped_model.named_parameters() for i in blocks if f"transformer_blocks.{i}." in name
        ]

        #### V1: only FFN and attn out projection are trainable in the specified blocks
        if self.version == "v1":
            trainable_names = [
                name for name in trainable_names if any(keyword in name for keyword in ["ff", "attn.to_out"])
            ]
        #### V2: only FFN and all attn projections are trainable in the specified blocks
        elif self.version == "v2":
            trainable_names = [
                name
                for name in trainable_names
                if any(keyword in name for keyword in ["ff", "attn.to_out", "attn.to_k", "attn.to_q", "attn.to_v"])
            ]
        else:
            raise ValueError(f"Unknown DIT blocks MLP version: {self.version}")

        print("Trainable parameters for DIT blocks MLP:")
        trainable_names = sorted(list(set(trainable_names)))
        for name in trainable_names:
            print(f"  - {name}")

        for _, p in unwrapped_model.named_parameters():
            p.requires_grad = False

        for n, p in unwrapped_model.named_parameters():
            if n in trainable_names:
                p.requires_grad = True
