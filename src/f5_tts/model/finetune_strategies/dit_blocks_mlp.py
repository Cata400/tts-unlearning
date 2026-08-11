from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch.nn as nn

from f5_tts.model.finetune_strategies.base import FineTuningStrategy

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


VALID_DIT_BLOCKS_MLP_VERSIONS = ("v1", "v2")

# v1 = FFN + attn output projection; v2 = FFN + all attn projections.
_VERSION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "v1": ("ff", "attn.to_out"),
    "v2": ("ff", "attn.to_out", "attn.to_k", "attn.to_q", "attn.to_v"),
}


def select_dit_block_mlp_param_names(
    unwrapped_model: nn.Module,
    blocks: Iterable[int],
    version: str,
    block_module_prefix: str = "transformer_blocks",
) -> list[str]:
    """Return the sorted list of parameter names inside `blocks` matching the v1/v2 keyword filter."""
    if version not in _VERSION_KEYWORDS:
        raise ValueError(f"Unknown DIT blocks MLP version: {version}")
    keywords = _VERSION_KEYWORDS[version]
    block_prefixes = tuple(f"{block_module_prefix}.{i}." for i in blocks)
    trainable_names: set[str] = set()
    for name, _ in unwrapped_model.named_parameters():
        if not any(prefix in name for prefix in block_prefixes):
            continue
        if any(keyword in name for keyword in keywords):
            trainable_names.add(name)
    return sorted(trainable_names)


class DitBlocksMlpStrategy(FineTuningStrategy):
    """Restricts training to selected sub-modules inside a chosen subset of DiT transformer blocks.

    Only parameters under `transformer_blocks.{i}.` for each `i` in `config["blocks"]` are considered;
    everything else is frozen. Two variants further narrow the selection within those blocks:
    - `v1`: FFN (`ff`) and attention output projection (`attn.to_out`).
    - `v2`: FFN plus all attention projections (`attn.to_out`, `attn.to_k`, `attn.to_q`, `attn.to_v`).

    Commonly composed with `SVDiff` / `SVDiff-UV`, in which case the SVD parametrization is only
    registered on the tensors this strategy leaves trainable.
    """

    name = "DitBlocksMlp"

    def __init__(self, config: dict):
        self.config = config
        version = config.get("version")
        if version not in VALID_DIT_BLOCKS_MLP_VERSIONS:
            raise ValueError(f"Unknown DIT blocks MLP version: {version}")
        self.version = version
        self.blocks = config.get("blocks", [])

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        trainable_names = select_dit_block_mlp_param_names(unwrapped_model, self.blocks, self.version)

        print("Trainable parameters for DIT blocks MLP:")
        for name in trainable_names:
            print(f"  - {name}")

        trainable_set = set(trainable_names)
        for n, p in unwrapped_model.named_parameters():
            p.requires_grad = n in trainable_set
