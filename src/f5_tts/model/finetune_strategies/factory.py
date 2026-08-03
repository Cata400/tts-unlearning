from __future__ import annotations

from f5_tts.model.finetune_strategies.base import FineTuningStrategy
from f5_tts.model.finetune_strategies.composite import CompositeStrategy
from f5_tts.model.finetune_strategies.diffit import DiffITStrategy
from f5_tts.model.finetune_strategies.dit_blocks_mlp import DitBlocksMlpStrategy
from f5_tts.model.finetune_strategies.noop import NoOpStrategy
from f5_tts.model.finetune_strategies.svdiff import SVDiffStrategy
from f5_tts.model.finetune_strategies.svdiff_uv import SVDiffUVStrategy


def build_finetune_strategy(model_cfg_dict: dict) -> FineTuningStrategy:
    """Build the fine-tuning strategy declared by `model.finetune` in the config.

    At most one parameter-freezing strategy (diffit XOR dit_blocks_mlp) and at most one SVD
    strategy (svdiff XOR svdiff_uv) may be enabled; both categories may coexist and are then
    wrapped in a `CompositeStrategy`.
    """
    finetune_cfg = model_cfg_dict.get("model", {}).get("finetune", {})

    strategies: list[FineTuningStrategy] = []

    diffit_cfg = finetune_cfg.get("diffit", {})
    dit_blocks_mlp_cfg = finetune_cfg.get("dit_blocks_mlp", {})
    if diffit_cfg.get("use", False) and dit_blocks_mlp_cfg.get("use", False):
        raise ValueError("Only one of diffit or dit_blocks_mlp can be enabled at a time.")
    if diffit_cfg.get("use", False):
        strategies.append(DiffITStrategy(diffit_cfg))
    elif dit_blocks_mlp_cfg.get("use", False):
        strategies.append(DitBlocksMlpStrategy(dit_blocks_mlp_cfg))

    svdiff_cfg = finetune_cfg.get("svdiff", {})
    svdiff_uv_cfg = finetune_cfg.get("svdiff_uv", {})
    if svdiff_cfg.get("use", False) and svdiff_uv_cfg.get("use", False):
        raise ValueError("Only one of svdiff or svdiff_uv can be enabled at a time.")
    if svdiff_cfg.get("use", False):
        strategies.append(SVDiffStrategy(svdiff_cfg))
    elif svdiff_uv_cfg.get("use", False):
        strategies.append(SVDiffUVStrategy(svdiff_uv_cfg))

    if not strategies:
        return NoOpStrategy()
    if len(strategies) == 1:
        return strategies[0]
    return CompositeStrategy(strategies)
