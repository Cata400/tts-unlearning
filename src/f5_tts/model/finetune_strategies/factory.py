from __future__ import annotations

from f5_tts.model.finetune_strategies.base import FineTuningStrategy
from f5_tts.model.finetune_strategies.composite import CompositeStrategy
from f5_tts.model.finetune_strategies.diffit import DiffFitStrategy
from f5_tts.model.finetune_strategies.dit_blocks_mlp import DitBlocksMlpStrategy
from f5_tts.model.finetune_strategies.fim import FIMStrategy
from f5_tts.model.finetune_strategies.fim_dit_blocks_mlp import FimDitBlocksMlpStrategy
from f5_tts.model.finetune_strategies.noop import NoOpStrategy
from f5_tts.model.finetune_strategies.svdiff import SVDiffStrategy
from f5_tts.model.finetune_strategies.svdiff_uv import SVDiffUVStrategy


def build_finetune_strategy(model_cfg_dict: dict) -> FineTuningStrategy:
    """Build the fine-tuning strategy declared by `model.finetune` in the config.

    At most one parameter-freezing strategy (diffit XOR dit_blocks_mlp XOR fim_dit_blocks_mlp),
    at most one selector strategy (fim), and at most one SVD strategy (svdiff XOR svdiff_uv) may
    be enabled. When more than one category is enabled they are composed in the order
    `[freezer, selector, svd]` inside a `CompositeStrategy`; FIM's `apply()` runs after the
    freezer (so its Fisher profiling respects the freezer's mask) and before the SVD strategy
    (so SVD parametrization is only registered on FIM-selected tensors).
    """
    finetune_cfg = model_cfg_dict.get("model", {}).get("finetune", {})

    strategies: list[FineTuningStrategy] = []

    diffit_cfg = finetune_cfg.get("diffit", {})
    dit_blocks_mlp_cfg = finetune_cfg.get("dit_blocks_mlp", {})
    fim_dit_blocks_mlp_cfg = finetune_cfg.get("fim_dit_blocks_mlp", {})
    freezer_flags = {
        "diffit": diffit_cfg.get("use", False),
        "dit_blocks_mlp": dit_blocks_mlp_cfg.get("use", False),
        "fim_dit_blocks_mlp": fim_dit_blocks_mlp_cfg.get("use", False),
    }
    enabled_freezers = [name for name, flag in freezer_flags.items() if flag]
    if len(enabled_freezers) > 1:
        raise ValueError(
            "At most one of diffit / dit_blocks_mlp / fim_dit_blocks_mlp can be enabled at a time; "
            f"got {enabled_freezers}."
        )
    if freezer_flags["diffit"]:
        strategies.append(DiffFitStrategy(diffit_cfg))
    elif freezer_flags["dit_blocks_mlp"]:
        strategies.append(DitBlocksMlpStrategy(dit_blocks_mlp_cfg))
    elif freezer_flags["fim_dit_blocks_mlp"]:
        strategies.append(FimDitBlocksMlpStrategy(fim_dit_blocks_mlp_cfg))

    fim_cfg = finetune_cfg.get("fim", {})
    fim_enabled = fim_cfg.get("use", False)
    if fim_enabled:
        strategies.append(FIMStrategy(fim_cfg))

    svdiff_cfg = finetune_cfg.get("svdiff", {})
    svdiff_uv_cfg = finetune_cfg.get("svdiff_uv", {})
    if svdiff_cfg.get("use", False) and svdiff_uv_cfg.get("use", False):
        raise ValueError("Only one of svdiff or svdiff_uv can be enabled at a time.")
    svd_enabled = svdiff_cfg.get("use", False) or svdiff_uv_cfg.get("use", False)

    if fim_enabled and svd_enabled and str(fim_cfg.get("granularity", "layer")).lower() == "weight":
        # Per-element grad hooks on original weights are bypassed once svdiff parametrizes them,
        # so weight-granularity FIM composed with any SVD strategy would silently no-op.
        raise ValueError(
            "fim.granularity='weight' cannot be combined with svdiff / svdiff_uv; "
            "use fim.granularity='layer' when composing FIM with an SVD strategy."
        )

    if svdiff_cfg.get("use", False):
        strategies.append(SVDiffStrategy(svdiff_cfg))
    elif svdiff_uv_cfg.get("use", False):
        strategies.append(SVDiffUVStrategy(svdiff_uv_cfg))

    if not strategies:
        return NoOpStrategy()
    if len(strategies) == 1:
        return strategies[0]
    return CompositeStrategy(strategies)
