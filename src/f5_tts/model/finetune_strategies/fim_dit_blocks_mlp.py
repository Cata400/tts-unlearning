from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch.nn as nn

import wandb
from f5_tts.model.finetune_strategies._fisher_profiling import (
    VALID_LOSS_SOURCES as _VALID_LOSS_SOURCES,
)
from f5_tts.model.finetune_strategies._fisher_profiling import (
    VALID_PARAM_FILTERS as _VALID_PARAM_FILTERS,
)
from f5_tts.model.finetune_strategies._fisher_profiling import (
    compute_fisher,
    select_scored_param_names,
)
from f5_tts.model.finetune_strategies.base import FineTuningStrategy
from f5_tts.model.finetune_strategies.dit_blocks_mlp import (
    VALID_DIT_BLOCKS_MLP_VERSIONS as _VALID_VERSIONS,
)
from f5_tts.model.finetune_strategies.dit_blocks_mlp import (
    select_dit_block_mlp_param_names,
)

if TYPE_CHECKING:
    from f5_tts.model.trainer_unlearn import TrainerUnlearn


_VALID_LAYER_SCORES = ("mean", "sum")


class FimDitBlocksMlpStrategy(FineTuningStrategy):
    """Picks the top-K DiT blocks by aggregated Fisher, then keeps only the v1/v2 MLP layers inside them.

    The per-block score is `FIM_k = sum` (or `mean`) of the diagonal empirical Fisher of every
    parameter under `transformer_blocks.{k}.`. After profiling, the top-K blocks are chosen and
    the same keyword filter as `DitBlocksMlpStrategy` is applied (v1: `ff` + `attn.to_out`;
    v2: FFN + all attn projections). Everything else is frozen.

    Acts as a freezer (mutually exclusive with `diffit` and `dit_blocks_mlp` in the factory);
    composes downstream with `fim` (for finer per-tensor/per-element selection inside the chosen
    blocks) and with `svdiff` / `svdiff_uv`.
    """

    name = "FIM-DitBlocksMlp"

    def __init__(self, config: dict):
        self.config = config

        top_k = config.get("top_k")
        if top_k is None or int(top_k) <= 0:
            raise ValueError(f"fim_dit_blocks_mlp.top_k must be a positive integer, got {top_k!r}")
        self.top_k = int(top_k)

        version = config.get("version")
        if version not in _VALID_VERSIONS:
            raise ValueError(f"fim_dit_blocks_mlp.version must be one of {_VALID_VERSIONS}, got {version!r}")
        self.version = version

        layer_score = str(config.get("layer_score", "sum")).lower()
        if layer_score not in _VALID_LAYER_SCORES:
            raise ValueError(
                f"fim_dit_blocks_mlp.layer_score must be one of {_VALID_LAYER_SCORES}, got {layer_score!r}"
            )
        self.layer_score = layer_score

        profile_steps = int(config.get("profile_steps", 0))
        if profile_steps <= 0:
            raise ValueError(f"fim_dit_blocks_mlp.profile_steps must be a positive integer, got {profile_steps!r}")
        self.profile_steps = profile_steps

        loss_source = str(config.get("loss_source", "combined")).lower()
        if loss_source not in _VALID_LOSS_SOURCES:
            raise ValueError(
                f"fim_dit_blocks_mlp.loss_source must be one of {_VALID_LOSS_SOURCES}, got {loss_source!r}"
            )
        self.loss_source = loss_source

        param_filter = str(config.get("param_filter", "all")).lower()
        if param_filter not in _VALID_PARAM_FILTERS:
            raise ValueError(
                f"fim_dit_blocks_mlp.param_filter must be one of {_VALID_PARAM_FILTERS}, got {param_filter!r}"
            )
        self.param_filter = param_filter

        self.log_wandb = bool(config.get("log_wandb", False))
        self.block_module_prefix = str(config.get("block_module_prefix", "transformer_blocks"))

        # Prefix may appear at the start (unwrapped model) or after a dot (e.g. "transformer.transformer_blocks.N.").
        self._block_index_re = re.compile(rf"(?:^|\.){re.escape(self.block_module_prefix)}\.(\d+)\.")

    def register_wandb_metrics(self, logger: str | None) -> None:
        if logger != "wandb" or not self.log_wandb:
            return
        wandb.define_metric("fim_dit_blocks_mlp_step")
        wandb.define_metric("fim_dit_blocks_mlp/*", step_metric="fim_dit_blocks_mlp_step")

    def apply(self, unwrapped_model: nn.Module, trainer: "TrainerUnlearn | None" = None) -> None:
        if trainer is None:
            print(f"[{self.name}] no trainer context provided; skipping apply().")
            return

        prefix_token = f"{self.block_module_prefix}."
        scored_names = [
            name for name in select_scored_param_names(unwrapped_model, self.param_filter) if prefix_token in name
        ]
        if not scored_names:
            print(f"[{self.name}] no trainable DiT-block parameters match the current filter; skipping.")
            return

        fisher = compute_fisher(
            trainer,
            unwrapped_model,
            scored_names,
            profile_steps=self.profile_steps,
            loss_source=self.loss_source,
            log_prefix=self.name,
        )
        if fisher is None:
            print(f"[{self.name}] Fisher profiling collected zero effective steps; skipping.")
            return

        block_agg: dict[int, dict] = {}
        for name, fisher_tensor in fisher.items():
            match = self._block_index_re.search(name)
            if match is None:
                continue
            idx = int(match.group(1))
            entry = block_agg.setdefault(idx, {"sum": 0.0, "numel": 0, "scored_params": 0})
            entry["sum"] += float(fisher_tensor.sum().item())
            entry["numel"] += int(fisher_tensor.numel())
            entry["scored_params"] += 1

        if not block_agg:
            print(f"[{self.name}] no DiT-block parameters accumulated any Fisher mass; skipping.")
            return

        ranked = sorted(
            (
                (
                    idx,
                    agg["sum"] / max(agg["numel"], 1) if self.layer_score == "mean" else agg["sum"],
                    agg["numel"],
                    agg["scored_params"],
                )
                for idx, agg in block_agg.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        k = min(self.top_k, len(ranked))
        if self.top_k > len(ranked):
            print(
                f"[{self.name}] requested top_k={self.top_k} but only {len(ranked)} DiT blocks are "
                f"available; clamping to {k}."
            )
        selected_blocks = sorted(idx for idx, _, _, _ in ranked[:k])

        print(f"[{self.name}] per-block FIM_k ranking (score={self.layer_score}, kept {k}/{len(ranked)} blocks):")
        selected_set = set(selected_blocks)
        for idx, score, numel, n_params in ranked:
            marker = "+" if idx in selected_set else "-"
            print(f"  {marker} block {idx:>2}  FIM_k={score:.3e}  scored_numel={numel}  scored_params={n_params}")

        trainable_names = select_dit_block_mlp_param_names(
            unwrapped_model, selected_blocks, self.version, self.block_module_prefix
        )

        print(f"[{self.name}] trainable parameters (version={self.version}, blocks={selected_blocks}):")
        for name in trainable_names:
            print(f"  - {name}")

        trainable_set = set(trainable_names)
        for n, p in unwrapped_model.named_parameters():
            p.requires_grad = n in trainable_set

        effective_elements = sum(p.numel() for n, p in unwrapped_model.named_parameters() if n in trainable_set)
        num_trainable_elems = sum(p.numel() for p in unwrapped_model.parameters() if p.requires_grad)
        print(
            f"[{self.name}] trainable blocks: {selected_blocks} "
            f"(elements kept: {effective_elements:,}; total requires_grad elems now: {num_trainable_elems:,} "
            f"= {num_trainable_elems / 1e6:.3f}M)"
        )

        if self.log_wandb and trainer.logger == "wandb" and trainer.accelerator.is_local_main_process:
            trainer.accelerator.log(
                {
                    "fim_dit_blocks_mlp_step": 1,
                    "fim_dit_blocks_mlp/top_k": float(k),
                    "fim_dit_blocks_mlp/version": self.version,
                    "fim_dit_blocks_mlp/layer_score": self.layer_score,
                    "fim_dit_blocks_mlp/loss_source": self.loss_source,
                    "fim_dit_blocks_mlp/param_filter": self.param_filter,
                    "fim_dit_blocks_mlp/profile_steps": float(self.profile_steps),
                    "fim_dit_blocks_mlp/num_selected_blocks": float(len(selected_blocks)),
                    "fim_dit_blocks_mlp/selected_block_indices": ",".join(str(i) for i in selected_blocks),
                    "fim_dit_blocks_mlp/effective_elements": float(effective_elements),
                },
                step=0,
            )
