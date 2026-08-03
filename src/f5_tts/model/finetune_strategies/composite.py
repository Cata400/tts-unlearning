from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from f5_tts.model.finetune_strategies.base import FineTuningStrategy

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from f5_tts.model.trainer_unlearn import TrainerUnlearn


class CompositeStrategy(FineTuningStrategy):
    """Applies a list of strategies in order (e.g. `DitBlocksMlp` freezing + `SVDiff-UV` parametrization)."""

    def __init__(self, children: list[FineTuningStrategy]):
        if not children:
            raise ValueError("CompositeStrategy requires at least one child strategy.")
        self.children = children
        self.name = "+".join(child.name for child in children)

    @property
    def requires_optimizer_reset(self) -> bool:
        return any(child.requires_optimizer_reset for child in self.children)

    def register_wandb_metrics(self, logger: str | None) -> None:
        for child in self.children:
            child.register_wandb_metrics(logger)

    def apply(self, unwrapped_model: nn.Module) -> None:
        for child in self.children:
            child.apply(unwrapped_model)

    def run_pre_training_hook(
        self,
        trainer: TrainerUnlearn,
        *,
        unlearn_method: str,
        train_dataset: Dataset,
        num_workers: int,
        resumable_with_seed: int | None,
    ) -> None:
        for child in self.children:
            child.run_pre_training_hook(
                trainer,
                unlearn_method=unlearn_method,
                train_dataset=train_dataset,
                num_workers=num_workers,
                resumable_with_seed=resumable_with_seed,
            )
