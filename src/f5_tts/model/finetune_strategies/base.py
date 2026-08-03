from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from f5_tts.model.trainer_unlearn import TrainerUnlearn


class FineTuningStrategy:
    """Base class for parameter-selection / parametrization strategies applied before training.

    Subclasses control which parameters are trainable (freezing, parametrization) and can
    optionally run a pre-training hook (e.g. gradient profiling + mask installation).
    """

    name: str = "base"

    def register_wandb_metrics(self, logger: str | None) -> None:
        return

    def apply(self, unwrapped_model: nn.Module) -> None:
        return

    @property
    def requires_optimizer_reset(self) -> bool:
        return False

    def run_pre_training_hook(
        self,
        trainer: TrainerUnlearn,
        *,
        unlearn_method: str,
        train_dataset: Dataset,
        num_workers: int,
        resumable_with_seed: int | None,
    ) -> None:
        return
