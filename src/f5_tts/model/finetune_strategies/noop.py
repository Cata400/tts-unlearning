from __future__ import annotations

from f5_tts.model.finetune_strategies.base import FineTuningStrategy


class NoOpStrategy(FineTuningStrategy):
    """Placeholder strategy used when no fine-tuning method is enabled: all parameters stay trainable."""

    name = "none"
