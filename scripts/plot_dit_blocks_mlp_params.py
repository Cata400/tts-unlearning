"""Plot the trainable-parameter counts produced by the `dit_blocks_mlp` v1 finetune strategy.

Loads the unlearn config, builds the F5-TTS DiT model exactly like
`src/f5_tts/train/unlearn.py`, optionally loads the pretrained safetensors
weights, applies `build_finetune_strategy` (so the resulting `requires_grad`
mask matches what training would produce), then renders two bar plots:

  * Plot A: one bar per Linear layer inside each DiT block (project_in,
    project_out of the feed-forward, plus attn.to_out — everything v1
    unfreezes). Weight + bias are summed into a single bar per Linear.
  * Plot B: one bar per DiT block, aggregating the three Linear-layer bars
    from Plot A.

Bars in both plots share the same block-index colormap.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from f5_tts.model import CFM  # noqa: E402
from f5_tts.model.backbones.dit import DiT  # noqa: E402
from f5_tts.model.finetune_strategies.dit_blocks_mlp import (  # noqa: E402
    DitBlocksMlpStrategy,
)
from f5_tts.model.utils import get_tokenizer  # noqa: E402

# state_dict key patterns matched by the v1 selection, per block index i:
#   transformer_blocks.{i}.ff.ff.0.0.{weight,bias}    -> "ff.project_in"
#   transformer_blocks.{i}.ff.ff.2.{weight,bias}      -> "ff.project_out"
#   transformer_blocks.{i}.attn.to_out.0.{weight,bias} -> "attn.to_out"
_ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ff.project_in", re.compile(r"^ff\.ff\.0\.0\.(weight|bias)$")),
    ("ff.project_out", re.compile(r"^ff\.ff\.2\.(weight|bias)$")),
    ("attn.to_out", re.compile(r"^attn\.to_out\.0\.(weight|bias)$")),
]
_ROLE_ORDER = [role for role, _ in _ROLE_PATTERNS]

_BLOCK_RE = re.compile(r"^transformer(?:\.transformer_blocks|_blocks)\.(\d+)\.(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base_unlearn.yaml"),
        help="Path to the unlearn config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "scripts"),
        help="Directory where the two PNG plots will be written.",
    )
    parser.add_argument(
        "--load-pretrained",
        dest="load_pretrained",
        action="store_true",
        default=True,
        help="Load pretrained safetensors (default). Does not affect param counts.",
    )
    parser.add_argument(
        "--no-load-pretrained",
        dest="load_pretrained",
        action="store_false",
        help="Skip loading pretrained weights (faster; param counts are identical).",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="Skip the v1 selection and plot every parameter in every DiT block.",
    )
    return parser.parse_args()


def build_model(cfg) -> CFM:
    tokenizer = cfg.model.tokenizer
    tokenizer_path = cfg.datasets.name if tokenizer != "custom" else cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    diffit_use = cfg.model.get("finetune", {}).get("diffit", {}).get("use", False)
    diffit_blocks = list(cfg.model.get("finetune", {}).get("diffit", {}).get("diffit_blocks", []))

    return CFM(
        transformer=DiT(
            **cfg.model.arch,
            text_num_embeds=vocab_size,
            mel_dim=cfg.model.mel_spec.n_mel_channels,
            diffit=diffit_use,
            diffit_blocks=diffit_blocks,
        ),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )


def load_pretrained_into(model: CFM, pretrained_path: str) -> None:
    if not pretrained_path or not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    if not pretrained_path.endswith(".safetensors"):
        raise ValueError(f"Expected .safetensors checkpoint, got: {pretrained_path}")

    from safetensors.torch import load_file

    raw = load_file(pretrained_path, device="cpu")
    # Same normalization as TrainerUnlearn.load_pretrained_checkpoint: strip the
    # ema_model. prefix (when present) and drop bookkeeping scalars.
    state_dict = {k.replace("ema_model.", ""): v for k, v in raw.items() if k not in ("initted", "update", "step")}
    # Backward-compat: drop stale mel_stft buffers when they appear in the ckpt.
    for stale in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state_dict.pop(stale, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_pretrained] {len(missing)} missing keys (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[load_pretrained] {len(unexpected)} unexpected keys (showing up to 5): {unexpected[:5]}")


def collect_trainable_by_block(
    model: CFM, expected_blocks: list[int]
) -> tuple[dict[int, dict[str, int]], dict[int, int], list[str]]:
    """Group trainable-parameter counts by (block_index, role).

    Returns:
        per_layer: {block_idx: {role: numel}} for the three v1 roles.
        per_block: {block_idx: total_numel} summed across roles.
        unexpected: list of trainable param names that did not match any expected role.
    """
    per_layer: dict[int, dict[str, int]] = {b: {r: 0 for r in _ROLE_ORDER} for b in expected_blocks}
    per_block: dict[int, int] = {b: 0 for b in expected_blocks}
    unexpected: list[str] = []

    expected_set = set(expected_blocks)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match = _BLOCK_RE.match(name)
        if not match:
            unexpected.append(name)
            continue
        block_idx = int(match.group(1))
        tail = match.group(2)
        if block_idx not in expected_set:
            unexpected.append(name)
            continue

        role: str | None = None
        for candidate_role, pattern in _ROLE_PATTERNS:
            if pattern.match(tail):
                role = candidate_role
                break
        if role is None:
            unexpected.append(name)
            continue

        n = int(param.numel())
        per_layer[block_idx][role] += n
        per_block[block_idx] += n

    return per_layer, per_block, unexpected


def collect_all_by_block(
    model: CFM, expected_blocks: list[int]
) -> tuple[dict[int, dict[str, int]], dict[int, int], list[str]]:
    # Roles are discovered from tensor names: strip trailing .weight/.bias to get the owning submodule path.
    per_layer: dict[int, dict[str, int]] = {b: {} for b in expected_blocks}
    per_block: dict[int, int] = {b: 0 for b in expected_blocks}
    expected_set = set(expected_blocks)
    for name, param in model.named_parameters():
        match = _BLOCK_RE.match(name)
        if not match:
            continue
        block_idx = int(match.group(1))
        if block_idx not in expected_set:
            continue
        tail = match.group(2)
        role = re.sub(r"\.(weight|bias)$", "", tail)
        n = int(param.numel())
        per_layer[block_idx][role] = per_layer[block_idx].get(role, 0) + n
        per_block[block_idx] += n
    return per_layer, per_block, []


def print_summary(
    per_layer: dict[int, dict[str, int]],
    per_block: dict[int, int],
    role_order: list[str],
    total_label: str = "grand total trainable params",
) -> None:
    cols = ["block", *role_order, "total"]
    widths = [max(6, len(c)) for c in cols]
    header = " | ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for b in sorted(per_layer.keys()):
        row = per_layer[b]
        cells = [f"{b:>{widths[0]}d}"]
        for r, w in zip(role_order, widths[1:-1]):
            cells.append(f"{row.get(r, 0):>{w},d}")
        cells.append(f"{per_block[b]:>{widths[-1]},d}")
        print(" | ".join(cells))
    total = sum(per_block.values())
    print("-" * len(header))
    print(f"{total_label:>{len(header) - 16}}: {total:>14,d}")


def plot_per_layer(
    per_layer: dict[int, dict[str, int]],
    output_path: Path,
    role_order: list[str],
    title: str,
    ylabel: str = "Trainable parameters (weight + bias)",
) -> None:
    blocks_sorted = sorted(per_layer.keys())
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(blocks_sorted) - 1)

    x_labels: list[str] = []
    y_values: list[int] = []
    colors: list[tuple[float, float, float, float]] = []

    for pos, b in enumerate(blocks_sorted):
        color = cmap(pos / denom)
        for role in role_order:
            x_labels.append(f"b{b:02d}.{role}")
            y_values.append(per_layer[b].get(role, 0))
            colors.append(color)

    xs = np.arange(len(x_labels))
    fig_width = max(12.0, 0.16 * len(x_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(xs, y_values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Layer (grouped by DiT block)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.margins(x=0.005)

    # Legend keyed by DiT block. With 22 blocks, keep it compact via multi-column layout.
    legend_handles = [
        Patch(facecolor=cmap(pos / denom), edgecolor="black", linewidth=0.3, label=f"block {b}")
        for pos, b in enumerate(blocks_sorted)
    ]
    ax.legend(
        handles=legend_handles,
        title="DiT block",
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=7,
        title_fontsize=8,
        ncol=2,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


def plot_per_block(
    per_block: dict[int, int],
    output_path: Path,
    title: str,
    ylabel: str = "Trainable parameters",
) -> None:
    blocks_sorted = sorted(per_block.keys())
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(blocks_sorted) - 1)
    colors = [cmap(pos / denom) for pos in range(len(blocks_sorted))]
    y_values = [per_block[b] for b in blocks_sorted]

    xs = np.arange(len(blocks_sorted))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(xs, y_values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in blocks_sorted])
    ax.set_xlabel("DiT block index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    y_max = max(y_values) if y_values else 1
    for bar, val in zip(bars, y_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.005,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=60,
        )
    ax.set_ylim(0, max(1, y_max) * 1.18)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {output_path}")


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.load(args.config)
    # `hydra.run.dir` uses the `${now:...}` resolver which is only registered
    # inside a Hydra run. This script does not need Hydra's runtime, so drop
    # the whole `hydra` block before resolving the rest of the config.
    if "hydra" in cfg:
        del cfg["hydra"]
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    dit_blocks_mlp_cfg = cfg_dict.get("model", {}).get("finetune", {}).get("dit_blocks_mlp", {})
    if not args.all_params:
        if not dit_blocks_mlp_cfg.get("use", False):
            raise SystemExit("Config has model.finetune.dit_blocks_mlp.use=False; nothing to plot.")
        version = dit_blocks_mlp_cfg.get("version")
        if version != "v1":
            raise SystemExit(f"This script targets v1; got version={version!r}.")
        expected_blocks = sorted({int(b) for b in dit_blocks_mlp_cfg.get("blocks", [])})
        if not expected_blocks:
            raise SystemExit("Config has an empty model.finetune.dit_blocks_mlp.blocks list.")
        print(f"[config] {args.config}")
        print(f"[config] dit_blocks_mlp version=v1 blocks={expected_blocks}")
    else:
        expected_blocks = []
        print(f"[config] {args.config}")
        print("[config] --all-params: plotting every parameter of every DiT block")

    with torch.device("cpu"):
        model = build_model(cfg)

    if args.load_pretrained:
        load_pretrained_into(model, cfg.ckpts.pretrained_path)
    else:
        print("[load_pretrained] skipped (--no-load-pretrained)")

    if args.all_params:
        # All 22 DiT blocks are always instantiated, regardless of the v1 `blocks` list.
        all_blocks = list(range(int(cfg.model.arch.depth)))
        per_layer, per_block, _ = collect_all_by_block(model, all_blocks)
        # Discover roles from block 0 (all blocks share the same structure).
        role_order = sorted(per_layer[all_blocks[0]].keys())

        grand_total_independent = sum(int(p.numel()) for name, p in model.named_parameters() if _BLOCK_RE.match(name))
        bucketed = sum(per_block.values())
        print(
            f"[check] grand total (all block params): {grand_total_independent:,d}; "
            f"bucketed: {bucketed:,d}; leftover: {grand_total_independent - bucketed:,d}"
        )
        print_summary(per_layer, per_block, role_order, total_label="grand total params in DiT blocks")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_per_layer(
            per_layer,
            output_dir / "dit_blocks_all_params_per_layer.png",
            role_order,
            title="All DiT-block params per layer, colored by DiT block",
            ylabel="Parameters (weight + bias)",
        )
        plot_per_block(
            per_block,
            output_dir / "dit_blocks_all_params_per_block.png",
            title="All DiT-block params aggregated per block",
            ylabel="Parameters",
        )
        return

    strategy = DitBlocksMlpStrategy(dit_blocks_mlp_cfg)
    strategy.apply(model)

    per_layer, per_block, unexpected = collect_trainable_by_block(model, expected_blocks)

    # Cross-check: any trainable param not attributable to one of the three v1 roles
    # inside an expected block. With v1 + all 22 blocks selected this should be empty
    # (aside from composite strategies producing SVDiff parametrizations, which the
    # current config disables via svdiff_uv triggering a separate SVD strategy).
    if unexpected:
        print("[warn] trainable parameters not matched by a v1 role in an expected block:")
        for n in unexpected[:20]:
            print(f"       {n}")
        if len(unexpected) > 20:
            print(f"       ... and {len(unexpected) - 20} more")

    # Sanity: independent grand total should match the sum of per_block (+ any unbucketed).
    grand_total_independent = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    bucketed = sum(per_block.values())
    print(
        f"[check] grand total (all trainable): {grand_total_independent:,d}; "
        f"bucketed (ff+attn.to_out inside expected blocks): {bucketed:,d}; "
        f"leftover (unbucketed): {grand_total_independent - bucketed:,d}"
    )

    print_summary(per_layer, per_block, _ROLE_ORDER)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_layer(
        per_layer,
        output_dir / "dit_blocks_mlp_trainable_params_per_layer.png",
        _ROLE_ORDER,
        title="dit_blocks_mlp v1: trainable params per Linear layer, colored by DiT block",
    )
    plot_per_block(
        per_block,
        output_dir / "dit_blocks_mlp_trainable_params_per_block.png",
        title="dit_blocks_mlp v1: trainable params aggregated per DiT block",
    )


if __name__ == "__main__":
    main()
