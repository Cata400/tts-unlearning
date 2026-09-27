"""Build the GitHub Pages audio demo page for the continual speaker-unlearning run.

For each showcased speaker it picks `N_PER_SPEAKER` utterances that exist in all three
sources (ground truth, pretrained generations, unlearned generations), copies them to
`docs/assets/audio/`, and writes `docs/index.md` with one comparison table per speaker.

The unlearned column always comes from the *final* continual step, so the forgotten
speakers are shown after the whole 5-speaker sequence has been unlearned.

Run with:  python scripts/build_demo_page.py
"""

from __future__ import annotations

import html
import json
import shutil
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GT_ROOT = Path("/home/catalin/Desktop/Datasets/LibriTTS/train-clean-100_val_intra_speaker_split_0.2")
INFERENCE_SUBDIR = "seed42_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"
PRETRAINED_DIR = REPO_ROOT / "results/pretrained/train-clean-100_val_intra_speaker_split_0.2" / INFERENCE_SUBDIR
UNLEARNED_DIR = (
    REPO_ROOT
    / "results/server"
    / (
        "F5TTS_v1_Base_vocos_pinyin_LibriTTS_100_train_intra_speaker_split_0.2_continual_SGU_oversampling"
        "_svdiff_u_top64_plus_fim_dit_blocks_mlp_v1_top_11_blocks_ewc_0.1_fim_load_from_ema_4_epochs"  # todo change with the correct run directory if needed
    )
    / "step05_spk87_train-clean-100_val_intra_speaker_split_0.2"
    / INFERENCE_SUBDIR
)

DOCS_DIR = REPO_ROOT / "docs"
AUDIO_DIR = DOCS_DIR / "assets" / "audio"
INDEX_PATH = DOCS_DIR / "index.md"

# Continual unlearning order: step01 -> step05.
FORGOTTEN_SPEAKERS = [196, 26, 40, 78, 87]
RETAINED_SPEAKERS = [1116, 200, 1040]
N_PER_SPEAKER = 3

# Preferred clip length in seconds; long enough to judge timbre, short enough to skim.
MIN_DURATION = 4.0
MAX_DURATION = 12.0
FALLBACK_MAX_DURATION = 15.0

SOURCES = ("gt", "pretrained", "unlearned")


def load_metainfo(run_dir: Path) -> dict[str, str]:
    """Map utterance id -> the text that was synthesised, from a run's metainfo.txt."""
    texts: dict[str, str] = {}
    with (run_dir / "metainfo.txt").open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                texts[fields[0]] = fields[1].strip()
    return texts


def load_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    """Map utterance id -> {sim, wer, utmosv2} from a run's per-utterance result files."""
    metrics: dict[str, dict[str, float]] = {}
    for filename, key in (
        ("_sim_results_speechbrain_ecapa.json", "sim"),
        ("_wer_results.json", "wer"),
        ("_utmosv2_results.json", "utmosv2"),
    ):
        path = run_dir / filename
        if not path.exists():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8"))["all_results"]:
            metrics.setdefault(entry["wav"], {})[key] = entry[key]
    return metrics


def speaker_of(utterance_id: str) -> str:
    return utterance_id.split("_", 1)[0]


def available_ids(run_dir: Path) -> set[str]:
    return {path.stem for path in run_dir.glob("*.wav")}


def gt_path(utterance_id: str) -> Path:
    return GT_ROOT / speaker_of(utterance_id) / f"{utterance_id}.wav"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def select_utterances(speaker: int, common_ids: set[str]) -> list[str]:
    """Pick N_PER_SPEAKER ids for `speaker`, preferring clips in the MIN..MAX duration window."""
    candidates = sorted(uid for uid in common_ids if speaker_of(uid) == str(speaker))
    if not candidates:
        raise SystemExit(f"No utterance is present in all three sources for speaker {speaker}.")

    durations = {uid: wav_duration(gt_path(uid)) for uid in candidates}
    preferred = [uid for uid in candidates if MIN_DURATION <= durations[uid] <= MAX_DURATION]
    if len(preferred) >= N_PER_SPEAKER:
        return preferred[:N_PER_SPEAKER]

    fallback = sorted(
        (uid for uid in candidates if durations[uid] <= FALLBACK_MAX_DURATION),
        key=lambda uid: durations[uid],
        reverse=True,
    )
    chosen = preferred + [uid for uid in fallback if uid not in preferred]
    if len(chosen) < N_PER_SPEAKER:
        raise SystemExit(
            f"Speaker {speaker} only has {len(chosen)} usable utterance(s) across all three sources, "
            f"need {N_PER_SPEAKER}. Pick a different speaker."
        )
    return chosen[:N_PER_SPEAKER]


def copy_audio(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    shutil.copy2(src, dest)


def export_audio(utterance_id: str, speaker: int) -> dict[str, str]:
    """Copy the three variants of one utterance; return source -> page-relative path."""
    sources = {
        "gt": gt_path(utterance_id),
        "pretrained": PRETRAINED_DIR / f"{utterance_id}.wav",
        "unlearned": UNLEARNED_DIR / f"{utterance_id}.wav",
    }
    relative: dict[str, str] = {}
    for name, src in sources.items():
        rel = f"assets/audio/{speaker}/{utterance_id}__{name}.wav"
        copy_audio(src, DOCS_DIR / rel)
        relative[name] = rel
    return relative


def audio_cell(rel_path: str, caption: str = "") -> str:
    player = f'<audio controls preload="none" src="{rel_path}"></audio>'
    if caption:
        player += f'<div class="metrics">{caption}</div>'
    return f'<td class="audio">{player}</td>'


def format_metrics(entry: dict[str, float] | None) -> str:
    if not entry:
        return ""
    parts = []
    if "sim" in entry:
        sim = 0.0 if abs(entry["sim"]) < 0.005 else entry["sim"]  # avoid rendering "-0.00"
        parts.append(f"SIM {sim:.2f}")
    if "wer" in entry:
        parts.append(f"WER {entry['wer'] * 100:.1f}%")
    return " · ".join(parts)


def speaker_table(
    speaker: int,
    utterance_ids: list[str],
    texts: dict[str, str],
    pretrained_metrics: dict[str, dict[str, float]],
    unlearned_metrics: dict[str, dict[str, float]],
) -> str:
    rows = []
    for utterance_id in utterance_ids:
        audio = export_audio(utterance_id, speaker)
        text = html.escape(texts.get(utterance_id, ""))
        rows.append(
            "<tr>"
            f'<td class="text"><span class="utt-id">{utterance_id}</span>{text}</td>'
            + audio_cell(audio["gt"])
            + audio_cell(audio["pretrained"], format_metrics(pretrained_metrics.get(utterance_id)))
            + audio_cell(audio["unlearned"], format_metrics(unlearned_metrics.get(utterance_id)))
            + "</tr>"
        )
    return (
        f'<h3 id="speaker-{speaker}">Speaker {speaker}</h3>\n'
        '<div class="table-wrap">\n<table>\n'
        "<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th>"
        "<th>Unlearned</th></tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n</div>\n"
    )


def summary_line(run_dir: Path) -> str:
    """One-line aggregate for the final unlearned model, read from its result files."""
    sim = json.loads((run_dir / "_sim_results_speechbrain_ecapa.json").read_text())["unlearning_avg_sim_results"]
    wer = json.loads((run_dir / "_wer_results.json").read_text())["unlearning_avg_wer_results"]
    utmos = json.loads((run_dir / "_utmosv2_results.json").read_text())["unlearning_avg_utmosv2_results"]
    return (
        f"After all five unlearning steps, speaker similarity on the forgotten speakers drops to "
        f"**{sim['forget_avg']:.3f}** while the retained speakers stay at **{sim['retain_avg']:.3f}**. "
        f"Intelligibility and naturalness are preserved: WER **{wer['retain_avg'] * 100:.1f}%** and "
        f"UTMOSv2 **{utmos['retain_avg']:.2f}** on retained speakers."
    )


def build_page(
    forgotten: dict[int, list[str]],
    retained: dict[int, list[str]],
    texts: dict[str, str],
    pretrained_metrics: dict[str, dict[str, float]],
    unlearned_metrics: dict[str, dict[str, float]],
) -> str:
    order = " → ".join(str(spk) for spk in FORGOTTEN_SPEAKERS)
    sections = [
        "---",
        "layout: default",
        "title: TTS Unlearning",
        "---",
        "",
        "# Continual Speaker Unlearning for Zero-Shot TTS",
        "",
        "Samples from an F5-TTS v1 Base model finetuned on LibriTTS `train-clean-100`, then put through "
        "**five sequential unlearning steps** that remove one speaker each, in the order "
        f"{order}. Every clip below is generated with the *final* model, after all five speakers have "
        "been unlearned.",
        "",
        "All samples use the same zero-shot setup: seed 42, Euler solver, NFE 32, CFG 2.0, sway sampling "
        "−1, speed 1.0, Vocos vocoder. The reference prompt is always another utterance from the same "
        "speaker, so a successful unlearn means the model can no longer copy the prompt's voice.",
        "",
        "`SIM` is ECAPA speaker similarity against the ground-truth recording — lower is better for "
        "forgotten speakers, higher is better for retained ones. `WER` is Whisper word error rate.",
        "",
        summary_line(UNLEARNED_DIR),
        "",
        "*This page is for research demonstration purposes only.*",
        "",
        "## Forgotten speakers",
        "",
        "The pretrained model clones these voices from the prompt. The unlearned model should not.",
        "",
    ]
    for speaker, ids in forgotten.items():
        sections.append(speaker_table(speaker, ids, texts, pretrained_metrics, unlearned_metrics))

    sections += [
        "## Retained speakers",
        "",
        "Speakers that were never targeted — voice cloning quality should be unchanged.",
        "",
    ]
    for speaker, ids in retained.items():
        sections.append(speaker_table(speaker, ids, texts, pretrained_metrics, unlearned_metrics))

    return "\n".join(sections) + "\n"


def main() -> None:
    for path in (GT_ROOT, PRETRAINED_DIR, UNLEARNED_DIR):
        if not path.is_dir():
            raise SystemExit(f"Missing required directory: {path}")

    # The sampled utterance subsets differ between runs, so only ids present everywhere are usable.
    common_ids = available_ids(PRETRAINED_DIR) & available_ids(UNLEARNED_DIR)
    common_ids = {uid for uid in common_ids if gt_path(uid).exists()}

    texts = load_metainfo(UNLEARNED_DIR)
    pretrained_metrics = load_metrics(PRETRAINED_DIR)
    unlearned_metrics = load_metrics(UNLEARNED_DIR)

    forgotten = {spk: select_utterances(spk, common_ids) for spk in FORGOTTEN_SPEAKERS}
    retained = {spk: select_utterances(spk, common_ids) for spk in RETAINED_SPEAKERS}

    INDEX_PATH.write_text(
        build_page(forgotten, retained, texts, pretrained_metrics, unlearned_metrics),
        encoding="utf-8",
    )

    n_speakers = len(forgotten) + len(retained)
    n_clips = sum(len(ids) for ids in (*forgotten.values(), *retained.values())) * len(SOURCES)
    print(f"Wrote {INDEX_PATH.relative_to(REPO_ROOT)} — {n_speakers} speakers, {n_clips} audio files.")
    print(f"Audio under {AUDIO_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
