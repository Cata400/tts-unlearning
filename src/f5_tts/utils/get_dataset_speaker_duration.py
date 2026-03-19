import os
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

dataset_path = "/home/catalin/Desktop/Datasets/LibriTTS/train-clean-100_train_intra_speaker_split_0.2"
dataset_name = Path(dataset_path).name
speaker_duration = {}

for speaker in tqdm(sorted(os.listdir(dataset_path))):
    speaker_path = os.path.join(dataset_path, speaker)
    total_duration = 0.0

    for file in os.listdir(os.path.join(dataset_path, speaker)):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, speaker, file)
            audio, sr = sf.read(file_path)
            duration = len(audio) / sr
            total_duration += duration

    speaker_duration[int(speaker)] = round(total_duration / 60.0, 2)


speaker_duration = dict(sorted(speaker_duration.items(), key=lambda x: int(x[0])))
df = pd.DataFrame.from_dict(speaker_duration, orient="index", columns=["Duration [min]"])
df.index.name = "Speaker"
df.to_csv(
    f"/home/catalin/Desktop/Projects/tts-unlearning/data/LibriTTS_100_train_intra_speaker_split_0.2_pinyin/{dataset_name}_speaker_duration.csv",
    index=True,
)
