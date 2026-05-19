import os
import shutil

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

libritts_100_path = "/home/catalin/Desktop/Datasets/LibriTTS/train-clean-100"
libritts_360_path = "/data/LibriTTS/train-clean-360"
forget_speaker = 196
destination_path = f"/home/catalin/Desktop/Datasets/LibriTTS/train-clean-360_top_k_similar_speakers_to_{forget_speaker}"
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
model = model.cuda("cuda")
model.eval()
k = 5

print(f"Calculating average embedding for speaker {forget_speaker}...")
forget_speaker_embeddings = []
for directory in os.listdir(os.path.join(libritts_100_path, str(forget_speaker))):
    for file in os.listdir(os.path.join(libritts_100_path, str(forget_speaker), directory)):
        if not file.endswith(".wav"):
            continue
        wav, sr = torchaudio.load(os.path.join(libritts_100_path, str(forget_speaker), directory, file))

        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resample(wav)
            sr = 16000

        wav = wav.cuda("cuda")
        wav_sb = wav.mean(dim=0)
        wavs = wav_sb.unsqueeze(0)
        embeddings = model.encode_batch(wavs)
        emb = embeddings[0]
        forget_speaker_embeddings.append(emb)

avg_forget_speaker_embedding = torch.mean(torch.stack(forget_speaker_embeddings), dim=0)

print("Calculating average embeddings for all other speakers and computing similarities...")
speaker_embeddings = {}
for speaker in tqdm(os.listdir(libritts_360_path)):
    if speaker == str(forget_speaker):
        continue

    speaker_embedding_list = []

    for directory in os.listdir(os.path.join(libritts_360_path, speaker)):
        for file in os.listdir(os.path.join(libritts_360_path, speaker, directory)):
            if not file.endswith(".wav"):
                continue
            wav, sr = torchaudio.load(os.path.join(libritts_360_path, speaker, directory, file))

            if sr != 16000:
                resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resample(wav)
                sr = 16000

            wav = wav.cuda("cuda")
            wav_sb = wav.mean(dim=0)
            wavs = wav_sb.unsqueeze(0)
            embeddings = model.encode_batch(wavs)
            emb = embeddings[0]
            speaker_embedding_list.append(emb)

    avg_speaker_embedding = torch.mean(torch.stack(speaker_embedding_list), dim=0)
    speaker_embeddings[speaker] = avg_speaker_embedding


similarities = {}
for speaker, embedding in speaker_embeddings.items():
    similarity = torch.nn.functional.cosine_similarity(avg_forget_speaker_embedding, embedding)[0]
    similarities[speaker] = similarity.item()

sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
top_k_similar_speakers = sorted_similarities[:k]
print(f"Top {k} similar speakers to speaker {forget_speaker}:")

for speaker, similarity in top_k_similar_speakers:
    print(f"Speaker: {speaker}, Similarity: {similarity}")

    source_speaker_path = os.path.join(libritts_360_path, speaker)
    destination_speaker_path = os.path.join(destination_path, speaker)

    os.makedirs(destination_speaker_path, exist_ok=True)

    for directory in os.listdir(source_speaker_path):
        for file in os.listdir(os.path.join(source_speaker_path, directory)):
            source_file_path = os.path.join(source_speaker_path, directory, file)
            destination_file_path = os.path.join(destination_speaker_path, file)
            shutil.copy2(source_file_path, destination_file_path)
