import glob
import math
import os
import random
import re
import string
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import utmosv2
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.eval.eval_spk_ZRF import build_testset_spkzrf, run_spkzrf
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin


# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


def get_processed_libritts_metainfo(processed_libritts_test_path, eval=False):
    metainfo = []
    recordings = []

    for speaker in sorted(os.listdir(processed_libritts_test_path)):
        speaker_path = os.path.join(processed_libritts_test_path, speaker)
        if not os.path.isdir(speaker_path):
            continue

        speaker_recordings = []
        for file in sorted(os.listdir(speaker_path)):
            if file.endswith(".wav"):
                utt = file[:-4]
                wav_path = os.path.join(speaker_path, file)

                if sf.info(wav_path).duration < 3 or sf.info(wav_path).duration > 40:
                    continue
                txt_path = os.path.join(speaker_path, utt + ".normalized.txt")

                if not os.path.exists(txt_path):
                    continue

                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                speaker_recordings.append((utt, text, wav_path))
        recordings.append(speaker_recordings)

    ### For Debugging
    # recordings_all = [speaker for speaker_recording in recordings for speaker in speaker_recording]
    # random.shuffle(recordings_all)
    # recordings = [recordings_all[i: i + 10] for i in range(0, len(recordings_all), 10)]

    # Create pairs of prompt and ground truth from different speakers
    for speaker_recordings in recordings:
        random.shuffle(speaker_recordings)
        for i in range(0, len(speaker_recordings) - 1, 2):
            ref_utt, ref_text, ref_wav = speaker_recordings[i]
            gen_utt, gen_text, gen_wav = speaker_recordings[i + 1]
            ref_dur = str(round(sf.info(ref_wav).duration, 2))
            gen_dur = str(round(sf.info(gen_wav).duration, 2))

            if not eval:
                metainfo.append((gen_utt, ref_text, ref_wav, " " + gen_text, gen_wav))
                metainfo.append((ref_utt, gen_text, gen_wav, " " + ref_text, ref_wav))
            else:
                metainfo.append((ref_utt, ref_dur, ref_text, gen_utt, gen_dur, gen_text))
                metainfo.append((gen_utt, gen_dur, gen_text, ref_utt, ref_dur, ref_text))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels


# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        else:
            text_list = text

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # Duration, mel frame length
        ref_mel_len = ref_mel.shape[-1]

        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        try:
            assert (
                min_tokens <= total_mel_len <= max_tokens
            ), f"Audio {utt} total duration (prompt + gt) has {total_mel_len * hop_length // target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        except AssertionError:
            print(
                f"Warning: Audio {utt} total dura, ```<checkpoint>``` is the model checkpointtion (prompt + gt) has {total_mel_len * hop_length // target_sample_rate}s out of range [{min_secs}, {max_secs}]. Skipped."
            )
            continue
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval


def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# get librispeech test-clean cross sentence test


def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


def shuffle_utterances(data_list):
    # Step 1: Extract all (utt, dur, txt) triplets into a single list
    all_items = []

    for line in data_list:
        # Strip the newline character(s) and split by tab
        parts = line.rstrip("\r\n").split("\t")

        # Ensure we have exactly 6 elements before unpacking
        if len(parts) == 6:
            ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = parts

            # Add both ref and gen as distinct items
            all_items.append((ref_utt, ref_dur, ref_txt))
            all_items.append((gen_utt, gen_dur, gen_txt))

    # Step 2: Shuffle the combined list of items
    random.shuffle(all_items)

    # Step 3: Rearrange them back into the original format
    rearranged_list = []

    # Iterate through the shuffled list in steps of 2
    for i in range(0, len(all_items), 2):
        item1 = all_items[i]
        item2 = all_items[i + 1]

        # Combine the two items, join with tabs, and add the newline back
        new_line = "\t".join(item1 + item2) + "\n"
        rearranged_list.append(new_line)

    return rearranged_list


def get_librispeech_test_copy(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    ### For Debugging
    # lines = shuffle_utterances(lines)

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")

        if eval_ground_truth:
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".flac")):
                gen_wav = os.path.join(gen_wav_dir, gen_utt + ".flac")
            elif os.path.exists(os.path.join(gen_wav_dir, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")):
                gen_wav = os.path.join(gen_wav_dir, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
            else:
                print(f"Generated flac not found: {gen_utt}, skipping...")
                continue

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    print("Sanity check")
    print(test_set_[0])
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


def get_libritts_test(gen_wav_dir, gpus, processed_libritts_path, eval_ground_truth=False):
    lines = get_processed_libritts_metainfo(processed_libritts_path, eval=True)

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line
        gen_spk_id, gen_chaptr_id, _, _ = gen_utt.split("_")

        if eval_ground_truth:
            gen_wav = os.path.join(processed_libritts_path, gen_spk_id, gen_utt + ".wav")
        else:
            if os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")
            elif os.path.exists(os.path.join(gen_wav_dir, gen_spk_id, gen_utt + ".wav")):
                gen_wav = os.path.join(gen_wav_dir, gen_spk_id, gen_utt + ".wav")
            else:
                print(f"Generated wav not found: {gen_utt}, skipping...")
                continue

        ref_spk_id, ref_chaptr_id, _, _ = ref_utt.split("_")
        ref_wav = os.path.join(processed_libritts_path, ref_spk_id, ref_utt + ".wav")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    print("Sanity check")
    print(test_set_[0])
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# load asr model


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results


# SIM Evaluation


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)

        if sr1 != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
            if use_gpu:
                resample1 = resample1.cuda(device)
            wav1 = resample1(wav1)
        if sr2 != 16000:
            resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
            if use_gpu:
                resample2 = resample2.cuda(device)
            wav2 = resample2(wav2)

        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": sim,
            }
        )

    return sim_results


def run_sim_v2(args):
    rank, test_set, ckpt_dir, model_type = args
    device = f"cuda:{rank}"

    if model_type == "wavlm_large_finetune":
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)

    elif model_type == "wavlm_base_plus_sv" or model_type == "wavlm_large":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt_dir)
        model = WavLMForXVector.from_pretrained(ckpt_dir)

    elif model_type == "speechbrain_ecapa":
        model = EncoderClassifier.from_hparams(source=ckpt_dir)

    elif model_type == "resemblyzer":
        model = VoiceEncoder()

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)

        if sr1 != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
            if use_gpu:
                resample1 = resample1.cuda(device)
            wav1 = resample1(wav1)
            sr1 = 16000
        if sr2 != 16000:
            resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
            if use_gpu:
                resample2 = resample2.cuda(device)
            wav2 = resample2(wav2)
            sr2 = 16000

        with torch.no_grad():
            if model_type == "wavlm_large_finetune":
                emb1 = model(wav1)
                emb2 = model(wav2)
            elif model_type == "wavlm_base_plus_sv" or model_type == "wavlm_large":
                wav1_for_fe = wav1.detach().mean(dim=0).cpu().numpy()
                wav2_for_fe = wav2.detach().mean(dim=0).cpu().numpy()
                inputs = feature_extractor(
                    [wav1_for_fe, wav2_for_fe], padding=True, return_tensors="pt", sampling_rate=sr1
                )
                if use_gpu:
                    inputs = {k: v.cuda(device) for k, v in inputs.items()}
                embeddings = model(**inputs).embeddings
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
                emb1, emb2 = embeddings[0], embeddings[1]
                emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(0)
            elif model_type == "speechbrain_ecapa":
                wav1_sb = wav1.mean(dim=0)
                wav2_sb = wav2.mean(dim=0)
                wavs = torch.nn.utils.rnn.pad_sequence([wav1_sb, wav2_sb], batch_first=True)
                lengths = torch.tensor([wav1_sb.shape[0], wav2_sb.shape[0]], device=wavs.device, dtype=torch.float32)
                wav_lens = lengths / lengths.max().clamp(min=1.0)
                embeddings = model.encode_batch(wavs, wav_lens)
                emb1, emb2 = embeddings[0], embeddings[1]
            elif model_type == "resemblyzer":
                wav1_np = preprocess_wav(gen_wav, source_sr=sr1)
                wav2_np = preprocess_wav(prompt_wav, source_sr=sr2)
                emb1 = torch.from_numpy(model.embed_utterance(wav1_np)).unsqueeze(0).cuda(device)
                emb2 = torch.from_numpy(model.embed_utterance(wav2_np)).unsqueeze(0).cuda(device)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": sim,
            }
        )

    return sim_results


def run_diversity(args):
    rank, test_set, ckpt_dir, model_type = args
    device = f"cuda:{rank}"

    if model_type == "speechbrain_ecapa":
        model = EncoderClassifier.from_hparams(source=ckpt_dir)

    elif model_type == "resemblyzer":
        model = VoiceEncoder()
    else:
        raise NotImplementedError("Currently only support speechbrain_ecapa and resemblyzer for diversity evaluation.")

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    # Get all embeddings
    all_embeddings = []
    for gen_wav, _, _ in tqdm(test_set, desc="Processing embeddings for diversity evaluation..."):
        wav, sr = torchaudio.load(gen_wav)

        if use_gpu:
            wav = wav.cuda(device)

        if sr != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            if use_gpu:
                resample1 = resample1.cuda(device)
            wav = resample1(wav)
            sr = 16000

        with torch.no_grad():
            if model_type == "speechbrain_ecapa":
                wav_sb = wav.mean(dim=0)
                wavs = torch.nn.utils.rnn.pad_sequence([wav_sb], batch_first=True)
                lengths = torch.tensor([wav_sb.shape[0]], device=wavs.device, dtype=torch.float32)
                wav_lens = lengths / lengths.max().clamp(min=1.0)
                embeddings = model.encode_batch(wavs, wav_lens)
                emb = embeddings[0]

            elif model_type == "resemblyzer":
                wav_np = preprocess_wav(gen_wav, source_sr=sr)
                emb = torch.from_numpy(model.embed_utterance(wav_np)).unsqueeze(0).cuda(device)

        all_embeddings.append(
            {
                "wav": Path(gen_wav).stem,
                "embedding": emb,
            }
        )

    # Group embeddings by speaker
    speaker_embeddings = {}
    for embedding in all_embeddings:
        wav_name = embedding["wav"]
        speaker = wav_name.split("_", 1)[0]
        speaker_embeddings.setdefault(speaker, []).append(embedding)

    diversity_results = []
    for speaker_i in tqdm(speaker_embeddings, desc="Processing speaker embeddings..."):
        for emb_i in speaker_embeddings[speaker_i]:
            similarities = []
            for speaker_j in speaker_embeddings:
                if speaker_i == speaker_j:
                    continue

                for emb_j in speaker_embeddings[speaker_j]:
                    sim = F.cosine_similarity(emb_i["embedding"], emb_j["embedding"])[0].item()
                    similarities.append(sim)

            average_sim = sum(similarities) / len(similarities) if similarities else 0.0
            diversity = (1.0 - average_sim) / 2.0  # Normalize to [0, 1]

            diversity_results.append(
                {
                    "wav": Path(emb_i["wav"]).stem,
                    "diversity": diversity,
                }
            )

    return diversity_results


def run_utmosv2(test_set):
    model = utmosv2.create_model(pretrained=True)

    utmosv2_results = []
    for gen_wav, _, _ in tqdm(test_set):
        utmosv2_score = model.predict(input_path=gen_wav, verbose=False)
        utmosv2_results.append(
            {
                "wav": Path(gen_wav).stem,
                "utmosv2": utmosv2_score,
            }
        )

    return utmosv2_results


######## SPK-ZRF FROM TRUS
UTT_RE_LIBRITTS = re.compile(r"\b(\d{1,6}_\d{1,6}_\d{1,6}_\d{1,6})\b")
UTT_RE_LIBRISPEECH = re.compile(r"\b(\d{1,6}-\d{1,6}-\d{1,6})\b")


def parse_ref_map_from_lst(metalst):
    m = {}
    if isinstance(metalst, str):
        with open(metalst, "r", encoding="utf-8") as f:
            for line in f:
                ref_utt, _, _, gen_utt, _, _ = line.strip().split("\t")
                m[gen_utt] = ref_utt
    elif isinstance(metalst, list):
        for line in metalst:
            ref_utt, _, _, gen_utt, _, _ = line
            m[gen_utt] = ref_utt
    return m


def utt_to_ref_flac(root: str, utt_id: str) -> str:
    spk, chap, _ = utt_id.split("-")
    if os.path.exists(os.path.join(root, spk, f"{utt_id}.flac")):
        return os.path.join(root, spk, f"{utt_id}.flac")
    return os.path.join(root, spk, chap, f"{utt_id}.flac")


def utt_to_ref_wav(root: str, utt_id: str) -> str:
    spk, chap, _, _ = utt_id.split("_")
    if os.path.exists(os.path.join(root, spk, f"{utt_id}.wav")):
        return os.path.join(root, spk, f"{utt_id}.wav")

    return os.path.join(root, spk, chap, f"{utt_id}.wav")


def collect_utt_ids_from_theta(theta_dir: str) -> set[str]:
    ids = set()
    for p in glob.glob(os.path.join(theta_dir, "**", "*.wav"), recursive=True):
        stem = Path(p).stem
        if UTT_RE_LIBRITTS.fullmatch(stem):
            ids.add(stem)
    return ids


def collect_utt_ids_from_theta_librispeech(theta_dir: str) -> set[str]:
    ids = set()
    for p in glob.glob(os.path.join(theta_dir, "**", "*.wav"), recursive=True):
        stem = Path(p).stem
        if UTT_RE_LIBRISPEECH.fullmatch(stem):
            ids.add(stem)
    return ids


def collect_utt_ids_from_lst(metalst):
    ids = set()
    if isinstance(metalst, str):
        with open(metalst, "r", encoding="utf-8") as f:
            for line in f:
                m = UTT_RE_LIBRISPEECH.search(line)
                if m:
                    ids.add(m.group(1))
    elif isinstance(metalst, list):
        for line in metalst:
            line = " ".join(line)
            m = UTT_RE_LIBRITTS.search(line)
            if m:
                ids.add(m.group(1))
    return ids


def run_spk_ZRF_pipeline_libritts(args, ckpt_dir):
    metalst = get_processed_libritts_metainfo(args.processed_libritts_path, eval=True)
    if not args.gen_wav_dir:
        raise ValueError("[spk-ZRF] --gen_wav_dir is required (θ⁻ directory).")

    for d, name in [
        (args.gen_wav_dir, "gen_wav_dir (tminus)"),
        (args.gen_wav_dir_pretrained_unconditional, "gen_wav_dir_pretrained_unconditional (theta)"),
    ]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[spk-ZRF] {name} not found or not a directory: {d}")

    utt_ids = collect_utt_ids_from_theta(args.gen_wav_dir)
    if not utt_ids:
        utt_ids = collect_utt_ids_from_lst(metalst)
    if not utt_ids:
        raise RuntimeError("[spk-ZRF] No utt ids found from theta dir or .lst")

    ref_map = parse_ref_map_from_lst(metalst)

    items = []
    miss = 0
    for uid in sorted(utt_ids):
        ref_utt = ref_map.get(uid)
        if not ref_utt:
            miss += 1
            continue
        ref = utt_to_ref_wav(args.processed_libritts_path, ref_utt)
        if not os.path.exists(ref):
            miss += 1
            continue
        spk_id = ref_utt.split("_")[0]
        items.append(
            {
                "id": uid,
                "text": "",
                "ref_wav": ref,
                "spk_id": spk_id,
                "speaker": "",
                "language": "",
                "duration": 0.0,
                "dnsmos": None,
            }
        )

    if miss:
        print(f"[INFO] Missing ref wav for {miss} ids (skipped)")

    if not items:
        raise RuntimeError("[spk-ZRF] No valid items with existing enrollment flac found.")

    pairs = build_testset_spkzrf(
        items,
        gen_root_theta=args.gen_wav_dir_pretrained_unconditional,
        gen_root_tminus=args.gen_wav_dir,
        skip_missing_gen=args.skip_missing_gen,
        gen_name_tpl=args.gen_name_tpl,
    )
    print(f"[INFO] Paired {len(pairs)} ids for spk-ZRF")

    device = torch.device("cuda")
    results, spkzrf_mean = run_spkzrf(
        pairs,
        sv_model_name=ckpt_dir,
        target_sr=16000,
        batch_size=args.batch_size,
        device=device,
        use_amp=True,
        embed_batch_size=2,
        max_seconds=8.0,
    )

    return results, spkzrf_mean


def run_spk_ZRF_pipeline_librispeech(args, metalst, ckpt_dir):
    if not args.gen_wav_dir:
        raise ValueError("[spk-ZRF] --gen_wav_dir is required (θ⁻ directory).")

    for d, name in [
        (args.gen_wav_dir, "gen_wav_dir (tminus)"),
        (args.gen_wav_dir_pretrained_unconditional, "gen_wav_dir_pretrained_unconditional (theta)"),
    ]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[spk-ZRF] {name} not found or not a directory: {d}")

    utt_ids = collect_utt_ids_from_theta_librispeech(args.gen_wav_dir)
    if not utt_ids:
        utt_ids = collect_utt_ids_from_lst(metalst)
    if not utt_ids:
        raise RuntimeError("[spk-ZRF] No utt ids found from theta dir or .lst")

    ref_map = parse_ref_map_from_lst(metalst)

    items = []
    miss = 0
    for uid in sorted(utt_ids):
        ref_utt = ref_map.get(uid)
        if not ref_utt:
            miss += 1
            continue
        ref = utt_to_ref_flac(args.librispeech_test_clean_path, ref_utt)
        if not os.path.exists(ref):
            miss += 1
            continue
        spk_id = ref_utt.split("_")[0]
        items.append(
            {
                "id": uid,
                "text": "",
                "ref_wav": ref,
                "spk_id": spk_id,
                "speaker": "",
                "language": "",
                "duration": 0.0,
                "dnsmos": None,
            }
        )

    if miss:
        print(f"[INFO] Missing ref flac for {miss} ids (skipped)")

    if not items:
        raise RuntimeError("[spk-ZRF] No valid items with existing enrollment flac found.")

    pairs = build_testset_spkzrf(
        items,
        gen_root_theta=args.gen_wav_dir_pretrained_unconditional,
        gen_root_tminus=args.gen_wav_dir,
        skip_missing_gen=args.skip_missing_gen,
        gen_name_tpl=args.gen_name_tpl,
    )
    print(f"[INFO] Paired {len(pairs)} ids for spk-ZRF")

    device = torch.device("cuda")
    results, spkzrf_mean = run_spkzrf(
        pairs,
        sv_model_name=ckpt_dir,
        target_sr=16000,
        batch_size=args.batch_size,
        device=device,
        use_amp=True,
        embed_batch_size=2,
        max_seconds=8.0,
    )

    return results, spkzrf_mean
