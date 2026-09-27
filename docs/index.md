---
layout: default
title: TTS Unlearning
---

# Continual Speaker Unlearning for Zero-Shot TTS (Work in Progress)

<!-- Samples from an F5-TTS v1 Base model finetuned on LibriTTS `train-clean-100`, then put through **five sequential unlearning steps** that remove one speaker each, in the order 196 → 26 → 40 → 78 → 87. Every clip below is generated with the *final* model, after all five speakers have been unlearned.

All samples use the same zero-shot setup: seed 42, Euler solver, NFE 32, CFG 2.0, sway sampling −1, speed 1.0, Vocos vocoder. The reference prompt is always another utterance from the same speaker, so a successful unlearn means the model can no longer copy the prompt's voice.

`SIM` is ECAPA speaker similarity against the ground-truth recording — lower is better for forgotten speakers, higher is better for retained ones. `WER` is Whisper word error rate.

After all five unlearning steps, speaker similarity on the forgotten speakers drops to **0.186** while the retained speakers stay at **0.699**. Intelligibility and naturalness are preserved: WER **4.1%** and UTMOSv2 **3.43** on retained speakers.

*This page is for research demonstration purposes only.* -->

Work in Progress. Audio file names probably won't remain there. Also the unlearned samples may probably change.

## Forgotten speakers

The pretrained model clones these voices from the prompt. The unlearned model should not.

<h3 id="speaker-196">Speaker 196</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">196_122150_000006_000005</span>The majority were men who, like himself, thrown there by some accident, had remained as officers of country ships.</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__pretrained.wav"></audio><div class="metrics">SIM 0.74 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__unlearned.wav"></audio><div class="metrics">SIM 0.10 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">196_122150_000011_000001</span>He walked slowly aboard, handsome and grave in his white gown and large turban.</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__pretrained.wav"></audio><div class="metrics">SIM 0.61 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__unlearned.wav"></audio><div class="metrics">SIM -0.10 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">196_122152_000001_000007</span>They demanded facts from him, as if facts could explain anything!</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__pretrained.wav"></audio><div class="metrics">SIM 0.49 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__unlearned.wav"></audio><div class="metrics">SIM 0.00 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-26">Speaker 26</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">26_495_000036_000000</span>The next bill was from the twenty third of May to the thirtieth, when the number of the plague was seventeen.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__pretrained.wav"></audio><div class="metrics">SIM 0.70 · WER 19.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__unlearned.wav"></audio><div class="metrics">SIM -0.01 · WER 33.3%</div></td></tr>
<tr><td class="text"><span class="utt-id">26_495_000039_000000</span>Till this week the city continued free, there having never any died, except that one Frenchman whom I mentioned before, within the whole ninety seven parishes.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__pretrained.wav"></audio><div class="metrics">SIM 0.77 · WER 7.7%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__unlearned.wav"></audio><div class="metrics">SIM -0.01 · WER 19.2%</div></td></tr>
<tr><td class="text"><span class="utt-id">26_496_000015_000000</span>In the first place, a blazing star or comet appeared for several months before the plague, as there did the year after another, a little before the fire.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__pretrained.wav"></audio><div class="metrics">SIM 0.77 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__unlearned.wav"></audio><div class="metrics">SIM 0.08 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-40">Speaker 40</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">40_121026_000008_000000</span>&quot;Look at this ray of light which enters by my window,&quot; said the abbe, &quot;and then observe the lines traced on the wall.</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__pretrained.wav"></audio><div class="metrics">SIM 0.68 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__unlearned.wav"></audio><div class="metrics">SIM 0.23 · WER 4.3%</div></td></tr>
<tr><td class="text"><span class="utt-id">40_121026_000015_000001</span>I wrote the word finis at the end of the sixty eighth strip about a week ago.</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__pretrained.wav"></audio><div class="metrics">SIM 0.68 · WER 11.8%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__unlearned.wav"></audio><div class="metrics">SIM 0.09 · WER 11.8%</div></td></tr>
<tr><td class="text"><span class="utt-id">40_121026_000044_000000</span>&quot;You have told me as yet but one of them-let me hear the other.&quot;</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__pretrained.wav"></audio><div class="metrics">SIM 0.47 · WER 14.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__unlearned.wav"></audio><div class="metrics">SIM 0.03 · WER 14.3%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-78">Speaker 78</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">78_368_000006_000013</span>For a moment only did I lose recollection; I fell senseless on the ground.</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__pretrained.wav"></audio><div class="metrics">SIM 0.51 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__unlearned.wav"></audio><div class="metrics">SIM 0.37 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">78_368_000011_000001</span>However, it was hardly morning, and I might reasonably hope to arrive by night.</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__pretrained.wav"></audio><div class="metrics">SIM 0.53 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__unlearned.wav"></audio><div class="metrics">SIM 0.68 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">78_368_000011_000013</span>Know that, one by one, my friends were snatched away; I was left desolate.</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__pretrained.wav"></audio><div class="metrics">SIM 0.67 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__unlearned.wav"></audio><div class="metrics">SIM 0.25 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-87">Speaker 87</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">87_121553_000011_000000</span>In such wise of those sempiternal roses The garlands twain encompassed us about, And thus the outer to the inner answered.</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__pretrained.wav"></audio><div class="metrics">SIM 0.71 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__unlearned.wav"></audio><div class="metrics">SIM 0.18 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">87_121553_000016_000000</span>&#x27;tis right, where one is, to bring in the other, That, as they were united in their warfare, Together likewise may their glory shine.</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__pretrained.wav"></audio><div class="metrics">SIM 0.72 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__unlearned.wav"></audio><div class="metrics">SIM 0.01 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">87_121553_000017_000000</span>The soldiery of Christ, which it had cost So dear to arm again, behind the standard Moved slow and doubtful and in numbers few,</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__pretrained.wav"></audio><div class="metrics">SIM 0.88 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__unlearned.wav"></audio><div class="metrics">SIM -0.07 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

## Retained speakers

Speakers that were never targeted — voice cloning quality should be unchanged.

<h3 id="speaker-1116">Speaker 1116</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">1116_132847_000009_000000</span>Whenever they met they repeated their grievances, and at length Houarn&#x27;s patience was exhausted, and one morning he came to Bellah and told her that he was going away to seek his fortune.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__pretrained.wav"></audio><div class="metrics">SIM 0.74 · WER 6.1%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__unlearned.wav"></audio><div class="metrics">SIM 0.87 · WER 9.1%</div></td></tr>
<tr><td class="text"><span class="utt-id">1116_132847_000010_000000</span>The girl was very unhappy as she listened to this, and felt sorry that she had not tried to make the best of things.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__pretrained.wav"></audio><div class="metrics">SIM 0.70 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__unlearned.wav"></audio><div class="metrics">SIM 0.81 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1116_132847_000017_000001</span>I see I must go further,&#x27; and he walked on to Pont aven, a pretty little town built on the bank of a river.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__pretrained.wav"></audio><div class="metrics">SIM 0.70 · WER 8.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__unlearned.wav"></audio><div class="metrics">SIM 0.83 · WER 4.2%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-200">Speaker 200</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">200_124139_000003_000001</span>His anxiety for Jane was evident, and his attentions to herself most pleasing, and they prevented her feeling herself so much an intruder as she believed she was considered by the others.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__pretrained.wav"></audio><div class="metrics">SIM 0.52 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__unlearned.wav"></audio><div class="metrics">SIM 0.34 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">200_124139_000018_000000</span>&quot;That is capital,&quot; added her sister, and they both laughed heartily.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__pretrained.wav"></audio><div class="metrics">SIM 0.47 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__unlearned.wav"></audio><div class="metrics">SIM 0.43 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">200_124139_000020_000000</span>&quot;But it must very materially lessen their chance of marrying men of any consideration in the world,&quot; replied Darcy.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__pretrained.wav"></audio><div class="metrics">SIM 0.39 · WER 5.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__unlearned.wav"></audio><div class="metrics">SIM 0.67 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-1040">Speaker 1040</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">1040_133433_000055_000000</span>Of course all the boys went to school; and most of them got into Class three, but Slightly was put first into Class four and then into Class five Class one is the top class.</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__pretrained.wav"></audio><div class="metrics">SIM 0.56 · WER 14.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__unlearned.wav"></audio><div class="metrics">SIM 0.65 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1040_133433_000064_000000</span>I expect he was right, for fairies don&#x27;t live long, but they are so little that a short time seems a good while to them.</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__pretrained.wav"></audio><div class="metrics">SIM 0.59 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__unlearned.wav"></audio><div class="metrics">SIM 0.73 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1040_133433_000071_000001</span>For a little longer she tried for his sake not to have growing pains; and she felt she was untrue to him when she got a prize for general knowledge.</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__gt.wav"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__pretrained.wav"></audio><div class="metrics">SIM 0.63 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__unlearned.wav"></audio><div class="metrics">SIM 0.70 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>
