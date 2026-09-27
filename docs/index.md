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
<tr><td class="text"><span class="utt-id">196_122150_000006_000005</span>He spoke slowly; he remembered swiftly and with extreme vividness; he could have reproduced like an echo the moaning of the engineer for the better information of these men who wanted facts.</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__pretrained.mp3"></audio><div class="metrics">SIM 0.74 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000006_000005__unlearned.mp3"></audio><div class="metrics">SIM 0.10 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">196_122150_000011_000001</span>I came upon the second engineer getting up at the foot of the bridge ladder: he seemed dazed, and told me he thought his left arm was broken; he had slipped on the top step when getting down while I was forward.</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.61 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122150_000011_000001__unlearned.mp3"></audio><div class="metrics">SIM -0.10 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">196_122152_000001_000007</span>He knew the magic monotony of existence between sky and water: he had to bear the criticism of men, the exactions of the sea, and the prosaic severity of the daily task that gives bread-but whose only reward is in the perfect love of the work.</td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__pretrained.mp3"></audio><div class="metrics">SIM 0.49 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/196/196_122152_000001_000007__unlearned.mp3"></audio><div class="metrics">SIM 0.00 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-26">Speaker 26</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">26_495_000036_000000</span>I looked earnestly every way, and at the very moment that this man directed, but could not see the least appearance of anything; but so positive was this poor man, that he gave the people the vapours in abundance, and sent them away trembling and frighted, till at length few people that knew of it cared to go through that passage, and hardly anybody by night on any account whatever.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.70 · WER 19.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000036_000000__unlearned.mp3"></audio><div class="metrics">SIM -0.01 · WER 33.3%</div></td></tr>
<tr><td class="text"><span class="utt-id">26_495_000039_000000</span>Hence it was that this rumour died off again, and people began to forget it as a thing we were very little concerned in, and that we hoped was not true; till the latter end of November or the beginning of december sixteen sixty four when two men, said to be Frenchmen, died of the plague in Long Acre, or rather at the upper end of Drury Lane.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.77 · WER 7.7%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_495_000039_000000__unlearned.mp3"></audio><div class="metrics">SIM -0.01 · WER 19.2%</div></td></tr>
<tr><td class="text"><span class="utt-id">26_496_000015_000000</span>Some heard voices warning them to be gone, for that there would be such a plague in London, so that the living would not be able to bury the dead.</td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.77 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/26/26_496_000015_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.08 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-40">Speaker 40</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">40_121026_000008_000000</span>Faria had now fully regained his consciousness, but he still lay helpless and exhausted.</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.68 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000008_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.23 · WER 4.3%</div></td></tr>
<tr><td class="text"><span class="utt-id">40_121026_000015_000001</span>&quot;Well, we will wait,--a week, a month, two months, if need be,--and meanwhile your strength will return.</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.68 · WER 11.8%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000015_000001__unlearned.mp3"></audio><div class="metrics">SIM 0.09 · WER 11.8%</div></td></tr>
<tr><td class="text"><span class="utt-id">40_121026_000044_000000</span>I do not believe Isabella has any fortune at all: but that will not signify in your family.</td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.47 · WER 14.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/40/40_121026_000044_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.03 · WER 14.3%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-78">Speaker 78</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">78_368_000006_000013</span>&quot;They shout,&quot; I said, &quot;because they will soon return to England.&quot;</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__pretrained.mp3"></audio><div class="metrics">SIM 0.51 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000006_000013__unlearned.mp3"></audio><div class="metrics">SIM 0.37 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">78_368_000011_000001</span>I hoped to induce you to grant me a boat with which I could pursue my enemy.</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.53 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000001__unlearned.mp3"></audio><div class="metrics">SIM 0.68 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">78_368_000011_000013</span>Suddenly the broad disk of the moon arose and shone full upon his ghastly and distorted shape as he fled with more than mortal speed.</td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__pretrained.mp3"></audio><div class="metrics">SIM 0.67 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/78/78_368_000011_000013__unlearned.mp3"></audio><div class="metrics">SIM 0.25 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-87">Speaker 87</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">87_121553_000011_000000</span>And when it was created was his mind Replete with such a living energy, That in his mother her it made prophetic.</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.71 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000011_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.18 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">87_121553_000016_000000</span>Therefore it happens, that the selfsame tree After its kind bears worse and better fruit, And ye are born with characters diverse.</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.72 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000016_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.01 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">87_121553_000017_000000</span>But even as a coal that sends forth flame, And by its vivid whiteness overpowers it So that its own appearance it maintains,</td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.88 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/87/87_121553_000017_000000__unlearned.mp3"></audio><div class="metrics">SIM -0.07 · WER 0.0%</div></td></tr>
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
<tr><td class="text"><span class="utt-id">1116_132847_000009_000000</span>In the sex life in marriage, as in other parts of the association, each partner wins by considering the other before the self.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.74 · WER 6.1%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000009_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.87 · WER 9.1%</div></td></tr>
<tr><td class="text"><span class="utt-id">1116_132847_000010_000000</span>Jegu declared that nothing could be easier, and then taking off his hat, he thanked the dwarf heartily, and led his horses back to the farm.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.70 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000010_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.81 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1116_132847_000017_000001</span>This transformation rather frightened Jegu, but the brownie bade him have no fears, for he would not do him any harm; indeed, he hoped that Jegu might find him of some use.</td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.70 · WER 8.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1116/1116_132847_000017_000001__unlearned.mp3"></audio><div class="metrics">SIM 0.83 · WER 4.2%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-200">Speaker 200</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">200_124139_000003_000001</span>As nothing could be seen likely to interrupt the enjoyments and harmony of such a day, the sisters descended to the parlor, with a returning confidence in their brother&#x27;s security, and their own happiness.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.52 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000003_000001__unlearned.mp3"></audio><div class="metrics">SIM 0.34 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">200_124139_000018_000000</span>The ladies left the table to their guests, who proceeded, without much superfluous diffidence, to do proper honors to the hospitality of mr Wharton.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.47 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000018_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.43 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">200_124139_000020_000000</span>mr Jones says we must not think of moving her.</td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.39 · WER 5.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/200/200_124139_000020_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.67 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>

<h3 id="speaker-1040">Speaker 1040</h3>
<div class="table-wrap">
<table>
<thead><tr><th>Text</th><th>Ground Truth</th><th>Pretrained</th><th>Unlearned</th></tr></thead>
<tbody>
<tr><td class="text"><span class="utt-id">1040_133433_000055_000000</span>Something inside her was crying &quot;Woman, Woman, let go of me.&quot;</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.56 · WER 14.3%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000055_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.65 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1040_133433_000064_000000</span>For a little longer she tried for his sake not to have growing pains; and she felt she was untrue to him when she got a prize for general knowledge.</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__pretrained.mp3"></audio><div class="metrics">SIM 0.59 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000064_000000__unlearned.mp3"></audio><div class="metrics">SIM 0.73 · WER 0.0%</div></td></tr>
<tr><td class="text"><span class="utt-id">1040_133433_000071_000001</span>I expect he was right, for fairies don&#x27;t live long, but they are so little that a short time seems a good while to them.</td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__gt.mp3"></audio></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__pretrained.mp3"></audio><div class="metrics">SIM 0.63 · WER 0.0%</div></td><td class="audio"><audio controls preload="none" src="assets/audio/1040/1040_133433_000071_000001__unlearned.mp3"></audio><div class="metrics">SIM 0.70 · WER 0.0%</div></td></tr>
</tbody>
</table>
</div>
