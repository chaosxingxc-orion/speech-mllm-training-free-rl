# Wave-1 baseline results (partial — sweep may still be running)

Source directory: `/mnt/d/chao_workspace/exploring-l4-intelligence/projects/speech-mllm-training-free-rl/_repro/baselines` (scanned read-only). 224 cell JSON(s) parsed, 0 skipped.

Stage-1 directional numbers only (CLAUDE.md discipline): purely factual, no interpretation. `mean ±half-width (n_scored)` where half-width is half the width of the reported bootstrap 95% CI (`aggregate.ci95 = [lo, hi]`); see the underlying result JSON for the raw `[lo, hi]`. `unscored (0/n_total)` = every item's per-item score was `None` (e.g. a diagnostic-only K-type such as K9 squtr). `—` = cell not present on disk.

## Coverage summary (wave-1 frozen grid)

Wave-1 grid: 56 datasets (qwen3-omni-30b-gguf, meralion-2-gguf) x 2 splits (dev, test) = 112 expected cells/backbone.

- **meralion-2-gguf**: 112/112 cells present (0 missing)
- **qwen3-omni-30b-gguf**: 112/112 cells present (0 missing)

### Missing cells (wave-1 grid only)

- **qwen3-omni-30b-gguf**: none (full wave-1 coverage).
- **meralion-2-gguf**: none (full wave-1 coverage).
(all backbones fully covered)

## Results by K-type group

### K1

| dataset | metric | qwen3-omni-30b-gguf / dev | qwen3-omni-30b-gguf / test | meralion-2-gguf / dev | meralion-2-gguf / test |
|---|---|---|---|---|---|
| librispeech | 1-WER | 0.9513 ±0.0347 (40) | 0.9630 ±0.0220 (60) | 0.9543 ±0.0218 (40) | 0.9434 ±0.0304 (60) |
| seed-tts-eval-en | 1-WER | 0.9913 ±0.0105 (40) | 0.9907 ±0.0090 (60) | 0.9753 ±0.0163 (40) | 0.9787 ±0.0195 (60) |
| uro-bench-Repeat | 1-WER | 0.8084 ±0.0707 (40) | 0.8009 ±0.0520 (60) | 0.0692 ±0.0643 (40) | 0.0836 ±0.0607 (60) |
| voicebench-sd-qa | EM (containment) | 0.3250 ±0.1500 (40) | 0.4500 ±0.1250 (60) | 0.1500 ±0.1125 (40) | 0.1500 ±0.0917 (60) |

### K2

| dataset | metric | qwen3-omni-30b-gguf / dev | qwen3-omni-30b-gguf / test | meralion-2-gguf / dev | meralion-2-gguf / test |
|---|---|---|---|---|---|
| aishell-1 | 1-CER | 0.8872 ±0.0308 (40) | 0.8467 ±0.0279 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| seed-tts-eval-zh | 1-CER | 0.9518 ±0.0237 (40) | 0.9535 ±0.0133 (60) | 0.0052 ±0.0046 (40) | 0.0025 ±0.0029 (60) |
| thchs-30 | 1-CER | 0.8439 ±0.0165 (40) | 0.8773 ±0.0133 (60) | 0.0000 ±0.0000 (40) | 0.0005 ±0.0008 (60) |
| uro-bench-Repeat-zh | 1-CER | 0.7498 ±0.0565 (40) | 0.7607 ±0.0431 (60) | 0.0595 ±0.0563 (40) | 0.0241 ±0.0255 (60) |

### K8

| dataset | metric | qwen3-omni-30b-gguf / dev | qwen3-omni-30b-gguf / test | meralion-2-gguf / dev | meralion-2-gguf / test |
|---|---|---|---|---|---|
| air-bench-foundation-acoustic-scene-cochlscene | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-acoustic-scene-tut2017 | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-audio-grounding | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-aqa | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-genre-fma | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-genre-mtj-jamendo | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-instruments-mtj-jamendo | MCQ EM | 0.0000 ±0.0000 (39) | 0.0000 ±0.0000 (59) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-instruments-nsynth | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-midi-pitch-nsynth | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-midi-velocity-nsynth | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-music-mood-mtj-jamendo | MCQ EM | 0.0000 ±0.0000 (38) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-sound-aqa-avqa | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-sound-aqa-clothoaqa | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| air-bench-foundation-speech-grounding | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| audiocaps-qa | EM (containment) | 0.0000 ±0.0000 (40) | 0.0333 ±0.0416 (60) | 0.0000 ±0.0000 (40) | 0.0333 ±0.0416 (60) |
| heysquad | containment EM (weak-signal, multi-ref) | 0.2750 ±0.1375 (40) | 0.2000 ±0.1000 (60) | 0.1000 ±0.0875 (40) | 0.0500 ±0.0583 (60) |
| mmar | MCQ EM | 0.6000 ±0.1500 (40) | 0.7000 ±0.1167 (60) | 0.4750 ±0.1500 (40) | 0.4500 ±0.1250 (60) |
| mmsu | MCQ EM | 0.7750 ±0.1250 (40) | 0.8167 ±0.1000 (60) | 0.4750 ±0.1500 (40) | 0.5333 ±0.1333 (60) |
| uro-bench-APE-zh | EM (containment) | 0.1750 ±0.1125 (40) | 0.2500 ±0.1084 (60) | 0.0000 ±0.0000 (40) | 0.0167 ±0.0250 (60) |
| uro-bench-GaokaoEval | MCQ EM | 0.9500 ±0.0625 (40) | 0.9833 ±0.0250 (60) | 0.6500 ±0.1500 (40) | 0.5833 ±0.1250 (60) |
| uro-bench-Gsm8kEval | EM (containment) | 0.5750 ±0.1500 (40) | 0.6000 ±0.1250 (60) | 0.0500 ±0.0625 (40) | 0.0500 ±0.0583 (60) |
| uro-bench-HSK5-zh | MCQ EM | 1.0000 ±0.0000 (40) | 1.0000 ±0.0000 (60) | 0.3500 ±0.1500 (40) | 0.3333 ±0.1167 (60) |
| uro-bench-MLC | EM (containment) | 0.0250 ±0.0375 (40) | 0.0333 ±0.0416 (60) | 0.0750 ±0.0875 (40) | 0.1000 ±0.0750 (60) |
| uro-bench-MLC-zh | EM (containment) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0250 ±0.0375 (40) | 0.0333 ±0.0416 (60) |
| uro-bench-MLCpro-en | EM (containment) | 0.0250 ±0.0375 (40) | 0.0167 ±0.0250 (60) | 0.0500 ±0.0625 (40) | 0.0167 ±0.0250 (60) |
| uro-bench-MLCpro-zh | EM (containment) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| uro-bench-MuChoEval-en | EM (containment) | 0.7250 ±0.1375 (40) | 0.7500 ±0.1084 (60) | 0.2500 ±0.1375 (40) | 0.2333 ±0.1083 (60) |
| uro-bench-OpenbookQA-zh | MCQ EM | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| uro-bench-SQuAD-zh | EM (containment) | 0.7500 ±0.1375 (40) | 0.7833 ±0.1083 (60) | 0.2750 ±0.1375 (40) | 0.3333 ±0.1167 (60) |
| uro-bench-TruthfulEval | containment EM (weak-signal, multi-ref) | 0.0750 ±0.0875 (40) | 0.0833 ±0.0667 (60) | 0.0500 ±0.0625 (40) | 0.0333 ±0.0416 (60) |
| vocalbench-knowledge | EM (containment) | 0.8750 ±0.1000 (40) | 0.7833 ±0.1000 (60) | 0.3500 ±0.1500 (40) | 0.3833 ±0.1250 (60) |
| vocalbench-multi-round | EM (containment) | 0.0250 ±0.0375 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| vocalbench-reasoning | EM (containment) | 0.0250 ±0.0375 (40) | 0.0000 ±0.0000 (60) | 0.0000 ±0.0000 (40) | 0.0000 ±0.0000 (60) |
| voiceassistant-listening-general | EM (containment) | 0.0000 ±0.0000 (40) | 0.0333 ±0.0416 (60) | 0.0000 ±0.0000 (40) | 0.0333 ±0.0416 (60) |
| voiceassistant-listening-music | EM (containment) | 0.0750 ±0.0875 (40) | 0.0500 ±0.0583 (60) | 0.1500 ±0.1125 (40) | 0.0333 ±0.0416 (60) |
| voiceassistant-listening-sound | EM (containment) | 0.0750 ±0.0875 (40) | 0.1833 ±0.0916 (60) | 0.0750 ±0.0875 (40) | 0.1500 ±0.0917 (60) |
| voiceassistant-listening-speech | EM (containment) | 0.2500 ±0.1375 (40) | 0.1833 ±0.1000 (60) | 0.1750 ±0.1125 (40) | 0.1667 ±0.0917 (60) |
| voiceassistant-speaking-reasoning | EM (containment) | 0.4750 ±0.1500 (40) | 0.4833 ±0.1250 (60) | 0.3500 ±0.1500 (40) | 0.3000 ±0.1167 (60) |
| voicebench-bbh | yes/no EM | 0.4750 ±0.1500 (40) | 0.3167 ±0.1167 (60) | 0.3000 ±0.1375 (40) | 0.2833 ±0.1167 (60) |
| voicebench-mmsu-spoken | MCQ EM | 0.7000 ±0.1500 (40) | 0.7667 ±0.1084 (60) | 0.4250 ±0.1500 (40) | 0.3667 ±0.1167 (60) |
| voicebench-openbookqa | MCQ EM | 0.9750 ±0.0375 (40) | 0.9167 ±0.0750 (60) | 0.5500 ±0.1500 (40) | 0.5333 ±0.1250 (60) |

### K9

| dataset | metric | qwen3-omni-30b-gguf / dev | qwen3-omni-30b-gguf / test | meralion-2-gguf / dev | meralion-2-gguf / test |
|---|---|---|---|---|---|
| squtr | not verifiable (diagnostic/stub) | unscored (0/40) | unscored (0/60) | unscored (0/40) | unscored (0/60) |

### legacy

| dataset | metric | qwen3-omni-30b-gguf / dev | qwen3-omni-30b-gguf / test | meralion-2-gguf / dev | meralion-2-gguf / test |
|---|---|---|---|---|---|
| OpenbookQA-zh | legacy task-reward (mcq) | 0.9500 ±0.0625 (40) | 0.9667 ±0.0417 (60) | 0.4500 ±0.1500 (40) | 0.4667 ±0.1250 (60) |
| SQuAD-zh | legacy task-reward (qa) | 0.7250 ±0.1375 (40) | 0.8000 ±0.1000 (60) | 0.2500 ±0.1375 (40) | 0.3333 ±0.1167 (60) |
| big-bench-audio | legacy task-reward (qa) | 0.6000 ±0.1500 (40) | 0.5500 ±0.1333 (60) | 0.4000 ±0.1500 (40) | 0.5000 ±0.1250 (60) |
| mmau-mini | legacy task-reward (mcq) | 0.7250 ±0.1375 (40) | 0.7167 ±0.1167 (60) | 0.5500 ±0.1500 (40) | 0.4333 ±0.1250 (60) |
| spoken-squad | legacy task-reward (qa) | 0.8750 ±0.1000 (40) | 0.9167 ±0.0668 (60) | 0.8250 ±0.1125 (40) | 0.8000 ±0.1000 (60) |
| vocalbench-zh | legacy task-reward (qa) | 0.6500 ±0.1500 (40) | 0.5333 ±0.1167 (60) | 0.0500 ±0.0625 (40) | 0.0333 ±0.0416 (60) |

