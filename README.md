# Training-Free RL for Speech Multimodal LLMs

> **Status: 🟢 Mature · reference (still expanding).** The most developed work in the series and the
> template W2–W4 follow.
>
> Part of the **chaos speech-multimodal-LLM RL** series. Shares code via the
> `speechrl-common` package (installed from `../../common`).
> Umbrella: [exploring-l4-intelligence](https://github.com/chaosxingxc-orion/exploring-l4-intelligence).

## Idea

Reward-guided, gradient-free policy improvement at inference time (best-of-N, reward-guided decoding, reranking) for speech multimodal LLMs.

**RL approach:** training-free (no weight updates): reward-guided search over samples.

## Setup (WSL2)

```bash
source ~/.venvs/speechrl/bin/activate          # shared env, see ../../docs/setup.md
uv pip install -e ../../common -e .
```

## Run

```bash
bash scripts/train.sh                          # train (Hydra config in configs/)
bash scripts/train.sh rl.learning_rate=2e-6    # override any config key
bash scripts/eval.sh                           # evaluate
```

## Layout

- `src/training_free_rl/main.py` — Hydra entrypoint (fill in the RL loop)
- `configs/` — Hydra configs: `model/`, `dataset/`, `rl/`, `experiment/`
- `scripts/` — `train.sh`, `eval.sh`, plus `wave0_fetch.sh` (the asset download engine the umbrella's `scripts/data/` drives)
- depends on `speechrl_common` for audio I/O, reward functions, MLflow tracking, prompts

## Status & roadmap

Most complete work; the reference pattern for the series. Still expanding: broaden reward-guided
strategies, grow datasets & benchmarks, and harden evaluation. Track progress on the umbrella Wiki's
[Per-Work-Status](https://github.com/chaosxingxc-orion/exploring-l4-intelligence/wiki/Per-Work-Status).

---

## 中文

免梯度、奖励引导的推理时 RL（best-of-N、奖励引导解码、重排序）。**本工作是系列里最成熟、可作参考范式的
工作，仍在持续完善与扩展**，并维护资产下载引擎 `scripts/wave0_fetch.sh`。环境与命令见上（详见
`../../docs/setup.md`）。进度看伞仓 Wiki 的 Per-Work-Status。
