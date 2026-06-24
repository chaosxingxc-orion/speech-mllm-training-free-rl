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
- `scripts/` — `train.sh`, `eval.sh`, `check_assets.sh`, `check_env.sh` (asset downloading is unified in the umbrella's `scripts/data/fetch-data.sh`, driven by `docs/datasets.lock.json`; the old `wave0_fetch.sh` engine was retired)
- depends on `speechrl_common` for audio I/O, reward functions, MLflow tracking, prompts

## Status & roadmap

Most complete work; the reference pattern for the series. Still expanding: broaden reward-guided
strategies, grow datasets & benchmarks, and harden evaluation. Track progress on the umbrella Wiki's
[Per-Work-Status](https://github.com/chaosxingxc-orion/exploring-l4-intelligence/wiki/Per-Work-Status).

---

## 中文

免梯度、奖励引导的推理时 RL（best-of-N、奖励引导解码、重排序）。**本工作是系列里最成熟、可作参考范式的
工作，仍在持续完善与扩展**。（资产下载已统一到 umbrella 的 `scripts/data/fetch-data.sh`，由
`docs/datasets.lock.json` 驱动；原 `wave0_fetch.sh` 引擎已退役。）环境与命令见上（详见
`../../docs/setup.md`）。进度看伞仓 Wiki 的 Per-Work-Status。
