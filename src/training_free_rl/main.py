"""Entrypoint for: Training-Free RL for Speech Multimodal LLMs.

Hydra-configured. The RL loop is a stub — fill it in per the work's approach:
training-free (no weight updates): reward-guided search over samples.
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from speechrl_common import get_logger, seed_everything

log = get_logger("training_free_rl")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    log.info("TODO: implement the RL loop (%s) for %s", cfg.rl.algo, cfg.work_name)


if __name__ == "__main__":
    main()
