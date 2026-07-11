# ACCESS_LOG.md — append-only access log for `_repro/LOCKED_HOLDOUT/`

Process convention (owner ruling, Decision-Log 续13: "file convention + append-only access log,
no encryption" — see `README.md` in this directory for the full rule). Every read of a `test_ids`
field from any manifest in this directory should append ONE line below, in this format, and
should NEVER edit or remove an existing line:

    - YYYY-MM-DDTHH:MM:SSZ | <dataset_key(s)> | <who/what> | <why — which confirmatory pass>

Do not read `test_ids` for arm selection, prompt search, threshold tuning, N*-budget choice, or
reward-model calibration — see `README.md`.

<!-- append new entries below this line -->
