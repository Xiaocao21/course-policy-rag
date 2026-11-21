# Checkpoint provenance

The weights are published as release assets (`v0.1.5-repro`), not committed. The official
checkpoint is the partial fine-tune used for **every** reported result (test and test2):

| item | value |
|---|---|
| directory | `models/reranker/runs/rk-1e-05-n4-s20261027-partial-3060-FROZEN/` |
| weights | `reranker-FROZEN-model.safetensors`, sha256 `547060da6310cfd6cb31b75521f75c6af4161aa916196330ef1492ee91719ba2` |
| `config.json` | sha256 `f8921fc136966127e56b76518da385bade346820a18178a3637b6bb891341739` (differs from the base model) |
| `tokenizer.json` | sha256 `7132ed20a040fba715eb868be52c6a7278ce29f0985dae642cd91e573b8d4f3f` (loaded from the checkpoint dir by `19`) |
| `tokenizer_config.json` | sha256 `7f8e591fccaa155b910473a53702be26990095e8904323a4241f7750ecc0273e` |
| `train_log.json` | sha256 `8d7ef1d4671c41c4094e63a8de3609a7a1b27e6e34ffe982470ac0b9235b4851` |
| `FROZEN.json` | sha256 `68e06ad79e7ab3c49ad4bcb0687c394a2eddad754872cdb9f0e34b7aeb21911d` |
| training run id | `8c0e9c786bb6` |

**Do not confuse this with** `models/reranker/runs/rk-1e-05-n4-s20261027/`, whose weights are
`d69da8db47755ab7…` after a 2026-11-11 re-run overwrote the original directory. That checkpoint
is **not** the model behind the paper; it is not published.

`python tools/fetch_models.py --download` installs and verifies the six files above;
`--check` verifies them in place. `--base` downloads `BAAI/bge-small-en-v1.5` and
`BAAI/bge-reranker-base` from Hugging Face for index building and retraining.
