# Provenance: which artefacts produced which numbers

## Hash definitions (do not mix these up)

| name | value | what it hashes |
|---|---|---|
| `frozen_json_sha256` | `68e06ad79e7ab3c4…` | the **file** `models/reranker/FROZEN.json` |
| `checkpoint_sha256` | `547060da6310cfd6…` | the **weights file** `model.safetensors` |
| `tokenizer_json_sha256` | `7132ed20a040fba7…` | `tokenizer.json` inside the checkpoint directory |
| `config_json_sha256` | `f8921fc136966127…` | `config.json` inside the checkpoint directory |
| `scripts_sha256` | `cefb847e50c273a8…` | a **field** in `FROZEN.json`, computed over `scripts/*.py` at training time |
| `train_run_id` | `8c0e9c786bb6` | a **field** in `FROZEN.json` |

`results/HASHES.csv` lists every value with its kind (`file` or `field`), definition and
source. Never call the file hash a "model state hash" or vice versa.

## Official checkpoint lineage

| directory | weights sha256 | mtime | status |
|---|---|---|---|
| `runs/rk-1e-05-n4-s20261027` | `d69da8db47755ab7…` | 2026-11-11 | **not official** — overwritten by a later run |
| `runs/rk-1e-05-n4-s20261027-partial-3060-FROZEN` | `547060da6310cfd6…` | 2026-10-25 | **official** — matches `FROZEN.model.checkpoint_sha256` |

Reconstruction: `FROZEN.json` was written on 2026-10-25 with checkpoint `547060da…`; the test
run on 2026-10-28 selected the only matching directory (then `runs/rk-1e-05-n4-s20261027`,
as its log shows) because that directory still held `547060da…`; on 2026-11-11 that directory
was overwritten with `d69da8db…`; test2 on 2026-11-13 therefore selected the preserved
`-partial-3060-FROZEN` copy. **Both official runs used `547060da…`.** The public package ships
only the `-FROZEN` directory, so `find_checkpoint()` has exactly one match.

## Code versions differ between the official runs

`FINAL_RUN.code_sha256` is `6f0bcff6…` (test, 2026-10-28), `da6a039b…` (test2, 2026-11-13),
and `FROZEN.code.scripts_sha256` is `cefb847e…` (training, 2026-10-25). The evaluation code
advanced between the runs, so test-versus-test2 differences mix split, hardware and code
version. `results/artifact_map.csv` records the code hash on every row. No claim is made that
the two runs used byte-identical code.

## Historical absolute paths in hash-locked files

`models/reranker/FROZEN.json` (`model.source`, `tokenizer.source`) and
`annotations/LOCKED_TEST2/LOCKED2.json` (`questions_source`) contain the author's original
machine paths. These files are covered by recorded hashes (`FINAL_RUN.frozen_sha256`,
`PREREGISTRATION` lock hash), so editing them would break the provenance chain. They are
informational only; no script uses those strings to locate files.

## Public repository code edits

The public release keeps every script and changes only three lines that point at the author's
desktop (two in `make_progress_report.py`, one in `make_teacher_brief.py`). The eight files in
`EVALUATION_CODE` (`13, 18, 19, 16, goldset, common, 11, 12`) are byte-identical to the
originals; `data/CODE_MANIFEST.original.csv` records every original hash, and the public
release adds `CODE_MANIFEST.csv` with both columns plus `CODE_DELTA.md`. Because `code_sha()`
hashes `scripts/*.py` as a set, the public tree's overall `code_sha256` **will differ** from
the recorded values; that is expected and is not a change to experiment logic.

## Crawler contact update (2026-11-22)

The crawler User-Agent contact `student@wku.edu.cn` was replaced by
`Xiaocao21@users.noreply.github.com` in `config/settings.yaml`, `MANIFEST.json`,
`COMPLIANCE.md`, `reports/robots_audit.json` and the three crawler scripts. This removes a
personal email address from the public repository; the crawler remains contactable.

`MANIFEST.settings_sha256` was recomputed (`8489027f…`) because it hashes `settings.yaml`,
`previous_settings_sha256` keeps the pre-change value, and
`settings_changed_since_last_freeze` is now `true`. The corpus artefact hashes are unchanged.
The test2 pre-registration does **not** cover `MANIFEST.json` or `settings.yaml`, so R1/R2 are
unaffected; `results/HASHES.csv` records the new `MANIFEST.json` file hash.
