# Public dataset acquisition status

Updated 2026-09-01. This record distinguishes metadata inspection from a
successful local acquisition; a directory alone is not evidence that data are
available. The 2026-07-30 entries below (proxy failures) are superseded: the
user copied all three datasets to the external drive directly. The ACTUAL
paths differ in capitalization from the paths this project's config
originally assumed — the lowercase paths below do not exist or are stale
zero-byte stubs; do not use them. Verified by `unzip -t` (zip integrity) and
by content inspection (README session counts, directory structure) on
2026-07-31; see `provenance/data_acquisition_decisions.json` for the updated
per-dataset decision records.

Current data root: `/media/amin/ADATA HD710 PRO/Research/Representation/Working Memory/data`.
Analysis processes receive this through `WM_DYNAMICS_DATA_ROOT`; the former `EXTERNAL_USB` root is
retired. `config/datasets.json` now records relative `local_path` entries for the registered corpora,
so the older statement below that those entries were absent is historical and closed.

| Dataset | Verified path on data root | Required official asset | Status |
|---|---|---|---|
| Panichello et al. 2024 (Dryad `10.5061/dryad.kkwh70sct`) | `Panichello_2024/` (NOT `panichello_2024/`) | 25 MAT session files + README (145 MB on disk) | ACQUIRED. 25 `.mat` files present (10 monkey A + 8 monkey H + 7 monkey J, matches README), plus README.md. Ready to use. |
| Watters et al. 2026 (OSF `vyw49`) | `Watters/` (NOT `watters_2026/`) | `data_for_figures.zip` (~4.5 GB compressed) | ACQUIRED. `data_for_figures.zip` (4.12 GB) + extracted, `data_for_modeling.zip` (382 MB) + extracted (`spikes_per_trial/<monkey>/<date>/...`). Total ~4.5 GB compressed, matches the audited estimate — this is the processed cache, not raw DANDI 000620 (correctly not downloaded). Both zips pass `unzip -t`. Ready to use. |
| Inagaki et al. ALM (Figshare `10.25378/janelia.7489253`) | `Inagaki/` (NOT `alm_5/`; `alm_5/SiliconProbeData.zip` is a 114 KB broken stub, do not use) | Smallest silicon-probe perturbation/recovery subset | ACQUIRED (and more): `Inagaki/SiliconProbeData.zip` (367 MB, `unzip -t` clean) extracted to `SiliconProbeData/SiliconProbeData/{FixedDelayTask,RandomDelayTask}/` plus a `WholeCellData.zip` (12.7 GB, also extracted) that was not requested but is present. Ready to use; `SiliconProbeData/` is what 5.2/6.3 need. |

The proxy variables were used only in the 2026-07-30 download process
environment and are no longer relevant now that the data arrived by direct
copy. No system-level, shell-profile, or other-terminal proxy setting was
changed at any point.

Closed: `config/datasets.json` has local-path entries and shared path resolution is implemented in
`src/project_config.py`. Remaining hard-coded historical notebook paths are non-production records;
current scripts must resolve the root from `WM_DYNAMICS_DATA_ROOT`.
