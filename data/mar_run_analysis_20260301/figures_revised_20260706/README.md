# Revised paper figures (prograde cohort)

All figures regenerate from the scripts in this directory using the repo venv:

    MPLCONFIGDIR=.mplconfig .venv/bin/python data/mar_run_analysis_20260301/figures_revised_20260706/<script>.py

| Script | Output |
|---|---|
| `make_paper_figures.py` | fig1-fig9 individual paper figures (`APPLY_PROGRADE` flag: PROGRADE vs REVISED sets) |
| `make_discovery_status_bars.py` | discovery-rate figure (solid discovered / hollow observed-not-discovered) |
| `make_combined_2x2.py` | combined 2x2 (discovered / obs-not-disc / IAWN / arc length), bars-step-line variants |
| `make_ip_range_bars.py` | max-IP IQR floating-bar figure |
| `make_ip_alternatives.py` | IP outcome-mix candidates (stack / lines / violins / heatmap) |
| `make_ip_value_plots.py` | IP-value candidates (boxplot / strip / percentile envelope, log-y) |
| `make_ip_distribution_by_leadtime.py` | IP distribution N years before impact (**needs window_results.parquet**) |

Inputs: `../summary_results.parquet` and `../impactor_orbits.parquet` are committed.
`../window_results.parquet` (72 MB) and `../observations.parquet` (211 MB) are NOT in
git (GitHub size limits) — copy them from the original machine
(`/Users/kathleenkiker/impacts_paper/adam_impacts_study/data/mar_run_analysis_20260301/`)
to run `make_ip_distribution_by_leadtime.py` and the window-based cells of
`notebooks/paper_claim_verification.ipynb`.

Cohort conventions (adopted 2026-07): prograde only (i < 90 deg); discovery statistics
include Sorcha-complete/pipeline-incomplete runs; all IP-dependent statistics use
`status == "complete"`. Full audit: `notebooks/paper_claim_verification.ipynb`.
