# Comparative Evaluation of Wearable Oxygen-Uptake Estimators — code and result artefacts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20768308.svg)](https://doi.org/10.5281/zenodo.20768308)

Analysis code and result artefacts for *Comparative evaluation of models for estimating oxygen uptake from wearable sensors across datasets, activity transitions and participants* (Li, Zhu, Zhang & Whittle).

The pipeline benchmarks eleven models (linear: OLS, ridge, lasso, elastic net; tree: random forest, XGBoost;
temporal: LSTM, GRU, TCN, TFT, PatchTST) on three public datasets with nested leave-one-participant-out (LOPO)
cross-validation and causal per-participant filtering, and evaluates them with baseline-relative,
transition-resolved, multiplicity-controlled/equivalence, and per-participant agreement analyses. Models are compared
pair by pair (all 55 pairs on each dataset, Holm-adjusted exact Wilcoxon tests and Holm-adjusted two one-sided
equivalence tests); these model-level analyses and the post-hoc stratum analyses are in `paper_analysis/`.

## Contents

```
r1/                     analysis package (data loading, preprocessing, nested CV, models, statistics, evaluation)
scripts_r1/             command-line scripts (training runners, aggregation, figures and tables)
paper_analysis/         model-level (55-pair) and post-hoc stratum analyses, their tables and figures
  results/              their outputs (model_pairs_all.csv, model_pairs_margin.csv, *_summary.txt, strata_*_out.txt)
data/
  TB-File-01.csv        primary dataset (Beltrame et al. 2017), see data/README.md
  external/             place for the two external datasets (not redistributed), see data/external/README.md
results_r1/
  K100/                 final results reported in the paper (100-trial temporal budget)
    <DS>/<MODEL>/outer_XX/   per-fold predictions (predictions_seed*.csv) and selected hyperparameters
    <DS>/analysis/           per-participant metrics, leaderboards, family-level pairwise tests and contrasts
                             (omnibus-gated; not used for the model comparison of the paper, see paper_analysis/),
                             Bland-Altman, transition strata, baseline-relative gain summary
    *.csv                    cross-dataset summary (table5.csv), replication, budget and margin sensitivity
  <DS>/analysis/        results with a 25-trial temporal budget (BEL, BEL_CORE, ING; read by summarize_k100.py)
  <DS>/w4/              lag diagnostic (w4a_*) and baseline-relative gain (w4b_*) outputs
  <DS>/<RF|XGB>/outer_XX/inner_trials.csv   Optuna trial logs of the tree models (tuning-budget table)
  <DS>_T100/            Optuna studies (SQLite) of the 100-trial temporal-model tuning
legacy/                 code and artefacts of a previous, non-nested version of this analysis
```

Dataset codes: `BEL` primary dataset; `BEL_CORE` primary dataset with the sensor types shared by all datasets
(HR, ΔHR, acceleration); `WEEE` (n = 15); `WEEE_HRQ` WEEE after the heart-rate plausibility screen (n = 11);
`WEEE_ALL` WEEE with all 17 participants (linear and tree models only); `ING` Ingraham et al. (n = 10).
Class codes used in some result files are explained in `results_r1/K100/README_classes.md`.

## Requirements

Python 3.11. Exact versions used are listed in `environment.txt` (`requirements.txt` gives the package names).
The temporal models were trained on a CUDA GPU (PyTorch); GPU training is not bit-reproducible, so re-runs can
differ slightly from the stored predictions.

```
pip install -r requirements.txt
```

## Reproducing the results

All commands are run from this folder. Steps 1–3 retrain the models (the temporal models take many GPU hours);
steps 4–5 only need the stored results in `results_r1/`.

1. **Data.** The primary dataset is included. Download the external datasets as described in
   `data/external/README.md` (or set the environment variable `VO2_EXTERNAL_DATA` to their location).
2. **Linear and tree models** (nested LOPO; outputs in `results_r1/<DS>/<MODEL>/`):
   ```
   python scripts_r1/run_linear.py BEL
   python scripts_r1/run_trees.py BEL --models RF,XGB
   ```
3. **Temporal models** (100 Optuna trials per outer fold; outputs in `results_r1/<DS>_T100/`):
   ```
   python scripts_r1/run_temporal.py BEL --models LSTM,GRU,TCN,TFT,PATCHTST --trials 100 --out-name BEL_T100
   ```
   `scripts_r1/supervise_temporal.py` restarts the runner after interruptions; completed folds are skipped and
   interrupted studies resume. Then assemble and evaluate the final results:
   ```
   python scripts_r1/assemble_primary.py BEL BEL_CORE WEEE WEEE_HRQ ING
   python scripts_r1/run_w4.py BEL --part both
   ```
   Repeat steps 2–3 for each dataset code.
4. **Summaries and sensitivity analyses** (read `results_r1/`):
   ```
   python scripts_r1/summarize_k100.py
   python scripts_r1/supplementary_k100.py BEL ING WEEE WEEE_HRQ BUDGET WEEE_ALL
   python scripts_r1/ing_quality_rescore.py
   python scripts_r1/weee_quality_tables.py
   python scripts_r1/margin_sensitivity.py
   ```
5. **Figures and tables** (written to `outputs/figures/` and `outputs/tables/`; the order matters, because the
   scripts in `paper_analysis/` read or overwrite some of the files written by `scripts_r1/`):
   ```
   python scripts_r1/make_figures.py
   python scripts_r1/make_tables.py
   python scripts_r1/make_supp_tables.py
   python paper_analysis/model_pairs.py
   python paper_analysis/model_level_tables.py
   python paper_analysis/supp_tables_extra.py
   python paper_analysis/make_supp_strata.py
   python paper_analysis/model_level_figures.py
   python paper_analysis/strata_level.py
   python paper_analysis/strata_support.py > paper_analysis/results/strata_support_out.txt
   python paper_analysis/strata_pairs.py > paper_analysis/results/strata_pairs_out.txt
   ```
   `make_figures.py` (`fig2`, `fig9`), `make_tables.py` (Table 1) and the stratum scripts also read the raw datasets.

| Paper item | File | Script |
|---|---|---|
| Fig. 1 | `fig3_metrics` | `scripts_r1/make_figures.py` |
| Fig. 2 | `fig5_gain` | `scripts_r1/make_figures.py` |
| Fig. 3 | `fig6_transition` | `scripts_r1/make_figures.py` |
| Fig. 4 | `fig8_pairs` | `paper_analysis/model_level_figures.py` |
| Fig. 5 | `fig10_vs_rf` | `paper_analysis/model_level_figures.py` |
| Tables 1, 2 | `table1_datasets`, `table3_leaderboard` | `scripts_r1/make_tables.py` |
| Table 3 | `table5_datasets` | `scripts_r1/make_tables.py`, then `paper_analysis/model_level_tables.py` |
| Supplementary Fig. S1 | protocol figure | reproduced from Beltrame et al. (2017) |
| Supplementary Figs S2, S5, S6, S7 | `fig2_causal_filter`, `fig4_bca`, `fig9_timeseries`, `fig7_bland_altman` | `scripts_r1/make_figures.py` |
| Supplementary Figs S3, S4 | mutual information, random-forest reproduction | `legacy/experiments/pmea_p1/make_data_figures.py` |
| Supplementary Table S1 | `table2_search` | `scripts_r1/make_tables.py` |
| Supplementary Tables S2, S3, S4, S6, S7, S10 | `supp_s1_body_mass`, `supp_s8_hr_screen`, `supp_s3_leaderboards`, `supp_s2_activity`, `supp_s4_ba`, `supp_s6_w4a` | `scripts_r1/make_supp_tables.py` |
| Supplementary Tables S5, S11, S13 | `supp_pairs`, `supp_s7_sensitivity`, `supp_s10_margin` | `paper_analysis/model_level_tables.py` |
| Supplementary Tables S8, S9, S12 | `supp_s11_ba_external`, `supp_s5_w4b`, `supp_s9_budget` | `paper_analysis/supp_tables_extra.py` |
| Supplementary Table S14 | `supp_s12_strata` | `paper_analysis/make_supp_strata.py` |
| Values cited as "code repository" | `paper_analysis/results/` and `results_r1/K100/` | scripts above |
| WEEE clock offsets, inclusion list | `data/external/` | `scripts_r1/make_weee_offsets.py`, `make_weee_inclusion.py` |

The other files written by `make_figures.py`, `make_tables.py` and `make_supp_tables.py` belong to the earlier,
family-level presentation of the same results (version 2.0.0) and are not used in the paper.

## Licence and citation

Code: MIT licence (`LICENSE`). Please cite the article and this archive: version 2.1.0, https://doi.org/10.5281/zenodo.24209830 (all versions: https://doi.org/10.5281/zenodo.20768308; see also `CITATION.cff`). The primary dataset is
redistributed under its original CC BY 4.0 licence; see `data/README.md`.
