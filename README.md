# Comparative Evaluation of Wearable Oxygen-Uptake Estimators — code and result artefacts

Analysis code and result artefacts for *Comparative Evaluation of Wearable Oxygen-Uptake Estimators Using
Baseline-Relative, Transition, and Equivalence Analyses* (Li, Zhu, Zhang & Whittle).

The pipeline benchmarks eleven models (linear: OLS, ridge, lasso, elastic net; tree: random forest, XGBoost;
temporal: LSTM, GRU, TCN, TFT, PatchTST) on three public datasets with nested leave-one-participant-out (LOPO)
cross-validation and causal per-participant filtering, and evaluates them with baseline-relative,
transition-resolved, multiplicity-controlled/equivalence, and per-participant agreement analyses.

## Contents

```
r1/                     analysis package (data loading, preprocessing, nested CV, models, statistics, evaluation)
scripts_r1/             command-line scripts (training runners, aggregation, figures and tables)
data/
  TB-File-01.csv        primary dataset (Beltrame et al. 2017), see data/README.md
  external/             place for the two external datasets (not redistributed), see data/external/README.md
results_r1/
  K100/                 final results reported in the paper (100-trial temporal budget)
    <DS>/<MODEL>/outer_XX/   per-fold predictions (predictions_seed*.csv) and selected hyperparameters
    <DS>/analysis/           per-participant metrics, leaderboards, pairwise and family contrasts,
                             Bland-Altman, transition strata, baseline-relative gain summary
    *.csv                    cross-dataset summary (table5.csv), replication, budget and margin sensitivity
  <DS>/analysis/        results with a 25-trial temporal budget (BEL, BEL_CORE, ING; read by summarize_k100.py)
  <DS>/w4/              lag diagnostic (w4a_*) and baseline-relative gain (w4b_*) outputs
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
5. **Figures and tables** (written to `outputs/figures/` and `outputs/tables/`):
   ```
   python scripts_r1/make_figures.py
   python scripts_r1/make_tables.py
   python scripts_r1/make_supp_tables.py
   ```
   `make_figures.py` (Fig. 2 and Fig. 9) and `make_tables.py` (Table 1) also read the raw datasets.

| Paper item | Script | Main inputs |
|---|---|---|
| Figs 2–10 | `make_figures.py` | `results_r1/K100/`, raw data (Figs 2, 9) |
| Tables 1–5 | `make_tables.py` | `results_r1/K100/`, raw data (Table 1) |
| Tables S1–S9 | `make_supp_tables.py` | `results_r1/K100/`, `results_r1/<DS>/analysis/`, `results_r1/<DS>/w4/` |
| Table S10, Fig. S3 | `margin_sensitivity.py` | `results_r1/K100/<DS>/analysis/` |
| Fig. S1 (mutual information) | `legacy/experiments/pmea_p1/make_data_figures.py` | primary dataset |
| WEEE clock offsets, inclusion list | `make_weee_offsets.py`, `make_weee_inclusion.py` | raw WEEE data |

## Licence and citation

Code: MIT licence (`LICENSE`). Please cite the article and this archive (`CITATION.cff`). The primary dataset is
redistributed under its original CC BY 4.0 licence; see `data/README.md`.
