# Household-Community Measles SBI Pipeline

Simulation-based inference for a household-community stochastic SEIR model of
measles transmission, comparing ABC, regression-adjusted ABC, Bayesian
synthetic likelihood (BSL), and neural posterior estimation (NPE, including
learned-embedding and ABC-preconditioned variants). Full write-up: shared
separately as a Word document ("Simulation-Based Inference for a
Household-Community Measles Transmission Model...").

## Requirements

- Python >= 3.9
- `numpy`, `pandas`, `matplotlib`, `scipy`
- [`episbi`](https://github.com/geunsoojang/EpiSBI) (for Steps 8-13; installed
  in editable mode via `pip install -e .` from a clone of that repository)

## Running the pipeline

Steps must be run **in order** -- each one reads outputs saved by an earlier
step rather than regenerating them, starting from a single population and
observed outbreak generated once in Step 1.

| Step | Script | What it does | Depends on |
|---|---|---|---|
| 1 | `step1_data_preparation.py` | Builds the population and simulates the synthetic "observed" outbreak | -- |
| 2 | `step2_abc_inference.py` | ABC rejection posterior for (beta_H, beta_C) | Step 1 |
| 3 | `step3_identifiability_experiment.py` | ABC under curve-only / household-only / combined summaries | Step 1 |
| 4 | `step4_forecasting_scenarios.py` | Posterior-predictive vaccination-coverage forecasts | Steps 1, 2 |
| 5 | `step5_regression_adjustment.py` | Local-linear and ridge ABC regression adjustment | Steps 1, 3 |
| 6 | `step6_bsl.py` | Bayesian synthetic likelihood via Metropolis-Hastings | Steps 1, 2, 3 |
| 7 | `step7_curve_household_bank.py` | Shared NPE training bank (summaries + household histogram + curve) | Step 1 |
| 8 | `step8_npe_smoke_test.py` | Toolbox validation on episbi's own built-in model | -- |
| 9 | `step9_npe_recovery.py` | NPE under curve-only / household-only / combined summaries | Step 7 |
| 10 | `step10_npe_lstm.py` | Learned curve embedding (LSTM) vs. hand-picked curve summaries | Step 7 |
| 11 | `step11_npe_curve_household.py` | Curve LSTM + household summaries vs. hand-picked combined | Step 7 |
| 12 | `step12_npe_household_learned.py` | Learned household-histogram embedding vs. hand-picked household summaries | Step 7 |
| 13 | `step13_pnpe_all_representations.py` | ABC-preconditioned NPE across all three representations | Steps 1, 7 |

Each script creates its own `output/stepN/` folder with its results (CSVs and
figures); these are not committed to this repository (see `.gitignore`) since
they are fully regenerable and the pipeline is deterministic given a fixed
seed -- confirmed by exact reproduction across independent runs and machines
during development.

Steps 1-7 need only `numpy`/`pandas`/`matplotlib`/`scipy`. Steps 8-13
additionally require `episbi` and its dependencies (`torch`, `sbi`).
