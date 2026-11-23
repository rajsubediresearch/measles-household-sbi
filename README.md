# Household-Community Measles SBI Pipeline

Simulation-based inference for a household-community stochastic SEIR model of
measles transmission, comparing ABC, regression-adjusted ABC, Bayesian
synthetic likelihood (BSL), and neural posterior estimation (NPE, including
learned-embedding and ABC-preconditioned variants), plus a one-at-a-time
sensitivity sweep across disease natural history and population parameters.
Full write-up: shared separately as a Word document ("Simulation-Based
Inference for a Household-Community Measles Transmission Model...").

## Requirements

- Python >= 3.10
- `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn` (Step 5)
- [`episbi`](https://github.com/geunsoojang/EpiSBI) (for Steps 8-14; installed
  in editable mode via `pip install -e .` from a clone of that repository)

## Shared modules

- `hc_model.py` -- parameterised household-community stochastic SEIR
  simulator and bank-building utilities, factored out of the simulation logic
  duplicated across Steps 1-7 into a single reusable module. Exposes a
  `ScenarioConfig` dataclass (population size, household-size distribution,
  latent/infectious periods, beta_H, beta_C, seed, etc.) so a scenario is an
  explicit argument rather than a module-level constant. Steps 1-13 do not
  import this module and are unaffected by it; Step 14 is built on it.

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
| 14 | `step14_sensitivity_sweep.py` | One-at-a-time sensitivity sweep (incubation period, infectious period, beta_H/beta_C split, population size) repeating the curve-only / household-only / combined NPE identifiability contrast across 11 scenarios | `hc_model.py`; independent of Steps 1-13's saved outputs (each scenario regenerates its own population, observed outbreak, and training bank) |

Each script creates its own `output/stepN/` folder with its results (CSVs and
figures); these are not committed to this repository (see `.gitignore`) since
they are fully regenerable from the fixed seeds in each script. Steps 1-7
(the simulator, ABC, regression adjustment and BSL) and the Step 13 ABC pilot
and Step 14 training banks reproduce exactly across machines and operating
systems (confirmed Windows vs macOS). The neural steps (8-14) do not reproduce
bit-for-bit across machines or torch/sbi versions: network training differs at
the floating-point level, so posterior medians and interval bounds shift
slightly (typically by 0.02-0.04, up to ~0.17 in Step 9) while every
qualitative conclusion is unchanged. Use the pinned versions in
`requirements.txt` to minimise this. Step 14 additionally creates one subfolder per scenario,
`output/step14/<scenario_name>/`, each holding that scenario's own bank,
per-summary-set posteriors, and recovery table, alongside a pooled
`step14_sensitivity_summary.csv` and two summary figures at the top level of
`output/step14/`.

Steps 1-7 need only `numpy`/`pandas`/`matplotlib`/`scipy`. Steps 8-14
additionally require `episbi` and its dependencies (`torch`, `sbi`).

Step 14 is the slowest step by design (11 scenarios, each simulating its own
10,000-run training bank and training three separate NPE posteriors) --
reduce `N_SIM_BANK` and `N_POSTERIOR_SAMPLES` at the top of the script for a
quick dry run before committing to the full sweep.

## Running on macOS

`run_all_mac.sh` creates a native Python environment (`../.venv_mac`), installs
the pinned requirements plus `episbi` in editable mode, and runs any range of
steps while logging per-step wall time to `output/mac_timing_<date>.csv`:

    bash run_all_mac.sh          # setup (first time) + Steps 1-14
    bash run_all_mac.sh 9 9      # Step 9 only

Each script `cd`s to its own folder on start-up, so it can also be run directly
(`python step9_npe_recovery.py`) from any working directory.

## Runtime

Full pipeline (Steps 1-14) on two machines, CPU only:

| | Windows laptop (MSI, 21 workers) | Mac mini M6 (12 cores, 11 workers) |
|---|---|---|
| Step 14 sensitivity sweep | 29.6 min | 5.1 min |
| -- bank simulation per scenario | 20-89 s | 5-17 s |
| -- NPE training per summary set | 14-82 s | 3-10 s |
| All 14 steps | -- | ~30 min |

On the Mac the slowest steps are 10 (LSTM, ~10.5 min), 13 (~6 min),
14 (~5 min) and 11 (~4.6 min); everything else finishes in under 2 minutes.
The Mac run was done from an external SSD with no slowdown.
