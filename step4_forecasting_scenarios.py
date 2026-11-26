# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 4: SCENARIO FORECASTING
# Take the posterior recovered in Step 2 and ask the public-health question:
#   "Given our UNCERTAINTY about transmission, how big would this outbreak be
#    under different vaccination-coverage levels?"
#
# Method (posterior-predictive):
#   - Sample (beta_H, beta_C) pairs from the Step-2 ABC posterior.
#   - For each vaccination-coverage scenario, simulate one outbreak per pair.
#   - The spread of outcomes = parameter uncertainty + stochastic uncertainty.
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step4_forecasting_scenarios.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP1_DIR = "output/step1"
STEP2_DIR = "output/step2"
OUT_DIR   = "output/step4"

N_FORECAST_DRAWS = 400
COVERAGE_LEVELS  = [0.30, 0.50, 0.70, 0.85, 0.95]
LARGE_THRESHOLD  = 50

LATENT_DAYS = 10
INFECTIOUS_DAYS = 8
SIGMA = 1 / LATENT_DAYS
GAMMA = 1 / INFECTIOUS_DAYS
INIT_INFECTED = 5
T_MAX = 320

WORKERS = max(1, (os.cpu_count() or 2) - 1)


def load_population(step1_dir):
    hh_final = pd.read_csv(f"{step1_dir}/step1_household_final_sizes.csv")
    hh_sizes = hh_final["hh_size"].to_numpy()
    hh_id = np.repeat(np.arange(len(hh_sizes)), hh_sizes)
    return hh_id, len(hh_id)


def rowsum(x, group, n_groups):
    return np.bincount(group, weights=x, minlength=n_groups)


def simulate_final_size(rng, hh_id, beta_H, beta_C, sigma, gamma, N, t_max,
                         init_infected, init_immune_frac):
    n_hh = int(hh_id.max()) + 1
    state = np.zeros(N, dtype=np.int8)

    if init_immune_frac > 0:
        state[rng.choice(N, size=round(init_immune_frac * N), replace=False)] = 3
    seeds = rng.choice(np.flatnonzero(state == 0), size=init_infected, replace=False)
    state[seeds] = 2

    ever_I = np.zeros(N, dtype=bool)
    ever_I[seeds] = True

    p_EI = 1 - np.exp(-sigma)
    p_IR = 1 - np.exp(-gamma)

    for t in range(1, t_max + 1):
        is_S = state == 0
        is_E = state == 1
        is_I = state == 2
        if not is_I.any() and not is_E.any():
            break

        hh_I = rowsum(is_I.astype(float), hh_id, n_hh)
        I_housemates = hh_I[hh_id] - is_I.astype(float)
        hazard = beta_H * I_housemates + beta_C * is_I.sum() / N
        p_inf = 1 - np.exp(-hazard)

        new_E = is_S & (rng.random(N) < p_inf)
        new_I = is_E & (rng.random(N) < p_EI)
        new_R = is_I & (rng.random(N) < p_IR)

        state[new_R] = 3; state[new_I] = 2; state[new_E] = 1
        ever_I[new_I] = True

    return int(ever_I.sum())


def _worker(task):
    beta_H, beta_C, coverage, sim_seed, hh_id, N = task
    rng = np.random.default_rng(sim_seed)
    return simulate_final_size(rng, hh_id, beta_H, beta_C, SIGMA, GAMMA, N, T_MAX,
                                INIT_INFECTED, coverage)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {WORKERS} worker processes.")

    hh_id, N = load_population(STEP1_DIR)
    print(f"Loaded population from Step 1: N={N}, households={hh_id.max()+1}")

    post = pd.read_csv(f"{STEP2_DIR}/step2_abc_posterior_draws.csv")
    idx = np.random.default_rng(2026).integers(0, len(post), N_FORECAST_DRAWS)
    samp_bH = post["beta_H"].to_numpy()[idx]
    samp_bC = post["beta_C"].to_numpy()[idx]

    # scenario x draw grid
    rows = []
    for cov in COVERAGE_LEVELS:
        for draw in range(N_FORECAST_DRAWS):
            rows.append((cov, draw))
    grid = pd.DataFrame(rows, columns=["coverage", "draw"])
    grid["beta_H"] = samp_bH[grid["draw"]]
    grid["beta_C"] = samp_bC[grid["draw"]]

    n_total = len(grid)
    print(f"\nForecasting {len(COVERAGE_LEVELS)} scenarios x {N_FORECAST_DRAWS} "
          f"posterior draws = {n_total} sims on {WORKERS} workers...")

    seed_seq = np.random.SeedSequence(2026)
    sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq.spawn(n_total)]
    tasks = [(grid["beta_H"].iloc[i], grid["beta_C"].iloc[i], grid["coverage"].iloc[i],
              sim_seeds[i], hh_id, N) for i in range(n_total)]

    chunk_size = max(1, n_total // (WORKERS * 20))
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        final_sizes = list(executor.map(_worker, tasks, chunksize=chunk_size))
    grid["final_size"] = final_sizes
    grid["coverage_lab"] = grid["coverage"].apply(lambda c: f"{round(100*c)}% vaccinated")
    coverage_order = [f"{round(100*c)}% vaccinated" for c in COVERAGE_LEVELS]
    grid["coverage_lab"] = pd.Categorical(grid["coverage_lab"], categories=coverage_order, ordered=True)

    # ---- summary table per scenario ------------------------------------------
    scenario_tab = grid.groupby("coverage_lab", observed=True).agg(
        median_size=("final_size", "median"),
        pi_lo=("final_size", lambda x: x.quantile(.025)),
        pi_hi=("final_size", lambda x: x.quantile(.975)),
        p_large=("final_size", lambda x: (x >= LARGE_THRESHOLD).mean()),
    ).reset_index()

    grid.to_csv(os.path.join(OUT_DIR, "step4_forecast_draws.csv"), index=False)
    scenario_tab.to_csv(os.path.join(OUT_DIR, "step4_scenario_summary.csv"), index=False)

    # ---- figure 1: boxplot of final size by coverage --------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    grid.boxplot(column="final_size", by="coverage_lab", ax=ax, grid=False,
                 patch_artist=True, showfliers=True,
                 boxprops=dict(facecolor="#a6d96a"), medianprops=dict(color="black"))
    ax.set_title("Forecasted Measles Outbreak Size by Vaccination Coverage")
    ax.set_xlabel(""); ax.set_ylabel("Final epidemic size (total cases)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step4_forecast_boxplot.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- figure 2: probability of a large outbreak by coverage -----------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(scenario_tab["coverage_lab"].astype(str), scenario_tab["p_large"],
                   color="#fdae61", width=0.65, edgecolor="grey")
    for b, v in zip(bars, scenario_tab["p_large"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{100*v:.0f}%",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Probability of a Large Outbreak (>= {LARGE_THRESHOLD} cases) by Coverage")
    ax.set_ylabel("P(large outbreak)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step4_takeoff_probability.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- console summary ---------------------------------------------------------
    print("\n========= STEP 4 (FORECAST) SUMMARY =========")
    print("Baseline (30% vaccinated) is the setting our data came from.\n")
    disp = scenario_tab.copy()
    disp["median_size"] = disp["median_size"].round().astype(int)
    disp["pi_lo"] = disp["pi_lo"].round().astype(int)
    disp["pi_hi"] = disp["pi_hi"].round().astype(int)
    disp["p_large"] = disp["p_large"].round(3)
    print(disp.to_string(index=False))
    print("\nReading guide: median_size = typical outbreak; pi_lo/pi_hi = 95% range")
    print(f"(reflecting our uncertainty in beta_H/beta_C); p_large = chance of a")
    print(f">= {LARGE_THRESHOLD}-case outbreak. Watch both collapse as coverage rises.")
    print(f"Saved to {os.path.abspath(OUT_DIR)}/  (2 CSVs + 2 PNGs)")
    print("=============================================")


if __name__ == "__main__":
    main()
