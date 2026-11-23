# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 3: IDENTIFIABILITY EXPERIMENT
# Question: does the household-clustering signal actually make beta_H
#           identifiable, or is the epidemic curve alone enough?
#
# Method: run the SAME ABC acceptance three ways, changing ONLY which
#         summary statistics are used:
#           (a) Curve only        : total cases, peak day, peak size
#           (b) Household only     : #hh affected, mean hh size, frac singletons
#           (c) Curve + Household  : all six
#         Then compare the recovered posteriors for beta_H and beta_C.
#
# Expectation: beta_H goes from flat (curve only) -> peaked (with households),
#              while beta_C stays sharp throughout.
#
# Run from /Volumes/ssd/MSI/APPEX/EpiSBI-main/measles_project/ (or anywhere -- the script cd's to its own folder):
#     python step3_identifiability_experiment.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP1_DIR = "output/step1"
OUT_DIR   = "output/step3"

N_SIM_ABC   = 20_000
ACCEPT_FRAC = 0.02     # keep top 2% per method (-> ~400 posterior draws each)

PRIOR_BH = (0.0, 0.6)
PRIOR_BC = (0.0, 0.5)

LATENT_DAYS = 10
INFECTIOUS_DAYS = 8
SIGMA = 1 / LATENT_DAYS
GAMMA = 1 / INFECTIOUS_DAYS
INIT_INFECTED = 5
INIT_IMMUNE_FRAC = 0.30
T_MAX = 320

WORKERS = max(1, (os.cpu_count() or 2) - 1)
SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]
CURVE_COLS = ["total", "peak_day", "peak_size"]
HH_COLS    = ["n_hh_aff", "mean_hh", "frac_singl"]


def load_population(step1_dir):
    hh_final = pd.read_csv(f"{step1_dir}/step1_household_final_sizes.csv")
    hh_sizes = hh_final["hh_size"].to_numpy()
    hh_id = np.repeat(np.arange(len(hh_sizes)), hh_sizes)
    return hh_id, len(hh_id)


def rowsum(x, group, n_groups):
    return np.bincount(group, weights=x, minlength=n_groups)


def simulate_hc(rng, hh_id, beta_H, beta_C, sigma, gamma, N, t_max, init_infected, init_immune_frac):
    n_hh = int(hh_id.max()) + 1
    state = np.zeros(N, dtype=np.int8)

    if init_immune_frac > 0:
        state[rng.choice(N, size=round(init_immune_frac * N), replace=False)] = 3
    seeds = rng.choice(np.flatnonzero(state == 0), size=init_infected, replace=False)
    state[seeds] = 2

    ever_I = np.zeros(N, dtype=bool)
    ever_I[seeds] = True
    incidence = np.zeros(t_max + 1, dtype=np.int64)
    incidence[0] = init_infected

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
        incidence[t] = new_I.sum()

    cph = rowsum(ever_I.astype(float), hh_id, n_hh)
    return incidence, cph


def summary_stats(inc, cph):
    aff = cph[cph > 0]
    return np.array([
        inc.sum(),
        len(aff),
        aff.mean() if len(aff) else 0,
        np.mean(aff == 1) if len(aff) else 0,
        int(np.argmax(inc)),
        inc.max(),
    ], dtype=float)


def _worker(task):
    beta_H, beta_C, sim_seed, hh_id, N = task
    rng = np.random.default_rng(sim_seed)
    inc, cph = simulate_hc(rng, hh_id, beta_H, beta_C, SIGMA, GAMMA, N, T_MAX,
                            INIT_INFECTED, INIT_IMMUNE_FRAC)
    s = summary_stats(inc, cph)
    return np.concatenate(([beta_H, beta_C], s))


def abc_subset(params, S_mat, obs_S, cols_idx, accept_frac):
    sm = S_mat[:, cols_idx]
    sds = sm.std(axis=0)
    sds[sds == 0] = 1
    d = np.sqrt((((sm - obs_S[cols_idx]) / sds) ** 2).sum(axis=1))
    threshold = np.quantile(d, accept_frac)
    return params[d <= threshold].copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {WORKERS} worker processes.")

    hh_id, N = load_population(STEP1_DIR)
    print(f"Loaded population from Step 1: N={N}, households={hh_id.max()+1}")

    obs_inc = pd.read_csv(f"{STEP1_DIR}/step1_observed_incidence.csv")["new_onsets"].to_numpy()
    obs_cph = pd.read_csv(f"{STEP1_DIR}/step1_household_final_sizes.csv")["n_cases"].to_numpy()
    obs_S = summary_stats(obs_inc, obs_cph)
    tp = pd.read_csv(f"{STEP1_DIR}/step1_true_params.csv")
    beta_H_true = float(tp.loc[tp["param"] == "beta_H", "value"].iloc[0])
    beta_C_true = float(tp.loc[tp["param"] == "beta_C", "value"].iloc[0])

    # ---- ONE shared bank of simulations, all six summaries recorded --------
    print(f"\nSimulating {N_SIM_ABC} outbreaks on {WORKERS} workers...")
    master_rng = np.random.default_rng(2026)
    beta_H = master_rng.uniform(*PRIOR_BH, N_SIM_ABC)
    beta_C = master_rng.uniform(*PRIOR_BC, N_SIM_ABC)
    seed_seq = np.random.SeedSequence(2026)
    sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq.spawn(N_SIM_ABC)]
    tasks = [(beta_H[i], beta_C[i], sim_seeds[i], hh_id, N) for i in range(N_SIM_ABC)]

    chunk_size = max(1, N_SIM_ABC // (WORKERS * 20))
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(executor.map(_worker, tasks, chunksize=chunk_size))
    rows = np.vstack(rows)
    params = pd.DataFrame(rows[:, :2], columns=["beta_H", "beta_C"])
    S_mat = rows[:, 2:]

    sim_table = pd.concat([params, pd.DataFrame(S_mat, columns=SUMMARY_COLS)], axis=1)
    sim_table.to_csv(os.path.join(OUT_DIR, "step3_simulation_table.csv"), index=False)

    # ---- ABC acceptance with different summary subsets ----------------------
    curve_idx = [SUMMARY_COLS.index(c) for c in CURVE_COLS]
    hh_idx    = [SUMMARY_COLS.index(c) for c in HH_COLS]
    both_idx  = list(range(len(SUMMARY_COLS)))

    post_curve = abc_subset(params, S_mat, obs_S, curve_idx, ACCEPT_FRAC)
    post_curve["method"] = "1. Curve only"
    post_hh = abc_subset(params, S_mat, obs_S, hh_idx, ACCEPT_FRAC)
    post_hh["method"] = "2. Household only"
    post_both = abc_subset(params, S_mat, obs_S, both_idx, ACCEPT_FRAC)
    post_both["method"] = "3. Curve + Household"

    post = pd.concat([post_curve, post_hh, post_both], ignore_index=True)
    post.to_csv(os.path.join(OUT_DIR, "step3_posteriors_by_method.csv"), index=False)

    # ---- quantify: 95% CrI width by summary set -----------------------------
    rows_w = []
    for method, sub in post.groupby("method"):
        for p in ["beta_H", "beta_C"]:
            med = sub[p].median(); lo = sub[p].quantile(.025); hi = sub[p].quantile(.975)
            rows_w.append({"method": method, "param": p, "median": med,
                            "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo})
    width_tab = pd.DataFrame(rows_w).sort_values(["param", "method"]).reset_index(drop=True)
    width_tab.to_csv(os.path.join(OUT_DIR, "step3_ci_widths.csv"), index=False)

    # ---- headline figure: posterior densities by summary set ---------------
    colors = {"1. Curve only": "#e6731a", "2. Household only": "#1a9850", "3. Curve + Household": "#3b6fb6"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, p, truth, label in zip(axes, ["beta_H", "beta_C"], [beta_H_true, beta_C_true],
                                    ["beta_H (household)", "beta_C (community)"]):
        for method, sub in post.groupby("method"):
            sub[p].plot(kind="density", ax=ax, color=colors[method], label=method, linewidth=1.8)
        ax.axvline(truth, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(label); ax.set_xlabel("value"); ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Which data makes each parameter identifiable? (black dashed = true value)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step3_identifiability.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- console summary ------------------------------------------------------
    print("\n========= STEP 3 (IDENTIFIABILITY) SUMMARY =========")
    print(f"True values: beta_H = {beta_H_true:.2f}, beta_C = {beta_C_true:.2f}\n")
    print("95% credible-interval WIDTH per parameter per summary set (smaller = better identified):\n")
    print(width_tab.round(3).to_string(index=False))
    print("\nWatch beta_H's width shrink as household summaries are added,")
    print("while beta_C stays narrow throughout.")
    print(f"Saved to {os.path.abspath(OUT_DIR)}/  (3 CSVs + 1 PNG)")
    print("====================================================")


if __name__ == "__main__":
    main()
