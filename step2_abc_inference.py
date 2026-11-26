# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 2: ABC INFERENCE
# Recover (beta_H, beta_C) from the synthetic outbreak made in Step 1,
# WITHOUT telling the estimator the true values.
#
# Method: ABC rejection
#   1. Draw thousands of (beta_H, beta_C) guesses from a wide prior.
#   2. Simulate one outbreak per guess (SAME population as Step 1 -- read
#      from step1_household_final_sizes.csv, not redrawn).
#   3. Summarise each outbreak into 6 numbers (curve + household signals).
#   4. Keep the guesses whose summaries are closest to the observed ones.
#   5. The kept guesses form the approximate posterior.
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step2_abc_inference.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP1_DIR = "output/step1"
OUT_DIR   = "output/step2"

# ============================================================================
# ABC SETTINGS (adjust as needed)
# ============================================================================
N_SIM_ABC   = 20_000     # number of prior draws / simulations
ACCEPT_FRAC = 0.01       # keep this fraction closest to observed (-> posterior)

PRIOR_BH = (0.0, 0.6)
PRIOR_BC = (0.0, 0.5)

# ---- FIXED MODEL SETTINGS -- must match Step 1 exactly ---------------------
LATENT_DAYS = 10
INFECTIOUS_DAYS = 8
SIGMA = 1 / LATENT_DAYS
GAMMA = 1 / INFECTIOUS_DAYS
INIT_INFECTED = 5
INIT_IMMUNE_FRAC = 0.30
T_MAX = 320

WORKERS = max(1, (os.cpu_count() or 2) - 1)
SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]


def load_population(step1_dir):
    # SAME population as Step 1 -- read directly, never redrawn. This is what
    # keeps every step consistent without relying on matching random seeds.
    hh_final = pd.read_csv(f"{step1_dir}/step1_household_final_sizes.csv")
    hh_sizes = hh_final["hh_size"].to_numpy()
    hh_id = np.repeat(np.arange(len(hh_sizes)), hh_sizes)
    return hh_id, len(hh_id)


def rowsum(x, group, n_groups):
    return np.bincount(group, weights=x, minlength=n_groups)


def simulate_hc(rng, hh_id, beta_H, beta_C, sigma, gamma, N, t_max, init_infected, init_immune_frac):
    n_hh = int(hh_id.max()) + 1
    state = np.zeros(N, dtype=np.int8)                     # 0=S 1=E 2=I 3=R

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

    print("Observed summaries:", dict(zip(SUMMARY_COLS, np.round(obs_S, 3))))

    # ---- RUN ABC: simulate from prior, summarise, measure distance --------
    print(f"\nRunning {N_SIM_ABC} simulations on {WORKERS} workers...")
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

    # ---- normalise each summary by its SD, then Euclidean distance --------
    sd_vec = S_mat.std(axis=0)
    sd_vec[sd_vec == 0] = 1
    S_norm = S_mat / sd_vec
    obs_norm = obs_S / sd_vec
    dist = np.sqrt(((S_norm - obs_norm) ** 2).sum(axis=1))

    threshold = np.quantile(dist, ACCEPT_FRAC)
    keep = dist <= threshold
    posterior = params[keep].reset_index(drop=True)

    print(f"Accepted {len(posterior)} of {N_SIM_ABC} draws (tolerance = {threshold:.3f})")

    # ---- POSTERIOR SUMMARY vs TRUTH ----------------------------------------
    post_tab = pd.DataFrame({
        "param": ["beta_H", "beta_C"],
        "true": [beta_H_true, beta_C_true],
        "median": [posterior["beta_H"].median(), posterior["beta_C"].median()],
        "ci_lo": [posterior["beta_H"].quantile(.025), posterior["beta_C"].quantile(.025)],
        "ci_hi": [posterior["beta_H"].quantile(.975), posterior["beta_C"].quantile(.975)],
    })
    post_tab["in_95CI"] = (post_tab["true"] >= post_tab["ci_lo"]) & (post_tab["true"] <= post_tab["ci_hi"])
    post_tab["rel_error"] = (100 * (post_tab["median"] - post_tab["true"]) / post_tab["true"]).round(1)

    posterior.to_csv(os.path.join(OUT_DIR, "step2_abc_posterior_draws.csv"), index=False)
    post_tab.to_csv(os.path.join(OUT_DIR, "step2_abc_posterior_summary.csv"), index=False)

    # ---- PLOTS --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(posterior["beta_H"], posterior["beta_C"], alpha=0.35, color="#4575b4", s=14)
    ax.scatter([beta_H_true], [beta_C_true], color="red", marker="x", s=100, linewidths=2, label="truth")
    ax.scatter([post_tab["median"][0]], [post_tab["median"][1]], color="black", s=50, label="posterior median")
    ax.set_xlim(PRIOR_BH); ax.set_ylim(PRIOR_BC)
    ax.set_xlabel(r"$\beta_H$ (household)"); ax.set_ylabel(r"$\beta_C$ (community)")
    ax.set_title("ABC Joint Posterior"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step2_abc_posterior_joint.png"), dpi=200, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, p, truth in zip(axes, ["beta_H", "beta_C"], [beta_H_true, beta_C_true]):
        ax.hist(posterior[p], bins=30, color="#4575b4", edgecolor="white")
        ax.axvline(truth, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(p); ax.set_xlabel("value"); ax.set_ylabel("count")
    fig.suptitle("ABC Marginal Posteriors vs True Values (red dashed = true value)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step2_abc_posterior_marginals.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- CONSOLE SUMMARY -----------------------------------------------------
    print("\n================ STEP 2 (ABC) SUMMARY ================")
    print(post_tab.to_string(index=False))
    print("\nReading guide: 'in_95CI = True' means the true value falls inside the")
    print("recovered 95% credible interval. rel_error is the median's % error.")
    print(f"Saved to {os.path.abspath(OUT_DIR)}/  (2 CSVs + 2 PNGs)")
    print("======================================================")


if __name__ == "__main__":
    main()
