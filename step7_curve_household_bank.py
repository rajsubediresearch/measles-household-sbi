# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 7: SHARED NPE BANK
# Builds ONE bank of simulations recording theta, the 6 hand-picked summaries,
# the household final-size histogram (hh1..hh6), and the truncated daily
# curve (first 180 days) -- all from the SAME simulations, so every NPE-based
# step from here on (8-13) trains on identical draws rather than each
# re-simulating its own bank.
#
# Replaces the old R export scripts (export_bank_for_python.R /
# export_curve_bank_for_python.R) from the earlier hybrid R+Python pipeline
# -- this is now the single, pure-Python source of NPE training data.
#
# Run from E:/APPEX/EpiSBI-main/measles_project/ :
#     python step7_curve_household_bank.py
# ============================================================================
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

STEP1_DIR = "output/step1"
OUT_DIR   = "output/step7"

N_SIM    = 10_000
MAX_DAYS = 180

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
HH_HIST_COLS = [f"hh{i}" for i in range(1, 7)]
DAY_COLS     = [f"d{i}" for i in range(MAX_DAYS)]


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
        inc.sum(), len(aff), aff.mean() if len(aff) else 0,
        np.mean(aff == 1) if len(aff) else 0, int(np.argmax(inc)), inc.max(),
    ], dtype=float)


def hh_histogram(cph, nbins=6):
    aff = cph[cph > 0].astype(int)
    return np.bincount(aff, minlength=nbins + 1)[1:nbins + 1].astype(float)


_worker_hh_id = None
_worker_N = None


def _pool_init(hh_id, N):
    global _worker_hh_id, _worker_N
    _worker_hh_id = hh_id
    _worker_N = N


def _worker(task):
    beta_H, beta_C, sim_seed = task
    rng = np.random.default_rng(sim_seed)
    inc, cph = simulate_hc(rng, _worker_hh_id, beta_H, beta_C, SIGMA, GAMMA, _worker_N, T_MAX,
                            INIT_INFECTED, INIT_IMMUNE_FRAC)
    summ = summary_stats(inc, cph)
    hist = hh_histogram(cph)
    curve = inc[:MAX_DAYS].astype(float)
    return np.concatenate(([beta_H, beta_C], summ, hist, curve))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {WORKERS} worker processes.")

    hh_id, N = load_population(STEP1_DIR)
    print(f"Loaded population from Step 1: N={N}, households={hh_id.max()+1}")

    obs_inc = pd.read_csv(f"{STEP1_DIR}/step1_observed_incidence.csv")["new_onsets"].to_numpy()
    obs_cph = pd.read_csv(f"{STEP1_DIR}/step1_household_final_sizes.csv")["n_cases"].to_numpy()
    obs_S = summary_stats(obs_inc, obs_cph)
    obs_hist = hh_histogram(obs_cph)
    obs_curve = obs_inc[:MAX_DAYS].astype(float)
    tp = pd.read_csv(f"{STEP1_DIR}/step1_true_params.csv")
    beta_H_true = float(tp.loc[tp["param"] == "beta_H", "value"].iloc[0])
    beta_C_true = float(tp.loc[tp["param"] == "beta_C", "value"].iloc[0])

    print(f"\nSimulating {N_SIM} outbreaks (summaries + hh histogram + curve) "
          f"on {WORKERS} workers...")
    master_rng = np.random.default_rng(2026)
    beta_H = master_rng.uniform(*PRIOR_BH, N_SIM)
    beta_C = master_rng.uniform(*PRIOR_BC, N_SIM)
    seed_seq = np.random.SeedSequence(2026)
    sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq.spawn(N_SIM)]
    tasks = [(beta_H[i], beta_C[i], sim_seeds[i]) for i in range(N_SIM)]

    chunk_size = max(1, N_SIM // (WORKERS * 20))
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init, initargs=(hh_id, N)) as executor:
        rows = list(executor.map(_worker, tasks, chunksize=chunk_size))
    rows = np.vstack(rows)

    cols = ["beta_H", "beta_C"] + SUMMARY_COLS + HH_HIST_COLS + DAY_COLS
    bank = pd.DataFrame(rows, columns=cols)
    bank.to_csv(os.path.join(OUT_DIR, "curve_household_bank.csv"), index=False)

    observed = pd.DataFrame([np.concatenate((obs_S, obs_hist, obs_curve))],
                             columns=SUMMARY_COLS + HH_HIST_COLS + DAY_COLS)
    observed["beta_H_true"] = beta_H_true
    observed["beta_C_true"] = beta_C_true
    observed.to_csv(os.path.join(OUT_DIR, "curve_household_observed.csv"), index=False)

    print(f"\nSaved curve_household_bank.csv ({N_SIM} rows x {bank.shape[1]} cols) "
          f"and curve_household_observed.csv to {os.path.abspath(OUT_DIR)}/")
    print("Observed household histogram (hh1..hh6):", obs_hist.astype(int))


if __name__ == "__main__":
    main()
