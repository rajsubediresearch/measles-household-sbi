# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 6: BAYESIAN SYNTHETIC LIKELIHOOD
# The forward-looking method: unlike ABC, it does NOT degrade as you add
# summaries (it models them jointly as multivariate normal), and it gives a
# proper likelihood you can drop into MCMC.
#
# How BSL works, at each proposed theta:
#   1. Simulate many outbreaks at theta.
#   2. Estimate the MEAN and COVARIANCE of their summary statistics.
#   3. Synthetic likelihood of the observed summaries = Normal(obs; mean, cov).
#   4. Use that inside Metropolis-Hastings.
#
# IMPORTANT design choice: BSL assumes summaries are ~Normal. Raw size
# summaries are NOT -- outbreaks either TAKE OFF or FIZZLE, giving strong
# skewness. Since the observed outbreak took off, we CONDITION ON TAKEOFF
# (keep only simulated outbreaks with total > threshold). This repairs
# normality and is principled, not a hack.
#
# PARALLELISATION: the MCMC chain itself is strictly sequential (each step
# needs the previous step's accept/reject decision), so cross-iteration
# parallelism isn't possible. What IS parallel is the batch of N_SIM_RAW
# simulations run at every step -- those are independent of each other, so
# they're dispatched across a persistent worker pool. The pool is created
# ONCE outside the MCMC loop (not per-iteration) and hh_id is sent to each
# worker ONCE via an initializer, not repickled on every one of the ~135,000
# simulation tasks across the full chain -- that repeated-pickling cost would
# otherwise dominate, given each individual simulation is only ~10ms.
#
# One consequence: simulation randomness now comes from independently-seeded
# per-task RNGs (via numpy's SeedSequence) rather than one continuous RNG
# stream shared with the MCMC proposals. This is statistically equivalent
# (each simulation is still an iid draw at its theta) but not bit-identical
# to a fully serial run -- expected literal numbers will differ slightly
# from an unparallelised version, exactly as every other seeded-vs-parallel
# comparison in this project has already shown.
#
# Run from E:/APPEX/EpiSBI-main/measles_project/ :
#     python step6_bsl.py
# ============================================================================
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular, LinAlgError

STEP1_DIR = "output/step1"
STEP2_DIR = "output/step2"
STEP3_DIR = "output/step3"
OUT_DIR   = "output/step6"

# ============================================================================
# BSL SETTINGS
# ============================================================================
N_SIM_RAW   = 45      # outbreaks simulated per MCMC step (before takeoff filter)
MIN_KEEP    = 15      # need at least this many takeoffs, else penalise theta
TAKEOFF_THR = 50       # "took off" = total cases > this
N_MCMC      = 1500
BURN        = 300
PROP_SD     = np.array([0.025, 0.020])   # random-walk steps for (beta_H, beta_C)
MCMC_SEED   = 7

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


def skewness(x):
    m = x.mean(); s = x.std()
    return 0.0 if s == 0 else np.mean((x - m) ** 3) / s ** 3


# ---- pool worker: hh_id/N set ONCE per worker process via initializer -------
_worker_hh_id = None
_worker_N = None


def _pool_init(hh_id, N):
    global _worker_hh_id, _worker_N
    _worker_hh_id = hh_id
    _worker_N = N


def _sim_worker(task):
    beta_H, beta_C, sim_seed = task
    rng = np.random.default_rng(sim_seed)
    inc, cph = simulate_hc(rng, _worker_hh_id, beta_H, beta_C, SIGMA, GAMMA, _worker_N, T_MAX,
                            INIT_INFECTED, INIT_IMMUNE_FRAC)
    return summary_stats(inc, cph)


def log_synth_lik(executor, seed_seq, theta, obs_std, scale, n_sim_raw, min_keep, takeoff_thr):
    children = seed_seq.spawn(n_sim_raw)
    sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]
    tasks = [(theta[0], theta[1], s) for s in sim_seeds]
    chunk_size = max(1, n_sim_raw // max(1, WORKERS))
    S = np.array(list(executor.map(_sim_worker, tasks, chunksize=chunk_size)))

    S = S[S[:, 0] > takeoff_thr]           # condition on takeoff (col 0 = total)
    if len(S) < min_keep:
        return -1e6                        # can't reproduce a large outbreak

    Sk = S / scale
    mu = Sk.mean(axis=0)
    Sig = np.cov(Sk, rowvar=False) + np.eye(Sk.shape[1]) * 1e-6
    try:
        L = cholesky(Sig, lower=True)
    except LinAlgError:
        return -1e6

    diff = obs_std - mu
    z = solve_triangular(L, diff, lower=True)
    k = Sk.shape[1]
    return -0.5 * k * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(z ** 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {WORKERS} worker processes for the per-step simulation batch.")

    hh_id, N = load_population(STEP1_DIR)
    print(f"Loaded population from Step 1: N={N}, households={hh_id.max()+1}")

    obs_inc = pd.read_csv(f"{STEP1_DIR}/step1_observed_incidence.csv")["new_onsets"].to_numpy()
    obs_cph = pd.read_csv(f"{STEP1_DIR}/step1_household_final_sizes.csv")["n_cases"].to_numpy()
    obs_S = summary_stats(obs_inc, obs_cph)

    scale = pd.read_csv(f"{STEP3_DIR}/step3_simulation_table.csv")[SUMMARY_COLS].std().to_numpy().copy()
    scale[scale == 0] = 1
    obs_std = obs_S / scale

    # ---- Metropolis-Hastings MCMC, batch simulations parallelised -----------
    prop_rng = np.random.default_rng(MCMC_SEED)   # proposal draws + accept/reject only
    seed_seq = np.random.SeedSequence(MCMC_SEED)   # drives all simulation randomness

    chain = np.zeros((N_MCMC, 2))
    cur = np.array([0.30, 0.30])
    acc = 0

    print(f"\nRunning BSL-MCMC: {N_MCMC} steps x {N_SIM_RAW} sims/step "
          f"(parallel batch, be patient)...")
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init, initargs=(hh_id, N)) as executor:
        cur_ll = log_synth_lik(executor, seed_seq, cur, obs_std, scale, N_SIM_RAW, MIN_KEEP, TAKEOFF_THR)

        for it in range(N_MCMC):
            prop = prop_rng.normal(cur, PROP_SD)
            if 0 < prop[0] < 0.6 and 0 < prop[1] < 0.5:
                prop_ll = log_synth_lik(executor, seed_seq, prop, obs_std, scale, N_SIM_RAW, MIN_KEEP, TAKEOFF_THR)
                if np.log(prop_rng.random()) < prop_ll - cur_ll:
                    cur, cur_ll = prop, prop_ll
                    acc += 1
            chain[it] = cur
            if (it + 1) % 250 == 0:
                print(f"  ...{it+1}/{N_MCMC} (accept {100*acc/(it+1):.0f}%)")

        # ---- (c) normality check at the posterior median, same pool --------
        post_preview = chain[BURN:]
        theta_hat = np.array([np.median(post_preview[:, 0]), np.median(post_preview[:, 1])])
        children = seed_seq.spawn(600)
        diag_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]
        diag_tasks = [(theta_hat[0], theta_hat[1], s) for s in diag_seeds]
        Sdiag = np.array(list(executor.map(_sim_worker, diag_tasks, chunksize=max(1, 600 // WORKERS))))

    post = pd.DataFrame(chain[BURN:], columns=["beta_H", "beta_C"])

    # ---- posterior summary ---------------------------------------------------
    summ = pd.DataFrame({
        "param": ["beta_H", "beta_C"], "true": [0.20, 0.15],
        "median": [post["beta_H"].median(), post["beta_C"].median()],
        "ci_lo": [post["beta_H"].quantile(.025), post["beta_C"].quantile(.025)],
        "ci_hi": [post["beta_H"].quantile(.975), post["beta_C"].quantile(.975)],
    })
    summ["in_95CI"] = (summ["true"] >= summ["ci_lo"]) & (summ["true"] <= summ["ci_hi"])

    post.to_csv(os.path.join(OUT_DIR, "bsl_posterior_draws.csv"), index=False)
    summ.to_csv(os.path.join(OUT_DIR, "bsl_posterior_summary.csv"), index=False)

    # ---- (a) trace plots -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, p, col in zip(axes, ["beta_H", "beta_C"], [0, 1]):
        ax.plot(chain[:, col], color="#7a3b9e", linewidth=0.4)
        ax.axvline(BURN, color="grey", linestyle="--")
        ax.set_title(f"BSL-MCMC trace: {p}"); ax.set_xlabel("iteration")
    fig.suptitle("dashed = end of burn-in")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bsl_traceplots.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- (b) BSL vs ABC posteriors ----------------------------------------------
    abc = pd.read_csv(f"{STEP2_DIR}/step2_abc_posterior_draws.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, p, truth in zip(axes, ["beta_H", "beta_C"], [0.20, 0.15]):
        abc[p].plot(kind="density", ax=ax, color="#e6731a", linewidth=1.8, label="ABC (Step 2)")
        post[p].plot(kind="density", ax=ax, color="#7a3b9e", linewidth=1.8, label="BSL")
        ax.axvline(truth, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(p); ax.set_xlabel("value"); ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=9)
    fig.suptitle("BSL vs ABC posteriors (black dashed = true value)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bsl_vs_abc.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- (c) normality check plot -----------------------------------------------
    Sdiag = Sdiag[Sdiag[:, 0] > TAKEOFF_THR]
    Sstd = Sdiag / scale
    skews = np.array([round(skewness(Sstd[:, j]), 2) for j in range(6)])

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for j, (ax, name) in enumerate(zip(axes.flat, SUMMARY_COLS)):
        ax.hist(Sstd[:, j], bins=25, density=True, color="#7a3b9e", alpha=0.6)
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(Sstd[:, j].min(), Sstd[:, j].max(), 200)
            ax.plot(xs, gaussian_kde(Sstd[:, j])(xs), color="black", linewidth=1.2)
        except Exception:
            pass
        ax.set_title(f"{name} (skew {skews[j]})", fontsize=9)
    fig.suptitle("BSL normality check (summaries at posterior median, takeoffs only)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bsl_normality_check.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- console summary -----------------------------------------------------------
    print("\n================ BSL SUMMARY ================")
    print(summ.round(3).to_string(index=False))
    print(f"\nAcceptance rate : {100*acc/N_MCMC:.1f}%")
    print("Summary skewness at posterior median (should all be |skew|<1):")
    print(dict(zip(SUMMARY_COLS, skews)))
    print(f"Saved to {os.path.abspath(OUT_DIR)}/  (2 CSVs + 3 PNGs)")
    print("=============================================")


if __name__ == "__main__":
    main()
