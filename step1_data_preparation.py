# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 1: DATA PREPARATION
# Household-Community stochastic SEIR simulator + synthetic "observed" outbreak
#
# This is the canonical source of the population for the ENTIRE Python
# pipeline. Every later step (2 onward) reads step1_household_final_sizes.csv
# to rebuild the SAME household partition, rather than redrawing it -- this
# is what keeps every step consistent without relying on matching random
# seeds across scripts (the R/Python hybrid pipeline needed a seed-matching
# workaround for exactly this reason; an all-Python pipeline avoids the
# problem structurally instead).
#
# Mirrors R's step1_data_preparation.R:
#   1. Builds a population grouped into households.
#   2. Runs ONE stochastic SEIR outbreak with KNOWN "true" parameters.
#   3. Saves that outbreak as the synthetic "observed data" at 3 levels:
#        Level 1: daily epidemic curve (onset counts)
#        Level 2: household final sizes (cases per household)
#        Level 3: full line list (one row per case)
#   4. Saves the true parameters so later steps can check recovery.
#
# Run from E:/APPEX/EpiSBI-main/measles_project/ :
#     python step1_data_preparation.py
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# PARAMETERS (edit these to test on different synthetic data)
# ============================================================================
SEED = 2026
rng = np.random.default_rng(SEED)

OUT_DIR = "output/step1"
os.makedirs(OUT_DIR, exist_ok=True)

# -- Population / household structure --
N_TARGET     = 1000
HH_SIZE_VALS = np.arange(1, 7)                                    # 1..6
HH_SIZE_PROB = np.array([0.10, 0.20, 0.25, 0.25, 0.12, 0.08])      # mean ~3.4

# -- Measles natural history (single-compartment, days) --
LATENT_DAYS     = 10
INFECTIOUS_DAYS = 8
SIGMA = 1 / LATENT_DAYS
GAMMA = 1 / INFECTIOUS_DAYS

# -- TRUE transmission parameters (what later steps try to recover) --
BETA_H_TRUE = 0.20   # within-household transmission (per infectious housemate / day)
BETA_C_TRUE = 0.15   # community transmission (per infectious person / day, scaled by N)

# -- Outbreak setup --
INIT_INFECTED    = 5
INIT_IMMUNE_FRAC = 0.30
T_MAX            = 320

print(f"Community-only R (no households) ~ {BETA_C_TRUE / GAMMA:.2f}")

# ============================================================================
# BUILD POPULATION (assign individuals to households)
# ============================================================================
def build_population(rng, n_target, size_vals, size_prob):
    sizes = []
    while sum(sizes) < n_target:
        sizes.append(int(rng.choice(size_vals, p=size_prob)))
    sizes = np.array(sizes)
    hh_id   = np.repeat(np.arange(len(sizes)), sizes)   # household label per person
    hh_size = np.repeat(sizes, sizes)                    # household size per person
    return hh_id, hh_size, len(hh_id), len(sizes)


hh_id, hh_sz, N, n_hh = build_population(rng, N_TARGET, HH_SIZE_VALS, HH_SIZE_PROB)
print(f"Population built: N = {N} people in {n_hh} households (mean size {hh_sz.mean():.2f})")

# ============================================================================
# SIMULATOR: stochastic individual-level Household-Community SEIR
#   Returns: daily onset incidence, per-person exposure/onset days, final states
# ============================================================================
def simulate_hc_seir_full(rng, hh_id, beta_H, beta_C, sigma, gamma, N, t_max,
                           init_infected, init_immune_frac):
    # states: 0 = S, 1 = E, 2 = I, 3 = R
    state = np.zeros(N, dtype=np.int8)
    n_hh = int(hh_id.max()) + 1

    if init_immune_frac > 0:
        immune = rng.choice(N, size=round(init_immune_frac * N), replace=False)
        state[immune] = 3

    seeds = rng.choice(np.flatnonzero(state == 0), size=init_infected, replace=False)
    state[seeds] = 2

    exposure_day = np.full(N, np.nan)
    onset_day    = np.full(N, np.nan)
    onset_day[seeds] = 0                      # seeds count as onset on day 0

    incidence = np.zeros(t_max + 1, dtype=np.int64)  # NEW onsets (E->I) per day
    incidence[0] = init_infected

    p_EI = 1 - np.exp(-sigma)
    p_IR = 1 - np.exp(-gamma)

    for t in range(1, t_max + 1):
        is_S = state == 0
        is_E = state == 1
        is_I = state == 2

        if not is_I.any() and not is_E.any():
            break

        # infectious housemates for each person (exclude self), community total
        I_in_hh      = np.bincount(hh_id, weights=is_I.astype(float), minlength=n_hh)
        I_housemates = I_in_hh[hh_id] - is_I.astype(float)
        I_community  = is_I.sum()

        # ---- force of infection: household + community ----
        hazard = beta_H * I_housemates + beta_C * I_community / N
        p_inf  = 1 - np.exp(-hazard)

        # draw transitions from CURRENT state (synchronous update)
        new_E = is_S & (rng.random(N) < p_inf)
        new_I = is_E & (rng.random(N) < p_EI)
        new_R = is_I & (rng.random(N) < p_IR)

        state[new_R] = 3
        state[new_I] = 2
        state[new_E] = 1

        exposure_day[new_E] = t
        onset_day[new_I]    = t
        incidence[t]        = new_I.sum()

    return {
        "incidence": incidence,
        "exposure_day": exposure_day,
        "onset_day": onset_day,
        "final_state": state,
    }


# ============================================================================
# synthetic "observed" outbreak
# ============================================================================
print("Simulating synthetic outbreak with true parameters...")
sim = simulate_hc_seir_full(rng, hh_id, BETA_H_TRUE, BETA_C_TRUE, SIGMA, GAMMA,
                             N, T_MAX, INIT_INFECTED, INIT_IMMUNE_FRAC)

# ---- LEVEL 1: daily epidemic curve ----------------------------------------
incidence_df = pd.DataFrame({
    "day": np.arange(T_MAX + 1),
    "new_onsets": sim["incidence"],
    "cumulative": np.cumsum(sim["incidence"]),
})

# ---- LEVEL 3: line list (one row per case) --------------------------------
case_idx = np.flatnonzero(~np.isnan(sim["onset_day"]))
linelist = pd.DataFrame({
    "id": case_idx,
    "hh_id": hh_id[case_idx],
    "hh_size": hh_sz[case_idx],
    "exposure_day": sim["exposure_day"][case_idx],
    "onset_day": sim["onset_day"][case_idx],
}).sort_values("onset_day").reset_index(drop=True)

# ---- LEVEL 2: household final sizes ---------------------------------------
has_onset = ~np.isnan(sim["onset_day"])
cases_per_hh = np.bincount(hh_id, weights=has_onset.astype(float), minlength=n_hh).astype(int)
hh_size_lookup = np.zeros(n_hh, dtype=int)
hh_size_lookup[hh_id] = hh_sz  # any member's hh_size works, all members share it
hh_final = pd.DataFrame({
    "hh_id": np.arange(n_hh),
    "hh_size": hh_size_lookup,
    "n_cases": cases_per_hh,
})

# ---- TRUE PARAMETERS (so later steps can score recovery) ------------------
true_params = pd.DataFrame({
    "param": ["beta_H", "beta_C", "sigma", "gamma", "latent_days", "infectious_days",
              "N", "init_infected", "init_immune_frac"],
    "value": [BETA_H_TRUE, BETA_C_TRUE, SIGMA, GAMMA, LATENT_DAYS, INFECTIOUS_DAYS,
              N, INIT_INFECTED, INIT_IMMUNE_FRAC],
})

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
incidence_df.to_csv(os.path.join(OUT_DIR, "step1_observed_incidence.csv"), index=False)
hh_final.to_csv(os.path.join(OUT_DIR, "step1_household_final_sizes.csv"), index=False)
linelist.to_csv(os.path.join(OUT_DIR, "step1_linelist.csv"), index=False)
true_params.to_csv(os.path.join(OUT_DIR, "step1_true_params.csv"), index=False)

# ============================================================================
# PLOTS
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(incidence_df["day"], incidence_df["new_onsets"], color="#4575b4", width=0.9)
ax.set_title("Synthetic Measles Outbreak - Daily Onsets (Level 1 data)")
ax.set_xlabel("Day"); ax.set_ylabel("New symptomatic onsets")
fig.suptitle(f"True beta_H = {BETA_H_TRUE:.2f}, beta_C = {BETA_C_TRUE:.2f} | N = {N}", fontsize=9, y=0.95)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step1_epidemic_curve.png"), dpi=200, facecolor="white")
plt.close(fig)

affected = hh_final[hh_final["n_cases"] > 0]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(affected["n_cases"], bins=np.arange(0.5, affected["n_cases"].max() + 1.5, 1),
        color="#d73027", edgecolor="white")
ax.set_title("Household Final Sizes among Affected Households (Level 2 data)")
ax.set_xlabel("Cases in household"); ax.set_ylabel("Number of households")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "step1_household_finalsize_hist.png"), dpi=200, facecolor="white")
plt.close(fig)

# ============================================================================
# CONSOLE SUMMARY
# ============================================================================
total_cases = len(linelist)
peak_day = int(incidence_df.loc[incidence_df["new_onsets"].idxmax(), "day"])
print("\n================ STEP 1 SUMMARY ================")
print(f"Total cases          : {total_cases}  (attack rate {100*total_cases/N:.1f}%)")
print(f"Peak day / peak size  : day {peak_day} / {incidence_df['new_onsets'].max()} onsets")
print(f"Outbreak duration     : {int(linelist['onset_day'].max())} days (last onset)")
print(f"Households affected   : {(hh_final['n_cases']>0).sum()} of {n_hh} "
      f"({100*(hh_final['n_cases']>0).mean():.1f}%)")
print(f"Mean cases / affected hh: {affected['n_cases'].mean():.2f}")
print(f"Saved to {os.path.abspath(OUT_DIR)}/  (4 CSVs + 2 PNGs)")
print("===============================================")
