# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 9: NPE RECOVERY
# Loads the Step 7 bank and trains NPE to recover (beta_H, beta_C) under
# three summary subsets, mirroring the Step 3 ABC identifiability experiment
# but via NPE instead of ABC rejection:
#   (a) Curve only        : total cases, peak day, peak size
#   (b) Household only     : #hh affected, mean hh size, frac singletons
#   (c) Curve + Household  : all six
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step9_npe_recovery.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from episbi.prior import UniformPrior
from episbi.inference import SBIEngine

STEP7_DIR = "output/step7"
OUT_DIR   = "output/step9"

TORCH_SEED = 0
N_POSTERIOR_SAMPLES = 4000

PARAM_COLS = ["beta_H", "beta_C"]
SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]
GROUPS = {
    "curve only":        ["total", "peak_day", "peak_size"],
    "household only":    ["n_hh_aff", "mean_hh", "frac_singl"],
    "curve + household": SUMMARY_COLS,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(TORCH_SEED)

    bank = pd.read_csv(f"{STEP7_DIR}/curve_household_bank.csv")
    obs = pd.read_csv(f"{STEP7_DIR}/curve_household_observed.csv").iloc[0]
    truth = {"beta_H": float(obs["beta_H_true"]), "beta_C": float(obs["beta_C_true"])}

    thetas = bank[PARAM_COLS].to_numpy(np.float32)
    S_all = bank[SUMMARY_COLS].to_numpy(np.float32)
    obs_all = obs[SUMMARY_COLS].to_numpy(np.float32)
    print(f"Loaded bank: {thetas.shape[0]} sims | truth beta_H={truth['beta_H']}, beta_C={truth['beta_C']}")

    prior = UniformPrior(bounds={"beta_H": (0.0, 0.6), "beta_C": (0.0, 0.5)})
    engine = SBIEngine(batch_size=256)

    def run(cols):
        idx = [SUMMARY_COLS.index(c) for c in cols]
        xs = S_all[:, idx]
        obs_x = obs_all[idx]
        res = engine.run_npe(obs_data=obs_x, prior=prior, thetas=thetas, xs=xs,
                             num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
        return pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)

    rows, posteriors = [], {}
    for label, cols in GROUPS.items():
        print(f"\n[*] NPE with summary set: {label}")
        df = run(cols)
        posteriors[label] = df
        for p in PARAM_COLS:
            s = df[p]; med = s.median(); lo, hi = s.quantile(.025), s.quantile(.975)
            rows.append({"summary_set": label, "param": p, "true": truth[p],
                         "median": round(float(med), 3), "ci_lo": round(float(lo), 3),
                         "ci_hi": round(float(hi), 3), "width": round(float(hi - lo), 3),
                         "in_95CI": bool(lo <= truth[p] <= hi)})

    tab = pd.DataFrame(rows)
    print("\n================ NPE RECOVERY vs TRUTH ================")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "npe_recovery_table.csv"), index=False)

    # ---- figure 1: marginal posteriors under the combined set ------------------
    df = posteriors["curve + household"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, p in zip(ax, PARAM_COLS):
        a.hist(df[p], bins=40, color="#4575b4", alpha=.85)
        a.axvline(truth[p], color="red", linestyle="--", linewidth=1.6)
        a.set_title(p); a.set_xlabel("posterior sample")
    fig.suptitle("NPE on measles bank -- curve + household (dashed = truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_marginals.png"), dpi=130)
    plt.close(fig)

    # ---- figure 2: beta_H across summary sets (the identifiability story) -------
    fig, a = plt.subplots(figsize=(6.2, 3.6))
    for label, df in posteriors.items():
        a.hist(df["beta_H"], bins=40, histtype="step", linewidth=2, density=True, label=label)
    a.axvline(truth["beta_H"], color="black", linestyle="--", linewidth=1.5, label="truth")
    a.set_title("beta_H posterior by summary set"); a.set_xlabel("beta_H"); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_betaH_identifiability.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved npe_recovery_table.csv, npe_marginals.png, "
          f"npe_betaH_identifiability.png to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
