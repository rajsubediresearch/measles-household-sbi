# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 12: HOUSEHOLD HISTOGRAM, LEARNED
# Mirrors Step 10's "hand-designed vs learned" test, but for household data
# instead of the curve. Curve columns are ignored entirely.
#   (A) hand-picked household summaries: n_hh_aff, mean_hh, frac_singl
#   (B) learned embedding of the raw household histogram: hh1..hh6
# Both use ONLY household-level data, same simulations (Step 7's bank).
#
# A small feedforward net is used here, not an LSTM -- hh1..hh6 is a
# fixed-length count vector, not a sequence, so this should train much
# faster than Steps 10-11.
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step12_npe_household_learned.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from episbi.prior import UniformPrior
from episbi.inference import SBIEngine

STEP7_DIR = "output/step7"
OUT_DIR   = "output/step12"

TORCH_SEED = 0
N_POSTERIOR_SAMPLES = 4000

PARAM_COLS = ["beta_H", "beta_C"]
HH_SUMMARY_COLS = ["n_hh_aff", "mean_hh", "frac_singl"]
HH_HIST_COLS = [f"hh{i}" for i in range(1, 7)]


class HouseholdHistogramEmbedding(nn.Module):
    def __init__(self, hist_dim=6, hidden=16, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim), nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        return self.net(x)


def summarize(df, truth, method, param_cols):
    out = []
    for p in param_cols:
        s = df[p]; med = s.median(); lo, hi = s.quantile(.025), s.quantile(.975)
        out.append({"method": method, "param": p, "true": truth[p],
                    "median": round(float(med), 3), "ci_lo": round(float(lo), 3),
                    "ci_hi": round(float(hi), 3), "width": round(float(hi - lo), 3),
                    "in_95CI": bool(lo <= truth[p] <= hi)})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(TORCH_SEED)

    bank = pd.read_csv(f"{STEP7_DIR}/curve_household_bank.csv")
    obs = pd.read_csv(f"{STEP7_DIR}/curve_household_observed.csv").iloc[0]
    truth = {"beta_H": float(obs["beta_H_true"]), "beta_C": float(obs["beta_C_true"])}

    thetas   = bank[PARAM_COLS].to_numpy(np.float32)
    X_sum    = bank[HH_SUMMARY_COLS].to_numpy(np.float32)
    obs_sum  = obs[HH_SUMMARY_COLS].to_numpy(np.float32)
    X_hist   = bank[HH_HIST_COLS].to_numpy(np.float32)
    obs_hist = obs[HH_HIST_COLS].to_numpy(np.float32)

    print(f"Loaded {thetas.shape[0]} sims | household-only (no curve) "
          f"| truth beta_H={truth['beta_H']}, beta_C={truth['beta_C']}")
    print(f"Observed histogram hh1..hh6: {obs_hist.astype(int)}")

    prior = UniformPrior(bounds={"beta_H": (0.0, 0.6), "beta_C": (0.0, 0.5)})
    engine = SBIEngine(batch_size=256)

    results = []

    # ---- (A) hand-picked household summaries, plain NPE ------------------------
    print("\n[*] NPE on hand-picked household summaries (baseline)...")
    res_sum = engine.run_npe(obs_data=obs_sum, prior=prior, thetas=thetas, xs=X_sum,
                             num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
    df_sum = pd.DataFrame(res_sum["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_sum, truth, "household summaries (hand-picked)", PARAM_COLS)

    # ---- (B) learned feedforward embedding of the raw histogram ----------------
    print("[*] NPE with learned embedding of the raw household histogram...")
    embed_net = HouseholdHistogramEmbedding(hist_dim=len(HH_HIST_COLS))
    embed_net = embed_net.to(engine.device)
    res_hist = engine.run_npe(obs_data=obs_hist, prior=prior, thetas=thetas, xs=X_hist,
                              input_dim=1, embedding_net=embed_net,
                              num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
    df_hist = pd.DataFrame(res_hist["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_hist, truth, "household histogram (learned)", PARAM_COLS)

    tab = pd.DataFrame(results)
    print("\n========= HOUSEHOLD-ONLY: HAND-PICKED vs LEARNED HISTOGRAM =========")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "npe_household_learned_table.csv"), index=False)

    # ---- figure -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, p in zip(ax, PARAM_COLS):
        a.hist(df_sum[p],  bins=40, histtype="step", linewidth=2, density=True, label="hand-picked hh summaries")
        a.hist(df_hist[p], bins=40, histtype="step", linewidth=2, density=True, label="learned hh histogram")
        a.axvline(truth[p], color="black", linestyle="--", linewidth=1.5, label="truth")
        a.set_title(p); a.set_xlabel(p)
    ax[0].legend(fontsize=8)
    fig.suptitle("Household only: learned histogram embedding vs hand-picked summaries (dashed = truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_household_learned_comparison.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved npe_household_learned_table.csv and npe_household_learned_comparison.png "
          f"to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
