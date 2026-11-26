# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 11: CURVE LSTM + HOUSEHOLD
# Once household summaries are already included, does adding a LEARNED curve
# embedding sharpen beta_H any further than adding the three hand-picked
# curve summaries did? Compares, on the SAME simulations (Step 7's bank):
#   (A) hand-picked curve + household summaries (Step 9's combined result)
#   (B) curve LSTM + hand-picked household summaries (NEW combination)
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step11_npe_curve_household.py
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
from episbi.embedding import LSTMembedding

STEP7_DIR = "output/step7"
OUT_DIR   = "output/step11"

TORCH_SEED = 0
N_POSTERIOR_SAMPLES = 4000
MAX_DAYS = 180

PARAM_COLS = ["beta_H", "beta_C"]
SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]
HH_SUMMARY_COLS = ["n_hh_aff", "mean_hh", "frac_singl"]
DAY_COLS = [f"d{i}" for i in range(MAX_DAYS)]


class CurveHouseholdEmbedding(nn.Module):
    def __init__(self, curve_len, hh_dim=3, hh_hidden=16, lstm_hidden=32):
        super().__init__()
        self.curve_len = curve_len
        self.lstm_embed = LSTMembedding(input_dim=1, hidden_dim=lstm_hidden)
        self.hh_net = nn.Sequential(nn.Linear(hh_dim, hh_hidden), nn.ReLU())

    def forward(self, x):
        curve = x[:, : self.curve_len, :]
        hh    = x[:, self.curve_len :, 0]
        return torch.cat([self.lstm_embed(curve), self.hh_net(hh)], dim=-1)


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

    thetas  = bank[PARAM_COLS].to_numpy(np.float32)
    X_sum   = bank[SUMMARY_COLS].to_numpy(np.float32)
    obs_sum = obs[SUMMARY_COLS].to_numpy(np.float32)
    X_curve = bank[DAY_COLS].to_numpy(np.float32)
    X_hh    = bank[HH_SUMMARY_COLS].to_numpy(np.float32)
    obs_curve = obs[DAY_COLS].to_numpy(np.float32)
    obs_hh    = obs[HH_SUMMARY_COLS].to_numpy(np.float32)
    print(f"Loaded {thetas.shape[0]} sims | truth beta_H={truth['beta_H']}, beta_C={truth['beta_C']}")

    prior = UniformPrior(bounds={"beta_H": (0.0, 0.6), "beta_C": (0.0, 0.5)})
    engine = SBIEngine(batch_size=256)

    results = []

    # ---- (A) hand-picked curve + household summaries, plain NPE (baseline) -----
    print("\n[*] NPE on hand-picked curve + household summaries (baseline)...")
    res_sum = engine.run_npe(obs_data=obs_sum, prior=prior, thetas=thetas, xs=X_sum,
                             num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
    df_sum = pd.DataFrame(res_sum["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_sum, truth, "curve+household summaries (hand-picked)", PARAM_COLS)

    # ---- (B) curve LSTM + hand-picked household summaries (NEW) ----------------
    print("[*] NPE with curve LSTM + hand-picked household summaries "
          "(this is the slow one)...")
    X_combo = np.concatenate([X_curve, X_hh], axis=1)
    obs_combo = np.concatenate([obs_curve, obs_hh])
    embed_net = CurveHouseholdEmbedding(curve_len=MAX_DAYS, hh_dim=len(HH_SUMMARY_COLS))
    embed_net = embed_net.to(engine.device)
    res_combo = engine.run_npe(obs_data=obs_combo, prior=prior, thetas=thetas, xs=X_combo,
                               input_dim=1, embedding_net=embed_net,
                               num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
    df_combo = pd.DataFrame(res_combo["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_combo, truth, "curve LSTM + household summaries", PARAM_COLS)

    tab = pd.DataFrame(results)
    print("\n========= CURVE+HOUSEHOLD: HAND-PICKED vs LSTM COMBO =========")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "npe_curve_household_table.csv"), index=False)

    # ---- figure -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, p in zip(ax, PARAM_COLS):
        a.hist(df_sum[p],   bins=40, histtype="step", linewidth=2, density=True, label="hand-picked curve+hh")
        a.hist(df_combo[p], bins=40, histtype="step", linewidth=2, density=True, label="curve LSTM + hh summaries")
        a.axvline(truth[p], color="black", linestyle="--", linewidth=1.5, label="truth")
        a.set_title(p); a.set_xlabel(p)
    ax[0].legend(fontsize=7)
    fig.suptitle("beta_H / beta_C: hand-picked vs curve LSTM + household summaries (dashed = truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_curve_household_comparison.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved npe_curve_household_table.csv and npe_curve_household_comparison.png "
          f"to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
