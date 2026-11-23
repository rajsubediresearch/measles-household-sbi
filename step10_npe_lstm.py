# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 10: NPE-LSTM vs HAND-PICKED
# Tests whether a LEARNED embedding (LSTM reading the whole daily curve) can
# extract beta_H where three HAND-PICKED curve summaries (total, peak_day,
# peak_size) cannot. Both use ONLY the curve and run on the SAME simulations
# (Step 7's bank), so it's a controlled "learned vs hand-designed" test.
#
# Run from /Volumes/ssd/MSI/APPEX/EpiSBI-main/measles_project/ (or anywhere -- the script cd's to its own folder):
#     python step10_npe_lstm.py
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
OUT_DIR   = "output/step10"

TORCH_SEED = 0
N_POSTERIOR_SAMPLES = 4000
MAX_DAYS = 180

PARAM_COLS = ["beta_H", "beta_C"]
CURVE_SUMMARY_COLS = ["total", "peak_day", "peak_size"]   # hand-picked baseline
DAY_COLS = [f"d{i}" for i in range(MAX_DAYS)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(TORCH_SEED)

    bank = pd.read_csv(f"{STEP7_DIR}/curve_household_bank.csv")
    obs = pd.read_csv(f"{STEP7_DIR}/curve_household_observed.csv").iloc[0]
    truth = {"beta_H": float(obs["beta_H_true"]), "beta_C": float(obs["beta_C_true"])}

    thetas    = bank[PARAM_COLS].to_numpy(np.float32)
    X_curve   = bank[DAY_COLS].to_numpy(np.float32)
    obs_curve = obs[DAY_COLS].to_numpy(np.float32)
    X_sum     = bank[CURVE_SUMMARY_COLS].to_numpy(np.float32)
    obs_sum   = obs[CURVE_SUMMARY_COLS].to_numpy(np.float32)
    print(f"Loaded {thetas.shape[0]} sims | curve length {len(DAY_COLS)} days "
          f"| truth beta_H={truth['beta_H']}, beta_C={truth['beta_C']}")

    prior = UniformPrior(bounds={"beta_H": (0.0, 0.6), "beta_C": (0.0, 0.5)})
    engine = SBIEngine(batch_size=256)

    def summarize(df, label):
        out = []
        for p in PARAM_COLS:
            s = df[p]; med = s.median(); lo, hi = s.quantile(.025), s.quantile(.975)
            out.append({"method": label, "param": p, "true": truth[p],
                        "median": round(float(med), 3), "ci_lo": round(float(lo), 3),
                        "ci_hi": round(float(hi), 3), "width": round(float(hi - lo), 3),
                        "in_95CI": bool(lo <= truth[p] <= hi)})
        return out

    # ---- (A) LSTM learns its own summary from the raw curve --------------------
    print("\n[*] NPE-LSTM on the raw daily curve (this is the slow one)...")
    res_lstm = engine.run_npe_lstm(obs_data=obs_curve, prior=prior, thetas=thetas, xs=X_curve,
                                   input_dim=1, num_samples=N_POSTERIOR_SAMPLES,
                                   show_train_summary=False)
    df_lstm = pd.DataFrame(res_lstm["samples"].detach().cpu().numpy(), columns=PARAM_COLS)

    # ---- (B) baseline: 3 hand-picked curve summaries, SAME sims ----------------
    print("[*] NPE on hand-picked curve summaries (baseline)...")
    res_sum = engine.run_npe(obs_data=obs_sum, prior=prior, thetas=thetas, xs=X_sum,
                             num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
    df_sum = pd.DataFrame(res_sum["samples"].detach().cpu().numpy(), columns=PARAM_COLS)

    # ---- report -----------------------------------------------------------------
    tab = pd.DataFrame(summarize(df_sum, "curve summaries (hand-picked)") +
                       summarize(df_lstm, "curve LSTM (learned)"))
    print("\n========= LEARNED vs HAND-PICKED, curve only =========")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "npe_lstm_table.csv"), index=False)

    # ---- figure: beta_H / beta_C, learned vs hand-picked -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, p in zip(ax, PARAM_COLS):
        a.hist(df_sum[p],  bins=40, histtype="step", linewidth=2, density=True, label="hand-picked summaries")
        a.hist(df_lstm[p], bins=40, histtype="step", linewidth=2, density=True, label="LSTM (learned)")
        a.axvline(truth[p], color="black", linestyle="--", linewidth=1.5, label="truth")
        a.set_title(p); a.set_xlabel(p)
    ax[0].legend(fontsize=8)
    fig.suptitle("Curve only: learned LSTM embedding vs hand-picked summaries (dashed = truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_lstm_vs_handpicked.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved npe_lstm_table.csv and npe_lstm_vs_handpicked.png to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
