# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 5: ABC REGRESSION ADJUSTMENT
# A near-free accuracy boost that reuses the simulations already saved in
# Step 3 (step3_simulation_table.csv). NO new simulations.
#
# Plain rejection ABC keeps accepted draws as-is. Regression adjustment
# (Beaumont et al. 2002) fits a locally weighted regression of parameters on
# summary statistics within the accepted set, then moves each draw to where
# its summaries would EXACTLY match the observed data -- correcting for the
# imperfect match and usually sharpening the posterior.
#
# We compare, at the SAME (loose) tolerance:
#   1. Rejection      -- no correction
#   2. Local-linear   -- Beaumont 2002 regression adjustment
#   3. Ridge          -- regularised regression adjustment (robust variant)
#
# Regression is done on logit-transformed parameters (keeps adjusted values
# inside the prior bounds) using Epanechnikov kernel weights (points closer
# to the observed summaries count more in the regression fit).
#
# Run from E:/APPEX/EpiSBI-main/measles_project/ :
#     python step5_regression_adjustment.py
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV

STEP1_DIR = "output/step1"
STEP3_DIR = "output/step3"
OUT_DIR   = "output/step5"

SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]
PARAM_BOUNDS = {"beta_H": (0.0, 0.6), "beta_C": (0.0, 0.5)}
TOL = 0.10   # loose tolerance on purpose, to make the adjustment's benefit visible


def logit(x, lo, hi, eps=1e-6):
    p = np.clip((x - lo) / (hi - lo), eps, 1 - eps)
    return np.log(p / (1 - p))


def inv_logit(y, lo, hi):
    p = 1 / (1 + np.exp(-y))
    return lo + p * (hi - lo)


def summary_stats(inc, cph):
    aff = cph[cph > 0]
    return np.array([
        inc.sum(), len(aff), aff.mean() if len(aff) else 0,
        np.mean(aff == 1) if len(aff) else 0, int(np.argmax(inc)), inc.max(),
    ], dtype=float)


def regression_adjust(theta_kept, diff_S, weights, bounds, ridge=False):
    """Beaumont et al. 2002 regression adjustment, one parameter at a time,
    in logit space. diff_S = standardised summaries (kept) minus observed."""
    lo, hi = bounds
    y = logit(theta_kept, lo, hi)
    if ridge:
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        model = LinearRegression()
    model.fit(diff_S, y, sample_weight=weights)
    # adjusted = y - beta^T * diff_S  (shift each draw to where diff_S = 0,
    # i.e. where its summaries would exactly equal the observed)
    adjusted_y = y - diff_S @ np.asarray(model.coef_).reshape(-1)
    return inv_logit(adjusted_y, lo, hi)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sim = pd.read_csv(f"{STEP3_DIR}/step3_simulation_table.csv")
    params = sim[["beta_H", "beta_C"]].to_numpy()
    sumstat = sim[SUMMARY_COLS].to_numpy()

    obs_inc = pd.read_csv(f"{STEP1_DIR}/step1_observed_incidence.csv")["new_onsets"].to_numpy()
    obs_cph = pd.read_csv(f"{STEP1_DIR}/step1_household_final_sizes.csv")["n_cases"].to_numpy()
    obs_S = summary_stats(obs_inc, obs_cph)

    beta_H_true, beta_C_true = 0.20, 0.15

    # ---- standardise summaries (same scale as distance calc) ---------------
    sds = sumstat.std(axis=0)
    sds[sds == 0] = 1
    S_std = sumstat / sds
    obs_std = obs_S / sds

    dist = np.sqrt(((S_std - obs_std) ** 2).sum(axis=1))
    threshold = np.quantile(dist, TOL)
    keep = dist <= threshold
    n_keep = keep.sum()
    print(f"Tolerance = {TOL} -> {n_keep} of {len(dist)} draws kept "
          f"(loose acceptance, on purpose)")

    theta_keep = params[keep]
    S_keep = S_std[keep]
    d_keep = dist[keep]
    diff_S = S_keep - obs_std

    # Epanechnikov kernel weights: 1 at the observed point, 0 at the tolerance
    # boundary -- points closer to the observed summaries count more.
    d_max = d_keep.max()
    weights = 1 - (d_keep / d_max) ** 2
    weights = np.clip(weights, 0, None)

    # ---- (1) plain rejection: no correction ---------------------------------
    post_rej = pd.DataFrame(theta_keep, columns=["beta_H", "beta_C"])
    post_rej["method"] = "1. Rejection"

    # ---- (2) local-linear regression adjustment ------------------------------
    adj_lin = np.column_stack([
        regression_adjust(theta_keep[:, j], diff_S, weights, PARAM_BOUNDS[p], ridge=False)
        for j, p in enumerate(["beta_H", "beta_C"])
    ])
    post_lin = pd.DataFrame(adj_lin, columns=["beta_H", "beta_C"])
    post_lin["method"] = "2. Local-linear adj."

    # ---- (3) ridge regression adjustment -------------------------------------
    adj_rid = np.column_stack([
        regression_adjust(theta_keep[:, j], diff_S, weights, PARAM_BOUNDS[p], ridge=True)
        for j, p in enumerate(["beta_H", "beta_C"])
    ])
    post_rid = pd.DataFrame(adj_rid, columns=["beta_H", "beta_C"])
    post_rid["method"] = "3. Ridge adj."

    post = pd.concat([post_rej, post_lin, post_rid], ignore_index=True)
    post.to_csv(os.path.join(OUT_DIR, "abc_adjusted_posteriors.csv"), index=False)

    # ---- summary table --------------------------------------------------------
    rows = []
    for method, sub in post.groupby("method"):
        for p in ["beta_H", "beta_C"]:
            med = sub[p].median(); lo = sub[p].quantile(.025); hi = sub[p].quantile(.975)
            rows.append({"method": method, "param": p, "median": med, "ci_lo": lo,
                         "ci_hi": hi, "ci_width": hi - lo})
    tab = pd.DataFrame(rows).sort_values(["param", "method"]).reset_index(drop=True)
    tab.to_csv(os.path.join(OUT_DIR, "abc_adjustment_summary.csv"), index=False)

    # ---- figure ------------------------------------------------------------------
    colors = {"1. Rejection": "#999999", "2. Local-linear adj.": "#3b6fb6", "3. Ridge adj.": "#1a9850"}
    truths = {"beta_H": beta_H_true, "beta_C": beta_C_true}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, p in zip(axes, ["beta_H", "beta_C"]):
        for method, sub in post.groupby("method"):
            sub[p].plot(kind="density", ax=ax, color=colors[method], label=method, linewidth=1.8)
        ax.axvline(truths[p], color="black", linestyle="--", linewidth=1.2)
        ax.set_title(p); ax.set_xlabel("value"); ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"ABC regression adjustment vs plain rejection (same simulations) | tol = {TOL} (loose)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "abc_adjustment_comparison.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- console summary -----------------------------------------------------------
    print("\n======= ABC REGRESSION ADJUSTMENT SUMMARY =======")
    print(f"True: beta_H = {beta_H_true:.2f}, beta_C = {beta_C_true:.2f} | tolerance = {TOL:.2f} (loose)\n")
    print(tab.round(3).to_string(index=False))
    print("\nWatch: at this loose tolerance, plain Rejection should be broad/biased,")
    print("while the adjusted methods pull the median toward truth and/or tighten.")
    print(f"Saved to {os.path.abspath(OUT_DIR)}/  (2 CSVs + 1 PNG)")
    print("=================================================")


if __name__ == "__main__":
    main()
