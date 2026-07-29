# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 14: SENSITIVITY SWEEP
# Repeats the curve-only / household-only / curve+household identifiability
# contrast from Step 9 across a grid of disease/population scenarios, to ask:
# does the household-clustering signal that makes beta_H identifiable hold up
# under different incubation periods, infectious periods, beta_H/beta_C
# splits, and sample sizes -- or is it specific to the Step 1 baseline?
#
# DESIGN: one-at-a-time (OAT) sweep around the Step 1 baseline
# (latent=10d, infectious=8d, beta_H=0.20, beta_C=0.15, N=1000), not a full
# factorial grid -- a factorial cross of 4 axes would multiply the number of
# NPE trainings (the expensive part) combinatorially for little extra
# inferential value. Each scenario below changes ONE axis from baseline, so
# each result isolates a single "why did identifiability change" story.
#
# WHAT'S PARALLEL vs SEQUENTIAL (same reasoning as Step 6's BSL-MCMC):
#   - Per scenario, the N_SIM_BANK training simulations ARE independent of
#     each other -> dispatched across a persistent worker pool per scenario,
#     same _pool_init pattern as steps 6/7 (hh_id/N/cfg sent once, not
#     repickled per task).
#   - Scenarios themselves are run SEQUENTIALLY, not nested-parallelized:
#     each scenario's NPE training step uses its own worker pool AND (if you
#     have one) the GPU/torch backend, and nesting process pools inside
#     process pools is fragile. If you have multiple GPUs or want to shard
#     scenarios across machines, that's a manual outer loop -- not attempted
#     here.
#
# COMPUTE NOTE: this is the slow step by design. N_SCENARIOS x 3 summary
# sets x (bank simulation + NPE training) will take a while -- that's
# expected and fine per your note. Reduce N_SIM_BANK / N_POSTERIOR_SAMPLES
# below for a fast dry run before committing to the full sweep.
#
# Run from your project root (same convention as steps 1-13):
#     python step14_sensitivity_sweep.py
# ============================================================================
import os
import time
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hc_model import (
    ScenarioConfig, SUMMARY_COLS, make_observed_outbreak, build_bank,
    hh_histogram, _pool_init,
)
from episbi.prior import UniformPrior
from episbi.inference import SBIEngine

OUT_DIR = "output/step14"

# ---- SWEEP SETTINGS (shrink these for a dry run) ---------------------------
N_SIM_BANK = 10_000          # bank simulations per scenario (Step 7 used 10,000)
MAX_DAYS = 180
N_POSTERIOR_SAMPLES = 4000
TORCH_SEED = 0
PRIOR_BH = (0.0, 0.6)
PRIOR_BC = (0.0, 0.5)
MIN_OBSERVED_CASES = 20      # retry the observed outbreak draw if it fizzles below this

WORKERS = max(1, (os.cpu_count() or 2) - 1)

PARAM_COLS = ["beta_H", "beta_C"]
GROUPS = {
    "curve only":        ["total", "peak_day", "peak_size"],
    "household only":    ["n_hh_aff", "mean_hh", "frac_singl"],
    "curve + household": SUMMARY_COLS,
}

BASELINE = ScenarioConfig(name="baseline")

# ---- OAT SCENARIOS, each changes ONE axis from BASELINE --------------------
SCENARIOS = [
    BASELINE,
    replace(BASELINE, name="short_incubation", latent_days=4),
    replace(BASELINE, name="long_incubation",  latent_days=18),
    replace(BASELINE, name="short_infectious", infectious_days=4),
    replace(BASELINE, name="long_infectious",  infectious_days=14),
    replace(BASELINE, name="weak_household",   beta_H=0.05),                 # beta_C fixed
    replace(BASELINE, name="strong_household", beta_H=0.40),                # beta_C fixed
    # same total transmission (beta_H + beta_C held ~constant at 0.35),
    # but the household/community SPLIT is reversed -- isolates whether
    # identifiability depends on the split rather than the total force
    replace(BASELINE, name="household_dominant_R0match", beta_H=0.30, beta_C=0.05),
    replace(BASELINE, name="community_dominant_R0match", beta_H=0.05, beta_C=0.30),
    replace(BASELINE, name="small_N", N_target=300),
    replace(BASELINE, name="large_N", N_target=3000),
]


def summarize_posterior(df, truth, scenario_name, varied_axis, varied_value, summary_set):
    rows = []
    for p in PARAM_COLS:
        s = df[p]
        med = s.median(); lo, hi = s.quantile(.025), s.quantile(.975)
        rows.append({
            "scenario": scenario_name, "varied_axis": varied_axis, "varied_value": varied_value,
            "summary_set": summary_set, "param": p, "true": truth[p],
            "median": float(med), "ci_lo": float(lo), "ci_hi": float(hi),
            "width": float(hi - lo), "bias": float(med - truth[p]),
            "rel_bias_pct": float(100 * (med - truth[p]) / truth[p]) if truth[p] != 0 else np.nan,
            "in_95CI": bool(lo <= truth[p] <= hi),
        })
    return rows


def varied_axis_and_value(cfg: ScenarioConfig, baseline: ScenarioConfig):
    """Identify which single field differs from baseline, for plotting."""
    for field_name in ["latent_days", "infectious_days", "beta_H", "beta_C", "N_target"]:
        if getattr(cfg, field_name) != getattr(baseline, field_name):
            return field_name, getattr(cfg, field_name)
    return "baseline", None


def run_scenario(cfg: ScenarioConfig, out_dir: str):
    scen_dir = os.path.join(out_dir, cfg.name)
    os.makedirs(scen_dir, exist_ok=True)
    print(f"\n{'='*70}\nSCENARIO: {cfg.name}  "
          f"(latent={cfg.latent_days}d, infectious={cfg.infectious_days}d, "
          f"beta_H={cfg.beta_H}, beta_C={cfg.beta_C}, N={cfg.N_target})\n{'='*70}")

    # ---- (a) observed outbreak, this scenario's own population -------------
    t0 = time.time()
    obs = make_observed_outbreak(cfg, min_total=MIN_OBSERVED_CASES)
    obs_S = hc_model_summary_stats(obs["incidence"], obs["cases_per_hh"])
    obs_hist = hh_histogram(obs["cases_per_hh"])
    obs_curve = obs["incidence"][:MAX_DAYS].astype(float)
    if len(obs_curve) < MAX_DAYS:
        obs_curve = np.concatenate([obs_curve, np.zeros(MAX_DAYS - len(obs_curve))])
    print(f"  observed outbreak: {int(obs['incidence'].sum())} total cases, "
          f"N={obs['N']}, n_hh={obs['n_hh']}  ({time.time()-t0:.1f}s)")

    pd.DataFrame([cfg.as_row()]).to_csv(os.path.join(scen_dir, "scenario_config.csv"), index=False)

    # ---- (b) training bank, parallelized ------------------------------------
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_pool_init,
                              initargs=(obs["hh_id"], obs["N"], cfg)) as executor:
        bank, hh_hist_cols, day_cols = build_bank(
            cfg, obs["hh_id"], obs["N"], n_sim=N_SIM_BANK, max_days=MAX_DAYS,
            prior_bh=PRIOR_BH, prior_bc=PRIOR_BC, executor=executor, chunk_workers=WORKERS,
        )
    bank.to_csv(os.path.join(scen_dir, "bank.csv"), index=False)
    print(f"  bank: {len(bank)} sims x {bank.shape[1]} cols  ({time.time()-t0:.1f}s, {WORKERS} workers)")

    # ---- (c) NPE identifiability contrast, 3 summary sets --------------------
    thetas = bank[PARAM_COLS].to_numpy(np.float32)
    S_all = bank[SUMMARY_COLS].to_numpy(np.float32)
    obs_all = obs_S.astype(np.float32)
    truth = {"beta_H": cfg.beta_H, "beta_C": cfg.beta_C}

    prior = UniformPrior(bounds={"beta_H": PRIOR_BH, "beta_C": PRIOR_BC})
    engine = SBIEngine(batch_size=256)

    axis, value = varied_axis_and_value(cfg, BASELINE)
    all_rows = []
    for label, cols in GROUPS.items():
        t0 = time.time()
        idx = [SUMMARY_COLS.index(c) for c in cols]
        xs = S_all[:, idx]
        obs_x = obs_all[idx]
        res = engine.run_npe(obs_data=obs_x, prior=prior, thetas=thetas, xs=xs,
                              num_samples=N_POSTERIOR_SAMPLES, show_train_summary=False)
        df = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
        df.to_csv(os.path.join(scen_dir, f"posterior_{label.replace(' ', '_').replace('+','and')}.csv"),
                   index=False)
        all_rows += summarize_posterior(df, truth, cfg.name, axis, value, label)
        print(f"  NPE [{label:20s}] done in {time.time()-t0:.1f}s")

    scen_tab = pd.DataFrame(all_rows)
    scen_tab.to_csv(os.path.join(scen_dir, "recovery_table.csv"), index=False)
    return scen_tab


# hc_model.summary_stats is imported under a clearer local alias to avoid
# shadowing the `summary_stats` name used loosely above/below in prose
from hc_model import summary_stats as hc_model_summary_stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(TORCH_SEED)
    print(f"Step 14 sensitivity sweep: {len(SCENARIOS)} scenarios, "
          f"{N_SIM_BANK} bank sims each, {WORKERS} workers for simulation, "
          f"NPE via episbi.")

    all_tables = []
    t_start = time.time()
    for cfg in SCENARIOS:
        try:
            all_tables.append(run_scenario(cfg, OUT_DIR))
        except RuntimeError as e:
            print(f"  !! SKIPPED {cfg.name}: {e}")

    master = pd.concat(all_tables, ignore_index=True)
    master.to_csv(os.path.join(OUT_DIR, "step14_sensitivity_summary.csv"), index=False)
    print(f"\nTotal sweep time: {(time.time()-t_start)/60:.1f} min")

    make_supplement_figures(master)
    print(f"\nSaved step14_sensitivity_summary.csv + figures to {os.path.abspath(OUT_DIR)}/")


# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================
def make_supplement_figures(master: pd.DataFrame):
    bh = master[master["param"] == "beta_H"].copy()

    # ---- Fig S1: beta_H CI width by summary set, faceted by scenario --------
    scen_order = [c.name for c in SCENARIOS]
    bh["scenario"] = pd.Categorical(bh["scenario"], categories=scen_order, ordered=True)
    pivot_w = bh.pivot_table(index="scenario", columns="summary_set", values="width", observed=True)
    pivot_w = pivot_w.reindex(scen_order)[list(GROUPS.keys())]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot_w.plot(kind="bar", ax=ax, width=0.75,
                 color=["#e6731a", "#1a9850", "#3b6fb6"])
    ax.set_ylabel("beta_H 95% CrI width (narrower = more identifiable)")
    ax.set_xlabel("")
    ax.set_title("Step 14: beta_H identifiability by summary set, across scenarios")
    ax.legend(title="", fontsize=8)
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step14_fig_S1_ci_width_by_scenario.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- Fig S2: beta_H median (+95% CrI) vs truth, combined summary set only
    combo = bh[bh["summary_set"] == "curve + household"].copy()
    combo["scenario"] = pd.Categorical(combo["scenario"], categories=scen_order, ordered=True)
    combo = combo.sort_values("scenario")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xpos = np.arange(len(combo))
    ax.errorbar(xpos, combo["median"], yerr=[combo["median"] - combo["ci_lo"], combo["ci_hi"] - combo["median"]],
                fmt="o", color="#3b6fb6", capsize=4, label="NPE posterior (curve+household)")
    ax.scatter(xpos, combo["true"], color="black", marker="x", s=70, label="true beta_H", zorder=5)
    ax.set_xticks(xpos); ax.set_xticklabels(combo["scenario"], rotation=40, ha="right")
    ax.set_ylabel("beta_H")
    ax.set_title("Step 14: beta_H recovery (median + 95% CrI) vs true value, by scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step14_fig_S2_recovery_vs_truth.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ---- Fig S3: coverage (in_95CI) heatmap-style summary table as CSV ------
    cov = master.pivot_table(index="scenario", columns=["summary_set", "param"],
                              values="in_95CI", observed=True)
    cov = cov.reindex(scen_order)
    cov.to_csv(os.path.join(OUT_DIR, "step14_fig_S3_coverage_table.csv"))


if __name__ == "__main__":
    main()
