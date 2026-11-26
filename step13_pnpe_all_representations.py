# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 13: PNPE, ALL REPRESENTATIONS
# ABC-preconditioned NPE tested against all three representations built in
# Steps 9-12: hand-picked summaries, curve LSTM + household summaries, and
# the learned household-histogram embedding.
#
# Structure:
#   A. Plain NPE baselines (full-prior bank = Step 7's bank), all 3 reps.
#   B. ABC pilot (live simulation, summary-distance based) -> proposal region.
#   C. ONE preconditioned training bank (live simulation, rich: summaries +
#      histogram + curve together) drawn from that proposal region.
#   D. PNPE trained on all 3 representations, same preconditioned bank.
#
# Each of the 6 training calls gets its own fixed torch seed (reseed() below)
# so results are independent of call order -- this was a real bug in an
# earlier version of this test (shared RNG stream across sequential
# trainings gave one representation an unlucky initialisation depending on
# what ran before it). Fixed here from the start.
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step13_pnpe_all_representations.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
from concurrent.futures import ProcessPoolExecutor

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from episbi.inference import SBIEngine
from episbi.prior import UniformPrior
from episbi.embedding import LSTMembedding

STEP1_DIR = "output/step1"
STEP7_DIR = "output/step7"
OUT_DIR   = "output/step13"

seed = 2026
torch_seed = 0
n_pilot = 20_000
n_train = 10_000
n_posterior_samples = 4_000
accept_frac = 0.01
proposal_padding = 0.1
workers = max(1, (os.cpu_count() or 2) - 1)
MAX_DAYS = 180

LATENT_DAYS = 10
INFECTIOUS_DAYS = 8
SIGMA = 1 / LATENT_DAYS
GAMMA = 1 / INFECTIOUS_DAYS
INIT_INFECTED = 5
INIT_IMMUNE_FRAC = 0.30
T_MAX = 320

PRIOR_BH = (0.0, 0.6)
PRIOR_BC = (0.0, 0.5)

PARAM_COLS      = ["beta_H", "beta_C"]
SUMMARY_COLS    = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]
HH_SUMMARY_COLS = ["n_hh_aff", "mean_hh", "frac_singl"]
HH_HIST_COLS    = [f"hh{i}" for i in range(1, 7)]
DAY_COLS        = [f"d{i}" for i in range(MAX_DAYS)]


def reseed(offset):
    torch.manual_seed(torch_seed + offset)


# ============================================================================
# simulator (shared by pilot + preconditioned bank)
# ============================================================================
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
    p_EI = 1 - np.exp(-sigma); p_IR = 1 - np.exp(-gamma)
    for t in range(1, t_max + 1):
        is_S = state == 0; is_E = state == 1; is_I = state == 2
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


def _summary_worker(task):
    """Used by the ABC pilot -- summaries only, fast."""
    beta_H, beta_C, sim_seed = task
    rng = np.random.default_rng(sim_seed)
    inc, cph = simulate_hc(rng, _worker_hh_id, beta_H, beta_C, SIGMA, GAMMA, _worker_N, T_MAX,
                            INIT_INFECTED, INIT_IMMUNE_FRAC)
    return np.concatenate(([beta_H, beta_C], summary_stats(inc, cph)))


def _full_worker(task):
    """Used by the preconditioned training bank -- summaries + histogram + curve."""
    beta_H, beta_C, sim_seed = task
    rng = np.random.default_rng(sim_seed)
    inc, cph = simulate_hc(rng, _worker_hh_id, beta_H, beta_C, SIGMA, GAMMA, _worker_N, T_MAX,
                            INIT_INFECTED, INIT_IMMUNE_FRAC)
    summ = summary_stats(inc, cph)
    hist = hh_histogram(cph)
    curve = inc[:MAX_DAYS].astype(float)
    return np.concatenate(([beta_H, beta_C], summ, hist, curve))


def distance_to_observed(S, obs_S):
    scale = S.std(axis=0)
    scale[scale == 0] = 1
    z = (S - obs_S) / scale
    return np.sqrt((z * z).sum(axis=1))


def make_proposal_bounds(theta_keep):
    proposal = {}
    original = {"beta_H": PRIOR_BH, "beta_C": PRIOR_BC}
    for j, name in enumerate(PARAM_COLS):
        lo = float(np.quantile(theta_keep[:, j], 0.01))
        hi = float(np.quantile(theta_keep[:, j], 0.99))
        width = hi - lo
        lo = lo - proposal_padding * width
        hi = hi + proposal_padding * width
        lo = max(original[name][0], lo)
        hi = min(original[name][1], hi)
        if hi <= lo:
            lo, hi = original[name]
        proposal[name] = (lo, hi)
    return proposal


# ============================================================================
# embedding architectures (same as Steps 11 and 12)
# ============================================================================
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


def summarize(df, truth, method, label):
    out = []
    for p in PARAM_COLS:
        s = df[p]; med = s.median(); lo, hi = s.quantile(.025), s.quantile(.975)
        out.append({"representation": label, "method": method, "param": p, "true": truth[p],
                    "median": round(float(med), 3), "ci_lo": round(float(lo), 3),
                    "ci_hi": round(float(hi), 3), "width": round(float(hi - lo), 3),
                    "in_95CI": bool(lo <= truth[p] <= hi)})
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {workers} worker processes.")

    # ========================================================================
    # STAGE A: plain NPE baselines, all 3 representations, Step 7's bank
    # ========================================================================
    print("\n[A] Loading Step 7 bank and training plain-NPE baselines ...", flush=True)
    bank = pd.read_csv(f"{STEP7_DIR}/curve_household_bank.csv")
    obs  = pd.read_csv(f"{STEP7_DIR}/curve_household_observed.csv").iloc[0]
    truth = {"beta_H": float(obs["beta_H_true"]), "beta_C": float(obs["beta_C_true"])}

    thetas_full     = bank[PARAM_COLS].to_numpy(np.float32)
    X_sum_full      = bank[SUMMARY_COLS].to_numpy(np.float32)
    X_curvehh_full  = np.concatenate(
        [bank[DAY_COLS].to_numpy(np.float32), bank[HH_SUMMARY_COLS].to_numpy(np.float32)], axis=1)
    X_hist_full     = bank[HH_HIST_COLS].to_numpy(np.float32)

    obs_sum     = obs[SUMMARY_COLS].to_numpy(np.float32)
    obs_curvehh = np.concatenate([obs[DAY_COLS].to_numpy(np.float32), obs[HH_SUMMARY_COLS].to_numpy(np.float32)])
    obs_hist    = obs[HH_HIST_COLS].to_numpy(np.float32)

    full_prior = UniformPrior(bounds={"beta_H": PRIOR_BH, "beta_C": PRIOR_BC})
    engine = SBIEngine(batch_size=256)

    results = []

    reseed(1)
    res = engine.run_npe(obs_data=obs_sum, prior=full_prior, thetas=thetas_full, xs=X_sum_full,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_plain_sum = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_plain_sum, truth, "plain NPE", "hand-picked")

    reseed(2)
    embed_curvehh = CurveHouseholdEmbedding(curve_len=MAX_DAYS, hh_dim=len(HH_SUMMARY_COLS)).to(engine.device)
    res = engine.run_npe(obs_data=obs_curvehh, prior=full_prior, thetas=thetas_full, xs=X_curvehh_full,
                         input_dim=1, embedding_net=embed_curvehh,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_plain_curvehh = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_plain_curvehh, truth, "plain NPE", "curve LSTM + household summaries")

    reseed(3)
    embed_hist = HouseholdHistogramEmbedding(hist_dim=len(HH_HIST_COLS)).to(engine.device)
    res = engine.run_npe(obs_data=obs_hist, prior=full_prior, thetas=thetas_full, xs=X_hist_full,
                         input_dim=1, embedding_net=embed_hist,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_plain_hist = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_plain_hist, truth, "plain NPE", "household histogram (learned)")

    print("Plain-NPE baselines done.", flush=True)

    # ========================================================================
    # STAGE B: ABC pilot (live simulation, summary-distance based)
    # ========================================================================
    print(f"\n[B] ABC pilot: {n_pilot} simulations on {workers} workers ...", flush=True)
    hh_id, N = load_population(STEP1_DIR)
    print(f"Population: N={N}, households={hh_id.max()+1}", flush=True)

    with ProcessPoolExecutor(max_workers=workers, initializer=_pool_init, initargs=(hh_id, N)) as executor:
        master_rng = np.random.default_rng(seed + 1)
        beta_H = master_rng.uniform(*PRIOR_BH, n_pilot)
        beta_C = master_rng.uniform(*PRIOR_BC, n_pilot)
        seed_seq = np.random.SeedSequence(seed + 1)
        sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq.spawn(n_pilot)]
        tasks = [(beta_H[i], beta_C[i], sim_seeds[i]) for i in range(n_pilot)]
        chunk_size = max(1, n_pilot // (workers * 20))
        rows = list(executor.map(_summary_worker, tasks, chunksize=chunk_size))
        rows = np.vstack(rows).astype(np.float32)
        theta_pilot, S_pilot = rows[:, :2], rows[:, 2:]

        d = distance_to_observed(S_pilot, obs_sum)
        keep_n = max(20, int(np.ceil(n_pilot * accept_frac)))
        keep_idx = np.argsort(d)[:keep_n]
        proposal_bounds = make_proposal_bounds(theta_pilot[keep_idx])
        print("Proposal bounds:", proposal_bounds, flush=True)

        pd.DataFrame([{"param": k, "proposal_lo": v[0], "proposal_hi": v[1]} for k, v in proposal_bounds.items()]) \
          .to_csv(os.path.join(OUT_DIR, "pnpe_proposal_bounds.csv"), index=False)

        # ====================================================================
        # STAGE C: preconditioned training bank -- rich version, same pool
        # ====================================================================
        print(f"\n[C] Preconditioned training bank: {n_train} simulations "
              f"on {workers} workers ...", flush=True)
        master_rng2 = np.random.default_rng(seed + 2)
        beta_H2 = master_rng2.uniform(*proposal_bounds["beta_H"], n_train)
        beta_C2 = master_rng2.uniform(*proposal_bounds["beta_C"], n_train)
        seed_seq2 = np.random.SeedSequence(seed + 2)
        sim_seeds2 = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq2.spawn(n_train)]
        tasks2 = [(beta_H2[i], beta_C2[i], sim_seeds2[i]) for i in range(n_train)]
        chunk_size2 = max(1, n_train // (workers * 20))
        rows2 = list(executor.map(_full_worker, tasks2, chunksize=chunk_size2))
        rows2 = np.vstack(rows2).astype(np.float32)

    theta_train = rows2[:, :2]
    summ_train  = rows2[:, 2:8]
    hist_train  = rows2[:, 8:14]
    curve_train = rows2[:, 14:14 + MAX_DAYS]

    hh_idx = [SUMMARY_COLS.index(c) for c in HH_SUMMARY_COLS]
    X_train_curvehh = np.concatenate([curve_train, summ_train[:, hh_idx]], axis=1)

    pnpe_prior = UniformPrior(bounds=proposal_bounds)

    # ========================================================================
    # STAGE D: train PNPE on all 3 representations, same preconditioned bank
    # ========================================================================
    print("\n[D] Training PNPE across all three representations ...", flush=True)

    reseed(1)
    res = engine.run_npe(obs_data=obs_sum, prior=pnpe_prior, thetas=theta_train, xs=summ_train,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_pnpe_sum = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_pnpe_sum, truth, "PNPE", "hand-picked")

    reseed(2)
    embed_curvehh2 = CurveHouseholdEmbedding(curve_len=MAX_DAYS, hh_dim=len(HH_SUMMARY_COLS)).to(engine.device)
    res = engine.run_npe(obs_data=obs_curvehh, prior=pnpe_prior, thetas=theta_train, xs=X_train_curvehh,
                         input_dim=1, embedding_net=embed_curvehh2,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_pnpe_curvehh = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_pnpe_curvehh, truth, "PNPE", "curve LSTM + household summaries")

    reseed(3)
    embed_hist2 = HouseholdHistogramEmbedding(hist_dim=len(HH_HIST_COLS)).to(engine.device)
    res = engine.run_npe(obs_data=obs_hist, prior=pnpe_prior, thetas=theta_train, xs=hist_train,
                         input_dim=1, embedding_net=embed_hist2,
                         num_samples=n_posterior_samples, show_train_summary=False)
    df_pnpe_hist = pd.DataFrame(res["samples"].detach().cpu().numpy(), columns=PARAM_COLS)
    results += summarize(df_pnpe_hist, truth, "PNPE", "household histogram (learned)")

    # ========================================================================
    # REPORT
    # ========================================================================
    tab = pd.DataFrame(results)[["representation", "method", "param", "true", "median",
                                  "ci_lo", "ci_hi", "width", "in_95CI"]]
    print("\n================ PNPE vs plain NPE, all representations ================")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT_DIR, "pnpe_all_representations_table.csv"), index=False)

    pairs = [
        ("hand-picked", df_plain_sum, df_pnpe_sum),
        ("curve LSTM + household summaries", df_plain_curvehh, df_pnpe_curvehh),
        ("household histogram (learned)", df_plain_hist, df_pnpe_hist),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))
    for row, (label, df_plain, df_pnpe) in enumerate(pairs):
        for col, p in enumerate(PARAM_COLS):
            a = axes[row, col]
            a.hist(df_plain[p], bins=40, histtype="step", linewidth=2, density=True, label="plain NPE")
            a.hist(df_pnpe[p],  bins=40, histtype="step", linewidth=2, density=True, label="PNPE")
            a.axvline(truth[p], color="black", linestyle="--", linewidth=1.2, label="truth")
            a.set_title(f"{label}\n{p}", fontsize=9)
            if row == 0 and col == 0:
                a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pnpe_all_representations_comparison.png"), dpi=130)
    plt.close(fig)

    print(f"\nSaved pnpe_all_representations_table.csv and "
          f"pnpe_all_representations_comparison.png to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
