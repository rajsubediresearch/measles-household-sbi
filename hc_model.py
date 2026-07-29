# ============================================================================
# APPEX Measles SBI Project (Python) -- SHARED MODEL MODULE
# Parameterized household-community stochastic SEIR simulator + bank-building
# utilities, factored out of steps 1/2/3/4/6/7 so that step14's sensitivity
# sweep (and any future step) can drive the SAME simulator across different
# natural-history / transmission / population settings without copy-pasting
# the simulation loop again.
#
# Nothing in here changes the model itself -- this is a straight refactor of
# simulate_hc_seir_full / simulate_hc / summary_stats / hh_histogram /
# build_population, unified into one parameterized version. Steps 1-13 keep
# working exactly as before if you leave them alone; step14 imports from here.
# ============================================================================
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    """Everything that determines one household-community SEIR scenario.

    Defaults reproduce the Step 1 baseline exactly (latent=10, infectious=8,
    beta_H=0.20, beta_C=0.15, N_target=1000).
    """
    name: str = "baseline"

    # -- population / household structure --
    N_target: int = 1000
    hh_size_vals: Sequence[int] = field(default_factory=lambda: tuple(range(1, 7)))  # 1..6
    hh_size_prob: Sequence[float] = field(
        default_factory=lambda: (0.10, 0.20, 0.25, 0.25, 0.12, 0.08))  # mean ~3.4

    # -- natural history (days) --
    latent_days: float = 10
    infectious_days: float = 8

    # -- true transmission parameters --
    beta_H: float = 0.20   # within-household transmission
    beta_C: float = 0.15   # community transmission

    # -- outbreak setup --
    init_infected: int = 5
    init_immune_frac: float = 0.30
    t_max: int = 320

    # -- reproducibility --
    seed: int = 2026

    @property
    def sigma(self) -> float:
        return 1.0 / self.latent_days

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_days

    def as_row(self) -> dict:
        """Flat dict for logging one row per scenario into a summary table."""
        d = asdict(self)
        d.pop("hh_size_vals"); d.pop("hh_size_prob")
        d["sigma"] = self.sigma
        d["gamma"] = self.gamma
        d["R0_community_only"] = self.beta_C / self.gamma
        return d


SUMMARY_COLS = ["total", "n_hh_aff", "mean_hh", "frac_singl", "peak_day", "peak_size"]


# ============================================================================
# POPULATION
# ============================================================================
def build_population(rng, cfg: ScenarioConfig):
    size_vals = np.asarray(cfg.hh_size_vals)
    size_prob = np.asarray(cfg.hh_size_prob)
    sizes = []
    while sum(sizes) < cfg.N_target:
        sizes.append(int(rng.choice(size_vals, p=size_prob)))
    sizes = np.array(sizes)
    hh_id = np.repeat(np.arange(len(sizes)), sizes)
    hh_size = np.repeat(sizes, sizes)
    return hh_id, hh_size, len(hh_id), len(sizes)


def load_population_from_csv(path):
    """Rebuild a household partition from a saved step1-style
    *_household_final_sizes.csv (hh_size column), same trick steps 2-7 use
    to stay consistent with a fixed 'observed' population without redrawing."""
    hh_final = pd.read_csv(path)
    hh_sizes = hh_final["hh_size"].to_numpy()
    hh_id = np.repeat(np.arange(len(hh_sizes)), hh_sizes)
    return hh_id, len(hh_id)


def rowsum(x, group, n_groups):
    return np.bincount(group, weights=x, minlength=n_groups)


# ============================================================================
# SIMULATOR
# ============================================================================
def simulate_hc_seir(rng, hh_id, cfg: ScenarioConfig, N, full_output=False):
    """One stochastic household-community SEIR run under cfg.

    full_output=True  -> also returns exposure/onset day per person (used by
                          the single "observed outbreak" draw, step1-style).
    full_output=False -> lightweight version (incidence + cases-per-household
                          only), used for the thousands of bank simulations.
    """
    n_hh = int(hh_id.max()) + 1
    state = np.zeros(N, dtype=np.int8)   # 0=S 1=E 2=I 3=R

    if cfg.init_immune_frac > 0:
        immune = rng.choice(N, size=round(cfg.init_immune_frac * N), replace=False)
        state[immune] = 3
    seeds = rng.choice(np.flatnonzero(state == 0), size=cfg.init_infected, replace=False)
    state[seeds] = 2

    ever_I = np.zeros(N, dtype=bool)
    ever_I[seeds] = True
    incidence = np.zeros(cfg.t_max + 1, dtype=np.int64)
    incidence[0] = cfg.init_infected

    if full_output:
        exposure_day = np.full(N, np.nan)
        onset_day = np.full(N, np.nan)
        onset_day[seeds] = 0

    p_EI = 1 - np.exp(-cfg.sigma)
    p_IR = 1 - np.exp(-cfg.gamma)

    for t in range(1, cfg.t_max + 1):
        is_S = state == 0
        is_E = state == 1
        is_I = state == 2
        if not is_I.any() and not is_E.any():
            break

        hh_I = rowsum(is_I.astype(float), hh_id, n_hh)
        I_housemates = hh_I[hh_id] - is_I.astype(float)
        hazard = cfg.beta_H * I_housemates + cfg.beta_C * is_I.sum() / N
        p_inf = 1 - np.exp(-hazard)

        new_E = is_S & (rng.random(N) < p_inf)
        new_I = is_E & (rng.random(N) < p_EI)
        new_R = is_I & (rng.random(N) < p_IR)

        state[new_R] = 3; state[new_I] = 2; state[new_E] = 1
        ever_I[new_I] = True
        incidence[t] = new_I.sum()

        if full_output:
            exposure_day[new_E] = t
            onset_day[new_I] = t

    cph = rowsum(ever_I.astype(float), hh_id, n_hh)

    if full_output:
        return {
            "incidence": incidence, "cases_per_hh": cph,
            "exposure_day": exposure_day, "onset_day": onset_day,
            "final_state": state,
        }
    return incidence, cph


# ============================================================================
# SUMMARIES
# ============================================================================
def summary_stats(inc, cph):
    aff = cph[cph > 0]
    return np.array([
        inc.sum(),
        len(aff),
        aff.mean() if len(aff) else 0,
        np.mean(aff == 1) if len(aff) else 0,
        int(np.argmax(inc)),
        inc.max(),
    ], dtype=float)


def hh_histogram(cph, nbins=6):
    aff = cph[cph > 0].astype(int)
    return np.bincount(aff, minlength=nbins + 1)[1:nbins + 1].astype(float)


def took_off(inc, threshold):
    return inc.sum() > threshold


# ============================================================================
# "OBSERVED" OUTBREAK (step1-equivalent, parameterized)
# Retries with fresh seeds if the epidemic fizzles (common with short
# incubation / small N / weak beta scenarios) so every scenario in the sweep
# actually has a real outbreak to condition on, rather than a silent stub.
# ============================================================================
def make_observed_outbreak(cfg: ScenarioConfig, min_total=20, max_tries=25):
    rng = np.random.default_rng(cfg.seed)
    hh_id, hh_sz, N, n_hh = build_population(rng, cfg)

    for attempt in range(max_tries):
        sim = simulate_hc_seir(rng, hh_id, cfg, N, full_output=True)
        if sim["incidence"].sum() >= min_total:
            break
    else:
        raise RuntimeError(
            f"[{cfg.name}] observed outbreak fizzled below min_total={min_total} "
            f"after {max_tries} tries -- widen beta_H/beta_C or lower min_total "
            f"for this scenario."
        )

    return {
        "hh_id": hh_id, "hh_size": hh_sz, "N": N, "n_hh": n_hh,
        "incidence": sim["incidence"],
        "cases_per_hh": sim["cases_per_hh"],
    }


# ============================================================================
# TRAINING BANK (step7-equivalent, parameterized)
# Uses a ProcessPoolExecutor with a pool initializer, same pattern as steps
# 6/7: hh_id/N/cfg sent to each worker ONCE, not repickled per task.
# ============================================================================
_worker_hh_id = None
_worker_N = None
_worker_cfg = None


def _pool_init(hh_id, N, cfg):
    global _worker_hh_id, _worker_N, _worker_cfg
    _worker_hh_id = hh_id
    _worker_N = N
    _worker_cfg = cfg


def _bank_worker(task):
    beta_H, beta_C, sim_seed, max_days = task
    rng = np.random.default_rng(sim_seed)
    cfg_i = ScenarioConfig(**{**asdict(_worker_cfg), "beta_H": beta_H, "beta_C": beta_C})
    inc, cph = simulate_hc_seir(rng, _worker_hh_id, cfg_i, _worker_N, full_output=False)
    summ = summary_stats(inc, cph)
    hist = hh_histogram(cph)
    curve = inc[:max_days].astype(float)
    if len(curve) < max_days:  # pad short (fizzled) runs with zeros
        curve = np.concatenate([curve, np.zeros(max_days - len(curve))])
    return np.concatenate(([beta_H, beta_C], summ, hist, curve))


def build_bank(cfg: ScenarioConfig, hh_id, N, n_sim, max_days, prior_bh, prior_bc,
                executor, chunk_workers, bank_seed=None):
    """Simulate n_sim prior draws in parallel and return the curve/household
    bank DataFrame, mirroring step7's curve_household_bank.csv."""
    hh_hist_cols = [f"hh{i}" for i in range(1, 7)]
    day_cols = [f"d{i}" for i in range(max_days)]
    cols = ["beta_H", "beta_C"] + SUMMARY_COLS + hh_hist_cols + day_cols

    seed = bank_seed if bank_seed is not None else cfg.seed
    master_rng = np.random.default_rng(seed)
    beta_H = master_rng.uniform(*prior_bh, n_sim)
    beta_C = master_rng.uniform(*prior_bc, n_sim)
    seed_seq = np.random.SeedSequence(seed)
    sim_seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seed_seq.spawn(n_sim)]
    tasks = [(beta_H[i], beta_C[i], sim_seeds[i], max_days) for i in range(n_sim)]

    chunk_size = max(1, n_sim // (chunk_workers * 20))
    rows = list(executor.map(_bank_worker, tasks, chunksize=chunk_size))
    rows = np.vstack(rows)
    return pd.DataFrame(rows, columns=cols), hh_hist_cols, day_cols
