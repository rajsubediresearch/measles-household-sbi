# ============================================================================
# APPEX Measles SBI Project (Python) -- STEP 8: NPE TOOLBOX SMOKE TEST
# Before trusting the episbi toolbox on the measles problem, confirm it
# recovers a KNOWN answer on its own built-in stochastic SEIR simulator:
# draw from the prior, simulate a training bank, train NPE, check that the
# planted parameters fall inside the posterior.
#
# This does not depend on Steps 1-7 -- it uses episbi's own built-in model,
# not our household-community simulator. Purely a "does the tool work"
# check before Steps 9+ trust it on our data.
#
# Run from the measles_project folder (or anywhere -- the script cd's to its own folder):
#     python step8_npe_smoke_test.py
# ============================================================================
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); os.chdir(_SCRIPT_DIR)  # run from any folder
import time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from episbi.models import Transition, compartment_model
from episbi.prior import UniformPrior
from episbi.inference import SBIEngine
from episbi import simulate_for_sbi

OUT_DIR = "output/step8"

# ---- Setup (deliberately simple: known truth, built-in model) --------------
simulator_kwargs = {"dt": 0.01}
initial_conditions = {"S": 99_990, "E": 0, "I": 10, "R": 0}
theta_true = {"beta": 0.80, "kappa": 0.30, "gamma": 0.35}
total_days, inference_days = 100, 80
output_name = "E_to_I"
NUM_SIMS = 1200


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seir = compartment_model(
        compartments=["S", "E", "I", "R"],
        transitions=[
            Transition("S", "E", "beta * S * I / N", name="S_to_E"),
            Transition("E", "I", "kappa * E", name="E_to_I"),
            Transition("I", "R", "gamma * I", name="I_to_R"),
        ],
        param_names=["beta", "kappa", "gamma"],
        model_type="stochastic",
        initial_conditions=initial_conditions,
        observed={"transitions": [output_name]},
    )

    prior = UniformPrior(bounds={"beta": (0.01, 1.5), "kappa": (0.01, 0.5), "gamma": (0.01, 0.5)})

    # ---- training bank from the prior --------------------------------------
    print(f"[1/4] Simulating training bank: {NUM_SIMS} sims ...", flush=True)
    t0 = time.time()
    thetas, xs = simulate_for_sbi(
        prior=prior, simulator=seir, num_simulations=NUM_SIMS,
        total_days=total_days, seed=0, simulator_kwargs=simulator_kwargs, n_jobs=1,
    )
    xs_train = xs[:, :inference_days, :]
    print(f"      done in {time.time()-t0:.1f}s | thetas {thetas.shape} xs {xs.shape}", flush=True)

    # ---- observed outbreak from KNOWN truth --------------------------------
    observed = seir(theta_true, total_days=total_days, seed=0, **simulator_kwargs)[output_name]
    obs_train = observed[:inference_days]

    # ---- NPE ------------------------------------------------------------------
    print("[2/4] Training NPE ...", flush=True)
    t0 = time.time()
    engine = SBIEngine(batch_size=256)
    res = engine.run_npe(
        obs_data=obs_train, prior=prior, thetas=thetas, xs=xs_train,
        input_dim=1, num_samples=2000, show_train_summary=False,
    )
    print(f"      NPE done in {time.time()-t0:.1f}s", flush=True)
    samples = res["samples"].detach().cpu().numpy()
    df = pd.DataFrame(samples, columns=prior.names)

    # ---- recovery report ------------------------------------------------------
    print("[3/4] Recovery vs truth (90% CrI):")
    rows = []
    for name in prior.names:
        s = df[name]
        med = s.median(); lo, hi = s.quantile(0.05), s.quantile(0.95)
        truth = theta_true[name]; covered = lo <= truth <= hi
        rows.append((name, truth, med, lo, hi, covered))
        print(f"   {name:6s}  true={truth:.3f}  median={med:.3f}  "
              f"90%CrI=[{lo:.3f}, {hi:.3f}]  truth_in_CrI={covered}")

    R0 = df["beta"] / df["gamma"]
    print(f"   R0=beta/gamma  true={theta_true['beta']/theta_true['gamma']:.2f}  "
          f"median={R0.median():.2f}  90%CrI=[{R0.quantile(0.05):.2f}, {R0.quantile(0.95):.2f}]")

    rec_table = pd.DataFrame(rows, columns=["param", "true", "median", "ci_lo", "ci_hi", "in_90CI"])
    rec_table.to_csv(os.path.join(OUT_DIR, "npe_smoke_recovery_table.csv"), index=False)

    # ---- figure ---------------------------------------------------------------
    print("[4/4] Saving figure ...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, name in zip(axes, prior.names):
        ax.hist(df[name], bins=40, color="steelblue", alpha=0.85)
        ax.axvline(theta_true[name], color="black", linestyle="--", linewidth=1.6, label="truth")
        ax.set_title(name); ax.set_xlabel("posterior sample"); ax.legend()
    fig.suptitle("EpiSBI NPE smoke test -- stochastic SEIR (synthetic, known truth)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "npe_smoke_recovery.png"), dpi=130)
    plt.close(fig)

    print(f"Saved to {os.path.abspath(OUT_DIR)}/")
    print("\nSMOKE TEST RESULT:", "PASS" if all(r[5] for r in rows) else "CHECK (a param's CrI missed truth)")


if __name__ == "__main__":
    main()
