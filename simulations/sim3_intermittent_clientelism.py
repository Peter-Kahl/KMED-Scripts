# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 3: Intermittent clientelism vs stable fiduciary — KMED (asymptotic)
#
# Author: Peter Kahl
# First published: London, 25 September 2025
# Revision: Rev A (25 September 2025)
#
# Repository: https://github.com/Peter-Kahl/KMED-Scripts
# Script URL: https://github.com/Peter-Kahl/KMED-Scripts/blob/main/sim3_intermittent_clientelism.py
#
# © 2025 Peter Kahl / Lex et Ratio Ltd.
#
# License: MIT
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided “AS IS”, without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the Software or the use or other dealings in
# the Software.
#
# Full license text: https://opensource.org/licenses/MIT

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# repo root is one level up from this script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- KMED asymptotic step ----------
def step_asymptotic(x, k, theta, noise_std):
    eps = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    x_next = x + k * (theta - x) + eps
    return float(np.clip(x_next, 0.0, 1.0))

# ---------- Targets (θ) / step sizes (k) per regime ----------
# Fiduciary: EA↑, DT↑, D↓
F_PARAMS = dict(EA_theta=0.74, EA_k=0.05,
                DT_theta=0.80, DT_k=0.06,
                D_theta =0.18, D_k =0.06)

# Clientelist: EA↓, DT↓, D↑
C_PARAMS = dict(EA_theta=0.03, EA_k=0.06,
                DT_theta=0.08, DT_k=0.06,
                D_theta =0.65, D_k =0.04)

def draw_regime(prev, p, rho=None):
    """
    Draw regime for this step.
    - If rho is None: IID draw; clientelist with prob p.
    - If rho in [0,1): two-state Markov with persistence rho.
      With prob rho, stay in previous regime; else flip w.r.t. base probs.
    Returns 'F' or 'C'.
    """
    if rho is None:
        return 'C' if rng.random() < p else 'F'
    # Markov persistence
    if prev is None or rng.random() > rho:
        # choose fresh according to base rate p
        return 'C' if rng.random() < p else 'F'
    # persist
    return prev

def simulate_intermit(
    T=600,
    EA0=0.5, DT0=0.5, D0=0.5,
    p_clientelist=0.2,   # share of clientelist steps
    rho=None,            # persistence (None = IID)
    noise_std=0.004
):
    EA = np.empty(T+1); DT = np.empty(T+1); D = np.empty(T+1)
    EA[0], DT[0], D[0] = EA0, DT0, D0

    regime = None
    for t in range(T):
        regime = draw_regime(regime, p_clientelist, rho)
        P = C_PARAMS if regime == 'C' else F_PARAMS

        EA[t+1] = step_asymptotic(EA[t], P['EA_k'], P['EA_theta'], noise_std)
        DT[t+1] = step_asymptotic(DT[t], P['DT_k'], P['DT_theta'], noise_std)
        D[t+1]  = step_asymptotic(D[t],  P['D_k'],  P['D_theta'],  noise_std)

    return EA, DT, D

# ---------- Run for multiple clientelist shares ----------
T = 600
t = np.arange(T+1)
p_grid = [0.0, 0.1, 0.2, 0.3, 0.5]   # 0% … 50% clientelist
rho = None                           # set to e.g. 0.8 to add streaks

EA_series, DT_series, D_series = {}, {}, {}
for p in p_grid:
    EA_series[p], DT_series[p], D_series[p] = simulate_intermit(
        T=T, p_clientelist=p, rho=rho, noise_std=0.004
    )

# ---------- Trajectory plots ----------
def plot_trajectories(series_dict, ylabel, title, fname):
    plt.figure(figsize=(10,5.8))
    for p, y in series_dict.items():
        plt.plot(t, y, label=f"p={p:.1f}")
    plt.xlabel("Time (micro-dissonance events)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Clientelist share")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.show()

plot_trajectories(EA_series, "Epistemic Autonomy (EA)",
                  "EA under Intermittent Clientelism (variable share) — KMED",
                  OUTPUT_DIR / "sim3_EA_trajectories.png")

plot_trajectories(DT_series, "Dissonance Tolerance (DT)",
                  "DT under Intermittent Clientelism (variable share) — KMED",
                  OUTPUT_DIR / "sim3_DT_trajectories.png")

plot_trajectories(D_series, "Dependence (D)",
                  "D under Intermittent Clientelism (variable share) — KMED",
                  OUTPUT_DIR / "sim3_D_trajectories.png")

# ---------- Equilibria vs p (last-100-step mean ± 95% CI) ----------
def tail_stats(y, tail=100):
    tail_vals = y[-tail:]
    m = float(np.mean(tail_vals))
    s = float(np.std(tail_vals, ddof=1))
    n = tail_vals.size
    ci = 1.96 * s / np.sqrt(n)
    return m, ci

EA_means, EA_cis = [], []
DT_means, DT_cis = [], []
D_means,  D_cis  = [], []

for p in p_grid:
    m, ci = tail_stats(EA_series[p]); EA_means.append(m); EA_cis.append(ci)
    m, ci = tail_stats(DT_series[p]); DT_means.append(m); DT_cis.append(ci)
    m, ci = tail_stats(D_series[p]);  D_means.append(m);  D_cis.append(ci)

plt.figure(figsize=(10,6))
for means, cis, label in [(EA_means, EA_cis, "EA"),
                          (DT_means, DT_cis, "DT"),
                          (D_means,  D_cis,  "D")]:
    means = np.array(means); cis = np.array(cis)
    plt.plot(p_grid, means, marker='o', label=label)
    plt.fill_between(p_grid, means-cis, means+cis, alpha=0.15)

plt.xlabel("Clientelist share (p)")
plt.ylabel("Equilibrium level (last-100-step mean)")
plt.title("Equilibrium EA / DT / D vs Clientelist Share — KMED")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim3_equilibria_vs_p.png", dpi=200)
plt.show()