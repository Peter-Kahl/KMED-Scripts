# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 2: Regime Switching (Rupture → Repair) — KMED
# (Single agent: clientelist → fiduciary)
#
# Author: Peter Kahl
# First published: London, 25 September 2025
# Revision: Rev A (25 September 2025)
#
# Repository: https://github.com/Peter-Kahl/KMED-Scripts
# Script URL: https://github.com/Peter-Kahl/KMED-Scripts/blob/main/sim2_regime_switching.py
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

def step_asymptotic(x, k, theta, noise_std):
    """One-step asymptotic update toward theta with step size k, clipped to [0,1]."""
    eps = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    x_next = x + k*(theta - x) + eps
    return float(np.clip(x_next, 0.0, 1.0))

def simulate_asym_schedule(
    T=300, Tswitch=150,
    EA0=0.5, DT0=0.5, D0=0.5,
    # Phase 1 (clientelist suppression) targets and step sizes
    EA_theta_1=0.05, EA_k_1=0.06,
    DT_theta_1=0.10, DT_k_1=0.06,
    D_theta_1 =0.70, D_k_1 =0.05,
    # Phase 2 (fiduciary scaffold) targets and step sizes
    EA_theta_2=0.74, EA_k_2=0.05,
    DT_theta_2=0.80, DT_k_2=0.06,
    D_theta_2 =0.18, D_k_2 =0.06,
    noise_std=0.004,
):
    """
    Piecewise-asymptotic simulation with a single regime switch at Tswitch.
    Returns EA, DT, D time series (length T+1).
    """
    EA = np.empty(T+1); DT = np.empty(T+1); D = np.empty(T+1)
    EA[0], DT[0], D[0] = EA0, DT0, D0

    for t in range(T):
        # choose regime by time
        if t < Tswitch:
            EA_theta, EA_k = EA_theta_1, EA_k_1
            DT_theta, DT_k = DT_theta_1, DT_k_1
            D_theta,  D_k  = D_theta_1,  D_k_1
        else:
            EA_theta, EA_k = EA_theta_2, EA_k_2
            DT_theta, DT_k = DT_theta_2, DT_k_2
            D_theta,  D_k  = D_theta_2,  D_k_2

        EA[t+1] = step_asymptotic(EA[t], EA_k, EA_theta, noise_std)
        DT[t+1] = step_asymptotic(DT[t], DT_k, DT_theta, noise_std)
        D[t+1]  = step_asymptotic(D[t],  D_k,  D_theta,  noise_std)

    return EA, DT, D

# ----------------------------
# Run the regime-switching sim
# ----------------------------
T = 300
Tswitch = 150
t = np.arange(T+1)

EA_s, DT_s, D_s = simulate_asym_schedule(
    T=T, Tswitch=Tswitch,
    # Phase 1: clientelist suppression (EA↓, DT↓, D↑)
    EA_theta_1=0.05, EA_k_1=0.06,
    DT_theta_1=0.10, DT_k_1=0.06,
    D_theta_1 =0.70, D_k_1 =0.05,
    # Phase 2: fiduciary scaffold (EA↑, DT↑, D↓)
    EA_theta_2=0.74, EA_k_2=0.05,
    DT_theta_2=0.80, DT_k_2=0.06,
    D_theta_2 =0.18, D_k_2 =0.06,
    noise_std=0.004
)

# ----------------------------
# Plots (mark the switch point)
# ----------------------------
def mark_switch(ax, Ts):
    ax.axvline(Ts, linestyle=":", linewidth=1.5)
    ax.text(Ts+2, 0.04, "switch\n(rupture → repair)", fontsize=9, va="bottom")

# repo root is one level up from this script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# (1) Epistemic Autonomy
plt.figure(figsize=(10, 5.6))
plt.plot(t, EA_s, label="EA (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Epistemic Autonomy (EA)")
plt.title("EA under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim2_EA.png", dpi=200); plt.show()

# (2) Dissonance Tolerance
plt.figure(figsize=(10, 5.6))
plt.plot(t, DT_s, label="DT (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dissonance Tolerance (DT)")
plt.title("DT under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim2_DT.png", dpi=200); plt.show()

# (3) Dependence
plt.figure(figsize=(10, 5.6))
plt.plot(t, D_s, label="D (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dependence (D)")
plt.title("D under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sim2_D.png", dpi=200); plt.show()