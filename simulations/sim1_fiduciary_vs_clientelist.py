# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 1: Fiduciary Scaffold vs Clientelist Suppression — KMED
# (Single agent, two regimes)
#
# Author: Peter Kahl
# First published: London, 25 September 2025
# Revision: Rev A (25 September 2025)
#
# Repository: https://github.com/Peter-Kahl/KMED-Scripts
# Script URL: https://github.com/Peter-Kahl/KMED-Scripts/blob/main/sim1_fiduciary_vs_clientelist.py
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
    eps = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    x_next = x + k*(theta - x) + eps
    return float(np.clip(x_next, 0.0, 1.0))

def simulate_asym(
    T=250,
    EA0=0.5, DT0=0.5, D0=0.5,
    # regime targets (theta) and step sizes (k) for each variable
    EA_theta=0.75, EA_k=0.05,
    DT_theta=0.80, DT_k=0.06,
    D_theta =0.18, D_k =0.06,
    noise_std=0.004,
):
    EA = np.empty(T+1); DT = np.empty(T+1); D = np.empty(T+1)
    EA[0], DT[0], D[0] = EA0, DT0, D0
    for t in range(T):
        EA[t+1] = step_asymptotic(EA[t], EA_k, EA_theta, noise_std)
        DT[t+1] = step_asymptotic(DT[t], DT_k, DT_theta, noise_std)
        D[t+1]  = step_asymptotic(D[t],  D_k,  D_theta,  noise_std)
    return EA, DT, D

# --- Run both regimes with clear, regime-appropriate targets ---
T = 250
t = np.arange(T+1)

# Fiduciary scaffold: EA↑, DT↑, D↓
EA_f, DT_f, D_f = simulate_asym(
    T=T,
    EA_theta=0.74, EA_k=0.05,
    DT_theta=0.80, DT_k=0.06,   # DT rises to ~0.8 (fixes inversion)
    D_theta =0.18, D_k =0.06,
    noise_std=0.004
)

# Clientelist suppression: EA↓, DT↓, D↑
EA_c, DT_c, D_c = simulate_asym(
    T=T,
    EA_theta=0.03, EA_k=0.06,
    DT_theta=0.08, DT_k=0.06,   # DT falls toward ~0.08
    D_theta =0.65, D_k =0.04,
    noise_std=0.004
)

# Quick sanity prints (first 6)
print("Fiduciary DT (first 6):", np.round(DT_f[:6], 4))   # strictly increasing
print("Clientelist DT (first 6):", np.round(DT_c[:6], 4)) # strictly decreasing

# repo root is one level up from this script
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Plots ---
plt.figure(figsize=(10,5.6))
plt.plot(t, EA_f, '-',  label="EA (fiduciary)")
plt.plot(t, EA_c, '--', label="EA (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Epistemic Autonomy (EA)")
plt.title("EA under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim1_EA.png", dpi=200); plt.show()

plt.figure(figsize=(10,5.6))
plt.plot(t, DT_f, '-',  label="DT (fiduciary)")
plt.plot(t, DT_c, '--', label="DT (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Dissonance Tolerance (DT)")
plt.title("DT under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim1_DT.png", dpi=200); plt.show()

plt.figure(figsize=(10,5.6))
plt.plot(t, D_f, '-',  label="D (fiduciary)")
plt.plot(t, D_c, '--', label="D (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Dependence (D)")
plt.title("D under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim1_D.png", dpi=200); plt.show()