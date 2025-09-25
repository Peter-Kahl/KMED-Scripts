# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 5 — Dissonance Training (Therapeutic Scaffolding)
#
# Author: Peter Kahl
# First published: London, 25 September 2025
# Revision: Rev A (25 September 2025)
#
# Repository: https://github.com/Peter-Kahl/KMED-Scripts
# Script URL: https://github.com/Peter-Kahl/KMED-Scripts/blob/main/sim5_dissonance_training.py
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

# ----- Core updater -----
def step_asymptotic(x, k, theta, noise_std):
    eps = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    x_next = x + k*(theta - x) + eps
    return float(np.clip(x_next, 0.0, 1.0))

# ----- Therapeutic training with session-driven target adaptation -----
def simulate_training(
    T=320,
    # initial (post-clientelist) state
    EA0=0.5, DT0=0.5, D0=0.5,
    # baseline (pre-therapy) targets θ_base and final therapy targets θ_final
    EA_theta_base=0.25, DT_theta_base=0.30, D_theta_base=0.60,
    EA_theta_final=0.75, DT_theta_final=0.80, D_theta_final=0.18,
    # step sizes outside sessions (slow drift) vs in sessions (faster learning)
    EA_k_base=0.025, DT_k_base=0.025, D_k_base=0.030,
    EA_k_sess=0.060, DT_k_sess=0.070, D_k_sess=0.060,
    # rate at which the *targets* move toward final values during a session
    theta_adapt_rate=0.20,        # how quickly therapy pushes θ toward θ_final
    theta_consolidate_rate=0.02,  # gentle consolidation between sessions
    # schedule: list of (start, duration) *in time steps* for therapy sessions
    sessions=None,
    noise_std=0.004
):
    """
    Sessions model structured 'dissonance training' (graded exposure + validation).
    During sessions: (i) larger k (faster movement toward θ), and
                     (ii) θ itself is nudged toward θ_final.
    Between sessions: small consolidation of θ toward θ_final.
    """
    if sessions is None:
        # Default: 8 sessions of 20 steps each, every 40 steps starting at t=20
        sessions = [(20 + i*40, 20) for i in range(8)]

    # Precompute boolean mask for "in session"
    in_session = np.zeros(T+1, dtype=bool)
    for s, d in sessions:
        in_session[s : min(s+d, T+1)] = True

    # time-varying targets (θ) start at base and adapt over time
    EA_theta = EA_theta_base
    DT_theta = DT_theta_base
    D_theta  = D_theta_base

    # state trajectories
    EA = np.empty(T+1); DT = np.empty(T+1); D = np.empty(T+1)
    EA[0], DT[0], D[0] = EA0, DT0, D0

    for t in range(T):
        if in_session[t]:
            # in-session: push targets toward final values (therapy work)
            EA_theta += theta_adapt_rate * (EA_theta_final - EA_theta)
            DT_theta += theta_adapt_rate * (DT_theta_final - DT_theta)
            D_theta  += theta_adapt_rate * (D_theta_final  - D_theta)
            # faster learning within sessions
            ea_k, dt_k, d_k = EA_k_sess, DT_k_sess, D_k_sess
        else:
            # between sessions: gentle consolidation toward final values
            EA_theta += theta_consolidate_rate * (EA_theta_final - EA_theta)
            DT_theta += theta_consolidate_rate * (DT_theta_final - DT_theta)
            D_theta  += theta_consolidate_rate * (D_theta_final  - D_theta)
            # slower drift between sessions
            ea_k, dt_k, d_k = EA_k_base, DT_k_base, D_k_base

        # one asymptotic step for each variable
        EA[t+1] = step_asymptotic(EA[t], ea_k, EA_theta, noise_std)
        DT[t+1] = step_asymptotic(DT[t], dt_k, DT_theta, noise_std)
        D[t+1]  = step_asymptotic(D[t],  d_k,  D_theta,  noise_std)

    return EA, DT, D, in_session

# ---------- Run & Plot ----------
T = 320
t = np.arange(T+1)
sessions = [(20 + i*40, 20) for i in range(8)]  # shown as green bands

EA_tr, DT_tr, D_tr, in_sess = simulate_training(T=T, sessions=sessions)

# Helper: shade therapy sessions
def shade_sessions(ax, in_session_mask, color_good="#e8f7e8", alpha=0.6):
    in_block = False
    start = None
    for i in range(len(in_session_mask)):
        if in_session_mask[i] and not in_block:
            in_block = True; start = i
        if (not in_session_mask[i] and in_block) or (in_block and i == len(in_session_mask)-1):
            end = i if not in_session_mask[i] else i+1
            ax.axvspan(start, end, color=color_good, alpha=alpha, lw=0)

# (1) Epistemic Autonomy
plt.figure(figsize=(10, 5.8))
ax = plt.gca()
shade_sessions(ax, in_sess)  # therapy windows
plt.plot(t, EA_tr, label="EA (training)")
plt.ylim(0, 1.0)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Epistemic Autonomy (EA)")
plt.title("EA under Dissonance Training (therapeutic scaffolding) — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim5_EA.png", dpi=200); plt.show()

# (2) Dissonance Tolerance
plt.figure(figsize=(10, 5.8))
ax = plt.gca()
shade_sessions(ax, in_sess)
plt.plot(t, DT_tr, label="DT (training)")
plt.ylim(0, 1.0)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dissonance Tolerance (DT)")
plt.title("DT under Dissonance Training (therapeutic scaffolding) — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim5_DT.png", dpi=200); plt.show()

# (3) Dependence
plt.figure(figsize=(10, 5.8))
ax = plt.gca()
shade_sessions(ax, in_sess)
plt.plot(t, D_tr, label="D (training)")
plt.ylim(0, 1.0)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dependence (D)")
plt.title("D under Dissonance Training (therapeutic scaffolding) — KMED")
plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sim5_D.png", dpi=200); plt.show()

# Quick sanity: last-100-step means (therapeutic “equilibria”)
print("Final (mean of last 100 steps):",
      f"EA={EA_tr[-100:].mean():.3f}, DT={DT_tr[-100:].mean():.3f}, D={D_tr[-100:].mean():.3f}")