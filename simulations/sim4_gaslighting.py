# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 4 — Gaslighting (oscillating recognition and suppression) — KMED
# (Single agent; alternating short blocks of fiduciary recognition and clientelist withdrawal)
#
# First published in London by Lex et Ratio Ltd, 25 September 2025.
#
# © 2025 Lex et Ratio Ltd.
#
# This code is released under the MIT Licence:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt

# --- RNG (fixed for reproducibility) ---
rng = np.random.default_rng(42)

# --- Asymptotic one-step updater (bounded in [0,1]) ---
def step_asymptotic(x, k, theta, noise_std):
    eps = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
    x_next = x + k * (theta - x) + eps
    return float(np.clip(x_next, 0.0, 1.0))

# --- Parameter presets for regimes (match Sim 1 style) ---
FIDUCIARY = dict(
    EA_theta=0.74, EA_k=0.05,  # autonomy grows to high plateau
    DT_theta=0.80, DT_k=0.06,  # tolerance consolidates under scaffolds
    D_theta =0.18, D_k =0.06   # dependence declines to dignified floor
)

CLIENTELIST = dict(
    EA_theta=0.03, EA_k=0.06,  # autonomy collapses toward 0
    DT_theta=0.08, DT_k=0.06,  # tolerance collapses
    D_theta =0.65, D_k =0.04   # dependence entrenches toward high level
)

def make_phase_pattern(T, start='fiduciary', min_len=18, max_len=32):
    """
    Build an alternating list of (regime, start_idx, end_idx) that covers 0..T-1.
    Each block length is uniformly sampled in [min_len, max_len).
    """
    phases = []
    t = 0
    current = start
    while t < T:
        block = int(rng.integers(min_len, max_len))
        end = min(T, t + block)
        phases.append((current, t, end))  # [t, end)
        t = end
        current = 'clientelist' if current == 'fiduciary' else 'fiduciary'
    return phases

def simulate_gaslighting(
    T=300,
    EA0=0.5, DT0=0.5, D0=0.5,
    start='fiduciary',
    min_len=18, max_len=32,
    noise_std=0.004
):
    """
    Alternate between fiduciary and clientelist blocks with random lengths
    to model oscillating recognition/withdrawal ("gaslighting").
    Returns EA, DT, D trajectories and the phase list for plotting.
    """
    EA = np.empty(T+1); DT = np.empty(T+1); D = np.empty(T+1)
    EA[0], DT[0], D[0] = EA0, DT0, D0

    phases = make_phase_pattern(T, start=start, min_len=min_len, max_len=max_len)
    # Pre-expand per-timestep regime parameters
    theta_k_series = []
    regime_by_t = np.empty(T, dtype=object)
    for regime, i0, i1 in phases:
        params = FIDUCIARY if regime == 'fiduciary' else CLIENTELIST
        for t in range(i0, i1):
            theta_k_series.append((
                params['EA_theta'], params['EA_k'],
                params['DT_theta'], params['DT_k'],
                params['D_theta'],  params['D_k']
            ))
            regime_by_t[t] = regime

    # Update loop
    for t in range(T):
        EA_theta, EA_k, DT_theta, DT_k, D_theta, D_k = theta_k_series[t]
        EA[t+1] = step_asymptotic(EA[t], EA_k, EA_theta, noise_std)
        DT[t+1] = step_asymptotic(DT[t], DT_k, DT_theta, noise_std)
        D[t+1]  = step_asymptotic(D[t],  D_k,  D_theta,  noise_std)

    return EA, DT, D, phases

# --- Run simulation ---
T = 300
t = np.arange(T+1)
EA, DT, D, phases = simulate_gaslighting(
    T=T,
    EA0=0.5, DT0=0.5, D0=0.5,
    start='fiduciary',      # begin with brief recognition bursts
    min_len=18, max_len=32, # short, destabilising blocks
    noise_std=0.004
)

# --- Quick sanity prints ---
print("First 6 steps (EA, DT, D):")
print("EA:", np.round(EA[:6], 4))
print("DT:", np.round(DT[:6], 4))
print("D :", np.round(D[:6], 4))

# --- Helper: shade phases on an axis ---
def shade_phases(ax, phases, ymax=1.0, fid_color=(0.6,0.9,0.6,0.25), cli_color=(0.9,0.6,0.6,0.25)):
    for regime, i0, i1 in phases:
        color = fid_color if regime == 'fiduciary' else cli_color
        ax.axvspan(i0, i1, color=color, linewidth=0)

# --- Plot 1: Epistemic Autonomy (EA) ---
fig, ax = plt.subplots(figsize=(10, 5.6))
shade_phases(ax, phases)
ax.plot(t, EA, label="EA (gaslighting)", lw=2)
ax.set_xlabel("Time (micro-dissonance events)")
ax.set_ylabel("Epistemic Autonomy (EA)")
ax.set_title("EA under Oscillating Recognition/Withdrawal (Gaslighting) — KMED")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig("A3_sim4_EA.png", dpi=200)
plt.show()

# --- Plot 2: Dissonance Tolerance (DT) ---
fig, ax = plt.subplots(figsize=(10, 5.6))
shade_phases(ax, phases)
ax.plot(t, DT, label="DT (gaslighting)", lw=2)
ax.set_xlabel("Time (micro-dissonance events)")
ax.set_ylabel("Dissonance Tolerance (DT)")
ax.set_title("DT under Oscillating Recognition/Withdrawal (Gaslighting) — KMED")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig("A3_sim4_DT.png", dpi=200)
plt.show()

# --- Plot 3: Dependence (D) ---
fig, ax = plt.subplots(figsize=(10, 5.6))
shade_phases(ax, phases)
ax.plot(t, D, label="D (gaslighting)", lw=2)
ax.set_xlabel("Time (micro-dissonance events)")
ax.set_ylabel("Dependence (D)")
ax.set_title("D under Oscillating Recognition/Withdrawal (Gaslighting) — KMED")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig("A3_sim4_D.png", dpi=200)
plt.show()