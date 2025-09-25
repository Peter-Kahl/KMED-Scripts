# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 2: Regime Switching (Rupture → Repair) — KMED
# (Single agent: clientelist → fiduciary)
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

# Quick sanity prints: first/around switch/last few steps
print("EA (first 6):", np.round(EA_s[:6], 4))
print("EA (around switch):", np.round(EA_s[Tswitch-2:Tswitch+4], 4))
print("EA (last 6):", np.round(EA_s[-6:], 4))
print("DT (first 6):", np.round(DT_s[:6], 4))
print("DT (around switch):", np.round(DT_s[Tswitch-2:Tswitch+4], 4))
print("DT (last 6):", np.round(DT_s[-6:], 4))
print("D  (first 6):", np.round(D_s[:6], 4))
print("D  (around switch):", np.round(D_s[Tswitch-2:Tswitch+4], 4))
print("D  (last 6):", np.round(D_s[-6:], 4))

# ----------------------------
# Plots (mark the switch point)
# ----------------------------
def mark_switch(ax, Ts):
    ax.axvline(Ts, linestyle=":", linewidth=1.5)
    ax.text(Ts+2, 0.04, "switch\n(rupture → repair)", fontsize=9, va="bottom")

# (1) Epistemic Autonomy
plt.figure(figsize=(10, 5.6))
plt.plot(t, EA_s, label="EA (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Epistemic Autonomy (EA)")
plt.title("EA under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig("A3_sim2_EA.png", dpi=200); plt.show()

# (2) Dissonance Tolerance
plt.figure(figsize=(10, 5.6))
plt.plot(t, DT_s, label="DT (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dissonance Tolerance (DT)")
plt.title("DT under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig("A3_sim2_DT.png", dpi=200); plt.show()

# (3) Dependence
plt.figure(figsize=(10, 5.6))
plt.plot(t, D_s, label="D (regime switching)")
mark_switch(plt.gca(), Tswitch)
plt.xlabel("Time (micro-dissonance events)")
plt.ylabel("Dependence (D)")
plt.title("D under Clientelist → Fiduciary Regime Switching — KMED")
plt.ylim(0, 1); plt.legend(); plt.tight_layout()
plt.savefig("A3_sim2_D.png", dpi=200); plt.show()