# The Kahl Model of Epistemic Dissonance (KMED)
# Simulation 1: Fiduciary Scaffold vs Clientelist Suppression — KMED
# (Single agent, two regimes)
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

# --- Plots ---
plt.figure(figsize=(10,5.6))
plt.plot(t, EA_f, '-',  label="EA (fiduciary)")
plt.plot(t, EA_c, '--', label="EA (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Epistemic Autonomy (EA)")
plt.title("EA under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig("A3_sim1_tuned_EA.png", dpi=200); plt.show()

plt.figure(figsize=(10,5.6))
plt.plot(t, DT_f, '-',  label="DT (fiduciary)")
plt.plot(t, DT_c, '--', label="DT (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Dissonance Tolerance (DT)")
plt.title("DT under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig("A3_sim1_tuned_DT.png", dpi=200); plt.show()

plt.figure(figsize=(10,5.6))
plt.plot(t, D_f, '-',  label="D (fiduciary)")
plt.plot(t, D_c, '--', label="D (clientelist)")
plt.xlabel("Time (micro-dissonance events)"); plt.ylabel("Dependence (D)")
plt.title("D under Fiduciary Scaffold vs Clientelist Suppression — KMED")
plt.legend(); plt.tight_layout(); plt.savefig("A3_sim1_tuned_D.png", dpi=200); plt.show()