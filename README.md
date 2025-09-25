#### Peter-Kahl >

# KMED Scripts

This repository contains the official Python reference implementations of the **Kahl Model of Epistemic Dissonance (KMED)**.
KMED is a mathematical and computational framework for modelling epistemic clientelism, dissonance, and resilience in intimate and institutional contexts.

The scripts reproduce the simulations documented in the paper:
[**_Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_**](https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships) (Lex et Ratio Ltd, London, 2025).

## Implemented Simulations

Each simulation corresponds to the versions published in Appendix A of the paper:

1. [**Simulation 1**](https://github.com/Peter-Kahl/KMED-Scripts/blob/main/simulations/sim1_fiduciary_vs_clientelist.py) — Fiduciary scaffold vs clientelist suppression (single agent)
2. [**Simulation 2**](https://github.com/Peter-Kahl/KMED-Scripts/blob/main/simulations/sim2_regime_switching.py) — Regime switching (rupture → repair)
3. [**Simulation 3**](https://github.com/Peter-Kahl/KMED-Scripts/blob/main/simulations/sim3_intermittent_clientelism.py) — Intermittent clientelism vs stable fiduciary
4. [**Simulation 4**](https://github.com/Peter-Kahl/KMED-Scripts/blob/main/simulations/sim4_gaslighting.py) — Gaslighting (oscillating recognition and suppression)
5. [**Simulation 5**](https://github.com/Peter-Kahl/KMED-Scripts/blob/main/simulations/sim5_dissonance_training.py) — Dissonance Training (therapeutic scaffolding)

All simulations share a consistent coding style and produce three primary plots:
- **Epistemic Autonomy (EA)**
- **Dissonance Tolerance (DT)**
- **Dependence (D)**

## Requirements

- Python 3.9+
- `numpy`
- `matplotlib`

Install dependencies via:

```bash
pip install numpy matplotlib
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Peter-Kahl/KMED-Scripts.git
```

## Usage

Run a simulation directly from the simulations/ folder. For example:

```
python simulations/sim1_fiduciary_vs_clientelist.py
```

Plots will be saved into the corresponding subdirectory of outputs/.

## License

- Code is released under the MIT License (see LICENSE).
- Accompanying paper and documentation are released under Creative Commons BY-NC-ND 4.0.

You may freely use, adapt, and extend the code for research and educational purposes. Please cite appropriately.


## Citation

If you use these scripts in academic work, please cite:

Kahl, P. (2025). _Epistemic clientelism in intimate relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_ (2nd ed.). Lex et Ratio Ltd. GitHub: https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships DOI: http://dx.doi.org/10.13140/RG.2.2.22662.43849
