#### Peter-Kahl >

# KMED Scripts

This repository contains the official Python reference implementations of the **Kahl Model of Epistemic Dissonance (KMED)**.
KMED is a mathematical and computational framework for modelling epistemic clientelism, dissonance, and resilience in intimate and institutional contexts.

The scripts reproduce the simulations documented in the paper:
[**_Epistemic Clientelism in Intimate Relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology_**](https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships) (Lex et Ratio Ltd, London, 2025).

## Implemented Simulations

Each simulation corresponds to the versions published in Appendix A of the paper:

1. **Simulation 1** — Fiduciary scaffold vs clientelist suppression (single agent)
2. **Simulation 2** — Regime switching (rupture → repair)
3. **Simulation 3** — Intermittent clientelism vs stable fiduciary
4. **Simulation 4** — Gaslighting (oscillating recognition and suppression)
5. **Simulation 5** — Dissonance Training (therapeutic scaffolding)

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

Kahl, P. (2025). Epistemic clientelism in intimate relationships: The Kahl Model of Epistemic Dissonance (KMED) and the Foundations of Epistemic Psychology (2nd ed.). Lex et Ratio Ltd. GitHub: https://github.com/Peter-Kahl/Epistemic-Clientelism-in-Intimate-Relationships DOI: http://dx.doi.org/10.13140/RG.2.2.22662.43849

## Links

- Full paper (Lex et Ratio Ltd): [link forthcoming]
- GitHub project: KMED-Scripts
