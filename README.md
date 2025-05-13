# RL-Explore-Bench

A light-weight playground for **efficient exploration algorithms** in reinforcement learning, centred on an implementation of **VAPOR** (Variational Approximation of Posterior Optimality) from *“Probabilistic Inference in Reinforcement Learning Done Right”* and a reference baseline of **Posterior Sampling RL (PSRL)**. This repository provides an unofficial implementation of the VAPOR algorithm.

| VAPOR in action | Average episodic return | Posterior reward uncertainty |
| :--: | :--: | :--: |
| ![Deep-Sea Exploration](grid_world/assets/env_reward.gif) | ![Return](grid_world/assets/mean_reward_30_horizon.png) | ![Uncertainty](grid_world/assets/reward_uncertainty_30_horizon.png) |

---

## 📂  Repository layout

```
rl-explore-bench/
│
├─ algorithms/          # VAPOR & PSRL implementations
│   ├─ vapor.py
│   └─ psrl.py
│
├─ assets/              # GIF + figures used in the README
├─ configs/             # Parameters for running experiments
│
├─ envs/                # Deep-Sea environment
│   ├─ grid_utils.py
│   └─ wrappers.py
│
├─ planning/            # Value-iteration & VAPOR solver
├─ sampling/            # Posterior samplers / distributions
├─ examples/            # Demo
│   └─ deep_sea_demo.py
└─ utils                # utilties for visualisation
```

---

## ⚡  Installation

> **Prerequisites** – Linux/macOS, Python ≥ 3.9.

```bash
git clone https://github.com/your-handle/rl-explore-bench.git
cd rl-explore-bench
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .                    # installs cvxpy, numpy, matplotlib, tqdm, ruff …
```

---

## 🚀  Quick start

Run the comparison between VAPOR and PSRL on the Deep-Sea environment (default: 30-time steps per episode, 100 episodes):

```bash
python examples/deep_sea_demo.py
```

The script prints cumulative regret and dumps the two PNGs you see above into
`assets/`. Edit the script or import `rl_explore_bench` in a notebook to play.

---

## 📖 References

### 1️⃣ Original VAPOR paper
```bibtex
@inproceedings{tarbouriech2023vapor,
  title     = {Probabilistic Inference in Reinforcement Learning Done Right},
  author    = {Tarbouriech, Jean and Lattimore, Tor and O'Donoghue, Brendan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{osband2013more,
  title     = {More Efficient Reinforcement Learning via Posterior Sampling},
  author    = {Osband, Ian and Russo, Daniel and Van Roy, Benjamin},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2013}
}

---
