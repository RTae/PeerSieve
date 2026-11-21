# Project Instructions (Figures)

## Figures

Publication-style figures follow the `paper-figures` (`figures4papers`) house style:

- **Style module (unedited copy):** `experiments/phase1/figstyle.py`
- **Project role module:** `experiments/phase1/figroles.py`
- **Plotting script:** `experiments/phase1/plot_figures.py`
- **Committed source data:** `report/data/phase1_paper_match_3seed.json`, `report/data/phase1_stabilized_ab_3seed.json`
- **Outputs:** 300 dpi PNG (with header & footnote) + header-free cropped PDF under `report/figures/`.

### Fixed Series Encodings (`figroles.py`)

| Series | Color (`PALETTE` key) | Marker | Line Style | Meaning |
|---|---|---|---|---|
| `TrustGraph (stab)` | `blue_main` (`#0F4D92`) | `o` | solid (`-`) | Stabilized TrustGraph-DFL (`descent_neg`, model-space clip `2 r_i`, $\tau=0.02$) |
| `BALANCE` | `teal` (`#42949E`) | `s` | dashed (`--`) | Calibrated BALANCE distance-threshold baseline |
| `Krum` | `red_strong` (`#B64342`) | `^` | dash-dot (`-.`) | Multi-Krum neighborhood baseline |
| `Trimmed Mean` | `violet` (`#9A4D8E`) | `D` | dotted (`:`) | Coordinate-wise trimmed mean baseline |
| `FedAvg` | `gray` (`#767676`) | `v` | solid (`-`) | Unprotected decentralized FedAvg |
| Ablation variants | `blue_secondary`, `green_3` (healthy $\alpha$); `red_2`, `red_strong` (Euclidean / No ReLU) | — | bars (black edge `1.8 pt`) | Section 5.5 component ablations |

Shaded bands (`fill_between`, $\alpha=0.12\text{--}0.14$) and vertical error bars always show $\pm 1$ standard deviation across the 3 seeds (`42, 43, 44`).
