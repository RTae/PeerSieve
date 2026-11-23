"""Fixed visual roles (colors, markers, dashes, labels) for TrustGraph-DFL Phase 1 figures.

All colors come strictly from ``figstyle.PALETTE`` (figures4papers house style).
"""

from __future__ import annotations

from experiments.phase1.figstyle import PALETTE

# Attack series styling (no-attack reference in gray, active attacks in blue/teal/red)
ATTACK_COLOR: dict[str, str] = {
    "none": PALETTE["gray"],
    "label_flip_all": PALETTE["blue_main"],
    "model_poison": PALETTE["red_strong"],
    "random_noise": PALETTE["teal"],
}

ATTACK_MARKER: dict[str, str] = {
    "none": "o",
    "label_flip_all": "s",
    "model_poison": "^",
    "random_noise": "D",
}

ATTACK_LINESTYLE: dict[str, str] = {
    "none": "-",
    "label_flip_all": "-",
    "model_poison": "--",
    "random_noise": "-.",
}

ATTACK_LABEL: dict[str, str] = {
    "none": "No Attack",
    "label_flip_all": "Label Flipping ($E_b=2$)",
    "model_poison": "Model Poisoning ($\\lambda=10$)",
    "random_noise": "Random Noise ($\\sigma=0.5$)",
}

ATTACK_ORDER: tuple[str, ...] = ("none", "label_flip_all", "model_poison", "random_noise")

# Component ablation roles (blue/green for healthy variants, red family for degraded)
ABLATION_COLOR: dict[str, str] = {
    "tg_stab": PALETTE["blue_main"],
    "tg_a03": PALETTE["blue_secondary"],
    "tg_a07": PALETTE["green_3"],
    "tg_euclidean": PALETTE["red_2"],
    "tg_no_relu": PALETTE["red_strong"],
}
