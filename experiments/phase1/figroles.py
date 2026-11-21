"""Fixed visual roles (colors, markers, dashes, labels) for PeerSieve Phase 1 figures.

All colors come strictly from ``figstyle.PALETTE`` (figures4papers house style).
Every method keeps the same color, marker, and line style across every figure.
"""

from __future__ import annotations

from experiments.phase1.figstyle import PALETTE

# Primary series colors (focus method = blue_main; comparators = red/violet/teal/gray)
SERIES_COLOR: dict[str, str] = {
    "tg_stab": PALETTE["blue_main"],
    "balance": PALETTE["teal"],
    "krum": PALETTE["red_strong"],
    "trimmed_mean": PALETTE["violet"],
    "fedavg": PALETTE["gray"],
    # Ablation variants (same blue/green family for healthy variants, red family for degraded)
    "tg_a03": PALETTE["blue_secondary"],
    "tg_a07": PALETTE["green_3"],
    "tg_euclidean": PALETTE["red_2"],
    "tg_no_relu": PALETTE["red_strong"],
}

SERIES_MARKER: dict[str, str] = {
    "tg_stab": "o",
    "balance": "s",
    "krum": "^",
    "trimmed_mean": "D",
    "fedavg": "v",
}

SERIES_LINESTYLE: dict[str, str] = {
    "tg_stab": "-",
    "balance": "--",
    "krum": "-.",
    "trimmed_mean": ":",
    "fedavg": "-",
}

SERIES_LABEL: dict[str, str] = {
    "tg_stab": "TrustGraph (stab)",
    "balance": "BALANCE",
    "krum": "Krum",
    "trimmed_mean": "Trimmed Mean",
    "fedavg": "FedAvg",
    "tg_a03": "TrustGraph ($\\alpha=0.3$)",
    "tg_a07": "TrustGraph ($\\alpha=0.7$)",
    "tg_euclidean": "Euclidean",
    "tg_no_relu": "No ReLU",
}

METHOD_ORDER: tuple[str, ...] = ("tg_stab", "balance", "krum", "trimmed_mean", "fedavg")
DEFENSE_ORDER: tuple[str, ...] = ("tg_stab", "balance", "krum", "trimmed_mean")
