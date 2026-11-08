"""Fixed series roles for KVMend figures (paper-figures skill). Import colors from here, never hard-code them."""
from figstyle import PALETTE

SERIES = {
    "KVMend, damage on one GPU": PALETTE["blue_main"],
    "KVMend, damage on every GPU": PALETTE["blue_secondary"],
    "re-prefill from the damaged page": PALETTE["red_strong"],
    "re-prefill the whole request": PALETTE["gray"],
    # measured results (Figure 12): GhostServe re-implemented, a comparator, in pink
    "GhostServe, told the GPU": PALETTE["red_2"],
    "GhostServe, told the GPU and the chunk": PALETTE["red_1"],
    "parity alone (GhostServe-style)": PALETTE["red_2"],
    # measured results (Figure 13): KVMend recomputing the damaged pages instead of rebuilding them from parity
    "KVMend recomputing pages": PALETTE["blue_main"],
}
# Dashes: solid = repair on one GPU or a comparator; dashed = KVMend when every GPU is damaged;
# dotted (with hollow markers) = KVMend recomputing pages instead of rebuilding them.
DASH = {"KVMend, damage on every GPU": (0, (5, 3)), "KVMend recomputing pages": (0, (1, 2))}
HOLLOW = {"KVMend recomputing pages"}
# Hatches: one meaning in every figure.
HATCH = {"next tokens still wrong after recovery": "//"}
