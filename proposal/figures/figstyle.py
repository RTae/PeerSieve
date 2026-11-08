"""The figure style of the paper-figures skill, as one module to copy into a project.

The style follows the house style of figures4papers by Chen Liu (https://github.com/ChenLiu-1996/figures4papers,
``scientific-figure-making/references/design-theory.md``, CC BY-NC 4.0): a sans-serif font, heavy left and bottom
spines with no top or right spine, frameless legends, a blue, green, red and gray palette with fixed roles, and
300 dpi export. The skill's ``SKILL.md`` says how to use it.

Copy this file into the project's plotting folder, commit it, and leave it unedited, so that a newer version of the
skill can replace it. Keep the project's fixed roles in a small module beside it, for example ``figroles.py``.

A figure in a document with no caption names its title and settings itself. :func:`header` draws that text, and
:func:`save` writes it into the PNG. The PDF leaves it out, for a paper whose caption carries it.

Use::

    import figstyle as fs
    fs.apply()
    fig, ax = plt.subplots(figsize=(9, 6))
    ...
    top = fs.header(fig, "Runtime against input size", "<hardware>, median of 5 runs")
    fig.subplots_adjust(top=top - 0.03)
    fs.save(fig, out_dir, "fig-runtime-vs-n")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# The palette of figures4papers, design-theory.md section 9, with its teal and violet accents and a darker gray for
# lines. Blue marks the method a figure is about, green its variants, red and pink the comparators, gray the rest.
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "gray": "#767676", "dark": "#4D4D4D", "ink": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}

# The color cycle for a plot call that names no color, so that even a default series stays in the palette.
CYCLE = [PALETTE[k] for k in ("blue_main", "red_strong", "green_3", "gray", "teal", "violet")]
INK, MUTED = PALETTE["ink"], PALETTE["gray"]

# design-theory.md section 8: 16 pt text and 2.5 pt spines for a compact figure, 24 pt and 3 pt for a large panel.
SIZES = {"compact": (16, 2.5), "large": (24, 3.0)}

_HEADER_GID = "figstyle-header"


def apply(size: str = "compact") -> None:
    """Set the rcParams of the style. Call once, before the first figure.

    It also selects the non-interactive Agg backend, so a notebook shows no inline figure after it."""
    matplotlib.use("Agg")
    font_size, spine = SIZES[size]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size * 0.875,
        "ytick.labelsize": font_size * 0.875,
        "legend.fontsize": font_size * 0.875,
        "axes.linewidth": spine,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": spine,
        "ytick.major.width": spine,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
        "legend.frameon": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
    })


def _edges(fig, texts):
    """The bottom and top of ``texts`` together, as figure fractions."""
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    height = fig.bbox.height
    return min(b.y0 for b in boxes) / height, max(b.y1 for b in boxes) / height


def header(fig, title: str, subtitle: str = "", top_inches: float = 0.25) -> float:
    """Draw the title and the settings line at the top left. :func:`save` leaves them out of the PDF.

    Returns the figure fraction of the header's lowest edge. Pass a little less to ``fig.subplots_adjust(top=...)``.
    A layout engine (``layout="constrained"``, ``tight_layout``) ignores this text, so do not use one with it."""
    height = fig.get_size_inches()[1]
    size = plt.rcParams["font.size"]
    fig.text(0.01, 1 - top_inches / height, title, fontsize=size * 1.25, fontweight="bold", color="black",
             ha="left", va="top", gid=_HEADER_GID)
    texts = [fig.texts[-1]]
    if subtitle:
        fig.text(0.01, 1 - (top_inches + size * 1.25 / 72 * 1.5) / height, subtitle, fontsize=size * 0.8,
                 color=MUTED, ha="left", va="top", gid=_HEADER_GID)
        texts.append(fig.texts[-1])
    return _edges(fig, texts)[0]


def footnote(fig, *lines: str, x: float = 0.01, bottom_inches: float = 0.12) -> float:
    """Muted notes at the bottom left, the last line lowest. Like the header they carry what a caption would, so
    :func:`save` leaves them out of the PDF.

    Returns the figure fraction of the notes' top edge, 0 for no lines. The x tick labels and the axis label need
    about 0.8 in above it: ``fig.subplots_adjust(bottom=footnote(...) + 0.8 / fig.get_size_inches()[1])``."""
    height = fig.get_size_inches()[1]
    size = plt.rcParams["font.size"] * 0.8
    step = size * 1.6 / 72
    texts = []
    for k, line in enumerate(reversed(lines)):
        texts.append(fig.text(x, (bottom_inches + k * step) / height, line, fontsize=size, color=MUTED, ha="left",
                              va="bottom", gid=_HEADER_GID))
    return _edges(fig, texts)[1] if texts else 0.0


def light_grid(ax, axis: str = "y") -> None:
    """A light grid, only where a reader must read values across a wide panel (``SKILL.md``, rule 1, "Style")."""
    ax.grid(axis=axis, color=PALETTE["neutral"], lw=0.9, alpha=0.8)
    ax.set_axisbelow(True)


def tint(colour: str, amount: float):
    """``colour`` mixed with white: ``amount`` 1 is the colour, 0 is white."""
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b))


def save(fig, out_dir, stem: str, *, dpi: int = 300, pdf: bool = True) -> list[Path]:
    """Write ``stem.png`` with the header at ``dpi``, and ``stem.pdf`` without it. Returns the paths.

    The PNG keeps the canvas as drawn, so the header stays in place and anything past the canvas edge is cut. The
    PDF is cropped to its content. This closes ``fig``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.png"]
    fig.savefig(paths[0], dpi=dpi)
    if pdf:
        head = [t for t in fig.texts if t.get_gid() == _HEADER_GID]
        for t in head:
            t.set_visible(False)
        paths.append(out / f"{stem}.pdf")
        fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.05)
        for t in head:
            t.set_visible(True)
    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    return paths
