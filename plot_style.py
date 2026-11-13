"""
Shared plot styling.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

TEXT_W = 6.27
TITLE_SIZE = 8.5

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
GRID = "#e1e0d9"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"

INCIDENT_COLORS = {
    "Sighting": BLUE,
    "Livestock or Pet Injury/Loss": ORANGE,
    "Human Safety Confrontation": AQUA,
}
INCIDENT_ORDER = list(INCIDENT_COLORS)

HIGHLIGHT = BLUE
CONTEXT = BASELINE

SEQ_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("cougar_seq", SEQ_STEPS)


def apply():
    """Set matplotlib defaults shared by every figure in the proj."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        "font.size": 7,
        "text.color": INK,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,

        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1,
        "patch.linewidth": 0.5,

        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA, RED]),
        "axes.grid": False,
        "grid.color": GRID,

        "xtick.color": BASELINE,
        "ytick.color": BASELINE,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,

        "legend.frameon": False,
        "legend.labelcolor": INK_2,

        "image.cmap": "cougar_seq",
    })
    if "cougar_seq" not in mpl.colormaps:
        mpl.colormaps.register(SEQ_CMAP)
