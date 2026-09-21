"""Unified publication style — Morandi editorial look (top-conference grade).

Palette sampled from the user's reference figures (docs/refs/style_refs):
warm cream + dusty blue + sage + clay/mauve, very low saturation.
Layout grammar adapted from the reference style sheet: bold panel titles,
annotation boxes, dashed reference lines, bottom-row frameless legend,
fine dotted grid, hairline full frame.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#3F3D3A"

# ---- Morandi palette (hex sampled from reference figures) ----
PALETTE = {
    "cream":      "#F6E0C1",
    "cream_pale": "#F7EDD9",
    "sand":       "#D9BE8A",
    "clay":       "#C98F7E",
    "clay_pale":  "#E7BDAD",
    "sage":       "#8FAE8B",
    "sage_pale":  "#C9D8C2",
    "moss":       "#7C8E70",
    "blue":       "#2F5AA8",   # strong accent (main line)
    "dusty_blue": "#9DB6C0",
    "mist":       "#C6D2DA",
    "navy":       "#23499D",
    "mauve":      "#A79BB5",
    "gray":       "#8A8681",
    "gray_light": "#D8D4CE",
    "ink": INK,
}

# Frozen semantics (same meaning -> same colour everywhere)
SEMANTICS = {
    "baseline": PALETTE["blue"],
    "heat": PALETTE["blue"],
    "none": PALETTE["clay"],
    "ideal": PALETTE["sage"],
    "steps2": PALETTE["sand"],
    "sharp050": PALETTE["dusty_blue"],
    "sharp200": PALETTE["mauve"],
    "sharp400": PALETTE["moss"],
    "info_loss": PALETTE["blue"],
    "broadband": PALETTE["clay"],
    "transformation": PALETTE["sage"],
    "actual": PALETTE["gray"],
    "predicted": PALETTE["blue"],
    "ci": PALETTE["mist"],
    "highlight": PALETTE["sand"],
}

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "morandi_seq", ["#FBF7EF", "#EFE5CF", "#D9C9A8", "#B9AE93", "#8FAE8B", "#5F7F6B"])
DIV_CMAP = "RdBu_r"

SINGLE_COL = 3.5
DOUBLE_COL = 7.2


THEMES = {
    # classic CVPR/Times look — figure type matches the paper body
    "times": {"family": "serif",
              "list": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "STIXGeneral"],
              "math": "stix"},
    # modern Nature/Helvetica look — clean, neutral
    "sans": {"family": "sans-serif",
             "list": ["Nimbus Sans", "Liberation Sans", "Arial", "Helvetica"],
             "math": "stixsans"},
    # soft serif (Nature editorial)
    "stix": {"family": "serif",
             "list": ["STIXGeneral", "Nimbus Roman"],
             "math": "stix"},
}


def apply_style(font_size: float = 8.0, theme: str = "times") -> None:
    t = THEMES[theme]
    list_key = "font.serif" if t["family"] == "serif" else "font.sans-serif"
    mpl.rcParams.update({
        "font.family": t["family"],
        list_key: t["list"],
        "mathtext.fontset": t["math"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": font_size,
        "axes.labelsize": font_size + 0.5,
        "axes.titlesize": font_size + 0.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "legend.fontsize": font_size - 0.5,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": PALETTE["gray_light"],
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "figure.facecolor": "white",
    })


def panel_label(ax, text: str, dx: float = -0.12, dy: float = 1.02) -> None:
    """Bold (a)(b)(c) panel label at the top-left, outside the axes (spec)."""
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def panel_title(ax, tag: str, text: str) -> None:
    """Bold panel title, reference style: 'A. Coefficient paths'."""
    ax.set_title(f"{tag}. {text}", loc="left", pad=6)


def annot_box(ax, text: str, xy=(0.04, 0.96), ha="left", va="top",
              fontsize=7.5) -> None:
    """White rounded annotation box with hairline edge (formula/stat callouts)."""
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=PALETTE["gray"], linewidth=0.6))


def ref_line(ax, x=None, y=None, color=None, ls=(0, (4, 2.5))) -> None:
    c = color or PALETTE["gray"]
    if x is not None:
        ax.axvline(x, color=c, lw=0.8, ls=ls, zorder=1)
    if y is not None:
        ax.axhline(y, color=c, lw=0.8, ls=ls, zorder=1)


def bottom_legend(ax, ncol=None, y: float = -0.34) -> None:
    """Frameless horizontal legend row below the axes (reference style)."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol or len(handles), frameon=False, handlelength=1.6,
              columnspacing=1.4, handletextpad=0.5)


def grad_fill(ax, x, y, color, alpha_near=0.30, alpha_far=0.03, n_layers=28):
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(np.nanmin(y))
    for i in range(n_layers):
        lo = y0 + (y - y0) * (i / n_layers)
        hi = y0 + (y - y0) * ((i + 1) / n_layers)
        a = alpha_far + (alpha_near - alpha_far) * (i / n_layers)
        ax.fill_between(x, lo, hi, color=color, alpha=a, linewidth=0)


def save_fig(fig, path_stem: str, png_dpi: int = 600) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path_stem)), exist_ok=True)
    fig.savefig(f"{path_stem}.png", dpi=png_dpi)
    fig.savefig(f"{path_stem}.pdf")
