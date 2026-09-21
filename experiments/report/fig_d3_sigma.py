"""D3 sigma-dose figure: (a) robustness curves per axis, (b) clean-vs-robust
trajectory over sigma, (c) gain bars (sigma=0.2 vs 0.1) at severe levels.

Style: experiments/report/style.py, Morandi palette, no in-figure titles.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.report.style import (  # noqa: E402
    PALETTE, apply_style, panel_label, save_fig,
)

OUT = "paper/figures"
PATHS = {
    0.05: "experiments/results/d3_sigma0.05.json",
    0.1: "experiments/results/e4_final_baseline.json",
    0.2: "experiments/results/d3_sigma0.2.json",
}
COLORS = {0.05: PALETTE["clay"], 0.1: PALETTE["dusty_blue"], 0.2: PALETTE["blue"]}
AXES_PANEL = [
    ("dropout", "point dropout"),
    ("fps", "FPS downsampling"),
    ("gaussian_noise", "Gaussian noise"),
    ("coarse_quantize", "coarse quantize"),
]
SEVERE = [("gaussian_noise", "0.05"), ("dropout", "0.875"), ("fps", "128"),
          ("coarse_quantize", "16")]


def _load():
    data = {}
    for sigma, path in PATHS.items():
        with open(path) as f:
            data[sigma] = json.load(f)
    return data


def _apply():
    plt.rcdefaults()
    apply_style(theme="times", font_size=7.0)
    plt.rcParams["axes.grid"] = False
    plt.rcParams["xtick.labelsize"] = 5.6
    plt.rcParams["ytick.labelsize"] = 6.0


def fig_d3():
    data = _load()
    sigmas = sorted(data)
    _apply()
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.85, 0.95], wspace=0.46,
                          left=0.065, right=0.985, top=0.80, bottom=0.28)

    sub = gs[0].subgridspec(2, 2, hspace=0.55, wspace=0.30)
    a_first = None
    for k, (axis, title) in enumerate(AXES_PANEL):
        a = fig.add_subplot(sub[k // 2, k % 2])
        if k == 0:
            a_first = a
        levels = sorted(data[0.1]["axes"][axis], key=float)
        xs = np.arange(len(levels))
        for sigma in sigmas:
            recs = data[sigma]["axes"][axis]
            ys = np.array([recs[lv]["acc_mean"] * 100 for lv in levels])
            ci = np.array([recs[lv]["ci95"] * 100 for lv in levels])
            a.fill_between(xs, ys - ci, ys + ci, color=COLORS[sigma], alpha=0.13, lw=0)
            a.plot(xs, ys, color=COLORS[sigma], lw=1.1, marker="o", ms=2.4,
                   label=f"$\\sigma$={sigma}")
        tick_idx = sorted({0, len(levels) // 2, len(levels) - 1})
        a.set_xticks(tick_idx)
        a.set_xticklabels([str(levels[i]) for i in tick_idx], fontsize=5.8)
        a.set_xlim(-0.35, len(levels) - 0.65)
        a.set_ylim(24, 88)
        a.set_yticks([40, 60, 80])
        a.tick_params(axis="both", length=2.0, pad=1.4)
        if k % 2 == 1:
            a.set_yticklabels([])
        a.text(0.04, 0.97, title, transform=a.transAxes, fontsize=6.0,
               color=PALETTE["ink"], va="top",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.9))
        if k == 0:
            a.legend(frameon=False, fontsize=5.8, handlelength=1.0, loc="lower left",
                     bbox_to_anchor=(0.0, -0.04), borderaxespad=0.0, labelspacing=0.25)
    fig.text(0.008, 0.52, "accuracy (%)", rotation=90, va="center", fontsize=7)
    panel_label(a_first, "(a)", dx=-0.22, dy=1.16)

    axb = fig.add_subplot(gs[1])
    clean = [data[s]["clean"]["acc"] * 100 for s in sigmas]
    robust = [np.mean([data[s]["axes"][a][lv]["acc_mean"] * 100 for a, lv in SEVERE])
              for s in sigmas]
    axb.plot(clean, robust, color=PALETTE["gray"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    for s, x, y in zip(sigmas, clean, robust):
        axb.scatter([x], [y], s=22, color=COLORS[s], zorder=3, edgecolors="white",
                    linewidths=0.4)
        axb.annotate(f"$\\sigma$={s}", (x, y), textcoords="offset points",
                     xytext=(4, 3.5), fontsize=6.4, color=PALETTE["ink"])
    axb.set_xlabel("clean accuracy (%)", fontsize=6.8)
    axb.set_ylabel("severe-corruption accuracy (%)", fontsize=6.8)
    axb.tick_params(labelsize=6.2)
    axb.set_xlim(73.0, 78.8)
    axb.set_ylim(40, 63)
    panel_label(axb, "(b)", dx=-0.24, dy=1.04)

    axc = fig.add_subplot(gs[2])
    labels = [f"{ax}\n{lv}" for ax, lv in SEVERE]
    gains = [(data[0.2]["axes"][a][lv]["acc_mean"] - data[0.1]["axes"][a][lv]["acc_mean"]) * 100
             for a, lv in SEVERE]
    colors = [PALETTE["blue"], PALETTE["blue"], PALETTE["clay"], PALETTE["sage"]]
    ypos = np.arange(len(labels))
    axc.barh(ypos, gains, color=colors, height=0.62, alpha=0.9)
    for y, g in zip(ypos, gains):
        axc.text(g + 0.5, y, f"+{g:.1f}", va="center", fontsize=6.2, color=PALETTE["ink"])
    axc.set_yticks(ypos)
    axc.set_yticklabels(labels, fontsize=6.0)
    axc.invert_yaxis()
    axc.set_xlabel("$\\Delta$ accuracy, $\\sigma$=0.2 vs 0.1 (pp)", fontsize=6.8)
    axc.tick_params(labelsize=6.2)
    axc.set_xlim(0, max(gains) * 1.22)
    panel_label(axc, "(c)", dx=-0.30, dy=1.04)

    save_fig(fig, os.path.join(OUT, "fig_d3_sigma"))
    print("[OK] fig_d3_sigma ->", OUT)


if __name__ == "__main__":
    fig_d3()
