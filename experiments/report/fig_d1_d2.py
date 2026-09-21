"""D1/D2 figures — enriched reference-grade layout.

D1: (a) batch-bounds heatmap, (b) per-sample heatmap (shared scale),
    (c) gap bars (batch@128 - per-sample) per variant.
D2: scatter + regression 95% CI band + direct category labels.
Morandi palette; (a)(b)(c) labels; no in-figure titles.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.report.style import (  # noqa: E402
    PALETTE, SEQ_CMAP, SEMANTICS, apply_style, panel_label, save_fig,
)

OUT = "paper/figures"


def _apply(theme="times"):
    plt.rcdefaults()
    apply_style(theme=theme)
    plt.rcParams["axes.grid"] = False


def _heat(ax, M, rows, cols, vmin, vmax, cmap=SEQ_CMAP):
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dark = M[i, j] > (vmin + vmax) / 2 + 4
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.6,
                    color="#FBFAF7" if dark else PALETTE["ink"])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(M.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", lw=1.4, zorder=2)
    for j in range(M.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", lw=1.4, zorder=2)
    return im


def fig_d1():
    d = json.load(open("experiments/results/e4_bounds_ab.json"))["results"]
    variants = list(d.keys())
    bs_list = [16, 64, 128]
    Mb = np.array([[d[v]["batch"][str(b)] * 100 for b in bs_list] for v in variants])
    Mp = np.array([[d[v]["per_sample"][str(b)] * 100 for b in bs_list] for v in variants])

    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.86], wspace=0.46,
                          left=0.085, right=0.915, top=0.80, bottom=0.20)

    ax = fig.add_subplot(gs[0])
    im = _heat(ax, Mb, variants, bs_list, 72, 92)
    ax.set_xlabel("Eval batch size")
    ax.set_ylabel("Variant")
    panel_label(ax, "(a)", dx=-0.30, dy=1.05)

    ax = fig.add_subplot(gs[1])
    _heat(ax, Mp, variants, bs_list, 72, 92, cmap=SEQ_CMAP)
    ax.set_yticklabels([])
    ax.set_xlabel("Eval batch size")
    panel_label(ax, "(b)", dx=-0.12, dy=1.05)
    cax = fig.add_axes([0.945, 0.26, 0.011, 0.46])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Accuracy (%)", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.2)
    cbar.outline.set_linewidth(0.5)

    ax = fig.add_subplot(gs[2])
    y = np.arange(len(variants))
    gap = Mb[:, -1] - Mp[:, -1]
    import matplotlib.patheffects as pe
    norm = (gap - gap.min()) / max(gap.max() - gap.min(), 1e-9)
    for i in range(len(variants)):
        ax.barh(i, gap[i], height=0.55, color=SEQ_CMAP(0.35 + 0.6 * norm[i]),
                edgecolor="white", linewidth=0.6)
    for i, g in enumerate(gap):
        ax.text(g + 1.0, i, f"{g:.0f} pp", va="center", fontsize=6.4,
                color=PALETTE["ink"],
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(len(variants) - 0.5, -0.5)
    ax.set_xlim(0, gap.max() * 1.22)
    ax.set_xlabel("Gap (pp)")
    panel_label(ax, "(c)", dx=-0.14, dy=1.05)
    save_fig(fig, f"{OUT}/fig_d1_bounds")
    plt.close(fig)
    print("[OK] fig_d1_bounds")


def fig_d2():
    d = json.load(open("experiments/results/d2_mechanism.json"))
    pts = d["points"]
    pred = np.array([p["pred"] for p in pts])
    act = np.array([p["actual"] for p in pts]) * 100
    fig, ax = plt.subplots(figsize=(4.0, 3.05))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.14)
    cmap = {"dropout": PALETTE["dusty_blue"], "coarse_quantize": PALETTE["sand"],
            "gaussian_noise": PALETTE["clay"], "fps": PALETTE["sage"]}
    order = np.argsort(pred)
    ax.plot(pred[order], act[order], "-", color=PALETTE["moss"], lw=0.7,
            alpha=0.7, zorder=2)
    ax.fill_between(pred[order], act[order], act[order].min() - 4,
                    color=PALETTE["moss"], alpha=0.14, linewidth=0, zorder=1)
    for axis, c in cmap.items():
        m = np.array([p["cond"].split(":")[0] == axis for p in pts])
        ax.scatter(pred[m], act[m], s=9, c=c, alpha=0.9, linewidths=0, zorder=3)
    z = np.polyfit(pred, act, 1)
    xs = np.linspace(pred.min(), pred.max(), 80)
    ys = np.polyval(z, xs)
    resid = act - np.polyval(z, pred)
    se = resid.std(ddof=2)
    ax.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, color=PALETTE["sage"],
                    alpha=0.18, linewidth=0, zorder=1)
    ax.plot(xs, ys, "-", color=PALETTE["ink"], lw=0.8, zorder=2)
    label_pos = {"dropout": (0.975, 0.05), "fps": (0.975, 0.14),
                 "gaussian_noise": (0.975, 0.23), "coarse_quantize": (0.975, 0.32)}
    for axis, c in cmap.items():
        ax.text(label_pos[axis][0], label_pos[axis][1], axis.replace("_", " "),
                transform=ax.transAxes, ha="right", fontsize=6.8, color=c)
    box = dict(boxstyle="square,pad=0.32", facecolor="white",
               edgecolor=PALETTE["ink"], linewidth=0.6)
    ax.text(0.04, 0.96, f"$R^2$ = {d['r2']:.2f}   $\\rho$ = {d['spearman_rho']:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="top", bbox=box, zorder=5)
    ax.set_xlabel("Predicted susceptibility (band product)")
    ax.set_ylabel("Actual accuracy drop (pp)")
    panel_label(ax, "(a)", dx=-0.16, dy=1.02)
    save_fig(fig, f"{OUT}/fig_d2_mechanism")
    plt.close(fig)
    print("[OK] fig_d2_mechanism")


if __name__ == "__main__":
    _apply("times")
    fig_d1()
    fig_d2()
