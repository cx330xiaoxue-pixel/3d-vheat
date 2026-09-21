"""Design options for user selection.

fig_d1_bounds_opt{A,B,C}:  (a) batch heatmap + (b) variant A/B/C + (c) gap bars
fig_d2_mechanism_opt{A,B,C}: D2 designs A (ECG trace) / B (marginals) / C (raincloud facets)
All PNG 600dpi + PDF via experiments/report/style.py.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.report.style import (  # noqa: E402
    PALETTE, SEQ_CMAP, apply_style, panel_label, save_fig,
)

OUT = "paper/figures"
plt.rcdefaults()
apply_style(theme="times")
plt.rcParams["axes.grid"] = False

D1 = json.load(open("experiments/results/e4_bounds_ab.json"))["results"]
VARIANTS = list(D1.keys())
BS = [16, 64, 128]
MB = np.array([[D1[v]["batch"][str(b)] * 100 for b in BS] for v in VARIANTS])
MP = np.array([[D1[v]["per_sample"][str(b)] * 100 for b in BS] for v in VARIANTS])


def _heat(ax, M, rows):
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=72, vmax=92)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dark = M[i, j] > 86
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.6,
                    color="#FBFAF7" if dark else PALETTE["ink"])
    for i in range(M.shape[0] + 1):
        ax.axhline(i - 0.5, color="white", lw=1.4, zorder=2)
    for j in range(M.shape[1] + 1):
        ax.axvline(j - 0.5, color="white", lw=1.4, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([str(b) for b in BS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    return im


def _gap_bars(ax, label_pp=True):
    y = np.arange(len(VARIANTS))
    gap = MB[:, -1] - MP[:, -1]
    norm = (gap - gap.min()) / max(gap.max() - gap.min(), 1e-9)
    for i in range(len(VARIANTS)):
        ax.barh(i, gap[i], height=0.55, color=SEQ_CMAP(0.35 + 0.6 * norm[i]),
                edgecolor="white", linewidth=0.6)
        ax.text(gap[i] + 1.0, i, f"{gap[i]:.0f} pp", va="center", fontsize=6.4,
                color=PALETTE["ink"],
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
    ax.set_xlim(0, gap.max() * 1.25)
    ax.set_xlabel("Gap (pp)")


def d1_layout(opt):
    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.86], wspace=0.46,
                          left=0.085, right=0.915, top=0.80, bottom=0.20)

    ax = fig.add_subplot(gs[0])
    im = _heat(ax, MB, VARIANTS)
    ax.set_xlabel("Eval batch size")
    ax.set_ylabel("Variant")
    panel_label(ax, "(a)", dx=-0.30, dy=1.05)
    cax = fig.add_axes([0.945, 0.26, 0.011, 0.46])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Accuracy (%)", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.2)
    cbar.outline.set_linewidth(0.5)

    ax = fig.add_subplot(gs[1])
    if opt == "A":  # raincloud over 7 variants per condition
        groups = [MP[:, -1], MB[:, -1]]
        for k, (vals, c, lab) in enumerate(zip(groups, [PALETTE["clay"], PALETTE["blue"]],
                                               ["per-sample", "batch$_{128}$"])):
            v = ax.violinplot([vals], positions=[k], vert=False, widths=0.8,
                              showextrema=False, showmedians=False)
            for b in v["bodies"]:
                b.set_facecolor(c)
                b.set_alpha(0.25)
                b.set_edgecolor(c)
                b.set_linewidth(0.6)
            bp = ax.boxplot([vals], positions=[k], vert=False, widths=0.22,
                            showfliers=False, patch_artist=True)
            for b in bp["boxes"]:
                b.set(facecolor="white", edgecolor=c, linewidth=1.0)
            for w in bp["whiskers"] + bp["caps"] + bp["medians"]:
                w.set(color=c, linewidth=0.9)
            rng = np.random.RandomState(0)
            ax.scatter(vals, k + rng.uniform(-0.06, 0.06, len(vals)), s=9,
                       c=c, alpha=0.9, linewidths=0)
            ax.text(vals.max() + 1.5, k, lab, fontsize=6.8, color=c, va="center")
        ax.set_yticks([0, 1])
        ax.set_yticklabels([])
        ax.set_ylim(1.6, -0.6)
        ax.set_xlim(20, 100)
    elif opt == "B":  # dumbbell dots
        y = np.arange(len(VARIANTS))
        for i in range(len(VARIANTS)):
            ax.plot([MP[i, -1], MB[i, -1]], [i, i], "-", color=PALETTE["gray_light"],
                    lw=1.4, solid_capstyle="round", zorder=1)
        ax.scatter(MP[:, -1], y, s=26, facecolors="white", edgecolors=PALETTE["clay"],
                   linewidths=1.0, zorder=3)
        ax.scatter(MB[:, -1], y, s=26, facecolors="white", edgecolors=PALETTE["blue"],
                   linewidths=1.0, zorder=3)
        ax.text(np.mean(MP[:, -1]), -0.9, "per-sample", ha="center", fontsize=6.5,
                color=PALETTE["clay"])
        ax.text(np.mean(MB[:, -1]), -0.9, "batch$_{128}$", ha="center", fontsize=6.5,
                color=PALETTE["blue"])
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
        ax.set_xlim(20, 96)
        ax.set_xlabel("Accuracy (%)")
    else:  # C: dot plot bs16/64/128
        y = np.arange(len(VARIANTS))
        shades = [0.25, 0.55, 0.9]
        for j, s in enumerate(shades):
            ax.scatter(MB[:, j], y + (j - 1) * 0.16, s=16, color=SEQ_CMAP(s),
                       edgecolors="white", linewidths=0.4, zorder=3)
        ax.axvline(float(MP[:, -1].mean()), color=PALETTE["clay"], lw=0.8,
                   ls=(0, (4, 2.5)), zorder=1)
        ax.text(float(MP[:, -1].mean()) + 0.8, 0.1, "per-sample\nmean", fontsize=6.3,
                color=PALETTE["clay"], va="top")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
        ax.set_xlim(20, 96)
        ax.set_xlabel("Accuracy (%)")
    panel_label(ax, "(b)", dx=-0.12, dy=1.05)

    ax = fig.add_subplot(gs[2])
    _gap_bars(ax)
    panel_label(ax, "(c)", dx=-0.14, dy=1.05)
    save_fig(fig, f"{OUT}/fig_d1_bounds_opt{opt}")
    plt.close(fig)
    print(f"[OK] fig_d1_bounds_opt{opt}")


D2 = json.load(open("experiments/results/d2_mechanism.json"))
PTS = D2["points"]
PRED = np.array([p["pred"] for p in PTS])
ACT = np.array([p["actual"] for p in PTS]) * 100
CATS = ["dropout", "coarse_quantize", "gaussian_noise", "fps"]
CMAP = {"dropout": PALETTE["dusty_blue"], "coarse_quantize": PALETTE["sand"],
        "gaussian_noise": PALETTE["clay"], "fps": PALETTE["sage"]}


def _cat_mask(cat):
    return np.array([p["cond"].split(":")[0] == cat for p in PTS])


def _fit(ax, pred, act, band=True):
    z = np.polyfit(pred, act, 1)
    xs = np.linspace(pred.min(), pred.max(), 80)
    ys = np.polyval(z, xs)
    if band:
        se = (act - np.polyval(z, pred)).std(ddof=2)
        ax.fill_between(xs, ys - 1.96 * se, ys + 1.96 * se, color=PALETTE["sage"],
                        alpha=0.18, linewidth=0, zorder=1)
    ax.plot(xs, ys, "-", color=PALETTE["ink"], lw=0.8, zorder=2)


def _statbox(ax):
    box = dict(boxstyle="square,pad=0.32", facecolor="white",
               edgecolor=PALETTE["ink"], linewidth=0.6)
    ax.text(0.04, 0.96, f"$R^2$ = {D2['r2']:.2f}   $\\rho$ = {D2['spearman_rho']:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="top", bbox=box, zorder=5)


def d2_layout(opt):
    if opt == "A":
        fig, ax = plt.subplots(figsize=(4.0, 3.05))
        fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.14)
        order = np.argsort(PRED)
        ax.plot(PRED[order], ACT[order], "-", color=PALETTE["moss"], lw=0.7,
                alpha=0.7, zorder=2)
        ax.fill_between(PRED[order], ACT[order], ACT[order].min() - 4,
                        color=PALETTE["moss"], alpha=0.14, linewidth=0, zorder=1)
        _fit(ax, PRED, ACT)
        for cat in CATS:
            m = _cat_mask(cat)
            ax.scatter(PRED[m], ACT[m], s=9, c=CMAP[cat], alpha=0.9, linewidths=0,
                       zorder=3)
            ax.text(0.975, 0.05 + 0.09 * CATS.index(cat), cat.replace("_", " "),
                    transform=ax.transAxes, ha="right", fontsize=6.8, color=CMAP[cat])
        _statbox(ax)
        ax.set_xlabel("Predicted susceptibility (band product)")
        ax.set_ylabel("Actual accuracy drop (pp)")
        panel_label(ax, "(a)", dx=-0.16, dy=1.02)
    elif opt == "B":
        fig = plt.figure(figsize=(4.3, 3.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.3], height_ratios=[1.3, 5],
                              wspace=0.08, hspace=0.08,
                              left=0.15, right=0.97, top=0.93, bottom=0.13)
        ax = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=ax)
        axr = fig.add_subplot(gs[1, 1], sharey=ax)
        _fit(ax, PRED, ACT)
        for cat in CATS:
            m = _cat_mask(cat)
            ax.scatter(PRED[m], ACT[m], s=9, c=CMAP[cat], alpha=0.9, linewidths=0)
        _statbox(ax)
        ax.set_xlabel("Predicted susceptibility (band product)")
        ax.set_ylabel("Actual accuracy drop (pp)")
        panel_label(ax, "(a)", dx=-0.20, dy=1.06)
        axt.hist(PRED, bins=18, color=PALETTE["dusty_blue"], alpha=0.75, linewidth=0)
        axt.tick_params(labelbottom=False)
        axr.hist(ACT, bins=18, orientation="horizontal", color=PALETTE["sage"],
                 alpha=0.75, linewidth=0)
        axr.tick_params(labelleft=False)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.4), sharey=True,
                                 gridspec_kw={"wspace": 0.12, "hspace": 0.42})
        fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.16)
        for ax, cat in zip(axes.reshape(-1), CATS):
            m = _cat_mask(cat)
            vals = ACT[m]
            v = ax.violinplot([vals], positions=[0], widths=0.9, showextrema=False,
                              showmedians=False)
            for b in v["bodies"]:
                b.set_facecolor(CMAP[cat])
                b.set_alpha(0.25)
                b.set_edgecolor(CMAP[cat])
                b.set_linewidth(0.6)
            bp = ax.boxplot([vals], positions=[0], widths=0.26, showfliers=False,
                            patch_artist=True)
            for b in bp["boxes"]:
                b.set(facecolor="white", edgecolor=CMAP[cat], linewidth=1.0)
            for w in bp["whiskers"] + bp["caps"] + bp["medians"]:
                w.set(color=CMAP[cat], linewidth=0.9)
            rng = np.random.RandomState(0)
            ax.scatter(rng.uniform(-0.14, 0.14, len(vals)), vals, s=7,
                       c=CMAP[cat], alpha=0.9, linewidths=0)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 0.6)
            ax.text(0.5, 0.08, cat.replace("_", " "), transform=ax.transAxes,
                    ha="center", fontsize=7, color=CMAP[cat])
            ax.set_ylabel("Drop (pp)")
        for ax, tag in zip(axes.reshape(-1), ["(a)", "(b)", "(c)", "(d)"]):
            panel_label(ax, tag, dx=-0.14, dy=1.03)
    save_fig(fig, f"{OUT}/fig_d2_mechanism_opt{opt}")
    plt.close(fig)
    print(f"[OK] fig_d2_mechanism_opt{opt}")


if __name__ == "__main__":
    for o in "ABC":
        d1_layout(o)
    for o in "ABC":
        d2_layout(o)
