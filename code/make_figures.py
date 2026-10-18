"""Figures for the paper, drawn from the released per-split results.

fig_power_alpha.pdf : power (left) and human effort (right) against alpha for
                      every dataset/annotator, HLCL vs the certified baselines.
fig_validity.pdf    : share of splits whose realised FDP exceeded alpha, all
                      methods; the contract is <= delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
VER = sys.argv[1] if len(sys.argv) > 1 else "v9"

SETTINGS = [("DBPedia (A)", "dbpedia"), ("WOS-46985 (A)", "wos"),
            ("Blurb (A)", "bgc"), ("Blurb (B)", "bgc_haiku")]
METHODS = [("HLCL", "HLCL (ours)", "#1b6ca8", "o", "-"),
           ("ItemBest", "Item-level certified (best test)", "#d1495b", "s", "--"),
           ("HSC", "Hierarchical selective cls.", "#00798c", "P", "--"),
           ("PerField-LTT", "Per-field risk control", "#66a182", "D", ":")]
ALL = METHODS + [("ConfLabel", "Item-level conformal", "#edae49", "^", "--"),
                 ("DepthBH", "Depth-wise BH", "#8d6a9f", "v", "-."),
                 ("Conf>=0.9", "Fixed threshold", "#555555", "x", "-.")]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "legend.fontsize": 7, "pdf.fonttype": 42})


def series(runs, m, key):
    al = sorted({r["alpha"] for r in runs})
    return al, [np.mean([r[m][key] for r in runs if r["alpha"] == a]) for a in al]


def viol(runs, m):
    al = sorted({r["alpha"] for r in runs})
    return al, [np.mean([r[m]["fdp"] > a + 1e-12 for r in runs if r["alpha"] == a]) for a in al]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: json.loads((ROOT / "results" / f"{VER}_{k}.json").read_text())["runs"]
            for _, k in SETTINGS}
    # Item-cert. with the best of its three test choices per alpha (as in the tables)
    for runs in data.values():
        for a in sorted({r["alpha"] for r in runs}):
            R = [r for r in runs if r["alpha"] == a]
            have = [m for m in ("ItemLTT", "ItemLTT-wsr", "ItemLTT-iut") if m in R[0]]
            b = min(have, key=lambda m: np.mean([r[m]["effort"] for r in R]))
            for r in R:
                r["ItemBest"] = r[b]

    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.4), sharex="col")
    for j, (title, k) in enumerate(SETTINGS):
        for m, lab, col, mk, ls in METHODS:
            a, p = series(data[k], m, "power")
            ax[0, j].plot(a, p, color=col, marker=mk, ls=ls, ms=3, lw=1, label=lab)
            a, e = series(data[k], m, "effort")
            ax[1, j].plot(a, e, color=col, marker=mk, ls=ls, ms=3, lw=1)
        ax[0, j].set_title(title)
        ax[1, j].set_xlabel(r"$\alpha$")
        ax[0, j].set_ylim(-0.02, 1.02)
        ax[1, j].set_ylim(-0.02, 1.02)
    ax[0, 0].set_ylabel("power (correct labels accepted)")
    ax[1, 0].set_ylabel("human effort")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_power_alpha.pdf")

    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=True)
    for j, (title, k) in enumerate(SETTINGS):
        for m, lab, col, mk, ls in ALL:
            a, v = viol(data[k], m)
            ax[j].plot(a, v, color=col, marker=mk, ls=ls, ms=3, lw=1, label=lab)
        ax[j].axhline(0.1, color="k", lw=0.6, ls=(0, (2, 2)))
        ax[j].set_title(title)
        ax[j].set_xlabel(r"$\alpha$")
        ax[j].set_ylim(-0.03, 1.03)
    ax[0].set_ylabel(r"share of splits with FDP $>\alpha$")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(OUT / "fig_validity.pdf")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
