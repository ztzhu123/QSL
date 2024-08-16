from functools import partial
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, num_to_alphabet, subplot
from path import DATA_DIR
import plot_sphere
from plot_toolbox import plot1d as _plot1d
from plot_toolbox import plot2d as _plot2d
from plot_toolbox import plot_ecdf as _plot_ecdf
from styles import (
    AX_LABEL_SIZE,
    AX_TITLE_SIZE,
    CAPSIZE,
    DASHES,
    FONTSIZE,
    LEGEND_SIZE,
    LW,
    MEW,
    MS,
    USE_TEX,
    article_style,
)

plot1d = partial(_plot1d, constrained_layout=False)


plot2d = partial(_plot2d, constrained_layout=False)


def collect_result():
    filename = DATA_DIR / "cal_Omega_qutrit.h5"
    data = {}
    with h5py.File(filename, "r") as f:
        for name in f.keys():
            group = f[name]
            data[name] = {}
            for k, v in group.items():
                data[name][k] = v[()]
    return data


@article_style()
def plot_all():
    fig = fig_in_a4(1, 0.18, dpi=200)
    ag = AxesGroup(2, 3, figs=fig)
    data = collect_result()
    Omega = 5
    case_name = f"2d_Omega={Omega}"
    data = data[case_name]

    ts = data["ts"]
    Ps = data["Ps"]
    amps = data["amps"]
    actual_Omegas = data["actual_Omegas"]
    opt_amp = data["opt_amp"]
    colors = ["#333c83", "k", "#c64756"]

    coeffs = np.polyfit(amps, actual_Omegas, 1)

    _, cbar = plot2d(
        Ps.T,
        xlim=[amps.min(), amps.max()],
        ylim=[ts.min(), ts.max()],
        interp="none",
        ax=ag.axes[0],
        return_cbar=1,
        vmin=0,
        vmax=1,
        cmap="Blues",
        # tick_fontsize=AX_LABEL_SIZE,
    )
    ax = plot1d(
        amps,
        actual_Omegas,
        marker="o",
        ls="",
        ax=ag.axes[1],
        zorder=np.inf,
        mew=MEW,
        ms=MS + 1,
        hollow=0,
        label="Exp.",
        color=colors[0],
    )
    ax = plot1d(
        amps,
        np.polyval(coeffs, amps),
        ls="--",
        ax=ag.axes[1],
        zorder=100,
        dashes=DASHES,
        lw=LW,
        label="Fit",
        color=colors[1],
    )
    ax.scatter(
        opt_amp,
        Omega,
        marker="*",
        color=colors[2],
        zorder=np.inf,
        s=70,
        label="Optimal",
    )
    ax.legend(
        labelspacing=0.1,
        framealpha=0.4,
        prop={"size": LEGEND_SIZE},
        loc="upper left",
    )

    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([0, 0.5, 1])
    cbar.ax.set_title(r"$P_{|1\mathsf{\rangle}}$", fontdict={"fontsize": AX_LABEL_SIZE})

    ag.set_xlabel("Control parameter", fontsize=AX_LABEL_SIZE)
    ag.set_ylabel("Driving time (ns)", fontsize=AX_LABEL_SIZE, axes=0, sharey=0)
    ag.set_ylabel(r"$|\Omega|~/~2\pi$ (MHz)", fontsize=AX_LABEL_SIZE, axes=1, sharey=0)

    xs = [0.11, 0.12, 0.13, 0.14, 0.15]
    ys = [int(ts.min()), 50, 100, 150, 200, 250]
    ag.set_xticks(xs, fontsize=AX_LABEL_SIZE, axes=0)
    ag.set_yticks(ys, fontsize=AX_LABEL_SIZE, axes=0, sharey=0)

    xs = [0.11, 0.12, 0.13, 0.14, 0.15]
    ys = np.arange(4.2, 6 + 0.1, 0.9)
    ag.set_xticks(
        xs, fontsize=AX_LABEL_SIZE, xlim=[xs[0], xs[-1]], axes=1, xlim_pad_ratio=0.02
    )
    ag.set_yticks(
        ys,
        fontsize=AX_LABEL_SIZE,
        ylim=[ys[0], ys[-1]],
        axes=1,
        sharey=0,
        ylim_pad_ratio=0.02,
    )
    ax.set_xticklabels([0.11, 0.12, 0.13, 0.14, 0.15])
    ax.vlines(opt_amp, *ax.get_ylim(), ls="--", color=colors[2], lw=LW, zorder=-np.inf)
    ax.text(opt_amp-0.001, 5.05, round(opt_amp, 3), fontsize=AX_LABEL_SIZE, ha="right", va='center', color=colors[2])

    ag.tick_params(direction="out")
    ag.grid(False)
    ag.annot_alphabet(fontsize=FONTSIZE, dx=-0.05)
    fig.subplots_adjust(wspace=0.25, top=0.89, bottom=0.22, left=0.15, right=0.85)