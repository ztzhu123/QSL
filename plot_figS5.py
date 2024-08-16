from functools import partial
from pathlib import Path
import re

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
    filename = DATA_DIR / "ac_phase.h5"
    data = {}
    with h5py.File(filename, "r") as f:
        for name in f.keys():
            group = f[name]
            data[name] = {"corr": {}, "no_corr": {}}
            for i in ["corr", "no_corr"]:
                for k, v in group[i].items():
                    data[name][i][k] = v[()]
    return data


@article_style()
def plot_all(save=False):
    all_data = collect_result()["X 21->10"]

    fig = fig_in_a4(1, 0.18, dpi=200)
    ag = AxesGroup(2, 3, figs=fig)

    cmap = "Spectral_r"

    for i, key in enumerate(["no_corr", "corr"]):
        ax = ag.axes[i]
        data = all_data[key]
        m_gates = data["m_gates"]
        phases = data["phases"]
        P0 = data["probs"][0]
        plot2d(
            P0.T,
            xlim=[m_gates.min(), m_gates.max()],
            ylim=[phases.min(), phases.max()],
            vmin=0,
            vmax=1,
            cmap=cmap,
            ax=ax,
            plot_cbar=0,
            interp="none",
        )

    pos = ag.axes[1].get_position()
    width = (pos.xmax - pos.xmin) * 0.0398950247000705  # according to cal_Omega
    height = pos.ymax - pos.ymin
    left = pos.xmax
    bottom = pos.ymin

    width = 0.012
    left = 0.88
    bottom = 0.2
    height = 0.69
    cax = fig.add_axes((left, bottom, width, height))
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([0, 0.5, 1])
    cbar.ax.set_title(r"$P_{|0\mathsf{\rangle}}$", fontdict={"fontsize": AX_LABEL_SIZE})

    ag.set_xlabel("Number of gates, $4m$", fontsize=AX_LABEL_SIZE)
    ag.set_ylabel("$\phi$", fontsize=AX_LABEL_SIZE)

    xs = np.arange(0, 81, 20)

    ag.set_xticks(xs, fontsize=AX_LABEL_SIZE)

    ys = [-np.pi, 0, np.pi]
    # ag.set_yticks(ticks=ys, labels=as_pi_str(ys, frac=0), fontsize=AX_LABEL_SIZE)

    ag.tick_params(direction="out")
    ag.grid(False)
    ag.annot_alphabet(fontsize=FONTSIZE)
    fig.subplots_adjust(wspace=0.2, top=0.89, bottom=0.2, left=0.15, right=0.85)