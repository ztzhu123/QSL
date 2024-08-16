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


@article_style()
def plot_parity():
    W=1.8
    filename = DATA_DIR / f"W{W}_parity.h5"
    with h5py.File(filename, "r") as f:
        ts = f["ts"][()]
        gammas = f["gammas"][()]
        gammas_fit = f["gammas_fit"][()]
        parities = f["parities"][()]
        parities_fit = f["parities_fit"][()]

    chosen_t = np.arange(0, 101, 20)
    fig = fig_in_a4(1, 0.3, dpi=200)
    ag = AxesGroup(6, 3, figs=fig)
    # ag.init_axes()
    ag.move_to_monitor(1)

    for i, t in enumerate(chosen_t):
        index = np.where(ts == t)[0].item()
        ax = ag.axes[i]
        # NOTE: we only plot the parity of the first sample here
        p = parities[0, index]
        p_fit = parities_fit[0, index]
        if ag.is_left(ax) and ag.is_bottom(ax):
            label_exp = "Exp."
            label_fit = "Fit"
        else:
            label_exp = label_fit = None
        plot1d(
            gammas,
            p,
            marker="o",
            label=label_exp,
            ax=ax,
            ls="",
            ms=MS,
            mew=MEW,
            hollow=0,
            zorder=np.inf,
            color="#ed9455",
        )
        plot1d(
            gammas_fit,
            p_fit,
            ls="--",
            label=label_fit,
            ax=ax,
            legend_size=LEGEND_SIZE,
            legend_loc="upper right",
            lw=LW,
            color="k",
        )
        ax.set_title(f"t = {t:d} ns", fontsize=AX_TITLE_SIZE, y=0.98)
    xs = [gammas.min(), 0, gammas.max()]
    # ag.set_xticks(
    #     xs,
    #     as_pi_str(xs, must_reduce_pi=1, frac=0),
    #     fontsize=AX_LABEL_SIZE,
    #     xlim=[-np.pi / 2, np.pi / 2],
    #     xlim_pad_ratio=0.02,
    # )
    ag.set_xlabel(r"$\gamma$", usetex=1, labelpad=0, fontsize=AX_LABEL_SIZE)

    ys = [-1, 0, 1]
    ag.set_yticks(ys, fontsize=AX_LABEL_SIZE, ylim=[ys[0], ys[-1]], ylim_pad_ratio=0.02)
    ag.set_ylabel(
        r"$\langle\mathcal{P}(\gamma)\rangle$",
        usetex=1,
        labelpad=0.5,
        fontsize=AX_LABEL_SIZE,
    )

    ag.tick_params(direction="out")
    ag.grid(False)
    ag.annot_alphabet(fontsize=FONTSIZE)
    fig.subplots_adjust(
        hspace=0.33, wspace=0.2, top=0.92, bottom=0.1, left=0.06, right=0.98
    )
