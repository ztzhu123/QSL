from functools import partial

import h5py
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from qutip import *
from scipy.special import comb
import seaborn as sns

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, subplot
from path import DATA_DIR
from plot_toolbox import plot1d as _plot1d
from plot_toolbox import plot2d
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

colors = {
    "evo": "#08519c",
    "sim": "#517ecc",
    "MT": "#e84a51",
    "ML": "#2e8b57",
    "ML_star": "#aa6eb3",
    "lower": "#e84a51",
    "upper": "#e84a51",
}
sim_alpha = 0.6

marker_colors = ["#6499e9", "#8644a2", "#f57d1f"]
markers = ["^", "^", "^"]
lw = 2


@article_style()
def plot_all():
    nrows = 3
    ncols = 2
    fig = fig_in_a4(1, 0.6, dpi=100)
    ag = AxesGroup(nrows * ncols, ncols, figs=fig)
    axs = np.reshape(ag.axes, (3, 2))

    fig.subplots_adjust(
        hspace=0.2,
        wspace=0.25,
        top=0.93,
        bottom=0.06,
        left=0.1,
        right=0.98,
    )

    ########################################################
    ###### Parameters ######################################
    ########################################################
    list_of_Ls = [6, 12, 24, 48, 96]
    J = -2.0
    W = 0.0

    Blues = plt.cm.Blues(np.linspace(0, 1, len(list_of_Ls) + 2, endpoint=True))
    Reds = plt.cm.Reds(np.linspace(0, 1, len(list_of_Ls) + 2))
    Greens = plt.cm.Greens(np.linspace(0, 1, len(list_of_Ls) + 2))

    path = DATA_DIR / "scaling_1d"
    # Fock
    for i_L, L in enumerate(list_of_Ls):
        t_values, overlaps, n_k0_t = np.loadtxt(
            path / ("1d_L%d_J%.2fW%.2fQSL.dat" % (L, J, W)),
            usecols=[0, 1, 5],
            unpack=True,
        )
        t_values = t_values / (2 * np.pi)

        ### Now plotting ########################
        plot1d(
            t_values * 1e3,
            overlaps,
            color=Blues[i_L + 2],
            ax=axs[0, 0],
            label="$L=%d$" % L,
        )
        plot1d(
            t_values * 1e3,
            overlaps,
            color=Blues[i_L + 2],
            ax=axs[1, 0],
            label="$L=%d$" % L,
        )
        plot1d(
            t_values * 1e3,
            n_k0_t,
            color=Greens[2 + i_L],
            ax=axs[2, 0],
            label="$L=%d$" % L,
        )

        E = 0  ## This is because our initial state is a product state (no hopping contribution) and the staggered potential is zero
        DeltaE = get_energy_uncertainty(L, E, J)

        ## MT bound
        MT_bound = np.cos(DeltaE * t_values * 2 * np.pi)
        subsetter2 = np.where(t_values <= np.pi / (2 * DeltaE * 2 * np.pi) + 0.004)[0]
        plot1d(
            t_values[subsetter2] * 1e3,
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            lw=lw,
            label="$L=%d$" % L,
            ax=axs[0, 0],
        )
        axs[1, 0].plot(
            t_values[subsetter2] * 1e3,
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            lw=lw,
            label="$L=%d$" % L,
        )

    # Superposition
    for i_L, L in enumerate(list_of_Ls):
        t_values, overlaps, n_k0_t = np.loadtxt(
            path / (("1d_L%d_J%.2fW%.2fQSL_superposition.dat") % (L, J, W)), unpack=True
        )
        t_values = t_values / (2 * np.pi)

        ### Now plotting ########################
        plot1d(
            t_values * 1e3,
            overlaps,
            color=Blues[i_L + 2],
            linestyle="-",
            lw=lw,
            ax=axs[0, 1],
        )
        plot1d(
            t_values * 1e3,
            overlaps,
            color=Blues[i_L + 2],
            linestyle="-",
            lw=lw,
            ax=axs[1, 1],
        )
        plot1d(
            t_values * 1e3,
            n_k0_t,
            color=Greens[2 + i_L],
            linestyle="-",
            lw=lw,
            ax=axs[2, 1],
        )

        E = 0  ## This is because our initial state is a product state (no hopping contribution) and the staggered potential is zero
        ## Notce that <H²> is also the same if using the superposition state... Interesting.
        DeltaE = get_energy_uncertainty(L, E, J)

        ## MT bound
        MT_bound = np.cos(DeltaE * t_values * 2 * np.pi)
        subsetter2 = np.where(t_values <= np.pi / (2 * DeltaE * 2 * np.pi) + 0.004)[0]
        plot1d(
            t_values[subsetter2] * 1e3,
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            lw=lw,
            label=r"$\rm ML\ bound$",
            ax=axs[0, 1],
        )
        plot1d(
            t_values[subsetter2] * 1e3,
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            lw=lw,
            label=r"$\rm ML\ bound$",
            ax=axs[1, 1],
        )

    ag.grid(False)
    ag.tick_params(direction="out")

    ag.figs[0].suptitle(
        r"$J~/~2\pi=%d\ {\rm MHz},\ W~/~2\pi = %d\ {\rm MHz}$" % (J, W),
        fontsize=AX_TITLE_SIZE,
        y=1,
    )

    ag.set_xlabel(r"$t$ (ns)", fontsize=AX_LABEL_SIZE)
    ag.set_ylabel(
        r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$",
        fontsize=AX_LABEL_SIZE,
        axes=axs[:2, :2].flatten(),
        usetex=True,
    )
    ag.set_ylabel(r"$n_{k=0}$", fontsize=AX_LABEL_SIZE, axes=axs[-1])

    ag.set_xticks([0, 50, 100, 150, 190], xlim=[0, 1200 / (2 * np.pi)])

    ag.set_yticks(np.arange(0, 1.001, 0.5), ylim=[0, 1], axes=axs[0], sharey=0)
    ag.set_yticks(
        np.arange(0.4, 0.801, 0.2), ylim=[0.4, 0.8], axes=axs[-1, 0], sharey=0
    )
    ag.set_yticks(
        np.arange(3.4, 3.9001, 0.25), ylim=[3.4, 3.9], axes=axs[-1, 1], sharey=0
    )

    axs[1, 0].set_yscale("log")
    ag.set_yticks([1e-32, 1e-16, 1], ylim=[1e-32, 1], axes=axs[1, 0], sharey=0)
    axs[1, 1].set_yscale("log")
    ag.set_yticks([1e-15, 1e-8, 1], ylim=[1e-15, 1], axes=axs[1, 1], sharey=0)

    for i in range(2):
        ax = axs[i, 0]
        handles, labels = ax.get_legend_handles_labels()
        indexes = np.arange(len(handles)).tolist()
        indexes = np.hstack([indexes[::2], indexes[1::2]])
        handles = [handles[i] for i in indexes]
        labels = [labels[i] for i in indexes]
        if i == 0:
            kw = {"loc": "upper center"}
        else:
            kw = {"loc": "lower right", "bbox_to_anchor": (0.79, 0.0, 0.2, 0.5)}
        ax.legend(
            handles,
            labels,
            framealpha=0.2,
            prop={"size": LEGEND_SIZE},
            ncol=2,
            **kw,
        )
    for ax in axs[-1]:
        handles, labels = ax.get_legend_handles_labels()
        indexes = np.arange(len(handles)).tolist()
        handles = [handles[i] for i in indexes]
        labels = [labels[i] for i in indexes]
        ax.legend(
            handles,
            labels,
            framealpha=0.2,
            prop={"size": LEGEND_SIZE},
            loc="lower center",
        )

    for i in range(3):
        ax = axs[i, 1]
        ax.get_legend().remove()
    axs[0, 0].set_title(
        r"$|\psi(0)\mathsf{\rangle} = |1010\ldots10\mathsf{\rangle}$",
        fontsize=AX_LABEL_SIZE + 2,
        usetex=1,
    )
    axs[0, 1].set_title(
        r"$|\psi(0)\mathsf{\rangle} = \frac{1}{2}(|1010\ldots 10\mathsf{\rangle} + \sqrt{3}|0101\ldots 01\mathsf{\rangle})$",
        fontsize=AX_LABEL_SIZE + 2,
        usetex=1,
    )
    annot_alphabet(
        axs.T.flatten(),
        fontsize=FONTSIZE,
        dx=-0.06,
        dy=0.015,
        transform="fig",
    )


def get_energy_uncertainty(L, E, J):
    H2 = J**2 * (L - 1)

    return np.sqrt(H2 - E * E)
