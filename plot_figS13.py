from functools import partial

import h5py
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from qutip import *
from scipy.special import binom, comb
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

nsites = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22, 9, 16, 25], dtype=int)

nsites_str = [
    "6",
    "8",
    "10",
    "12",
    "14",
    "16",
    "18",
    "20",
    "22",
    "9",
    "4x4",
    "25",
]  # name string of the numbers
n_up = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 8, 13], dtype=int
)  # number of spin-ups
Jn = -2.0  # the nearest neighbouring couplings
W_type = "C"  # potential type: C(checkerboard) S(step)
W = 0.0  # onsite potential
PI = np.pi

marker_colors = ["#6499e9", "#8644a2", "#f57d1f"]
markers = ["^", "^", "^"]
lw = 2

plot_line_color = [
    "darkorange",
    "g",
    "r",
    "fuchsia",
    "b",
    "purple",
    "darkviolet",
    "k",
    "olive",
]

Emin = np.zeros(len(nsites), dtype=np.double)  # E_min
Emax = np.zeros(len(nsites), dtype=np.double)  # E_max
Eave = np.zeros(len(nsites), dtype=np.double)  # E
Evar = np.zeros(len(nsites), dtype=np.double)  # standard variance
Wnew = np.zeros(len(nsites), dtype=np.double)
positions = np.zeros(
    (len(nsites), 2), dtype=np.double
)  # store the position of cases in the pahse diagram


def load_data():
    for i in range(len(nsites)):
        # input data, calculate the quantities we want
        energy_file = (
            DATA_DIR
            / "fock_gap"
            / "EnergyDetails_{:s}_{:d}_Jn{:.2f}{}{:.2f}.dat".format(
                nsites_str[i], n_up[i], abs(Jn), W_type, W
            )
        )
        # input the energy, etc
        EnergyDetails = Read_EnergyDetails(energy_file)

        Wnew[i] = -EnergyDetails[3] / 2.0 * 1000.0 / 2.0 / PI

        # then we input the data at the crossover point
        energy_file = (
            DATA_DIR
            / "fock_gap"
            / "EnergyDetails_{:s}_{:d}_Jn{:.2f}{}{:.2f}.dat".format(
                nsites_str[i], n_up[i], abs(Jn), W_type, Wnew[i]
            )
        )
        EnergyDetails = Read_EnergyDetails(energy_file)
        Emin[i] = EnergyDetails[0]
        Emax[i] = EnergyDetails[1]
        Eave[i] = EnergyDetails[2]
        Evar[i] = EnergyDetails[3]
        positions[i] = [EnergyDetails[2] - EnergyDetails[0], EnergyDetails[3]] / (
            EnergyDetails[1] - EnergyDetails[0]
        )


@article_style()
def plot_all():
    load_data()
    nrows = 3
    ncols = 2
    fig = fig_in_a4(1, 0.55, dpi=130)
    ag = AxesGroup(nrows * ncols, ncols, figs=fig)
    axes = np.reshape(ag.axes, (3, 2))

    fig.subplots_adjust(
        hspace=0.38,
        wspace=0.25,
        top=0.94,
        bottom=0.06,
        left=0.07,
        right=0.9,
    )

    # plot panel a
    i = 0
    plot_phase_2d(axes[0, 0])
    ag.set_xticks([0, 1], xlim=[0, 1], axes=i, sharex=0)
    ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=i, sharey=0)

    # plot panel b
    i = 1
    plot_panel_b(axes[0, 1])
    ag.set_xticks([5, 15, 25], xlim=[5, 25], axes=i, sharex=0, xlim_pad_ratio=0.05)
    ag.set_yticks([4, 9, 14], ylim=[4, 14], axes=i, sharey=0, ylim_pad_ratio=0.05)

    # plot panel gaps
    plot_gaps(ag)

    annot_alphabet(
        axes.flatten(),
        fontsize=FONTSIZE,
        dx=-0.04,
        dy=0.015,
        transform="fig",
        share_top=[0, 1],
    )

    ag.grid(False)
    ag.tick_params(direction="out")


def Read_EnergyDetails(filename):
    # open the file
    file_handler = open(filename, "r")
    lines_list = file_handler.readlines()

    # extract the values
    Emin = np.double(lines_list[0].split()[1])
    Emax = np.double(lines_list[1].split()[1])
    Energy = np.double(lines_list[2].split()[1])
    variance = np.double(lines_list[3].split()[1])

    return [Emin, Emax, Energy, variance]


def plot_phase_2d(ax):
    colors = [
        "#86a789",
        "#d37676",
        "#7077a1",
    ]
    x = np.arange(0, 1.001, 0.001, dtype=np.double)
    y = np.sqrt(x * (1.0 - x), dtype=np.double)
    pos = np.where(x == 0.5)[0][0]

    ax.set_aspect("equal")

    ax.plot(x, y, color="k", linewidth=1)
    ax.fill_between(
        x[0 : (pos + 1)],
        y[0 : (pos + 1)],
        x[0 : (pos + 1)],
        color=colors[0],
        lw=0,
        alpha=0.8,
    )
    ax.fill_between(
        list(x[0 : (pos + 1)]) + list(x[pos:]),
        list(x[0 : (pos + 1)]) + list(1.0 - x[pos:]),
        0,
        color=colors[1],
        alpha=0.7,
        lw=0,
    )
    ax.fill_between(x[pos:], y[pos:], 1.0 - x[pos:], color=colors[2], lw=0, alpha=0.8)

    ax.text(
        0.26,
        0.6,
        r"$\rm ML$",
        transform=ax.transAxes,
        fontsize=AX_LABEL_SIZE + 2,
        rotation=45,
        rotation_mode="anchor",
        horizontalalignment="center",
        color="black",
        verticalalignment="center",
    )
    ax.text(
        0.756,
        0.6,
        r"$\rm ML^{\star}$",
        transform=ax.transAxes,
        fontsize=AX_LABEL_SIZE + 2,
        horizontalalignment="center",
        rotation=-45,
        rotation_mode="anchor",
        color="black",
        verticalalignment="center",
    )

    ax.text(
        0.5,
        0.2,
        r"$\rm MT$",
        transform=ax.transAxes,
        fontsize=AX_LABEL_SIZE + 2,
        horizontalalignment="center",
        color="black",
        verticalalignment="center",
    )

    ax.set_xlabel(
        r"$(E-E_{\rm{min}})~/~(E_{\rm{max}}-E_{\rm{min}})$",
        fontsize=AX_LABEL_SIZE,
        labelpad=-1.5,
    )
    ax.set_ylabel(r"$\Delta E~/~(E_{\rm{max}}-E_{\rm{min}})$", fontsize=AX_LABEL_SIZE)

    cm = plt.get_cmap("viridis")

    for i in range(len(nsites) - 3):
        p = ax.scatter(
            positions[i][0],
            positions[i][1],
            marker="o",
            s=40,
            c=nsites[i],
            vmin=6,
            vmax=22,
            cmap=cm,
        )

    ax.scatter(
        positions[-3][0],
        positions[-3][1],
        marker="X",
        s=50,
        alpha=1,
        color=plot_line_color[3],
        label=r"$3\times 3,n=5$",
    )
    ax.scatter(
        positions[-2][0],
        positions[-2][1],
        marker="h",
        s=50,
        alpha=1,
        color=plot_line_color[4],
        label=r"$4\times 4,n=8$",
    )
    ax.scatter(
        positions[-1][0],
        positions[-1][1],
        marker="^",
        s=50,
        alpha=1,
        color=plot_line_color[5],
        label=r"$5\times 5,n=13$",
    )

    # add results of positive Ws of 3x3 case
    energy_file = (
        DATA_DIR
        / "fock_gap"
        / "EnergyDetails_9_5_Jn2.00C{:.2f}.dat".format(abs(Wnew[-3]))
    )
    # input the energy, etc
    EnergyDetails = Read_EnergyDetails(energy_file)
    temEmin = EnergyDetails[0]
    temEmax = EnergyDetails[1]
    temEave = EnergyDetails[2]
    temEvar = EnergyDetails[3]

    ax.scatter(
        (temEave - temEmin) / (temEmax - temEmin),
        temEvar / (temEmax - temEmin),
        marker="X",
        s=50,
        alpha=1,
        color=plot_line_color[6],
        label=r"$3\times 3,n=5$",
    )

    # add results of positive Ws of 4x4 case
    energy_file = (
        DATA_DIR
        / "fock_gap"
        / "EnergyDetails_4x4_8_Jn2.00C{:.2f}.dat".format(abs(Wnew[-2]))
    )
    # input the energy, etc
    EnergyDetails = Read_EnergyDetails(energy_file)
    temEmin = EnergyDetails[0]
    temEmax = EnergyDetails[1]
    temEave = EnergyDetails[2]
    temEvar = EnergyDetails[3]

    ax.scatter(
        (temEave - temEmin) / (temEmax - temEmin),
        temEvar / (temEmax - temEmin),
        marker="h",
        s=50,
        alpha=1,
        color=plot_line_color[7],
        label=r"$4\times 4,n=8$",
    )

    # add results of positive Ws of 5x5 case
    energy_file = (
        DATA_DIR
        / "fock_gap"
        / "EnergyDetails_25_13_Jn2.00C{:.2f}.dat".format(abs(Wnew[-1]))
    )
    # input the energy, etc
    EnergyDetails = Read_EnergyDetails(energy_file)
    temEmin = EnergyDetails[0]
    temEmax = EnergyDetails[1]
    temEave = EnergyDetails[2]
    temEvar = EnergyDetails[3]

    ax.scatter(
        (temEave - temEmin) / (temEmax - temEmin),
        temEvar / (temEmax - temEmin),
        marker="^",
        s=50,
        alpha=1,
        color=plot_line_color[8],
        label=r"$5\times 5,n=13$",
    )

    fig = ax.figure
    pos = ax.get_position()

    height = 0.01
    left = pos.xmin + 0.02
    right = pos.xmax
    bottom = pos.ymax + 0.03
    width = 0.08

    cax = fig.add_axes([left, bottom, width, height])

    cbar = fig.colorbar(p, ax=ax, cax=cax, orientation="horizontal")

    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.set_xlabel(r"Number of qubits", fontsize=AX_LABEL_SIZE - 1)
    cbar.ax.set_xticks([6, 22])
    cbar.ax.tick_params(axis="both", which="major", length=2, pad=1)

    ax.legend(
        framealpha=0.4,
        prop={"size": LEGEND_SIZE},
        handletextpad=0.1,
        ncol=2,
        bbox_to_anchor=(0.27, 1, 0.85, 0.4),
        loc="center",
        labelspacing=0.2,
        columnspacing=0.6,
    )


def plot_panel_b(ax):
    label_size = AX_LABEL_SIZE
    ax.plot(
        nsites[0 : (len(nsites) - 3)],
        Evar[0 : (len(nsites) - 3)] * 1000.0 / 2.0 / PI,
        marker="o",
        linestyle="None",
        color="#0e7c66",
        label=r"1d chain $\Delta E$",
    )
    ax.plot(
        nsites[0 : (len(nsites) - 3)],
        np.sqrt(nsites[0 : (len(nsites) - 3)] - 1.0) * 2.0,
        linestyle="-",
        color="#fead61",
        label=r"$J_1\sqrt{N-1}$",
        zorder=-np.inf,
    )

    ax.scatter(
        nsites[-3],
        Evar[-3] * 1000.0 / 2.0 / PI,
        marker="X",
        s=70,
        color=plot_line_color[3],
        zorder=np.inf,
    )
    ax.scatter(
        nsites[-2],
        Evar[-2] * 1000.0 / 2.0 / PI,
        marker="h",
        s=70,
        color=plot_line_color[4],
        zorder=np.inf,
    )
    ax.scatter(
        nsites[-1],
        Evar[-1] * 1000.0 / 2.0 / PI,
        marker="^",
        s=70,
        color=plot_line_color[5],
        zorder=np.inf,
    )

    ax.legend(loc="upper left", fontsize=label_size - 1)

    ax.set_xlabel("Number of qubits", fontsize=label_size, labelpad=0)
    ax.set_ylabel(r"$\Delta E~/~2\pi$ (MHz)", fontsize=label_size)


def plot_gaps(ag):
    nrows = 3
    ncols = 2
    ax = np.reshape(ag.axes, (nrows, ncols))
    label_size = AX_LABEL_SIZE

    nsites_str_latex = [
        "1\\times6",
        "1\\times8",
        "1\\times10",
        "1\\times12",
        "1\\times14",
        "1\\times16",
        "1\\times18",
        "1\\times20",
        "1\\times22",
        "3\\times3",
        "4\\times4",
        "5\\times5",
    ]  # name string of the numbers

    slected_cases = [0, 8, 9, 11]
    for k in range(0, 4):
        i = slected_cases[k]
        # degeneracy of the whole space
        n_fold = binom(nsites[i], n_up[i])
        # print(n_fold)
        dos_Fock = np.array([], dtype=np.double)
        energy_Fock = np.array([], dtype=np.double)

        # calculate the density of states and Fock energy of the Fock states
        for j in range(0, min(n_up[i], nsites[i] - n_up[i]) + 1):
            dos_Fock = np.append(
                dos_Fock, binom(n_up[i], j) * binom(nsites[i] - n_up[i], j) / n_fold
            )
            energy_Fock = np.append(
                energy_Fock,
                ((n_up[i] * Wnew[i] - 2.0 * j * Wnew[i]) * 2.0 * PI / 1000.0 - Emin[i])
                / (Emax[i] - Emin[i]),
            )

        # print("The sum of the dos: ", np.sum(dos_Fock))
        # print("dos of Fock states: \n", dos_Fock)
        # print("energy density of Fock states:\n", energy_Fock)
        i_row = (k + ncols) // ncols
        i_col = (k + ncols) % ncols

        # plotting
        ax[i_row, i_col].bar(energy_Fock, dos_Fock, width=0.05, alpha=0.7)
        ax[i_row, i_col].set_yscale("log")
        ax[i_row, i_col].tick_params(which="minor", length=0)

        # mark the initial energy and the variance
        fill_left = max((Eave[i] - Emin[i] - Evar[i]) / (Emax[i] - Emin[i]), 0.0)
        fill_right = (Eave[i] - Emin[i] + Evar[i]) / (Emax[i] - Emin[i])
        ax[i_row, i_col].axvline(
            x=(Eave[i] - Emin[i]) / (Emax[i] - Emin[i]),
            color="k",
            ls="--",
            label=r"$\frac{E_{\mathbf{s}}(t=0) - E_{\rm min}}{E_{\rm max} - E_{\rm min}}$",
            lw=LW,
        )

        ax[i_row, i_col].fill_between(
            [fill_left, fill_right],
            0,
            1,
            color="tab:orange",
            alpha=0.2,
            label=r"$\frac{\Delta E_{\mathbf{s}}}{E_{\rm max} - E_{\rm min}}$",
        )

        ax[i_row, i_col].set_title(
            r"$W~/~2\pi=%.2f$ MHz, $%s$, $n=%d$"
            % (Wnew[i], nsites_str_latex[i], n_up[i]),
            fontsize=AX_LABEL_SIZE,
        )

    ax[1, 1].legend(
        labelspacing=0.8,
        handletextpad=0.3,
        framealpha=0.2,
        prop={"size": LEGEND_SIZE - 1},
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1, 0.2, 0.2, 0.6),
    )

    axes = ax[1:].flatten()
    ag.set_xlabel(
        r"$(E_{\mathbf{s}} - E_{\rm min})~/~(E_{\rm max} - E_{\rm min})$",
        axes=axes,
        fontsize=AX_LABEL_SIZE,
        labelpad=0,
    )
    ag.set_xticks([0, 0.5, 1], xlim=[0, 1], axes=axes)
    ag.set_ylabel("State density", axes=axes, fontsize=AX_LABEL_SIZE, labelpad=0)

    for k, ax in enumerate(axes):
        ax.set_ylim(1e-8, 1)
        ax.set_yticks([1e-8, 1e-4, 1])
        if k in [1, 3]:
            ax.tick_params(labelleft=False)
