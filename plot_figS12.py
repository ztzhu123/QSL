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


@article_style()
def plot_all():
    nrows = 3
    ncols = 1
    fig = fig_in_a4(1, 0.6, dpi=100)
    ag = AxesGroup(nrows * ncols, ncols, figs=fig)
    axs = ag.axes

    fig.subplots_adjust(
        hspace=0.2,
        top=0.95,
        bottom=0.06,
        left=0.25,
        right=0.75,
    )

    colors = ["dodgerblue", "crimson", "forestgreen"]
    linestyles = ["-", ":", "--"]

    ########################################################
    ###### Parameters ######################################
    ########################################################
    list_of_Ls = [6, 12, 24, 48, 96]
    J = -2.0
    W = 0.0
    init_state_type = "density_wave"

    Blues = plt.cm.Blues(np.linspace(0, 1, len(list_of_Ls) + 2, endpoint=True))
    Reds = plt.cm.Reds(np.linspace(0, 1, len(list_of_Ls) + 2))
    Greens = plt.cm.Greens(np.linspace(0, 1, len(list_of_Ls) + 2))

    # sm1 = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=plt.Normalize(vmin=0, vmax=len(list_of_Ls)))
    # cbaxes = fig.add_axes([0.5, 0.85, 0.35, 0.015])
    # cb1 = plt.colorbar(sm1, cax = cbaxes, ticks = 2+ np.array([0, 1, 2, 3]), orientation = 'horizontal')
    # cb1.ax.set_xticklabels([6, 12, 24, 48],  fontsize = txt_size)  # vertically oriented colorbar
    # cb1.set_label(r'$L$', fontsize = txt_size)

    # sm2 = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=len(list_of_Ls)+2))

    for i_L, L in enumerate(list_of_Ls):
        # Initialize and define Hamiltonian H under which our state evolves
        H = build_H(L, J, W)

        # Note, we want to handle a many-body state (rectangular matrices)
        init_state = build_init_state(L, init_state_type)

        # Get mean energy and variance
        E = get_mean_energy(H, init_state)  # Note, the energy spread is wrong here

        DeltaE = get_energy_uncertainty(L, E)


        eigenv_H, U = np.linalg.eigh(H)
        idx = np.argsort(eigenv_H)
        eigenv_H = eigenv_H[idx]
        U = U[:, idx]
        E0 = np.sum(eigenv_H[: L // 2])

        path = DATA_DIR / "scaling_1d"
        t_values, overlaps, Sx_t, Sy_t, Sz_t, n_k0_t = np.loadtxt(
            path / ("1d_L%d_J%.2fW%.2fQSL.dat" % (L, J, W)), unpack=True
        )
        t_values = np.loadtxt(
            path / (("1d_L%d_J%.2fW%.2fQSL_rho_related.dat") % (L, J, W)),
            unpack=True,
            usecols=0,
        )
        eigenv_rho_ij = np.loadtxt(
            path / ("1d_L%d_J%.2fW%.2fQSL_rho_related.dat" % (L, J, W)),
            unpack=True,
            usecols=range(1, L + 1),
        )
        eigenvec_largest_eigenv = np.loadtxt(
            path / ("1d_L%d_J%.2fW%.2fQSL_rho_related.dat" % (L, J, W)),
            unpack=True,
            usecols=range(L + 1, 2 * L + 1),
        )

        ### Now plotting ########################
        plot1d(t_values, overlaps, color=Blues[i_L + 2], ax=axs[0], label="$L=%d$" % L)
        plot1d(t_values, overlaps, color=Blues[i_L + 2], ax=axs[1], label="$L=%d$" % L)
        plot1d(t_values, n_k0_t, color=Greens[2 + i_L], ax=axs[2], label="$L=%d$" % L)

        ## ML bound
        ML_bound = np.cos(np.sqrt(np.pi * (E - E0) * t_values / 2.0))
        subsetter = np.where(t_values <= np.pi / (2 * (E - E0)))[0]
        # axs[0].plot(t_values[subsetter], ML_bound[subsetter], color = Reds[2+i_L], linestyle = ':', label = r'$\rm ML\ bound$')
        # axs[1].plot(t_values[subsetter], ML_bound[subsetter], color = Reds[2+i_L], linestyle = ':', label = r'$\rm ML\ bound$')

        ## MT bound
        MT_bound = np.cos(DeltaE * t_values)
        subsetter2 = np.where(t_values <= np.pi / (2 * DeltaE) + 0.004)[0]
        plot1d(
            t_values[subsetter2],
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            label="$L=%d$" % L,
            ax=axs[0],
        )
        plot1d(
            t_values[subsetter2],
            MT_bound[subsetter2],
            color=Reds[2 + i_L],
            linestyle=":",
            label="$L=%d$" % L,
            ax=axs[1],
        )

    ag.grid(False)
    ag.tick_params(direction="out")

    axs[0].set_title(
        r"$J~/~2\pi=%d\ {\rm MHz},\ W~/~2\pi = %d\ {\rm MHz}$" % (J, W),
        fontsize=AX_TITLE_SIZE,
        pad=10,
    )

    ag.set_xlabel(r"$t$ ($\mu$s)", fontsize=AX_LABEL_SIZE)
    ag.set_ylabel(
        r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$",
        fontsize=AX_LABEL_SIZE,
        axes=axs[:-1],
        usetex=True,
    )
    ag.set_ylabel(r"$n_{k=0}$", fontsize=AX_LABEL_SIZE, axes=axs[-1])

    ag.set_xticks(np.arange(0, 1.201, 0.2), xlim=[0, 1.2])
    ag.set_yticks(np.arange(0, 1.001, 0.5), ylim=[0, 1], sharey=0, axes=axs[0])
    ag.set_yticks(np.arange(0.4, 0.801, 0.2), ylim=[0.4, 0.8], sharey=0, axes=axs[-1])

    axs[1].set_yscale("log")
    ag.set_yticks([1e-32, 1e-16, 1], ylim=[1e-32, 1], sharey=0, axes=axs[1])
    for i in range(2):
        ax = axs[i]
        handles, labels = ax.get_legend_handles_labels()
        indexes = np.arange(len(handles)).tolist()
        indexes = np.hstack([indexes[::2], indexes[1::2]])
        handles = [handles[i] for i in indexes]
        labels = [labels[i] for i in indexes]
        if i == 0:
            kw = {"loc": "upper center"}
        else:
            kw = {"loc": "lower right", "bbox_to_anchor": (0.7, 0.0, 0.2, 0.5)}
        ax.legend(
            handles,
            labels,
            framealpha=0.2,
            prop={"size": LEGEND_SIZE},
            ncol=2,
            **kw,
        )
    ax = axs[-1]
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

    annot_alphabet(
        ag.axes,
        fontsize=FONTSIZE,
        dx=-0.07,
        dy=0.015,
        transform="fig",
    )


#############################################################################################
##### Here starts the fun ###################################################################
#############################################################################################
def build_init_state(L, init_state_type):
    """
    Build the many-body initial state
    It can be represented as Nsites x Nelectrons rectangular matrices
    """

    init_state = np.zeros([L, L // 2], dtype=np.float64)
    if init_state_type == "density_wave":
        for i in range(L // 2):
            init_state[2 * i, i] = 1.0

    elif init_state_type == "domain_wall":
        for i in range(L // 2):
            init_state[i, i] = 1.0

    else:
        raise ("Invalid initial state!")

    return init_state


def build_H(L, J, W):
    H = np.zeros((L, L), dtype=np.float64)

    ## hoppings ##
    for i in range(L):
        if i > 0:
            H[i, i - 1] = H[i - 1, i] = J

    ## staggered potential ##
    for i in range(L):
        H[i, i] = (-1) ** i * W

    return H


def creation(state, site):
    Ne = np.shape(state)[1]
    L = np.shape(state)[0]
    new_state = state.copy()

    ## String operator contribution
    for i in range(site):
        for j in range(Ne):
            new_state[i, j] = -new_state[i, j]

    ## Creation of a fermion at site

    # add a column
    new_state = np.c_[new_state, np.zeros(L)]
    # add the new particle
    new_state[site, Ne] = 1.0

    return new_state


def get_mean_energy(H, init_state):
    inner_prod = np.linalg.det((init_state.conj().T) @ init_state)

    L = np.shape(init_state)[0]

    # Neighbors table
    xplus, xminus = indexsetsq(L)

    ## Mean energy ##
    E = 0

    ## Hopping terms
    for i in range(L):
        j = xplus[i]
        if abs(i - j) != L - 1:  # OBC testing
            # Compute terms as <a_i+1 a_i⁺>   (already using the commutation relation for bosons)

            ai_plus = creation(init_state, i)
            a_ip1_dag = creation(init_state, j)

            E += H[i, j] * np.linalg.det((a_ip1_dag.conj().T) @ ai_plus)

        j = xminus[i]
        if abs(i - j) != L - 1:  # OBC testing
            # Compute terms as <a_i-1 a_i⁺>   (already using the commutation relation for bosons)
            ai_plus = creation(init_state, i)
            a_im1_dag = creation(init_state, j)

            E += H[i, j] * np.linalg.det((a_im1_dag.conj().T) @ ai_plus)

    ## Diagonal terms
    for i in range(L):
        ai_plus = creation(init_state, i)
        E += H[i, i] * (1.0 - np.linalg.det((ai_plus.conj().T) @ ai_plus))

    return E


def get_energy_uncertainty(L, E):
    H2 = 4 * L - 4

    return np.sqrt(H2 - E * E)


############## This function finds the neighbors ##########################################################\
def indexsetsq(L):
    """
    It is a bit of an overkill but nice either way
    """

    # xplus = np.full(L, np.nan, dtype=np.intc)
    # xminus = np.full(L, np.nan, dtype=np.intc)
    xplus = np.full(L, 0, dtype=np.intc)
    xminus = np.full(L, 0, dtype=np.intc)

    for i in range(L):
        xplus[i] = (i + 1 + L) % L
        xminus[i] = (i - 1 + L) % L

    return xplus, xminus
