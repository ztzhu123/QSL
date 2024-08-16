from collections import Counter, defaultdict
from copy import deepcopy
from functools import partial
from itertools import combinations
from pathlib import Path
import pickle
import re

import h5py
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


def plot_bound_mask(
    ts,
    bound_mask,
    mask=None,
    plot_cbar=True,
    alpha=0.2,
    alphas=None,
    ax=None,
    cax=None,
    cmap=None,
    clabel=None,
    cbar_ylim=None,
    vmin=0,
    vmax=2,
    fill_value=None,
    orientation="vertical",
    **kwargs,
):
    if mask is not None:
        bound_mask = np.ma.masked_where(mask, bound_mask)
    if bound_mask.ndim == 1:
        bound_mask = bound_mask[None, :]
    if ax is None:
        ax = subplot([6.5, 4])

    labels = ax.get_legend_handles_labels()[1]
    label = kwargs.pop("label", None)

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = sns.color_palette(cmap, as_cmap=1)
    else:
        cmap = sns.color_palette("Reds", as_cmap=1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if plot_cbar:
        cbar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            cax=cax,
            orientation=orientation,
        )
        if cbar_ylim is None:
            cbar_ylim = [vmin, vmax]
        cbar.set_ticks([0, 1, 2])
        cbar.ax.tick_params(axis="both", which="major", length=2, pad=1)
        if orientation == "vertical":
            cbar.ax.set_ylim(*cbar_ylim)
            if clabel is not None:
                cbar.set_label(clabel, fontsize=AX_LABEL_SIZE)
        else:
            cbar.ax.set_xlim(*cbar_ylim)
            cbar.ax.xaxis.set_ticks_position("top")
            if clabel is not None:
                cbar.ax.set_ylabel(clabel, fontsize=AX_LABEL_SIZE, rotation=0)
                # cbar.ax.yaxis.set_label_coords(-0.9, -0.2)
                cbar.ax.yaxis.set_label_coords(-0.85, -0.2)

    if label in labels:
        label = None
    for i in range(len(bound_mask)):
        mask = bound_mask[i].mask
        if np.all(mask):
            continue
        if fill_value is None:
            t = ts[~mask]
            bound = bound_mask[i].data[~mask]
        else:
            t = ts
            bound = bound_mask[i].data.copy()
            bound[mask] = fill_value
        ax = plot1d(
            t,
            bound,
            ax=ax,
            alpha=alpha,
            # label=label,
            label=None,
            color=cmap(norm(alphas[i])),
            **kwargs,
        )
    return ax


def plot_phase_2d(ax, phase_data):
    plot_line_color = [
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
        color=plot_line_color[0],
        lw=0,
        alpha=0.8,
    )
    ax.fill_between(
        list(x[0 : (pos + 1)]) + list(x[pos:]),
        list(x[0 : (pos + 1)]) + list(1.0 - x[pos:]),
        0,
        color=plot_line_color[1],
        alpha=0.7,
        lw=0,
    )
    ax.fill_between(
        x[pos:], y[pos:], 1.0 - x[pos:], color=plot_line_color[2], lw=0, alpha=0.8
    )

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

    for i, W in enumerate([0, 3.4, 6.5]):
        delta_E, E_minus_Emin, Emax_minus_E, Emax_minus_Emin = phase_data[f"W={W}"]
        ax.scatter(
            E_minus_Emin / Emax_minus_Emin,
            delta_E / Emax_minus_Emin,
            marker=markers[i],
            color=marker_colors[i],
            s=40,
            clip_on=False,
            zorder=np.inf,
            label=f"{W} MHz",
        )

    # set_legend(ax, 1.295)

    ax.legend(
        labelspacing=0.0,
        framealpha=0.4,
        prop={"size": LEGEND_SIZE + 0},
        handletextpad=0.1,
        ncol=3,
        bbox_to_anchor=(0.07, 1, 0.85, 0.4),
        loc="center",
        columnspacing=0.6,
    )


def set_legend(ax, dx=0):
    ax.legend(
        labelspacing=0.8,
        handletextpad=0.3,
        framealpha=0.2,
        prop={"size": LEGEND_SIZE - 1},
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1 + dx, 0.2, 0.2, 0.6),
    )


def plot_phase_1d(ax, phase_data):
    xs = phase_data["Ws"]
    keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es"]
    labels = [
        r"$\Delta E~/~2\pi$",
        r"$(E-E_{\rm min})~/~2\pi$",
        r"$(E_{\rm max}-E)~/~2\pi$",
    ]
    plot1d(
        xs,
        np.array([phase_data[key] for key in keys]),
        ax=ax,
        lw=LW,
        label=labels,
        legend_size=LEGEND_SIZE - 1,
        color=["#dd5746", "#76885b", "#535c91"],
    )
    ax.scatter(
        0,
        phase_data["W=0"][0],
        marker=markers[0],
        color=marker_colors[0],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )
    ax.scatter(
        3.4,
        phase_data["W=3.4"][0],
        marker=markers[1],
        color=marker_colors[1],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )
    ax.scatter(
        6.5,
        phase_data["W=6.5"][2],
        marker=markers[2],
        color=marker_colors[2],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )


def plot_hamming(ax=None, W=0):
    for key in ["font.size", "xtick.labelsize", "ytick.labelsize", "axes.titlesize"]:
        mpl.rcParams[key] = 14

    result = collect_fock_result()

    ts = result[f"W={W}"]["ts"]
    probs = result[f"W={W}"]["probs"]
    spec2d = result[f"W={W}"]["spec2d"]
    alphas = result[f"W={W}"]["alphas"]
    eigvals = result[f"W={W}"]["eigvals"]
    E_init_state = result[f"W={W}"]["E_init_state"]
    delta_E = result[f"W={W}"]["delta_E"]
    Emin = eigvals.min()
    Emax = eigvals.max()
    alpha_to_hamming = lambda alpha: 5 - ((Emax - Emin) * alpha + Emin) / (
        W * 2 * np.pi / 1e3
    )
    E_to_hamming = lambda E: alpha_to_hamming((E - Emin) / (Emax - Emin))

    mask = ts <= 100
    ts = ts[mask]
    probs = probs[:, mask]
    spec2d = spec2d[:, mask]
    basis = get_basis(9, 5)  # (D, num_q)
    indexes = bs_to_decimal(basis.T)
    state0 = np.zeros((1, 9), dtype=int)
    state0[0, ::2] = 1
    hamming = np.count_nonzero(basis - state0, 1)

    h_probs = []
    all_alphas = []
    h_unique = np.unique(hamming)
    for h in h_unique:
        indexes2 = np.where(hamming == h)[0]
        h_probs.append(probs[indexes[indexes2]].sum(0))
        all_alphas.append(alphas[indexes[indexes2]])
    h_probs = np.array(h_probs)

    ys = np.unique(hamming)
    ax = plot2d(
        h_probs,
        interp="none",
        vmin=0,
        vmax=1,
        xlabel="t (ns)",
        ylabel="hamming distance",
        xlim=[ts[0], ts[-1]],
        ylim=ys,
        title=f"[exp] W={W}",
        clabel="probability",
        ax=ax,
    )
    if W != 0:
        ax.fill_between(
            ax.get_xlim(),
            E_to_hamming(E_init_state - delta_E),
            E_to_hamming(E_init_state + delta_E),
            color=(0.5, 0.5, 0.5, 0.4),
            edgecolor=(0, 0, 0, 0.4),
            linewidth=2,
        )
    else:
        ax.fill_between(
            ax.get_xlim(),
            -1,
            9,
            color=(0.5, 0.5, 0.5, 0.4),
            edgecolor=(0, 0, 0, 0.4),
            linewidth=2,
        )
    ax.set_ylim(-1, 9)


def plot_alpha(W=0, plot=True, ax=None):
    for key in ["font.size", "xtick.labelsize", "ytick.labelsize", "axes.titlesize"]:
        mpl.rcParams[key] = 14

    result = collect_fock_result()

    ts = result[f"W={W}"]["ts"]
    probs = result[f"W={W}"]["probs"]
    spec2d = result[f"W={W}"]["spec2d"]
    alphas = result[f"W={W}"]["alphas"]
    print(np.sort(alphas)[-1] - np.sort(alphas)[-3])
    eigvals = result[f"W={W}"]["eigvals"]
    E_init_state = result[f"W={W}"]["E_init_state"]
    delta_E = result[f"W={W}"]["delta_E"]
    Emin = eigvals.min()
    Emax = eigvals.max()
    E_to_alpha = lambda E: (E - Emin) / (Emax - Emin)
    print("Emax-E:", Emax - E_init_state)
    print("Emax-Emin:", Emax - Emin)

    mask = ts <= 100
    ts = ts[mask]
    probs = probs[:, mask]
    spec2d = spec2d[:, mask]
    basis = get_basis(9, 5)  # (D, num_q)
    indexes = bs_to_decimal(basis.T)
    state0 = np.zeros((1, 9), dtype=int)
    state0[0, ::2] = 1
    hamming = np.count_nonzero(basis - state0, 1)

    h_probs = []
    all_alphas = []
    h_unique = np.unique(hamming)
    for h in h_unique:
        indexes2 = np.where(hamming == h)[0]
        h_probs.append(probs[indexes[indexes2]].sum(0))
        all_alphas.append(alphas[indexes[indexes2]])
    h_probs = np.array(h_probs)

    if W != 0:
        mask = np.ones_like(spec2d, dtype=np.bool8)
        mask[-1 : -9 - 1 : -2, :] = 0
        spec2d = np.ma.masked_where(mask, spec2d)
    if plot:
        # if W == 0:
        if W == -1:
            ax = subplot()
            y = np.unique(alphas)
            X, Y = np.meshgrid(ts, y)
            ax.pcolormesh(X, Y, spec2d, cmap="Spectral_r", linewidth=0, rasterized=True)
            ax.set_ylim(0, 1)
            ax.set_xlabel("t (ns)")
            ax.set_ylabel("$\\alpha$")

        else:
            log = True
            if log:
                vmin = 0.001
            else:
                vmin = 0
            ax = plot2d(
                spec2d,
                interp="none",
                vmin=vmin,
                vmax=1,
                xlabel="t (ns)",
                ylabel="$\\alpha$",
                xlim=[ts[0], ts[-1]],
                ylim=alphas,
                title=f"[exp] W={W}",
                clabel="probability",
                log=log,
            )
            ax.set_ylim([0, 1])
        # if W != 0:
        ax.fill_between(
            ax.get_xlim(),
            E_to_alpha(E_init_state - delta_E),
            E_to_alpha(E_init_state + delta_E),
            color=(0.5, 0.5, 0.5, 0.4),
            edgecolor=(0, 0, 0, 0.4),
            linewidth=2,
        )


def plot_alpha_1d(W=0, plot=True, ax=None):
    for key in ["font.size", "xtick.labelsize", "ytick.labelsize", "axes.titlesize"]:
        mpl.rcParams[key] = 14

    result = collect_fock_result()

    ts = result[f"W={W}"]["ts"]
    probs = result[f"W={W}"]["probs"]
    spec2d = result[f"W={W}"]["spec2d"]
    alphas = result[f"W={W}"]["alphas"]
    print(np.sort(alphas)[-1] - np.sort(alphas)[-3])
    eigvals = result[f"W={W}"]["eigvals"]
    E_init_state = result[f"W={W}"]["E_init_state"]
    delta_E = result[f"W={W}"]["delta_E"]
    Emin = eigvals.min()
    Emax = eigvals.max()
    E_to_alpha = lambda E: (E - Emin) / (Emax - Emin)
    print("Emax-E:", Emax - E_init_state)
    print("Emax-Emin:", Emax - Emin)

    last_probs = probs[:, -1]

    alpha_N_map = Counter()
    for alpha in alphas:
        alpha_N_map[alpha] += 1
    xlabels = np.round(list(alpha_N_map.keys()), 3)
    indexes = np.argsort(xlabels)
    alphas_plot = np.array(list(alpha_N_map.values()))[indexes]
    xlabels = xlabels[indexes]

    ax = subplot()
    ax.bar(
        xlabels,
        alphas_plot,
        width=0.05,
        # x_tick_labels=xlabels,
        # xlabel="$\\alpha$",
        # ylabel="count",
        # title=f"W={W}",
    )
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("state count")
    ax.set_title(f"W={W}")
    ax.vlines((E_init_state - Emin) / (Emax - Emin), *ax.get_ylim(), ls="--", color="k")
    ax.vlines(
        (E_init_state - delta_E - Emin) / (Emax - Emin),
        *ax.get_ylim(),
        ls="--",
        color="tab:orange",
    )
    ax.vlines(
        (E_init_state + delta_E - Emin) / (Emax - Emin),
        *ax.get_ylim(),
        ls="--",
        color="tab:orange",
    )

    basis = get_basis(9, 5)  # (D, num_q)
    photon_indexes = bs_to_decimal(basis.T)

    alpha_prob_map = Counter()
    for i, alpha in enumerate(alphas):
        alpha_prob_map[alpha] += last_probs[i] * 512
    x = np.array(list(alpha_prob_map.keys()))
    y = np.array(list(alpha_prob_map.values()))
    indexes = np.argsort(x)

    plot1d(x, y, ax=ax, marker="o", color="tab:green", ls="", xlim=[0, 1])

    return

@article_style()
def plot_fig4(save=False, W=0):
    result = collect_fock_result()

    fig = fig_in_a4(1, 0.55, dpi=200)
    ag = AxesGroup(12, 3, figs=fig, mark_index=0)

    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.3,
        top=0.96,
        bottom=0.06,
        left=0.064,
        right=0.9,
    )

    # plot bounds
    for i, case_name in enumerate(result):
        if not case_name.startswith("W"):
            continue
        W = float(case_name.split("=")[-1])
        if np.isclose(W, 0):
            W = 0
        ax = ag.axes[i + 3]
        data = result[case_name]
        # ---- evolution ----
        t_max = 100
        ts = data["ts"]
        t_exp_mask = ts <= t_max
        ts = ts[t_exp_mask]
        ts_sim = data["ts_sim"]
        t_sim_mask = ts_sim <= t_max
        ts_sim = ts_sim[t_sim_mask]
        fids_ideal = data["fids_sim"][t_sim_mask]

        ts_bound = ts_sim

        fids = data["fids"][:, t_exp_mask]

        overlaps = np.sqrt(fids)

        plot1d(
            ts,
            overlaps.mean(0),
            overlaps.std(0, ddof=1),
            ax=ax,
            lw=LW,
            ms=MS,
            mew=MEW,
            capsize=CAPSIZE,
            # title=r"$W/2\pi$ = %.1f MHz" % (W),
            color=colors["evo"],
            label="Exp.",
            ls="",
            zorder=100,
            clip_on=False,
        )
        plot1d(
            ts_sim,
            np.sqrt(fids_ideal),
            lw=LW,
            color=colors["sim"],
            ax=ax,
            alpha=sim_alpha,
            label="Sim.",
        )
        # # ---- bound ----
        bound_names = ["ML", "ML_star", "MT"]
        for j, bound_name in enumerate(bound_names):
            bound = data[f"{bound_name}_bound"][t_sim_mask]
            mask = data[f"{bound_name}_bound_mask"][t_sim_mask]
            if "star" in bound_name:
                label = "ML*"
            else:
                label = bound_name
            plot1d(
                ts_bound[~mask],
                bound[~mask],
                ls="--",
                lw=LW,
                color=colors[bound_name],
                label=label,
                ax=ax,
                xlim=None,
                ylim=None,
                dashes=DASHES,
            )
        o = round(W, 3)
        if float(o).is_integer():
            o = int(o)
        ax.text(
            0.5,
            0.97,
            rf"$W~/~2\pi=${o} MHz",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
        )

        cmap_upper = sns.light_palette("#92ba92", as_cmap=True)
        cmap_lower = sns.light_palette("#ffa41b", as_cmap=True)

        if i == 0:
            height = 0.0066
            interval = 0.1
            pos = ag.axes[3].get_position()
            left = pos.xmin + 0.06
            right = pos.xmax
            bottom = pos.ymax + 0.009
            width = 0.045

            cax_lower = fig.add_axes([left, bottom, width, height])
            cax_upper = fig.add_axes([right - width, bottom, width, height])
        else:
            cax_upper = cax_lower = None
        kw = {
            "plot_cbar": i == 0,
            "vmin": 0,
            "vmax": 4,  # deliberately different from cbar_ylim
            "cbar_ylim": [0, 2],
            "cmap_lower": cmap_lower,
            "cmap_upper": cmap_upper,
            "cax_upper": cax_upper,
            "cax_lower": cax_lower,
            "orientation": "horizontal",
        }
        plot_fock_gen_bound(W, ax=ax, cbar_kwargs=kw)
        if i == 2:
            handles, labels = ax.get_legend_handles_labels()
            indexes = np.arange(len(handles) - 1).tolist()
            indexes.insert(0, -1)
            indexes[0], indexes[1] = indexes[1], indexes[0]
            handles = [handles[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            set_legend(ax)
        else:
            ax.get_legend().remove()

    ag.tick_params()
    ag.set_xticks(
        np.arange(0, 101, 50),
        xlim=[0, 100],
        fontsize=AX_LABEL_SIZE,
        axes="row1",
        sharex=0,
    )
    ag.set_xlabel(
        "Evolution time (ns)",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        labelpad=1,
        sharex=0,
        axes="row1",
    )
    ag.set_yticks([0, 0.5, 1], ylim=[0, 1], fontsize=AX_LABEL_SIZE, axes="row1")
    ag.set_ylabel(
        r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$",
        fontsize=AX_LABEL_SIZE,
        usetex=USE_TEX,
        axes="row1",
        labelpad=3,
        sharey=1,
        clean_others=1,
    )

    # plot lattice
    ag.axes[0].axis("off")
    # plot phase 2d
    i = 1
    plot_phase_2d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([0, 1], xlim=[0, 1], axes=i, sharex=0)
    ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=i, sharey=0)

    # plot phase 1d
    i = 2
    keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es", "Emax_minus_Emin"]
    Ws = result["phase_1d"]["Ws"]
    plot_phase_1d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([0, 8], xlim=[Ws[0], Ws[-1]], axes=i, sharex=0)
    ag.set_yticks([0, 40, 80], ylim=[0, 80], axes=i, sharey=0)
    ag.set_xlabel(
        r"$W~/~2\pi$ (MHz)",
        sharex=False,
        axes=i,
        fontsize=AX_LABEL_SIZE,
        labelpad=-2,
    )
    ag.set_ylabel(
        r"Energy / $2\pi$ (MHz)",
        sharey=False,
        axes=i,
        fontsize=AX_LABEL_SIZE,
        # labelpad=-1,
    )
    # ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=1, sharey=0)

    # plot evolution
    plot_evo(ag)

    # plot state density
    plot_state_density(ag)

    annot_alphabet(
        ag.axes,
        fontsize=FONTSIZE,
        dx=-0.03,
        dy=0.022,
        transform="fig",
        top_bond_dict={0: 2, 1: 2},
        left_bond_dict={0: 3, 1: 4},
    )

    ag.grid(False)
    ag.tick_params(direction="out")

    if save:
        fig_name = MAIN_FIG_DIR / "fig4.pdf"
        fig.savefig(fig_name, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


def plot_evo(ag: AxesGroup):
    result = collect_fock_result()
    axes = [9, 10, 11]
    for i, W in enumerate([0, 3.4, 6.5]):
        ax = ag.axes[axes[i]]
        ts = result[f"W={W}"]["ts"]
        probs = result[f"W={W}"]["probs"].mean(0)
        spec2d = result[f"W={W}"]["spec2d"]
        alphas = result[f"W={W}"]["alphas"]
        eigvals = result[f"W={W}"]["eigvals"]

        mask = ts <= 100
        ts = ts[mask]
        probs = probs[:, mask]
        unique_alphas = np.unique(alphas)

        spec2d = spec2d[:, mask]

        basis = get_basis(9, 5)  # (D, num_q)
        state0 = np.zeros((1, 9), dtype=int)
        state0[0, ::2] = 1
        hamming = np.count_nonzero(basis - state0, 1)

        h_probs = []
        h_unique = np.unique(hamming)
        for h in h_unique:
            indexes2 = np.where(hamming == h)[0]
            h_probs.append(probs[indexes2].sum(0))
        h_probs = np.array(h_probs)

        alpha_probs = []
        for a in unique_alphas:
            indexes2 = np.where(alphas == a)[0]
            alpha_probs.append(probs[indexes2].sum(0))
        alpha_probs = np.array(alpha_probs)

        # cmap = sns.light_palette("#0e46a3", as_cmap=True)
        cmap = sns.color_palette("Blues", as_cmap=True)
        ax = plot2d(
            h_probs,
            interp="none",
            vmin=0,
            vmax=1,
            cmap=cmap,
            xlim=[ts[0], ts[-1]],
            ylim=h_unique,
            ax=ax,
            plot_cbar=False,
            constrained_layout=False,
        )

        ag.set_xticks([0, 50, 100], axes=axes, sharex=0)
        ag.set_yticks([0, 4, h_unique[-1]], axes=axes)
        ag.set_xlabel(
            "Evolution time (ns)",
            axes=axes,
            fontsize=AX_LABEL_SIZE,
            clean_others=1,
            sharex=0,
        )
        ag.set_ylabel(r"$d$", axes=axes, fontsize=AX_LABEL_SIZE)

        pos = ag.axes[-1].get_position()
        height = (pos.ymax - pos.ymin) * 0.7
        left = pos.xmax + 0.02
        bottom = (pos.ymax - pos.ymin - height) / 2 + pos.ymin
        width = 0.01

        if i == 2:
            cax = ag.figs[0].add_axes([left, bottom, width, height])
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                cax=cax,
            )
            cbar.set_ticks([0, 1])
            # cbar.ax.tick_params(axis="both", which="major", length=2, pad=1)
            cbar.set_label("$\pi(d)$", fontsize=AX_LABEL_SIZE)


def plot_state_density(ag):
    axes = [6, 7, 8]
    my_result = collect_fock_result()
    for i, case_name in enumerate(["3x3"]):
        basis = np.load(DATA_DIR / f"{case_name}_basis.npy")
        for j, W in enumerate([0, 3.4, 6.5]):
            Ws = np.ones(basis.shape[1]) * W
            Ws[1::2] = -Ws[1::2]

            energies = np.sum(basis * Ws, 1)

            # name = f"{case_name}_{W}"
            name = f"W={W}"
            ax = ag.axes[axes[j]]
            Emin = my_result[name]["Emin"]
            Emax = my_result[name]["Emax"]
            E = my_result[name]["E"]
            delta_E = my_result[name]["delta_E"]
            E_to_alpha = lambda E: (E - Emin) / (Emax - Emin)
            alphas = E_to_alpha(energies * 2 * np.pi * 1e-3)
            x, y = np.unique(alphas.round(11), return_counts=True)
            ax.bar(x, y / len(energies), width=0.07, alpha=0.7)
            ax.vlines(
                E_to_alpha(E),
                0,
                1,
                color="k",
                ls="--",
                label=r"$\frac{E_{\mathbf{s}}(t=0) - E_{\rm min}}{E_{\rm max} - E_{\rm min}}$"
                if i == 0 and j == 2
                else None,
                lw=LW,
            )
            a = max(E_to_alpha(E - delta_E), 0)
            b = min(E_to_alpha(E + delta_E), 1)
            ax.fill_between(
                [a, b],
                0,
                1,
                color="tab:orange",
                alpha=0.2,
                label=r"$\frac{\Delta E_{\mathbf{s}}}{E_{\rm max} - E_{\rm min}}$"
                if i == 0 and j == 2
                else None,
            )
            ax.set_yscale("log")
            ax.tick_params(which="minor", length=0)
            if i == 0 and j == 2:
                set_legend(ax)

    ag.set_xlabel(
        r"$(E_{\mathbf{s}} - E_{\rm min})~/~(E_{\rm max} - E_{\rm min})$",
        axes=axes,
        fontsize=AX_LABEL_SIZE,
        labelpad=0,
    )
    ag.set_xticks([0, 0.5, 1], xlim=[0, 1], axes=axes)
    ag.set_ylabel("state density", axes=axes, fontsize=AX_LABEL_SIZE, labelpad=0)
    for k, i in enumerate(axes):
        ag.axes[i].set_ylim(1e-3, 1)
        ag.axes[i].set_yticks([1e-4, 1e-2, 1])
        if k > 0:
            ag.axes[i].tick_params(labelleft=False)


def collect_fock_result():
    Ws = [0, 3.4, 6.5]
    result = defaultdict(dict)

    for W in Ws:
        case_name = f"W={W}"
        filename = DATA_DIR / "fig4.h5"
        with h5py.File(filename, "r") as f:
            for key in f[case_name].keys():
                result[case_name][key] = f[case_name][key][()]

        with h5py.File(DATA_DIR / "fig4.h5", "r") as f:
            result[case_name]["probs"] = f[case_name]["probs"][()].mean(0)
        with h5py.File(DATA_DIR / "fig4.h5", "r") as f:
            alphas = f[case_name]["alphas"][()]
            ts = result[case_name]["ts"]
            probs = f[case_name]["probs"][()]
            result[case_name]["probs"] = probs
            result[case_name]["alphas"] = alphas
            unique_alphas = f[case_name]["unique_alphas"][()]
            result[case_name]["eigvals"] = f[case_name]["eigvals"][()]
            result[case_name]["delta_E"] = f[case_name]["delta_E"][()]
            result[case_name]["E_init_state"] = f[case_name]["E_init_state"][()]
            result[case_name]["unique_alphas"] = unique_alphas
            spec2d = np.zeros((len(unique_alphas), len(ts)))
            for i in range(len(ts)):
                p = probs[:, :, i].mean(0)
                for j, alpha in enumerate(unique_alphas):
                    indexes = np.where(alphas == alpha)[0]
                    spec2d[j, i] = p[indexes].sum()
            result[case_name]["spec2d"] = spec2d

    keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es", "Emax_minus_Emin"]
    case_name = "phase_1d"
    with h5py.File(DATA_DIR / "fig4.h5", "r") as f:
        Ws = f[case_name]["Ws"][()]
        result["phase_1d"]["Ws"] = Ws
        for key in keys:
            result["phase_1d"][key] = f[case_name][key][()]
        for W in [0, 3.4, 6.5]:
            index = np.where(W == Ws)[0].item()
            result["phase_1d"][f"W={W}"] = [
                f[case_name][key][()][index] for key in keys
            ]

    return dict(result)


@article_style()
def plot_fock_gen_bound(W, ax=None, colors=None, cbar_kwargs=None, **kwargs):
    filename = DATA_DIR / "fig4_fock_3x3_gen_bound.h5"
    if np.isclose(W, 0):
        W = int(W)
    with h5py.File(filename, "r") as f:
        group_name = f"W={W}"
        ts = f[group_name]["ts"][()]
        bounds = f[group_name]["bounds"][()]
        mask = f[group_name]["bounds_mask"][()]
        bounds = np.ma.masked_where(mask, bounds)
        alphas = f[group_name]["alphas"][()]
        assert f[group_name]["W"][()] == W

    if colors is None:
        colors = ["tab:blue", "tab:orange"]
        # colors = ["#d2e3f0", "#ffd8b6"]
    cbar_kwargs = cbar_kwargs or {}
    alpha = 1
    lower_bound = bounds[:, 0]
    upper_bound = bounds[:, 1]

    cmap_lower = cbar_kwargs.pop("cmap_lower", None)
    cmap_upper = cbar_kwargs.pop("cmap_upper", None)
    cax_lower = cbar_kwargs.pop("cax_lower", None)
    cax_upper = cbar_kwargs.pop("cax_upper", None)

    all_indexes = np.arange(len(alphas))
    indexes = all_indexes
    indexes = np.sort(np.unique(indexes))
    ax = plot_bound_mask(
        ts,
        lower_bound[indexes],
        ax=ax,
        cax=cax_lower,
        lw=LW,
        alphas=alphas[indexes],
        clabel=r"$\alpha$ (lower)",
        alpha=alpha,
        cmap=cmap_lower,
        zorder=-np.inf,
        fill_value=0,
        **cbar_kwargs,
        **kwargs,
    )
    ax = plot_bound_mask(
        ts,
        upper_bound[indexes],
        ax=ax,
        cax=cax_upper,
        xlabel="Evolution time (ns)",
        ylabel=r"$\sqrt{\mathcal{F}}$",
        xlim=[ts.min(), ts.max()],
        ylim=[0, 1],
        lw=LW,
        alphas=alphas[indexes],
        clabel=r"$\alpha$ (upper)",
        alpha=alpha,
        cmap=cmap_upper,
        zorder=-100,
        fill_value=1,
        **cbar_kwargs,
        **kwargs,
    )

    return ax


def get_positions(num_q, num_photons):
    D = comb(num_q, num_photons, True)

    positions = list(combinations(np.arange(num_q), num_photons))
    positions = positions[::-1]
    assert D == len(positions)
    return positions


def get_basis(num_q, num_photons):
    D = comb(num_q, num_photons, True)
    positions = get_positions(num_q, num_photons)

    assert D == len(positions)
    basis = np.zeros((D, num_q), dtype=int)
    for i, p in enumerate(positions):
        basis[i, list(p)] = 1
    return basis


def bs_to_decimal(bitstrings):
    bitstrings = np.asarray(bitstrings, dtype=np.uint8)
    num_q = bitstrings.shape[0]
    bin_coeffs = 1 << np.arange(num_q - 1, -1, -1, dtype=np.uint64)
    decimal_states = np.tensordot(bin_coeffs, bitstrings, axes=[0, 0])
    return decimal_states
