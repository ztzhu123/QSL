from collections import defaultdict
from copy import deepcopy
from functools import partial
from itertools import combinations
from pathlib import Path
import pickle
import re

import h5py
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
import numpy as np
from qutip import *
import seaborn as sns

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, subplot
from path import DATA_DIR
from plot_fig4 import collect_fock_result
from plot_toolbox import plot1d as _plot1d
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
sim_alpha = 1

marker_colors = ["#6499e9", "#8644a2", "#f57d1f"]
markers = ["*", "^", "x"]
marker_size = 40


def collect_result():
    fock_result = collect_fock_result()  # 3x3
    result = {}
    phase_data = {}
    for k, W in enumerate([0, 3.4, 6.5]):
        # 3x3, 4x4, 5x5
        for i in range(2, 5):
            if i == 2:
                num_q = 9
                excitation = 5
                name = f"3x3_{W}"
            elif i == 3:
                num_q = 16
                excitation = 8
                name = f"4x4_{W}"
            elif i == 4:
                num_q = 25
                excitation = 13
                name = f"5x5_{W}"

            if num_q == 9:
                result[name] = {}
                result[name]["fids_sim"] = fock_result[f"W={W}"]["fids_sim"]
                result[name]["ts_sim"] = fock_result[f"W={W}"]["ts_sim"]
                Emin = float(fock_result[f"W={W}"]["Emin"])
                Emax = float(fock_result[f"W={W}"]["Emax"])
                E = float(fock_result[f"W={W}"]["E"])
                delta_E = float(fock_result[f"W={W}"]["delta_E"])
                for key in ["Emin", "Emax", "E", "delta_E"]:
                    result[name][key] = eval(key)
            elif num_q in [16, 25]:
                path = DATA_DIR / "large_sim"
                data = np.fromfile(
                    path / f"fidelity_{num_q}_{excitation}_Jn2.00C{W:.2f}_coeff13.bin"
                )
                result[name] = {}
                result[name]["fids_sim"] = data[1::2] ** 2
                result[name]["ts_sim"] = data[::2]
                with open(
                    path
                    / f"EnergyDetails_{num_q}_{excitation}_Jn2.00C{W:.2f}_coeff13.dat",
                    "r",
                ) as f:
                    Emin = float(f.readline()[6:-1])
                    Emax = float(f.readline()[6:-1])
                    E = float(f.readline()[8:-1])
                    delta_E = float(f.readline()[10:])
                    for key in ["Emin", "Emax", "E", "delta_E"]:
                        result[name][key] = eval(key)
            else:
                raise

            phase_data[name] = [delta_E, E - Emin, Emax - E, Emax - Emin]
            ts = result[name]["ts_sim"]
            filename = DATA_DIR / "large_sim" / "bounds.h5"
            with h5py.File(filename, "r") as f:
                MT_bound = f[name]["MT_bound"][()]
                MT_bound_mask = f[name]["MT_bound_mask"][()]
                MT_bound = np.ma.masked_where(MT_bound_mask, MT_bound)

                ML_bound = f[name]["ML_bound"][()]
                ML_bound_mask = f[name]["ML_bound_mask"][()]
                ML_bound = np.ma.masked_where(ML_bound_mask, ML_bound)

                ML_star_bound = f[name]["ML_star_bound"][()]
                ML_star_bound_mask = f[name]["ML_star_bound_mask"][()]
                ML_star_bound = np.ma.masked_where(ML_star_bound_mask, ML_star_bound)
            for key in ["MT_bound", "ML_bound", "ML_star_bound"]:
                result[name][key] = eval(key)

        result["phase_data"] = phase_data
    return result


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

    W_shape = ["o", "*", "x"]
    for i, case_name in enumerate(["3x3", "4x4", "5x5"]):
        for j, W in enumerate([0, 3.4, 6.5]):
            delta_E, E_minus_Emin, Emax_minus_Emin, Emax_minus_Emin = phase_data[
                f"{case_name}_{W}"
            ]
            ax.scatter(
                E_minus_Emin / Emax_minus_Emin,
                delta_E / Emax_minus_Emin,
                # marker=markers[i],
                # color=marker_colors[i],
                s=20,
                clip_on=False,
                zorder=np.inf,
                label=f"$%d\\times%d,~%s$" % (i + 3, i + 3, W),
                marker=W_shape[j],
            )
    handles, labels = ax.get_legend_handles_labels()
    indexes = np.arange(len(handles))
    indexes = indexes.reshape(3, 3).T.flatten()
    handles = [handles[i] for i in indexes]
    labels = [labels[i] for i in indexes]
    ax.legend(
        handles,
        labels,
        labelspacing=0.2,
        framealpha=0.4,
        prop={"size": LEGEND_SIZE - 1},
        handletextpad=0.2,
        ncol=3,
        bbox_to_anchor=(0.16, 1.15, 0.65, 0.45),
        loc="center",
        columnspacing=0.6,
    )
    annot_alphabet(
        [ax],
        fontsize=FONTSIZE,
        dx=-0.06,
        dy=0.06,
        transform="fig",
        offset=9,
    )


@article_style()
def plot(save=False):
    result = collect_result()

    fig = fig_in_a4(1, 0.40, dpi=150)
    ag = AxesGroup(9, 3, figs=fig, mark_index=0)

    fig.subplots_adjust(
        hspace=0.5,
        wspace=0.3,
        top=0.93,
        bottom=0.1,
        left=0.07,
        right=0.68,
    )

    pos = ag.axes[3].get_position()
    left = pos.xmin
    right = pos.xmax
    top = pos.ymax
    bottom = pos.ymin - 0.1
    width = right - left
    height = top - bottom
    ax_phase = fig.add_axes(
        [1 - 0.025 - width * 1.2, bottom, width * 1.2, height * 1.2]
    )

    plot_phase_2d(ax_phase, result["phase_data"])
    ax_phase.set_xlim([0, 1])
    ax_phase.set_ylim([0, 0.5])
    ax_phase.set_xticks([0, 1])
    ax_phase.set_yticks([0, 0.5])

    # plot bounds
    for i, case_name in enumerate(["3x3", "4x4", "5x5"]):
        Ws = [0, 3.4, 6.5]
        for k, W in enumerate(Ws):
            ax = ag.axes[3 * i + k]
            data = result[case_name + f"_{W}"]
            # ---- evolution ----
            ts_sim = data["ts_sim"]
            plot1d(
                ts_sim,
                np.sqrt(data["fids_sim"]),
                lw=LW,
                color=colors["sim"],
                ax=ax,
                alpha=sim_alpha,
                label="Sim.",
            )
            # # ---- bound ----
            bound_names = ["ML", "ML_star", "MT"]
            for j, bound_name in enumerate(bound_names):
                bound = data[f"{bound_name}_bound"]
                mask = bound.mask
                bound = bound.data
                if "star" in bound_name:
                    label = "ML*"
                else:
                    label = bound_name
                plot1d(
                    ts_sim[~mask],
                    bound[~mask],
                    ls="--",
                    lw=LW,
                    color=colors[bound_name],
                    label=label,
                    ax=ax,
                    xlim=None,
                    ylim=None,
                    dashes=DASHES,
                    title="%s, $W~/~2\pi=%s$ MHz" % (case_name, W),
                    title_size=8,
                )

            if i == 0 and k == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles,
                    labels,
                    labelspacing=0.5,
                    handletextpad=0.3,
                    framealpha=0.2,
                    prop={"size": LEGEND_SIZE - 1},
                    loc="upper right",
                    ncol=1,
                )
            else:
                ax.get_legend().remove()
            if i == 0:
                if k == 2:
                    width = 0.0066
                    interval = 0.05
                    pos = ag.axes[2].get_position()
                    left = pos.xmax + 0.02
                    right = left + width
                    bottom = pos.ymax + 0.009
                    height = (pos.ymax - pos.ymin - interval) / 2

                    cax_lower = fig.add_axes([left, pos.ymin, width, height])
                    cax_upper = fig.add_axes(
                        [left, pos.ymin + height + interval, width, height]
                    )
                else:
                    cax_upper = cax_lower = None
                cmap_upper = sns.light_palette("#92ba92", as_cmap=True)
                cmap_lower = sns.light_palette("#ffa41b", as_cmap=True)
                kw = {
                    "plot_cbar": k == 2,
                    "vmin": 0,
                    "vmax": 4,
                    "cbar_ylim": [0, 2],
                    "cmap_lower": cmap_lower,
                    "cmap_upper": cmap_upper,
                    "cax_upper": cax_upper,
                    "cax_lower": cax_lower,
                    # "orientation": "horizontal",
                }
                plot_fock_gen_bound(W, ax=ax, cbar_kwargs=kw)
            if i == 1:
                cax_upper = cax_lower = None
                cmap_upper = sns.light_palette("#92ba92", as_cmap=True)
                cmap_lower = sns.light_palette("#ffa41b", as_cmap=True)
                kw = {
                    "plot_cbar": False,
                    "vmin": 0,
                    "vmax": 4, 
                    "cbar_ylim": [0, 2],
                    "cmap_lower": cmap_lower,
                    "cmap_upper": cmap_upper,
                    "cax_upper": cax_upper,
                    "cax_lower": cax_lower,
                    # "orientation": "horizontal",
                }
                plot_large_gen_bound(W, ax=ax, cbar_kwargs=kw)

    ag.set_xticks([0, 50, 100], xlim=[0, 100], fontsize=AX_LABEL_SIZE)
    ag.set_xlabel(
        "Evolution time (ns)",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        labelpad=0,
    )
    ag.set_yticks([0, 0.5, 1], ylim=[0, 1], fontsize=AX_LABEL_SIZE)
    ylabel = r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$"
    ag.set_ylabel(
        ylabel,
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        usetex=USE_TEX,
    )

    # general
    ag.tick_params(direction="out")
    ag.grid(False)
    annot_alphabet(
        ag.axes,
        fontsize=FONTSIZE,
        dx=-0.03,
        dy=0.02,
        transform="fig",
        offset=0,
    )


def read_large_result():
    results = {}
    for W in [0, 3.4, 6.5]:
        result = {}
        group_name = f"W={W}"
        filename = DATA_DIR / "large_sim" / "qutrit_4x4_GML.h5"
        with h5py.File(filename, "r") as f:
            group = f[group_name]
            for key in group.keys():
                result[key] = group[key][()]
        results[group_name] = result
    return results


def plot_large_gen_bound(W, ax=None, colors=None, cbar_kwargs=None, **kwargs):
    f = read_large_result()
    group_name = f"W={W}"
    ts = f[group_name]["ts"]
    bounds = f[group_name]["bounds"]
    mask = f[group_name]["bounds_mask"]
    bounds = np.ma.masked_where(mask, bounds)
    alphas = f[group_name]["alphas"]
    assert f[group_name]["W"] == W

    if colors is None:
        colors = ["tab:blue", "tab:orange"]
    cbar_kwargs = cbar_kwargs or {}
    alpha = 1
    lower_bound = bounds[:, 0]
    upper_bound = bounds[:, 1]

    cmap_lower = cbar_kwargs.pop("cmap_lower", None)
    cmap_upper = cbar_kwargs.pop("cmap_upper", None)
    cax_lower = cbar_kwargs.pop("cax_lower", None)
    cax_upper = cbar_kwargs.pop("cax_upper", None)

    all_indexes = np.arange(len(alphas))
    # mask = (alphas >= 0.0) | (alphas == 0.0)
    # all_indexes = all_indexes[mask]
    mask = (
        (alphas == 0.0)
        | (alphas == 0.5)
        | (alphas == 1)
        | (alphas == 1.5)
        | (alphas == 2)
    )
    # all_indexes = all_indexes[mask]
    indexes = all_indexes
    # indexes = all_indexes[::100].tolist() + [all_indexes[-1]]
    # indexes = indexes + all_indexes[:40:1].tolist()
    # indexes = indexes + all_indexes[40:100:10].tolist()
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
    cbar_kwargs = cbar_kwargs or {}
    alpha = 1
    lower_bound = bounds[:, 0]  # (len(alphas), len(ts))
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
