from collections import defaultdict
from functools import partial

import h5py
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
import numpy as np
import seaborn as sns

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, num_to_alphabet, subplot
from path import DATA_DIR
from plot_toolbox import plot1d as _plot1d

from . import plot_sphere
from .styles import (
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

    for i, Omega in enumerate([-2.5, 0, 2.5]):
        delta_E, E_minus_Emin, Emax_minus_E, Emax_minus_Emin = phase_data[
            f"Omega={Omega}"
        ]
        ax.scatter(
            E_minus_Emin / Emax_minus_Emin,
            delta_E / Emax_minus_Emin,
            marker=markers[i],
            color=marker_colors[i],
            s=40,
            clip_on=False,
            zorder=np.inf,
            label=f"{Omega} MHz",
        )
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


def plot_phase_1d(ax, phase_data):
    xs = phase_data["Omegas"]
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
        legend_loc="upper center",
    )
    ax.scatter(
        -2.5,
        phase_data["Omega=-2.5"][1],
        marker=markers[0],
        color=marker_colors[0],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )
    ax.scatter(
        0,
        phase_data["Omega=0"][0],
        marker=markers[1],
        color=marker_colors[1],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )
    ax.scatter(
        2.5,
        phase_data["Omega=2.5"][2],
        marker=markers[2],
        color=marker_colors[2],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )


def rotate_matrix(x, y, angle, x_center=0, y_center=0):
    """
    Rotates a point in the xy-plane counterclockwise through an angle about the origin
    https://en.wikipedia.org/wiki/Rotation_matrix
    :param x: x coordinate
    :param y: y coordinate
    :param x_shift: x-axis shift from origin (0, 0)
    :param y_shift: y-axis shift from origin (0, 0)
    :param angle: The rotation angle in degrees
    :param units: DEGREES (default) or RADIANS
    :return: Tuple of rotated x and y
    """

    # Shift to origin (0,0)
    x = x - x_center
    y = y - y_center

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * np.cos(angle)) - (y * np.sin(angle)) + x_center
    yr = (x * np.sin(angle)) + (y * np.cos(angle)) + y_center

    return xr, yr


# ----- qubit -----
def collect_qubit_result(full=False):
    result = defaultdict(dict)
    filename = DATA_DIR / "fig1.h5"
    with h5py.File(filename, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            for key in group.keys():
                result[group_name][key] = group[key][()]
        keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es", "Emax_minus_Emin"]
        # with h5py.File(Q11_9_DIR / "phase_1d.h5", "r") as f_phase:
        f_phase = f["phase_1d"]
        Omegas = f_phase["Omegas"][()]
        result["phase_1d"]["Omegas"] = Omegas
        for key in keys:
            result["phase_1d"][key] = f_phase[key][()]
        for Omega in [-2.5, 0, 2.5]:
            index = np.where(Omega == Omegas)[0].item()
            result["phase_1d"][f"Omega={Omega}"] = [
                f_phase[key][()][index] for key in keys
            ]
    return dict(result)


def plot_2level(ax):
    lw = LW
    fontsize = AX_LABEL_SIZE + 1
    colors = {
        "w10": "#5e5d9c",
        "delta": "#8c6a5d",
        "uwave": "#cf4747",
    }
    ax.axis("on")
    ax.set_xlim(-0.5, 1.5)

    f10 = 1  # only for plotting, not the real value
    df = 0.25  # only for plotting, not the real value
    ax.hlines([0, f10], 0, 1, ls="-", color="k", lw=lw, zorder=np.inf)
    ax.hlines([f10 - df], 0.5, 1, ls="--", color="k", lw=lw, zorder=np.inf)
    pad = 0.02

    ax.text(
        -0.3,
        0,
        r"$|0\mathsf{\rangle}$",
        ha="left",
        va="center",
        fontdict={"fontsize": fontsize},
    )
    ax.text(
        -0.3,
        1,
        r"$|1\mathsf{\rangle}$",
        ha="left",
        va="center",
        fontdict={"fontsize": fontsize},
    )
    dx = -0.15

    # Delta
    color = colors["delta"]
    arrow = FancyArrowPatch(
        (0.75 + dx, f10 - df - pad),
        (0.75 + dx, f10 + pad),
        arrowstyle="<->",
        mutation_scale=7,
        color=color,
    )
    ax.add_patch(arrow)
    ax.text(
        0.8 + dx,
        f10 - df / 2,
        r"$\Delta$",
        ha="left",
        va="center",
        fontdict={"fontsize": fontsize},
        color=color,
    )
    # f10
    color = colors["w10"]
    arrow = FancyArrowPatch(
        (0.25 + dx, 0 - pad),
        (0.25 + dx, f10 + pad),
        arrowstyle="<->",
        mutation_scale=7,
        color=color,
    )
    ax.add_patch(arrow)
    ax.text(
        0.3 + dx,
        0.5,
        r"$\omega_{10}$",
        ha="left",
        va="center",
        fontdict={"fontsize": fontsize},
        color=color,
    )
    # uwave
    color = colors["uwave"]
    ax.text(
        0.96 + dx,
        0.75 / 2,
        "$\omega_d$",
        ha="left",
        va="center",
        fontdict={"fontsize": fontsize},
        color=color,
    )
    omega = 45
    T = 2 * np.pi / omega
    start = 0.9 + dx
    y = np.linspace(0, T * 4, 1001)
    x = 0.06 * np.sin(omega * y) + start
    x[y <= T] = start
    ax.plot(x, y, lw=LW, color=color)

    arrow = FancyArrowPatch(
        (0.9 + dx, y.max() - pad),
        (0.9 + dx, f10 - df + pad),
        arrowstyle="->",
        mutation_scale=7,
        color=color,
    )
    ax.add_patch(arrow)


@article_style()
def plot_fig1():
    result = collect_qubit_result()

    fig = fig_in_a4(1, 0.3, dpi=200)
    ag = AxesGroup(9, 3, figs=fig, mark_index=0, indexes_3d=[6, 7, 8])
    for ax in ag.axes[-3:]:
        ax.patch.set_alpha(0)

    fig.subplots_adjust(
        hspace=0.4,
        wspace=0.25,
        top=0.93,
        bottom=-0.3,
        left=0.064,
        right=0.928,
    )

    # plot bounds
    for i, Omega in enumerate([-2.5, 0, 2.5]):
        case_name = f"Omega={Omega}"
        if not case_name.startswith("Omega"):
            continue
        Omega = float(case_name.split("=")[-1])
        if np.isclose(Omega, 0):
            Omega = 0
        ax = ag.axes[i + 3]
        data = result[case_name]
        # ---- evolution ----
        ts = data["ts_exp"]
        ts_sim = data["ts_sim"]
        ts_bound = data["ts_bound"]

        overlaps_mean = data["overlaps_exp_mean"]
        overlaps_std = data["overlaps_exp_std"]

        plot1d(
            ts,
            overlaps_mean,
            overlaps_std,
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
            data["overlaps_sim"],
            lw=LW,
            color=colors["sim"],
            ax=ax,
            alpha=sim_alpha,
            label="Sim.",
        )
        index = data["overlaps_sim"].argmin()
        s = 30
        ax.scatter(
            ts_sim[index],
            data["overlaps_sim"][index],
            s=s,
            zorder=100,
            marker="*",
            clip_on=False,
            color="#6c0345",
        )
        o = round(Omega, 3)
        ax.text(
            0.5,
            0.97,
            rf"$\Omega~/~2\pi=${o} MHz",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
        )
        # # ---- bound ----
        bound_names = ["ML", "ML_star", "MT"]
        for j, bound_name in enumerate(bound_names):
            bound = data[f"{bound_name}_bound"]
            mask = data[f"{bound_name}_bound_mask"]
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

        cmap_upper = sns.light_palette("#92ba92", as_cmap=True)
        cmap_lower = sns.light_palette("#ffa41b", as_cmap=True)

        if i == 0:
            height = 0.012
            pos = ag.axes[3].get_position()
            left = pos.xmin + 0.06
            right = pos.xmax
            bottom = pos.ymax + 0.013
            # width = (right - left - interval) / 2
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
        plot_qubit_gen_bound(Omega, ax=ax, cbar_kwargs=kw)
        if i == 2:
            handles, labels = ax.get_legend_handles_labels()
            indexes = np.arange(len(handles) - 1).tolist()
            indexes.insert(0, -1)
            indexes[0], indexes[1] = indexes[1], indexes[0]
            handles = [handles[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            ax.legend(
                handles,
                labels,
                labelspacing=0.5,
                handletextpad=0.3,
                framealpha=0.2,
                prop={"size": LEGEND_SIZE - 1},
                loc="center left",
                ncol=1,
                bbox_to_anchor=(1, 0.2, 0.2, 0.6),
            )
        else:
            ax.get_legend().remove()

    ag.set_xticks(
        [0, 20, 40, 60], xlim=[0, 60], fontsize=AX_LABEL_SIZE, axes="row_1", sharex=0
    )
    ag.set_xlabel(
        "Evolution time (ns)",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        axes="row_1",
        sharex=0,
    )
    ag.set_yticks([0, 0.5, 1], ylim=[0, 1], fontsize=AX_LABEL_SIZE, axes="row_1")
    ag.set_ylabel(
        r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        usetex=USE_TEX,
        axes="row_1",
    )

    # plot phase 2d
    i = 1
    plot_phase_2d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([0, 1], xlim=[0, 1], axes=i, sharex=0)
    ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=i, sharey=0)

    # plot phase 1d
    i = 2
    keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es"]
    Omegas = result["phase_1d"]["Omegas"]
    plot_phase_1d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([-5, 5], xlim=[Omegas[0], Omegas[-1]], axes=i, sharex=0)
    ag.set_yticks([0, 5, 10, 15], ylim=[0, 15], axes=i, sharey=0)
    ag.set_xlabel(
        r"$\Omega~/~2\pi$ (MHz)",
        sharex=False,
        axes=i,
        fontsize=AX_LABEL_SIZE,
        labelpad=-4,
    )
    ag.set_ylabel(
        r"Energy / $2\pi$ (MHz)",
        sharey=False,
        axes=i,
        fontsize=AX_LABEL_SIZE,
        labelpad=-1,
    )
    # ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=1, sharey=0)

    # plot Bloch
    size = 0.2
    left = 0.1
    for i, Omega in enumerate([-2.5, 0, 2.5]):
        ax = ag.axes[i + 6]
        parent_pos = ag.axes[i + 3].get_position()
        plot_sphere.plot(ax=ax, Omega=Omega, is_qubit=1)
        ax.set_position([parent_pos.xmin - 0.068, parent_pos.ymin - 0.04, size, size])

    # plot level
    plot_2level(ax=ag.axes[0])

    # general
    ag.axes[0].axis("off")
    ag.tick_params(direction="out")
    ag.grid(False)
    top = 0.25
    annot_alphabet(
        ag.axes[:6],
        fontsize=FONTSIZE,
        dx=-0.03,
        dy=0.03,
        transform="fig",
        share_top=[0, 1, 2],
        left_bond_dict={6: 0, 7: 1, 8: 2},
        top_dict={6: top, 7: top, 8: top},
    )


@article_style()
def plot_qubit_gen_bound(Omega, ax=None, colors=None, cbar_kwargs=None, **kwargs):
    group_name = f"Omega={Omega}"
    data = collect_qubit_result()[group_name]
    ts = data["ts_gen_bound"]
    mask = data["gen_bounds_mask"]
    bounds = np.ma.masked_where(mask, data["gen_bounds"])
    alphas = data["alphas"]

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
    # mask = (
    #     (alphas == 0.0)
    #     | (alphas == 0.5)
    #     | (alphas == 1)
    #     | (alphas == 1.5)
    #     | (alphas == 2)
    # )
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
