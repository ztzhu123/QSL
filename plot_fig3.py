from collections import defaultdict
import sys
from functools import partial

import h5py
import matplotlib as mpl
import numpy as np
import seaborn as sns


from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, subplot
from path import DATA_DIR
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
                cbar.ax.yaxis.set_label_coords(-0.9, -0.2)

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

    for i, W in enumerate([0, 1.8, 8]):
        delta_E, E_minus_Emin, Emax_minus_E, Emax_minus_Emin = phase_data[f"W={W}"]
        ax.scatter(
            E_minus_Emin / Emax_minus_Emin,
            delta_E / Emax_minus_Emin,
            marker=markers[i],
            color=marker_colors[i],
            s=40,
            clip_on=False,
            zorder=np.inf,
            label=f"${W}$ MHz",
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
        1.8,
        phase_data["W=1.8"][1],
        marker=markers[1],
        color=marker_colors[1],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )
    ax.scatter(
        8,
        phase_data["W=8"][1],
        marker=markers[2],
        color=marker_colors[2],
        s=40,
        clip_on=False,
        zorder=np.inf,
    )


@article_style()
def plot_fig3():
    result = collect_cat_result()

    fig = fig_in_a4(0.6, 0.4, dpi=200)
    ag = AxesGroup(6, 2, figs=fig, mark_index=0)

    fig.subplots_adjust(
        hspace=0.4,
        wspace=0.35,
        top=0.93,
        bottom=0.08,
        left=0.12,
        right=0.975,
    )

    # plot bounds
    for i, case_name in enumerate(result):
        if not case_name.startswith("W"):
            continue
        W = float(case_name.split("=")[-1])
        if np.isclose(W, 0):
            W = 0
        ax = ag.axes[2 * i + 1]
        data = result[case_name]
        # ---- evolution ----
        ts = data["ts"]
        ts_sim = data["ts_sim"]
        fids_ideal = data["fids_sim"]

        ts_bound = ts_sim

        fids = data["fids"]
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

        o = round(W, 3)
        if float(o).is_integer():
            o = int(o)
        ax.text(
            0.99,
            0.1,
            rf"$W~/~2\pi=${o} MHz",
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )

        cmap_upper = sns.light_palette("#92ba92", as_cmap=True)
        cmap_lower = sns.light_palette("#ffa41b", as_cmap=True)

        if i == 2:
            height = 0.012
            interval = 0.1
            pos = ag.axes[1].get_position()
            left = pos.xmin + 0.105
            right = pos.xmax
            bottom = pos.ymax + 0.013
            # width = (right - left - interval) / 2
            width = 0.067

            cax_lower = fig.add_axes([left, bottom, width, height])
            cax_upper = fig.add_axes([right - width, bottom, width, height])
        else:
            cax_upper = cax_lower = None
        kw = {
            "plot_cbar": i == 2,
            "vmin": 0,
            "vmax": 4,  # deliberately different from cbar_ylim
            "cbar_ylim": [0, 2],
            "cmap_lower": cmap_lower,
            "cmap_upper": cmap_upper,
            "cax_upper": cax_upper,
            "cax_lower": cax_lower,
            "orientation": "horizontal",
        }
        plot_cat_gen_bound(W, ax=ax, cbar_kwargs=kw)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            indexes = np.arange(len(handles) - 1).tolist()
            indexes.insert(0, -1)
            indexes[0], indexes[1] = indexes[1], indexes[0]
            handles = [handles[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            ax.legend(
                handles,
                labels,
                labelspacing=0.2,
                handletextpad=0.3,
                framealpha=0.2,
                prop={"size": LEGEND_SIZE - 1},
                loc="upper right",
                ncol=1,
            )
        else:
            ax.get_legend().remove()

    ag.tick_params()
    ag.set_xticks(
        np.arange(0, 101, 50),
        xlim=[0, 100],
        fontsize=AX_LABEL_SIZE,
        axes=[1, 3],
        sharex=0,
    )
    ag.set_xticks(
        np.arange(0, 41, 20), xlim=[0, 40], fontsize=AX_LABEL_SIZE, axes=5, sharex=0
    )
    ag.set_xlabel(
        "Evolution time (ns)",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        labelpad=0,
        sharex=1,
        axes="col_1",
    )
    ag.set_yticks(
        [0, 0.5, 1], ylim=[0, 1], fontsize=AX_LABEL_SIZE, sharey=0, axes="col_1"
    )
    ag.set_ylabel(
        r"$|\mathsf{\langle}\psi(0)|\psi(t)\mathsf{\rangle}|$",
        fontsize=AX_LABEL_SIZE,
        clean_others=1,
        usetex=USE_TEX,
        sharey=0,
        axes="col_1",
        labelpad=1,
    )

    # plot chain
    ax = ag.axes[0]
    ax.patch.set_alpha(0)
    pos = ax.get_position()
    ax.set_position([pos.xmin, pos.ymin - 0.05, pos.width, pos.height * 1.4])

    for o in ax.findobj():
        o.set_clip_on(False)
    # ax.axis("on")
    pos = ax.get_position()
    ymin, ymax = ax.get_ylim()
    ymin = ymin + 0.13
    dy = 1 / 6
    ag.axes[0].axis("off")

    # plot phase 2d
    i = 2
    plot_phase_2d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([0, 1], xlim=[0, 1], axes=i, sharex=0)
    ag.set_yticks([0, 0.5], ylim=[0, 0.5], axes=i, sharey=0)

    # plot phase 1d
    i = 4
    keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es", "Emax_minus_Emin"]
    Ws = result["phase_1d"]["Ws"]
    plot_phase_1d(ag.axes[i], result["phase_1d"])
    ag.set_xticks([0, 10], xlim=[Ws[0], Ws[-1]], axes=i, sharex=0)
    ag.set_yticks([0, 25, 50], ylim=[0, 50], axes=i, sharey=0)
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
    ag.grid(False)
    ag.tick_params(direction="out")

    annot_alphabet(
        np.reshape(np.reshape(ag.axes, (3, 2)), -1, order="F"),
        fontsize=FONTSIZE,
        dx=-0.06,
        dy=0.02,
        transform="fig",
        share_top=[1, 4],
        top_bond_dict={0: 3},
        left_bond_dict={0: 1},
    )


def collect_cat_result():
    Ws = [0, 1.8, 8]
    result = defaultdict(dict)
    filename = DATA_DIR / "fig3.h5"
    with h5py.File(filename, "r") as f:
        for W in Ws:
            case_name = f"W={W}"
            for key in f[case_name].keys():
                result[case_name][key] = f[case_name][key][()]
            result[case_name]["ts_tomo"] = f[case_name]["ts_tomo"][()]
            result[case_name]["fids_tomo"] = f[case_name]["fids_tomo"][()]
            result[case_name]["fids_sim_exp_init_state"] = f[case_name][
                "fids_sim_exp_init_state"
            ][()]
            result[case_name]["ts_sim_exp_init_state"] = f[case_name][
                "ts_sim_exp_init_state"
            ][()]

        keys = ["delta_Es", "E_minus_Emins", "Emax_minus_Es", "Emax_minus_Emin"]
        Ws = f["phase_1d"]["Ws"][()]
        result["phase_1d"]["Ws"] = Ws
        for key in keys:
            result["phase_1d"][key] = f["phase_1d"][key][()]
        for W in [0, 1.8, 8]:
            index = np.where(W == Ws)[0].item()
            result["phase_1d"][f"W={W}"] = [
                f["phase_1d"][key][()][index] for key in keys
            ]

    return dict(result)


@article_style()
def plot_cat_gen_bound(W, ax=None, colors=None, cbar_kwargs=None, **kwargs):
    filename = DATA_DIR / "fig3_cat_gen_bound.h5"
    if not np.isclose(W, 1.8):
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
