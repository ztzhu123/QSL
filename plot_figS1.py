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


@article_style()
def plot_all():
    fig = fig_in_a4(1, 0.3, dpi=200)
    ag = AxesGroup(6, 3, max_rows_or_cols_per_fig=6, figs=fig)
    plot_hist(ag, np.arange(-6, 0))

    annot_alphabet(ag.axes, fontsize=FONTSIZE)
    ag.tick_params(direction="out")
    ag.grid(False)
    fig.subplots_adjust(
        hspace=0.6, wspace=0.15, top=0.92, bottom=0.12, left=0.07, right=0.98
    )


def plot_hist(ag, axes):
    axes = ag.convert_axes(axes)
    keys = ["cat", "fock"]
    label_dict = {"cat": "6Q", "fock": "9Q"}
    colors = ["tab:blue", "tab:orange"]
    num_qs = [6, 9]
    dy_dict = {"cat": 0, "fock": -0.25}
    for i, key in enumerate(keys):
        num_q = num_qs[i]
        data = load_data(key)

        yticks = [0, 0.5, 1]

        xticks = [4.5, 4.6, 4.7, 4.8]
        ax = plot_ecdf(
            data["f10s"],
            None,
            axes[0],
            xlim=[xticks[0], xticks[-1]],
            xlabel=r"Frequency (GHz)",
            xticks=xticks,
            yticks=yticks,
            ag=ag,
            precision=3,
            unit="GHz",
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key],
        )
        ax.set_title(r"Qubit $\omega_{10}~/~2\pi$", fontdict={"size": AX_TITLE_SIZE})

        xticks = [80, 100, 120, 140, 160]
        ax = plot_ecdf(
            data["t1s"],
            None,
            axes[1],
            xlim=[80, 170],
            xlabel=r"$T_1$ ($\mu$s)",
            xticks=xticks,
            yticks=yticks,
            ag=ag,
            precision=1,
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key],
        )
        ax.set_title(r"Qubit $T_1$", fontdict={"size": AX_TITLE_SIZE})

        xticks = [10, 20, 30, 40]
        ax = plot_ecdf(
            data["t2_ses"],
            None,
            axes[2],
            xlim=[10, 40],
            xticks=xticks,
            yticks=yticks,
            ag=ag,
            precision=1,
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key],
        )
        ax.set_xlabel(r"$T_2^{\rm{SE}}$ ($\mu$s)", labelpad=-1, fontsize=AX_LABEL_SIZE)
        ax.set_title(r"Qubit $T_2^{\rm{SE}}$", fontdict={"size": AX_TITLE_SIZE})

        xticks = [0, 0.05, 0.1, 0.15, 0.2]
        annot_suffix = "single-qubit"
        c = colors[i]
        ax = plot_ecdf(
            data["sq_pauli_errors"],
            None,
            axes[3],
            xlabel=r"Pauli error (%)",
            xticks=xticks,
            yticks=yticks,
            xlabelpad=0,
            unit="%",
            ag=ag,
            annot_suffix=annot_suffix,
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key] * 2,
            xlim=None,
        )
        if key == "cat":
            annot_suffix = "CZ,cycle"
            c = "tab:red"
            ax = plot_ecdf(
                data["tq_pauli_errors"],
                None,
                axes[3],
                xlabel=r"Pauli error (%)",
                xticks=xticks,
                yticks=yticks,
                unit="%",
                ag=ag,
                dash_color=c,
                label="6Q",
                xlabelpad=0,
                annot_yshift=-0.25,
                annot_suffix=annot_suffix,
                annot_color=c,
                color=c,
                xlim=None,
            )
        ax.set_xscale("log")
        ax.set_xlim(0.002, 1)
        title = "Gate error"
        ax.set_title(title, fontdict={"size": AX_TITLE_SIZE})

        xticks = [96, 97, 98, 99, 100]
        ax = plot_ecdf(
            data["measureF0s"],
            None,
            axes[4],
            xlim=[xticks[0], xticks[-1]],
            xlabel=r"Fidelity (%)",
            xticks=xticks,
            yticks=yticks,
            unit="%",
            ag=ag,
            precision=1,
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key],
        )
        ax.set_title(
            r"$|0\mathsf{\rangle}$ readout fidelity",
            fontdict={"size": AX_TITLE_SIZE},
        )
        ax = plot_ecdf(
            data["measureF1s"],
            None,
            axes[5],
            xlim=[xticks[0], xticks[-1]],
            xlabel=r"Fidelity (%)",
            xticks=xticks,
            yticks=yticks,
            unit="%",
            ag=ag,
            precision=1,
            label=label_dict[key],
            color=colors[i],
            dash_color=colors[i],
            annot_color=colors[i],
            annot_yshift=dy_dict[key],
        )
        ax.set_title(
            r"$|1\mathsf{\rangle}$ readout fidelity",
            fontdict={"size": AX_TITLE_SIZE},
        )

    ag.grid(False)
    for i in range(6):
        if i < 3:
            loc = "lower right"
        else:
            loc = "lower left"
        if i == 3:
            ncol = 2
        else:
            ncol = 1
        axes[i].legend(
            labelspacing=0.1,
            framealpha=0.4,
            prop={"size": LEGEND_SIZE - 1},
            loc=loc,
            ncol=ncol,
            columnspacing=0.8,
        )


def plot_ecdf(
    values,
    ax_heatmap,
    ax_ecdf,
    xlim=None,
    ylim=[0, 1],
    xlabel=None,
    xticks=None,
    yticks=None,
    precision=3,
    dash_color=None,
    annot_color="k",
    xlabelpad=None,
    unit="$\mu$s",
    ag=None,
    **kwargs,
):
    qnames = list(values.keys())

    ax = _plot_ecdf(
        list(values.values()),
        ax=ax_ecdf,
        unit=unit,
        lw=LW,
        annot_size=AX_LABEL_SIZE,
        precision=precision,
        sep="\n",
        xlim=xlim,
        ylim=ylim,
        dash_color=dash_color,
        annot_color=annot_color,
        constrained_layout=False,
        **kwargs,
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontdict={"size": AX_LABEL_SIZE}, labelpad=xlabelpad)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if ag is not None:
        if not ag.is_left(ax_ecdf):
            ax.tick_params(labelleft=False)
            ax.set_ylabel(None)
    return ax


def is_qubit(name):
    return name.startswith("q")


def is_coupler(name):
    return name.startswith("c")


def sort_names(names):
    def sort_key(name):
        if name.startswith("q"):
            value = 0
        elif name.startswith("c"):
            value = 1e5
        else:
            raise Exception(f"Unknown name: {name}")
        row, col = split_name(name)
        row, col = int(row), int(col)
        value = value + row * 1e3 + col
        return value

    return sorted(list(names), key=sort_key)


def split_name(q_or_c_name):
    result = re.findall(r"([q|c])?(\d+)_(\d+)", q_or_c_name)
    if len(result) == 0:
        raise NameError(f"Cannot split `{q_or_c_name}`!")

    prefix, row, col = result[0]
    row, col = int(row), int(col)

    return row, col


def load_data(key):
    device_info_filename = DATA_DIR / "device_info.xlsx"
    xeb_filename = DATA_DIR / "xeb.xlsx"
    df = pd.read_excel(device_info_filename, sheet_name=key, index_col=0)
    df_xeb = pd.read_excel(xeb_filename, sheet_name=key, index_col=0)
    qnames = []
    cnames = []
    data = {}
    for c in df.columns:
        if is_qubit(c):
            qnames.append(c)
    for c in df_xeb.columns:
        if is_coupler(c):
            cnames.append(c)
    qnames = sort_names(qnames)
    cnames = sort_names(cnames)
    data["t1s"] = {qname: df.loc["T1", qname] for qname in qnames}
    data["t2_ses"] = {qname: df.loc["T2_SE", qname] for qname in qnames}
    data["f10s"] = {qname: df.loc["f10", qname] for qname in qnames}
    data["measureF0s"] = {qname: df.loc["measureF0", qname] * 100 for qname in qnames}
    data["measureF1s"] = {qname: df.loc["measureF1", qname] * 100 for qname in qnames}
    data["sq_pauli_errors"] = {qname: df_xeb.loc["ref", qname] for qname in qnames}
    if key == "cat":
        tq_pauli_errors = {}
        for j in range(3):
            for cname in cnames:
                error = df_xeb.loc[f"tq_layer{j}(part)", cname]
                isnan = df_xeb.isna().loc[f"tq_layer{j}(part)", cname]
                if not isnan:
                    assert cname not in tq_pauli_errors
                    tq_pauli_errors[cname] = error
        data["tq_pauli_errors"] = tq_pauli_errors
    return data
