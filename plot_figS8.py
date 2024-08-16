from functools import partial
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from fig_utils import AxesGroup, annot_alphabet, fig_in_a4, num_to_alphabet, subplot
from path import DATA_DIR
from plot_fig1 import collect_qubit_result
from plot_fig2 import collect_qutrit_result
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

colors = {
    "probs": "#ee7214",
    "probs_ideal": "#836096",
}
colors["probs_pulse_shape"] = colors["probs"]
sim_alpha = 0.7

qutrit_exp_omegas = [-15, -5, 8.5]

plot1d = partial(_plot1d, constrained_layout=False)
plot2d = partial(_plot2d, constrained_layout=False)


def collect_pulse_shape_data():
    filename = DATA_DIR / "impulse_data.h5"
    result = {}
    with h5py.File(filename, mode="r") as f:
        for key in f.keys():
            result[key] = f[key][()]
    return result


@article_style()
def plot_pulse():
    fig = fig_in_a4(1, 0.37, dpi=200)
    ag = AxesGroup(4, 2, figs=fig)
    colors = ["#b2b7bd", "#37669a", "#d84941"]

    data = collect_pulse_shape_data()

    for i, key in enumerate(["I", "Q"]):
        # plot pulse shape
        lw = 1
        ts_original = data["ts_original"]
        ts = data["ts"]
        new_ts = data["new_ts"]
        plot1d(
            ts_original,
            data[f"{key}_hilbert_original"],
            ax=ag.axes[2 * i],
            lw=lw,
            label="Exp.",
            color=colors[0],
        )
        plot1d(
            ts,
            data[f"{key}_hilbert"],
            ax=ag.axes[2 * i],
            lw=lw + 0.5,
            label="Exp. + SG filter",
            color=colors[1],
        )
        plot1d(
            new_ts,
            data[f"{key}_hilbert_interp"],
            ax=ag.axes[2 * i],
            lw=lw + 0.5,
            label="Exp. + SG filter + interp.",
            legend_size=LEGEND_SIZE,
            legend_loc="upper right",
            color=colors[2],
        )
        # plot rect pulse
        pulse = data[f"{key}_pulse"]
        transferred_pulse = data[f"{key}_transferred_pulse"]
        ts_pad = data["ts_pad"]
        plot1d(
            ts_pad,
            [pulse, transferred_pulse],
            xlabel="time (ns)",
            ax=ag.axes[2 * i + 1],
            lw=LW,
            label=["Ideal", "Applied\ntransfer function"],
            color=["#739955", "#e97f2a"],
            legend_size=LEGEND_SIZE,
            # legend_loc=(0.09, 0.75),
            legend_loc="lower right",
        )

        carrier = np.cos(2 * np.pi * 0.5 * ts_pad)
        plot1d(
            ts_pad,
            np.array([pulse, transferred_pulse]) * carrier,
            ax=ag.axes[2 * i + 1],
            lw=LW,
            color=["#739955", "#e97f2a"],
            alpha=0.3,
        )
    # finalize
    ag.grid(False)
    ag.tick_params(direction="out")

    ag.set_xlabel("time (ns)", fontsize=AX_LABEL_SIZE)

    ag.set_ylabel(
        f"Impulse response of $I$ (a.u.)",
        fontsize=AX_LABEL_SIZE,
        sharey=0,
        axes=0,
        labelpad=-0.5,
    )
    ag.set_ylabel(
        f"Signal amplitude of $I$ (a.u.)",
        fontsize=AX_LABEL_SIZE,
        sharey=0,
        axes=1,
        labelpad=-0.5,
    )
    ag.set_ylabel(
        f"Impulse response of $Q$ (a.u.)",
        fontsize=AX_LABEL_SIZE,
        sharey=0,
        axes=2,
        labelpad=-0.5,
    )
    ag.set_ylabel(
        f"Signal amplitude of $Q$ (a.u.)",
        fontsize=AX_LABEL_SIZE,
        sharey=0,
        axes=3,
        labelpad=-0.5,
    )

    ag.set_xticks([0, 2, 4, 6, 8], xlim=[0, 8], axes=[0, 2], xlim_pad_ratio=0.02)
    ag.set_yticks([0, 0.6, 1.2], ylim=[0, 1.2], axes=[0, 2], ylim_pad_ratio=0.02)

    ag.set_xticks([0, 10, 20, 30], xlim=[0, 30], axes=[1, 3], xlim_pad_ratio=0.02)
    # ag.set_yticks([0, 1], ylim=[-0.02, 1.02], axes=[1, 3], sharey=0)
    ag.set_yticks([-1, 0, 1], ylim=[-1, 1], axes=[1, 3], sharey=0, ylim_pad_ratio=0.02)
    ag.remove_legend([-2, -1])
    ag.annot_alphabet(fontsize=FONTSIZE)
    ag.set_constrained_layout()


@article_style()
def plot_probs():
    plot_qutrit_probs()
    plot_qubit_probs()


@article_style()
def plot_qubit_probs():
    fig = fig_in_a4(1, 0.4 / 3 * 2, dpi=200)
    ag = AxesGroup(6, 3, figs=fig)
    axes = ag.axes

    data = collect_qubit_result(full=1)
    Omegas = [-2.5, 0, 2.5]
    for i, Omega in enumerate(Omegas):
        case_name = f"Omega={Omega}"

        ts = data[case_name]["ts_exp"]
        ts_ideal = data[case_name]["ts_sim"]
        ts_pulse_shape = data[case_name]["ts_pulse_shape"]
        mask = ts <= np.inf
        mask_ideal = ts_ideal <= np.inf
        mask_pulse_shape = ts_pulse_shape <= np.inf

        ts = ts[mask]
        ts_ideal = ts_ideal[mask_ideal]
        ts_pulse_shape = ts_pulse_shape[mask_pulse_shape]

        probs = data[case_name]["probs"][..., mask]
        probs_ideal = data[case_name]["probs_ideal"][..., mask_ideal]
        probs_pulse_shape = data[case_name]["probs_pulse_shape"][..., mask_pulse_shape]

        ax = np.reshape(axes, (2, 3)).T.flatten()
        ax = ax[2 * i : 2 * (i + 1)]
        ax[0].set_title(f"$\Omega~/~2\pi = {Omega}$ MHz", fontsize=AX_TITLE_SIZE)

        for j in range(2):
            plot1d(
                ts,
                probs.mean(0)[j],
                probs.std(0, ddof=1)[j],
                ax=ax[j],
                label="Exp.",
                color=colors["probs"],
                marker="o",
                ls="",
                lw=LW,
                ms=MS,
                capsize=CAPSIZE,
                legend_size=LEGEND_SIZE,
                mew=MEW,
            )
            plot1d(
                ts_ideal,
                probs_ideal[j],
                ax=ax[j],
                label="Ideal",
                color=colors["probs_ideal"],
                legend_size=LEGEND_SIZE,
                lw=LW,
                alpha=sim_alpha,
            )
            plot1d(
                ts_pulse_shape,
                probs_pulse_shape[j],
                ax=ax[j],
                label="Sim. with imperfect pulse",
                color=colors["probs_pulse_shape"],
                legend_size=LEGEND_SIZE,
                lw=LW,
                alpha=sim_alpha,
            )

    # finalize
    ag.grid(False)
    ag.tick_params(direction="out")

    ag.set_xlabel("time (ns)", fontsize=AX_LABEL_SIZE)
    for i in range(2):
        ag.set_ylabel(
            r"$P_{|%d\mathsf{\rangle}}$" % i, fontsize=AX_LABEL_SIZE, axes=f"row_{i}"
        )

    ag.set_xticks([0, 20, 40, 60], xlim=[0, 60], xlim_pad_ratio=0.02)
    ag.set_yticks([0, 0.5, 1], ylim=[0, 1], ylim_pad_ratio=0.02)
    ag.annot_alphabet(fontsize=FONTSIZE)
    ag.set_constrained_layout()
    ag.remove_legend(["row1", 1, 2])


@article_style()
def plot_qutrit_probs():
    fig = fig_in_a4(1, 0.4, dpi=200)
    ag = AxesGroup(9, 3, figs=fig)
    axes = ag.axes

    data = collect_qutrit_result(full=True)
    Omegas = qutrit_exp_omegas
    for i, Omega in enumerate(Omegas):
        case_name = f"Omega={Omega}"

        ts = data[case_name]["ts"]
        ts_ideal = data[case_name]["ts_sim"]
        ts_pulse_shape = data[case_name]["ts_pulse_shape"]
        mask = ts <= 5
        mask_ideal = ts_ideal <= 5
        mask_pulse_shape = ts_pulse_shape <= 5

        ts = ts[mask]
        ts_ideal = ts_ideal[mask_ideal]
        ts_pulse_shape = ts_pulse_shape[mask_pulse_shape]

        probs = data[case_name]["probs"][..., mask]
        probs_ideal = data[case_name]["probs_ideal"][..., mask_ideal]
        probs_pulse_shape = data[case_name]["probs_pulse_shape"][..., mask_pulse_shape]

        ax = np.reshape(axes, (3, 3)).T.flatten()
        ax = ax[3 * i : 3 * (i + 1)]
        ax[0].set_title(f"$\Omega~/~2\pi = {Omega}$ MHz", fontsize=AX_TITLE_SIZE)

        for j in range(3):
            plot1d(
                ts,
                probs.mean(0)[j],
                probs.std(0, ddof=1)[j],
                ax=ax[j],
                label="Exp.",
                color=colors["probs"],
                marker="o",
                ls="",
                lw=LW,
                ms=MS,
                capsize=CAPSIZE,
                legend_size=LEGEND_SIZE,
                mew=MEW,
            )
            plot1d(
                ts_ideal,
                probs_ideal[j],
                ax=ax[j],
                label="Ideal",
                color=colors["probs_ideal"],
                legend_size=LEGEND_SIZE,
                lw=LW,
                alpha=sim_alpha + 0.15,
            )
            plot1d(
                ts_pulse_shape,
                probs_pulse_shape[j],
                ax=ax[j],
                label="Sim. with imperfect pulse",
                color=colors["probs_pulse_shape"],
                legend_size=LEGEND_SIZE,
                lw=LW,
                alpha=sim_alpha,
            )

    # finalize
    ag.grid(False)
    ag.tick_params(direction="out")

    ag.set_xlabel("time (ns)", fontsize=AX_LABEL_SIZE)
    for i in range(3):
        ag.set_ylabel(
            r"$P_{|%d\mathsf{\rangle}}$" % i, fontsize=AX_LABEL_SIZE, axes=f"row_{i}"
        )

    ag.set_xticks([0, 1, 2, 3, 4, 5], xlim=[0, 5], xlim_pad_ratio=0.02)
    ag.set_yticks([0, 0.2], ylim=[0, 0.2], axes="row_0", ylim_pad_ratio=0.02)
    ag.set_yticks([0.2, 0.6], ylim=[0.2, 0.6], axes="row_1", ylim_pad_ratio=0.02)
    ag.set_yticks([0.3, 0.7], ylim=[0.3, 0.7], axes="row_2", ylim_pad_ratio=0.02)
    ag.annot_alphabet(fontsize=FONTSIZE)
    ag.set_constrained_layout()
    ag.remove_legend(["row1", "row2", 1, 2])
