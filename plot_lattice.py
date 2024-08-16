from itertools import combinations

from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
import numpy as np

from qcexp.device import (
    RectLattice,
    get_adj_qnames,
    get_cnames,
    get_qnames,
    is_adj,
    is_coupler,
    sort_names,
)
from qcexp.lazy_modules import networkx as nx
from qcexp.lazy_modules import pandas as pd
from qcexp.utils import print_info, wrap_emph
from qcexp.visualization import fig_in_a4

from qsl.path import FORMAL_DIR, MAIN_FIG_DIR, SM_FIG_DIR

from .styles import article_style

node_size = 500
linewidth = 1
edge_width = 20
edge_color = "#b3c6e6"
node_color = "#fdad9d"
node_edge_color = "#305c93"


@article_style()
def _plot_3x3(ax=None, save=False):
    size = (3, 3)
    qnames = get_qnames(size, "q7_7")
    cnames = get_cnames(size, "q7_7")
    cross_pairs = []
    rl = RectLattice(size, start_from="q7_7")

    for q0, q1 in combinations(qnames, 2):
        if is_adj(q0, q1, cross=1):
            cross_pairs.append((q0, q1))
    labels = {}
    colors = {}
    # set node color
    for i, qname in enumerate(qnames):
        if i % 2 == 0:
            labels[qname] = "$+W$"
            colors[qname] = "#e5a29b"
            colors[qname] = "white"
        else:
            labels[qname] = "$-W$"
            colors[qname] = "#96b7e0"
            colors[qname] = "white"
    # set edge color
    g_color = "#de7c7c"
    gx_color = "#9dafd3"

    for cname in cnames:
        colors[cname] = g_color
    # set edge label
    edge_labels = {("q11_7", "q11_9"): "$J_1$", ("q9_7", "q11_9"): "$J_2$"}
    # draw
    if ax is None:
        fig = fig_in_a4(1, 0.25)
        ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    edge_width = 10
    rl.draw(
        ax=ax,
        labels=labels,
        edge_labels=edge_labels,
        update_color=colors,
        node_size=350,
        width=edge_width,
        font_family="arial",
        font_size=9,
        edge_label_font_size=9,
        edgecolors=node_edge_color,
        linewidths=1.2,
        rotate_edge_label=False,
        tight_layout=False,
    )
    # draw cross coupling later to make it in the bottom
    for pair in cross_pairs:
        colors[pair] = gx_color
        rl.G.add_edge(*pair)
    collection = nx.draw_networkx_edges(
        rl.G,
        rl.get_position_dict(),
        edgelist=cross_pairs,
        edge_color=gx_color,
        width=edge_width * 0.5,
        ax=ax,
    )
    collection.set_zorder(-np.inf)

    ax.figure.set_constrained_layout(1)

    if save:
        fig_name = MAIN_FIG_DIR / "3x3_lattice.pdf"
        fig.savefig(fig_name, transparent=True, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


@article_style()
def plot_1x6(
    ax=None, save=False, node_size=350, edge_width=10, font_size=9, annot=True
):
    size = (6, 1)
    qnames = get_qnames(size, "q1_9")
    cnames = get_cnames(size, "q1_9")
    cross_pairs = []
    rl = RectLattice(size, start_from="q1_9")

    for q0, q1 in combinations(qnames, 2):
        if is_adj(q0, q1, cross=1):
            cross_pairs.append((q0, q1))
    labels = {}
    colors = {}
    # set node color
    for i, qname in enumerate(qnames):
        if i % 2 == 0:
            labels[qname] = "$+W$"
            colors[qname] = "#de7d7d"
            colors[qname] = "white"
        else:
            labels[qname] = "$-W$"
            colors[qname] = "white"
    # set edge color
    g_color = "#de7c7c"
    gx_color = "#9dafd3"

    for cname in cnames:
        colors[cname] = g_color
    # set edge label
    edge_labels = {"c10_9": "$J_1$"}
    # draw
    if ax is None:
        fig = fig_in_a4(1, 0.1)
        ax = fig.add_subplot(111)
    # ax.set_aspect("equal")

    if not annot:
        labels = {}
        edge_labels = {}
    rl.draw(
        ax=ax,
        labels=labels,
        edge_labels=edge_labels,
        update_color=colors,
        node_size=node_size,
        width=edge_width,
        font_family="arial",
        font_size=font_size,
        edge_label_font_size=font_size,
        edgecolors=node_edge_color,
        linewidths=1.2,
        rotate_edge_label=False,
        tight_layout=False,
    )
    # ax.figure.set_constrained_layout(1)
    # fig.subplots_adjust(left=0.2, right=0.8)

    if save:
        fig_name = MAIN_FIG_DIR / "1x6_chain.pdf"
        fig.savefig(fig_name, transparent=True, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


@article_style()
def plot_3x3(ax=None, save=False, node_size=350, edge_width=10, font_size=9):
    size = (3, 3)
    qnames = get_qnames(size, "q7_7")
    cnames = get_cnames(size, "q7_7")
    cross_pairs = []
    rl = RectLattice(size, start_from="q7_7")

    for q0, q1 in combinations(qnames, 2):
        if is_adj(q0, q1, cross=1):
            cross_pairs.append((q0, q1))
    labels = {}
    colors = {}
    # set node color
    for i, qname in enumerate(qnames):
        if i % 2 == 0:
            labels[qname] = "$+W$"
            colors[qname] = "#e5a29b"
            colors[qname] = "white"
        else:
            labels[qname] = "$-W$"
            colors[qname] = "#96b7e0"
            colors[qname] = "white"
    # set edge color
    g_color = "#de7c7c"
    gx_color = "#9dafd3"

    for cname in cnames:
        colors[cname] = g_color
    # set edge label
    edge_labels = {("q11_7", "q11_9"): "$J_1$", ("q9_7", "q11_9"): "$J_2$"}
    # draw
    if ax is None:
        fig = fig_in_a4(1, 0.25)
        ax = fig.add_subplot(111)
        ax.figure.set_constrained_layout(1)
    ax.set_aspect("equal")
    rl.draw(
        ax=ax,
        labels=labels,
        edge_labels=edge_labels,
        update_color=colors,
        node_size=node_size,
        width=edge_width,
        font_family="arial",
        font_size=font_size,
        edge_label_font_size=font_size,
        edgecolors=node_edge_color,
        linewidths=1.2,
        rotate_edge_label=False,
        tight_layout=False,
    )
    # draw cross coupling later to make it in the bottom
    for pair in cross_pairs:
        colors[pair] = gx_color
        rl.G.add_edge(*pair)
    collection = nx.draw_networkx_edges(
        rl.G,
        rl.get_position_dict(),
        edgelist=cross_pairs,
        edge_color=gx_color,
        width=edge_width * 0.5,
        ax=ax,
    )
    collection.set_zorder(-np.inf)

    if save:
        fig_name = MAIN_FIG_DIR / "3x3_lattice.pdf"
        fig.savefig(fig_name, transparent=True, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


@article_style()
def plot_cat_lattice(
    save=False, node_size=300, edge_width=12, linewidth=1.2, axes=None
):
    patterns = {
        0: ["c3_6"],
        1: ["c3_4", "c3_8"],
        2: ["c3_2", "c3_10"],
    }
    num_patterns = len(patterns)
    if axes is None:
        fig = fig_in_a4(1, 0.3)
        gs = fig.add_gridspec(num_patterns, 1)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(num_patterns)]

    cnames = get_cnames((1, 6), start_from="q3_1")
    rl = RectLattice((1, 6), start_from="q3_1")
    inactive_color = "#e7dad2"
    history_color = "#b3c6e6"
    active_color = "#fdad9d"

    colors = {}
    for cname in cnames:
        colors[cname] = inactive_color

    for pattern, pattern_cnames in patterns.items():
        ax = axes[pattern]
        for cname in cnames:
            if colors[cname] == active_color:
                colors[cname] = history_color
            for qname in get_adj_qnames(cname):
                colors[qname] = inactive_color

        edge_labels = {}
        for cname in pattern_cnames:
            colors[cname] = active_color
            edge_labels[cname] = f"{2*pattern+1}"
            for qname in get_adj_qnames(cname):
                colors[qname] = active_color

        rl.draw(
            ax=ax,
            with_labels=False,
            edge_labels=edge_labels,
            update_color=colors,
            node_size=node_size,
            width=edge_width,
            font_family="arial",
            font_size=12,
            edge_label_font_size=11,
            edgecolors=to_rgba("k", 0.9),
            linewidths=linewidth,
            rotate_edge_label=False,
            tight_layout=False,
        )
        if pattern < num_patterns - 1:
            arrow = FancyArrowPatch(
                (0.5, 0),
                (0.5, -0.8),
                transform=ax.transAxes,
                zorder=np.inf,
                mutation_scale=17,
                color="k",
                arrowstyle="-|>",
                clip_on=False,
                linewidth=2,
            )
            ax.add_patch(arrow)

    fig = axes[0].figure
    fig.set_constrained_layout(1)

    if save:
        fig_name = SM_FIG_DIR / "cat_lattice.pdf"
        fig.savefig(fig_name, transparent=True, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


@article_style()
def plot_coupling(save=0):
    pass
    fig = fig_in_a4(1, 0.3)
    ax_cat = fig.add_subplot(211)
    ax_fock = fig.add_subplot(212)
    plot_cat_coupling(ax=ax_cat)
    plot_fock_coupling(ax=ax_fock)

    fig.set_constrained_layout(1)

    if save:
        fig_name = SM_FIG_DIR / "sm_coupling.pdf"
        fig.savefig(fig_name, transparent=True, pad_inches=0)
        print_info("Saved ", wrap_emph(fig_name.as_posix()))


def plot_fock_coupling(ax):
    filename = FORMAL_DIR / "coupling.xlsx"
    df_g = pd.read_excel(filename, sheet_name="fock_g", index_col=0)
    df_gx = pd.read_excel(filename, sheet_name="fock_gx", index_col=0)
    cnames = sort_names([c for c in df_g.columns if is_coupler(c)])
    qnames = set()
    for cname in cnames:
        qnames.update(get_adj_qnames(cname))
    qnames = sort_names(qnames)
    pairs = sort_names([p for p in df_g.columns if p.startswith("[")])

    rl = RectLattice((3, 3), start_from=qnames[0])

    colors = {}
    for cname in cnames:
        colors[cname] = edge_color
    for qname in qnames:
        colors[qname] = node_color

    edge_labels = {}

    rl.draw(
        ax=ax,
        with_labels=False,
        edge_labels=edge_labels,
        update_color=colors,
        node_size=node_size,
        width=edge_width,
        font_family="arial",
        font_size=12,
        edge_label_font_size=11,
        edgecolors=to_rgba("k", 0.9),
        linewidths=linewidth,
        rotate_edge_label=False,
        transpose=1,
        tight_layout=False,
    )


def plot_cat_coupling(ax):
    filename = FORMAL_DIR / "coupling.xlsx"
    df = pd.read_excel(filename, sheet_name="cat_g", index_col=0)
    cnames = sort_names([c for c in df.columns if is_coupler(c)])
    qnames = set()
    for cname in cnames:
        qnames.update(get_adj_qnames(cname))
    qnames = sort_names(qnames)

    rl = RectLattice((6, 1), start_from=qnames[0])

    colors = {}
    for cname in cnames:
        colors[cname] = edge_color
    for qname in qnames:
        colors[qname] = node_color

    edge_labels = {c: f"{df.loc['g',c]:.3f}" for c in cnames}

    rl.draw(
        ax=ax,
        with_labels=False,
        edge_labels=edge_labels,
        update_color=colors,
        node_size=node_size,
        width=edge_width,
        font_family="arial",
        font_size=12,
        edge_label_font_size=11,
        edgecolors=to_rgba("k", 0.9),
        linewidths=linewidth,
        rotate_edge_label=False,
        transpose=1,
        tight_layout=False,
    )
