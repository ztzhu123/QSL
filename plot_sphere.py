import h5py
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, _zalpha
import numpy as np
from skspatial.objects import Points, Sphere

from path import DATA_DIR
from styles import AX_LABEL_SIZE, article_style


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj3d.proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


@article_style()
def plot(
    ax=None,
    Omega=0,
    is_qubit=True,
    save=False,
    scheme=1,
    full=False,
    f=None,
    lw=0.5,
    elev=20,
    azim=50,
    pulse_shape=False,
):
    if ax is None:
        fig = fig_in_a4(1, 0.2)
        fig.set_constrained_layout(1)
        ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=elev, azim=azim, roll=0)

    sphere = Sphere([0, 0, 0], 1)
    sphere.plot_3d(ax, alpha=0.1)

    plot_circle(ax, direction="z", color="#c4bdc5", lw=0.5)
    plot_frame(ax, lw=0.8, color="k")
    plot_data(
        ax,
        color="#ff5733",
        Omega=Omega,
        scheme=scheme,
        full=full,
        f=f,
        lw=lw,
        pulse_shape=pulse_shape,
    )


def plot_data(
    ax, color, Omega=0, lw=1.2, scheme=0, full=False, f=None, pulse_shape=False
):
    filename = DATA_DIR / "fig1.h5"
    if f is None:
        with h5py.File(filename, "r") as f:
            group = f[f"Omega={Omega}"]
            overlaps_sim = group["overlaps_sim"][()]
            if not pulse_shape:
                xs_ideal = group["bloch_xs_ideal"][()]
                ys_ideal = group["bloch_ys_ideal"][()]
                zs_ideal = group["bloch_zs_ideal"][()]
            else:
                xs_ideal = group["bloch_xs_pulse_shape"][()]
                ys_ideal = group["bloch_ys_pulse_shape"][()]
                zs_ideal = group["bloch_zs_pulse_shape"][()]
            xs_exp = group["bloch_xs_exp_mean"][()]
            ys_exp = group["bloch_ys_exp_mean"][()]
            zs_exp = group["bloch_zs_exp_mean"][()]
    else:
        overlaps_sim = f["overlaps_sim"]
        xs_ideal = f["bloch_xs_ideal"]
        ys_ideal = f["bloch_ys_ideal"]
        zs_ideal = f["bloch_zs_ideal"]
        xs_exp = f["bloch_xs_exp_mean"]
        ys_exp = f["bloch_ys_exp_mean"]
        zs_exp = f["bloch_zs_exp_mean"]

    points_ideal = []
    for i in range(len(xs_ideal)):
        points_ideal.append([xs_ideal[i], ys_ideal[i], zs_ideal[i]])
    points_ideal = Points(points_ideal)

    points_exp = []
    for i in range(len(xs_exp)):
        points_exp.append([xs_exp[i], ys_exp[i], zs_exp[i]])
    if len(points_exp):
        points_exp = Points(points_exp)

        # points_ideal.plot_3d(ax, s=10, depthshade=True, zorder=np.inf, lw=0)
        points_exp.plot_3d(ax, s=6, depthshade=True, zorder=np.inf, lw=0, color=color)
    plot_line(ax, xs_ideal, ys_ideal, zs_ideal, color, lw=lw)

    alpha = 0.7
    kwargs = {
        "mutation_scale": 8,
        "arrowstyle": "->",
        "lw": 1,
        "shrinkA": 0,
        "alpha": alpha,
    }
    pad = 0
    length = 1.4
    x = xs_ideal[0]
    y = ys_ideal[0]
    z = zs_ideal[0]
    # arrow = Arrow3D(pad, 0, 0, x, y, z, color="tab:green", **kwargs)
    # ax.add_artist(arrow)

    ax.computed_zorder = False
    s = 30
    points = [[x, y, z]]
    points = Points(points)
    points.plot_3d(
        ax, s=s, depthshade=False, zorder=np.inf, lw=0, color="tab:green", marker="*"
    )

    index = np.argmin(np.abs(overlaps_sim))
    x = xs_ideal[index]
    y = ys_ideal[index]
    z = zs_ideal[index]
    # arrow = Arrow3D(pad, 0, 0, x, y, z, color="tab:blue", **kwargs)
    # ax.add_artist(arrow)
    points = [[x, y, z]]
    points = Points(points)
    points.plot_3d(
        ax, s=s, depthshade=False, zorder=np.inf, lw=0, color="#6c0345", marker="*"
    )


def plot_circle(ax, color="k", lw=0.3, direction="z"):
    theta = np.linspace(0, 2 * np.pi, 101)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x)
    if direction == "x":
        x, z = z, x
    elif direction == "y":
        y, z = z, y
    plot_line(ax, x, y, z, color, lw=lw, zorder=-np.inf)


def plot_line(ax, xs, ys, zs, color, **kwargs):
    colors = get_depth_colors(ax, xs, ys, zs, color)
    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = Line3DCollection(segments, colors=colors, **kwargs)
    ax.add_collection(line)


def get_depth_colors(ax, xs, ys, zs, color):
    alpha = ax.azim * np.pi / 180.0
    beta = ax.elev * np.pi / 180.0
    n = np.array(
        [np.cos(alpha) * np.sin(beta), np.sin(alpha) * np.cos(beta), np.sin(beta)]
    )
    ns = -np.dot(n, [xs, ys, zs])
    cs = _zalpha(color, ns)
    return cs


def plot_frame(ax, **kwargs):
    alpha = 0.7
    kwargs = {
        "mutation_scale": 8,
        "arrowstyle": "->",
        "lw": 0.5,
        "shrinkA": 0,
        "alpha": alpha,
    }
    pad = 0
    length = 1.4
    fontsize = AX_LABEL_SIZE - 2
    arrow = Arrow3D(pad, 0, 0, length, 0, 0, **kwargs)
    ax.add_artist(arrow)
    ax.text(length, 0.4, 0, "x", fontsize=fontsize, zorder=np.inf)

    arrow = Arrow3D(0, pad, 0, 0, length, 0, **kwargs)
    ax.add_artist(arrow)
    ax.text(0, length, 0, "y", fontsize=fontsize)

    arrow = Arrow3D(0, 0, pad, 0, 0, length, **kwargs)
    ax.add_artist(arrow)
    ax.text(0, 0, length, "z", fontsize=fontsize)
