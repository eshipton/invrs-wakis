"""Figures and animations of optimization runs (matplotlib + pillow only).

Inputs are the files written by `scripts/run_challenge.py` for one run
(`snapshots.npz`, `fields.npz`, `summary.json`); the gym challenge object is
rebuilt by name only for geometry (no simulation).
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SOLID = "#4f63d2"  # silicon / lens material
VOID = "#eceef4"  # cladding / air
PORT = "#e0549b"
INK = "#2b2d33"
MUTED = "#8a8f9c"
MATERIAL = LinearSegmentedColormap.from_list("material", [VOID, SOLID])
FIELD = "RdBu_r"
INTENSITY = "inferno"

SHORT = {
    "ceviche_lightweight_waveguide_bend": "waveguide bend",
    "ceviche_lightweight_mode_converter": "mode converter",
    "ceviche_lightweight_power_splitter": "power splitter",
    "ceviche_lightweight_beam_splitter": "beam splitter",
    "ceviche_lightweight_wdm": "wavelength demux",
    "metalens": "RGB metalens",
}


def set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
        }
    )


def short_name(name):
    return SHORT.get(name, name.replace("ceviche_", "").replace("_", " "))


# ---------------------------------------------------------------------------
# geometry helpers


def gym_challenge(name):
    from invrs_gym import challenges

    return challenges.BY_NAME[name]()


def scale_bar(ax, length_um, x, y, color=INK, label=None):
    """Horizontal scale bar starting at data coordinates (x, y) [um]."""
    ax.plot([x, x + length_um], [y, y], color=color, lw=2, solid_capstyle="butt")
    ax.text(x + length_um / 2, y, label or f"{length_um:g} µm", color=color,
            ha="center", va="bottom", fontsize=7.5)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_ports(ax, ports, color=PORT, arrows=True):
    """Port planes (and the input direction) from fields.npz 'ports'."""
    for k, (x, y, w, axis, sign) in enumerate(ports):
        if int(axis) == 0:
            ax.plot([x, x], [y - w / 2, y + w / 2], color=color, lw=1.2)
        else:
            ax.plot([x - w / 2, x + w / 2], [y, y], color=color, lw=1.2)
        if arrows and k == 0:
            d = 0.45 * sign
            dx, dy = (d, 0) if int(axis) == 0 else (0, d)
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))


def ceviche_extent_um(model):
    n0, n1 = model.shape
    dl = model.dl / 1e-6
    return [0, n0 * dl, 0, n1 * dl]


def draw_ceviche_density(ax, model, design, ports=None, pml_um=None, design_box=True):
    density = np.asarray(model.density(np.asarray(design, float)))
    ext = ceviche_extent_um(model)
    im = ax.imshow(density.T, origin="lower", extent=ext, cmap=MATERIAL, vmin=0, vmax=1,
                   interpolation="nearest")
    dl = model.dl / 1e-6
    if design_box:
        x0, y0, x1, y1 = model.design_region_coords
        ax.add_patch(Rectangle((x0 * dl, y0 * dl), (x1 - x0) * dl, (y1 - y0) * dl, fill=False,
                               ec=MUTED, lw=0.8, ls=(0, (3, 2))))
    if pml_um:
        _shade_pml(ax, ext, pml_um)
    if ports is not None:
        draw_ports(ax, ports)
    _bare(ax)
    return im


def _shade_pml(ax, ext, pml):
    x0, x1, y0, y1 = ext
    kw = dict(fc="black", alpha=0.08, lw=0)
    ax.add_patch(Rectangle((x0, y0), pml, y1 - y0, **kw))
    ax.add_patch(Rectangle((x1 - pml, y0), pml, y1 - y0, **kw))
    ax.add_patch(Rectangle((x0 + pml, y0), x1 - x0 - 2 * pml, pml, **kw))
    ax.add_patch(Rectangle((x0 + pml, y1 - pml), x1 - x0 - 2 * pml, pml, **kw))


def _ports_from_model(model):
    dl = model.dl / 1e-6
    ports = []
    for p in model.ports:
        axis = 0 if p.dir.is_along_x else 1
        if axis == 0:
            ports.append([(p.x + 0.5) * dl, p.y * dl, p.width * dl, 0, p.dir.sign])
        else:
            ports.append([p.x * dl, (p.y + 0.5) * dl, p.width * dl, 1, p.dir.sign])
    return np.asarray(ports)


# ---------------------------------------------------------------------------
# (a) design evolution


def _frame(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, path, fps=8, hold_last=2.0):
    images = [Image.fromarray(f) for f in frames]
    durations = [int(1000 / fps)] * len(images)
    durations[-1] = int(1000 * hold_last)
    # Shared adaptive palette keeps colours stable across frames
    pal = images[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=False, disposal=2)


def design_evolution_gif(name, snapshots, path, max_frames=120, fps=10):
    """
    Animated design through the optimization: density (with the fixed
    waveguides, ports and design region) next to the eval-metric trace.
    """
    set_style()
    designs = snapshots["designs"].astype(float)
    metric = snapshots["eval_metric"]
    stages = snapshots["stage"]
    n = len(designs)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int))
    metalens = name == "metalens"
    if metalens:
        fig = plt.figure(figsize=(7.2, 3.0), dpi=110)
        ax_d = fig.add_axes([0.03, 0.55, 0.94, 0.32])
        ax_m = fig.add_axes([0.1, 0.12, 0.86, 0.33])
        im = ax_d.imshow(designs[0].T, cmap=MATERIAL, vmin=0, vmax=1,
                         aspect="auto", interpolation="nearest",
                         extent=[0, 16, 0, designs.shape[2] * 0.02])
        _bare(ax_d)
        ax_d.set_ylim(0, designs.shape[2] * 0.02)
        ax_d.text(0, -0.08, "16 µm period × 1 µm lens (height stretched)", fontsize=6.5,
                  color=MUTED, transform=ax_d.transAxes, va="top")
    else:
        model = gym_challenge(name).component.ceviche_model
        fig = plt.figure(figsize=(7.2, 3.4), dpi=110)
        ax_d = fig.add_axes([0.01, 0.04, 0.46, 0.84])
        ax_m = fig.add_axes([0.58, 0.16, 0.39, 0.7])
        im = draw_ceviche_density(ax_d, model, designs[0], ports=_ports_from_model(model),
                                  pml_um=model.pml_width * model.dl / 1e-6)
        ext = ceviche_extent_um(model)
        scale_bar(ax_d, 1, ext[0] + 0.3, ext[2] + 0.12)
    # metric trace
    ax_m.plot(np.arange(n), metric, color=MUTED, lw=0.8)
    (cur,) = ax_m.plot([0], [metric[0]], "o", color=PORT, ms=4)
    (trace,) = ax_m.plot([0], [metric[0]], color=SOLID, lw=1.4)
    lo, hi = np.nanpercentile(metric, 2), np.nanmax(metric)
    pad = 0.08 * (hi - lo + 1e-9)
    ax_m.set_ylim(lo - pad, hi + pad)
    ax_m.set_xlim(0, n - 1)
    ax_m.set_xlabel("iteration")
    ax_m.set_ylabel("eval metric (Wakis)")
    if not metalens:
        ax_m.axhline(0, color=MUTED, lw=0.6, ls=":")
    starts = [i for i in range(1, n) if stages[i] != stages[i - 1]]
    for s in starts:
        ax_m.axvline(s, color=MUTED, lw=0.5, ls="--")
    for s, label in zip([0] + starts, [stages[0]] + [stages[s] for s in starts]):
        ax_m.text(s + 0.5, hi + pad * 0.6, label.replace("density_", "").replace("beta", "β="),
                  fontsize=6.5, color=MUTED, va="top")
    for sp in ("top", "right"):
        ax_m.spines[sp].set_visible(False)
    title = fig.suptitle("", x=0.02, ha="left", fontsize=10)
    frames = []
    for i in idx:
        if metalens:
            im.set_data(designs[i].T)
        else:
            im.set_data(np.asarray(model.density(designs[i])).T)
        cur.set_data([i], [metric[i]])
        trace.set_data(np.arange(i + 1), metric[: i + 1])
        title.set_text(f"{short_name(name)} · iteration {i}  ({stages[i]})  ·  eval metric {metric[i]:+.3f}")
        frames.append(_frame(fig))
    plt.close(fig)
    save_gif(frames, path, fps=fps)


# ---------------------------------------------------------------------------
# (b, c) field figures


def _outline(ax, x, y, eps, color, lw=0.6):
    level = 0.5 * (np.nanmin(eps) + np.nanmax(eps))
    ax.contour(x, y, eps.T, levels=[level], colors=color, linewidths=lw)


def _crop(f, pml):
    x, y = f["x_um"], f["y_um"]
    sx = (x > x.min() + pml) & (x < x.max() - pml)
    sy = (y > y.min() + pml) & (y < y.max() - pml)
    return sx, sy


def ceviche_field_figure(name, f, path):
    """Re(Ez) and |Ez|^2 at each challenge wavelength, design outline, ports."""
    set_style()
    sx, sy = _crop(f, float(f["pml_um"]))
    x, y = f["x_um"][sx], f["y_um"][sy]
    eps = f["eps"][np.ix_(sx, sy)]
    ez = f["Ez"][:, sx][:, :, sy]
    wl = f["wavelengths_nm"]
    nw = len(wl)
    aspect = (y.max() - y.min()) / (x.max() - x.min())
    fig, axes = plt.subplots(nw, 2, figsize=(5.6, 2.75 * nw * aspect + 0.5), dpi=150, squeeze=False)
    vmax = np.percentile(np.abs(ez), 99.7)
    for k in range(nw):
        for j, (data, cmap, lim, color) in enumerate(
            [
                (ez[k].real, FIELD, (-vmax, vmax), INK),
                (np.abs(ez[k]) ** 2, INTENSITY, (0, vmax**2), "white"),
            ]
        ):
            ax = axes[k, j]
            m = ax.pcolormesh(x, y, data.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto",
                              rasterized=True)
            _outline(ax, x, y, eps, color)
            draw_ports(ax, f["ports"], arrows=(j == 0))
            ax.set_aspect("equal")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            _bare(ax)
            cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6.5)
            cb.outline.set_linewidth(0.4)
            label = ("Re $E_z$" if j == 0 else "$|E_z|^2$") + f"  ·  λ = {wl[k]:.0f} nm"
            ax.set_title(label, fontsize=8)
        scale_bar(axes[k, 0], 1, x.min() + 0.2, y.min() + 0.15)
    fig.suptitle(f"{short_name(name)} — Wakis fields of the final design\n"
                 "(normalized to unit incident mode amplitude)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def metalens_field_figure(f, path, summary=None):
    """|E|^2 / |E_focus, no lens|^2 and Re(Ex) at the three wavelengths."""
    set_style()
    x, y, eps = f["x_um"], f["y_um"], f["eps"]
    wl = f["wavelengths_nm"]
    colors = {450: "#4f63d2", 550: "#2e9e5b", 650: "#d2493f"}
    fig, axes = plt.subplots(len(wl), 2, figsize=(9.0, 1.65 * len(wl) + 0.6), dpi=150,
                             squeeze=False)
    fx, fy = f["focus_um"]
    for k, w in enumerate(wl):
        enh = f["enhancement"][k]
        ex = f["Ex"][k]
        vmax = np.percentile(np.abs(ex), 99.7)
        for j, (data, cmap, lim, color) in enumerate(
            [(enh, INTENSITY, (0, np.nanmax(enh)), "white"), (ex.real, FIELD, (-vmax, vmax), INK)]
        ):
            ax = axes[k, j]
            m = ax.pcolormesh(x, y, data.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto",
                              rasterized=True)
            _outline(ax, x, y, eps, color, lw=0.4)
            ax.plot([fx], [fy], marker="o", mfc="none", mec=PORT, ms=6, mew=1)
            ax.set_aspect("equal")
            _bare(ax)
            cb = fig.colorbar(m, ax=ax, fraction=0.02, pad=0.01)
            cb.ax.tick_params(labelsize=6.5)
            cb.outline.set_linewidth(0.4)
            if j == 0:
                i_f = np.argmin(np.abs(x - fx))
                j_f = np.argmin(np.abs(y - fy))
                ax.set_title(f"intensity enhancement  ·  λ = {w:.0f} nm  ·  at focus "
                             f"{enh[i_f, j_f]:.1f}", fontsize=8,
                             color=colors.get(int(round(w)), INK))
            else:
                ax.set_title(f"Re $E_x$  ·  λ = {w:.0f} nm", fontsize=8)
        scale_bar(axes[k, 1], 2, x.min() + 0.3, y.min() + 0.15)
    title = "RGB metalens — Wakis fields of the final design (EX polarization)"
    if summary:
        title += f"\nofficial (fmmax) eval metric = {summary['official_eval_metric']:.2f}"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def phase_gif(name, f, path, n_phase=24, fps=12):
    """Wave propagation: Re(E exp(i phi)) of the single-frequency DFT fields."""
    set_style()
    if name == "metalens":
        x, y, eps = f["x_um"], f["y_um"], f["eps"]
        e = f["Ex"]
        figsize, label = (7.2, 1.25 * len(e) + 0.5), "Re $E_x e^{i\\varphi}$"
    else:
        sx, sy = _crop(f, float(f["pml_um"]))
        x, y = f["x_um"][sx], f["y_um"][sy]
        eps = f["eps"][np.ix_(sx, sy)]
        e = f["Ez"][:, sx][:, :, sy]
        aspect = (y.max() - y.min()) / (x.max() - x.min())
        figsize, label = (2.6 * len(e) + 0.2, 2.6 * aspect + 0.6), "Re $E_z e^{i\\varphi}$"
    wl = f["wavelengths_nm"]
    rows, cols = (len(e), 1) if name == "metalens" else (1, len(e))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=110, squeeze=False)
    axes = axes.ravel()
    meshes = []
    for k, ax in enumerate(axes):
        vmax = np.percentile(np.abs(e[k]), 99.7)
        meshes.append(ax.pcolormesh(x, y, e[k].real.T, cmap=FIELD, vmin=-vmax, vmax=vmax,
                                    shading="auto"))
        _outline(ax, x, y, eps, INK, lw=0.5)
        if name != "metalens":
            draw_ports(ax, f["ports"], arrows=False)
        ax.set_aspect("equal")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        _bare(ax)
        ax.set_title(f"λ = {wl[k]:.0f} nm", fontsize=8)
    fig.suptitle(f"{short_name(name)} — {label}", fontsize=9)
    fig.tight_layout()
    frames = []
    for phi in np.linspace(0, 2 * np.pi, n_phase, endpoint=False):
        for k, m in enumerate(meshes):
            m.set_array((e[k] * np.exp(1j * phi)).real.T.ravel())
        frames.append(_frame(fig))
    plt.close(fig)
    save_gif(frames, path, fps=fps, hold_last=1 / fps)


# ---------------------------------------------------------------------------
# (e) S-parameter spectra


def sparams_figure(name, f, path, summary=None):
    """|S_j1|^2 [dB] on the dense grid of the broadband run, the target
    windows at the challenge wavelengths and the official values."""
    set_style()
    ch = gym_challenge(name)
    wl = f["wavelengths_nm"]
    dense = f["dense_wavelengths_nm"]
    S2 = np.abs(f["dense_S"]) ** 2
    nport = S2.shape[1]
    from invrs_gym.challenges.ceviche.challenge import _wavelength_bound

    lb = np.asarray(_wavelength_bound(ch.transmission_lower_bound, (len(wl), 1, nport)))[:, 0]
    ub = np.asarray(_wavelength_bound(ch.transmission_upper_bound, (len(wl), 1, nport)))[:, 0]
    floor = -40.0
    db = lambda v: 10 * np.log10(np.maximum(v, 10 ** (floor / 10)))  # noqa: E731
    colors = [SOLID, PORT, "#2e9e5b", "#d08a1c"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=150)
    for j in range(nport):
        c = colors[j % len(colors)]
        ax.plot(dense, db(S2[:, j]), color=c, lw=1.4, label=f"$|S_{{{j + 1}1}}|^2$")
        for k, w in enumerate(wl):
            xw = w + (j - (nport - 1) / 2) * 1.6
            lo_db, hi_db = db(lb[k, j]), db(ub[k, j])
            ax.plot([xw, xw], [lo_db, hi_db], color=c, lw=4, alpha=0.35, solid_capstyle="butt")
            ax.plot([xw - 0.6, xw + 0.6], [hi_db, hi_db], color=c, lw=1.2)
            if lo_db > floor:
                ax.plot([xw - 0.6, xw + 0.6], [lo_db, lo_db], color=c, lw=1.2)
        if summary and "official_S2" in summary:
            off = np.asarray(summary["official_S2"])[:, 0, j]
            ax.plot(wl, db(off), "x", color=c, ms=5, mew=1.2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (dB)")
    ax.set_ylim(floor, 1)
    ax.set_xlim(dense.min(), dense.max())
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, ncol=nport, loc="lower center")
    t = f"{short_name(name)}: Wakis broadband spectrum (one run)"
    if summary:
        t += f"\nbars: target windows · ×: official ceviche · eval = {summary['official_eval_metric']:+.3f}"
    ax.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# run rendering and gallery


def design_figure(name, design, path, summary=None):
    """Static figure of the final design (ceviche: with its surroundings)."""
    set_style()
    metric = summary.get("official_eval_metric") if summary else None
    if name == "metalens":
        fig, ax = plt.subplots(figsize=(7.2, 1.3), dpi=150)
        ax.imshow(design.T, cmap=MATERIAL, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=[0, 16, 0, design.shape[1] * 0.02])
        _bare(ax)
        scale_bar(ax, 2, 0.3, 0.05, color=INK)
    else:
        model = gym_challenge(name).component.ceviche_model
        fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=150)
        draw_ceviche_density(ax, model, design, ports=_ports_from_model(model),
                             pml_um=model.pml_width * model.dl / 1e-6)
        ext = ceviche_extent_um(model)
        scale_bar(ax, 1, ext[0] + 0.3, ext[2] + 0.12)
    title = short_name(name)
    if metric is not None:
        title += f"  ·  official eval metric {metric:+.3f}"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_run(run_dir, name=None):
    """All figures of one run directory, written next to its data. Each
    figure is attempted independently; returns {file: 'ok' or error}."""
    run = pathlib.Path(run_dir)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    name = name or summary.get("challenge")
    jobs = []
    if (run / "design.npy").exists():
        design = np.load(run / "design.npy").astype(float)
        jobs.append(("design.png", lambda: design_figure(name, design, run / "design.png", summary)))
    if (run / "snapshots.npz").exists():
        snaps = dict(np.load(run / "snapshots.npz"))
        jobs.append(("optimization.gif",
                     lambda: design_evolution_gif(name, snaps, run / "optimization.gif")))
    if (run / "fields.npz").exists():
        f = dict(np.load(run / "fields.npz"))
        if name == "metalens":
            jobs.append(("fields.png", lambda: metalens_field_figure(f, run / "fields.png", summary)))
        else:
            jobs.append(("fields.png", lambda: ceviche_field_figure(name, f, run / "fields.png")))
            jobs.append(("spectrum.png",
                         lambda: sparams_figure(name, f, run / "spectrum.png", summary)))
        jobs.append(("wave.gif", lambda: phase_gif(name, f, run / "wave.gif")))
    status = {}
    for fname, job in jobs:
        try:
            job()
            status[fname] = "ok"
        except Exception as err:  # noqa: BLE001 - other figures still render
            status[fname] = f"{type(err).__name__}: {err}"
            plt.close("all")
    return status


def gallery(results_dir, path):
    """One column per challenge: final design (top), field intensity (bottom)."""
    set_style()
    runs = []
    for s in sorted(pathlib.Path(results_dir).glob("**/summary.json")):
        summ = json.loads(s.read_text())
        if (s.parent / "fields.npz").exists():
            runs.append((summ, s.parent))
    best = {}
    for summ, d in runs:
        c = summ["challenge"]
        if summ.get("official_eval_metric") is None:
            continue
        if c not in best or summ["official_eval_metric"] > best[c][0]["official_eval_metric"]:
            best[c] = (summ, d)
    order = [c for c in SHORT if c in best] + [c for c in best if c not in SHORT]
    n = len(order)
    if n == 0:
        return None
    widths = [2.0 if c == "metalens" else 1.0 for c in order]
    fig, axes = plt.subplots(2, n, figsize=(2.3 * sum(widths) + 0.4, 4.9), dpi=150,
                             squeeze=False, gridspec_kw=dict(width_ratios=widths))
    for k, c in enumerate(order):
        summ, d = best[c]
        f = dict(np.load(d / "fields.npz"))
        design = np.load(d / "design.npy")
        ax0, ax1 = axes[0, k], axes[1, k]
        if c == "metalens":
            x, y, eps = f["x_um"], f["y_um"], f["eps"]
            i = len(f["wavelengths_nm"]) // 2
            sel = (x > 2.5) & (x < 13.5)
            ax0.imshow(design.T, cmap=MATERIAL, vmin=0, vmax=1, aspect=4,
                       extent=[0, 16, 0, design.shape[1] * 0.02], interpolation="nearest")
            ax0.set_xlim(2.5, 13.5)
            enh = f["enhancement"][i][sel]
            ax1.pcolormesh(x[sel], y, enh.T, cmap=INTENSITY, shading="auto", rasterized=True)
            _outline(ax1, x[sel], y, eps[sel], "white", lw=0.4)
            ax1.plot(*f["focus_um"], marker="o", mfc="none", mec=PORT, ms=5)
            ax1.set_aspect("equal")
            metric = f"min enhancement {summ['official_eval_metric']:.2f}\n(height ×4)"
        else:
            model = gym_challenge(c).component.ceviche_model
            draw_ceviche_density(ax0, model, design, ports=f["ports"],
                                 pml_um=float(f["pml_um"]))
            ax0.set_aspect("equal")
            sx, sy = _crop(f, float(f["pml_um"]))
            x, y = f["x_um"][sx], f["y_um"][sy]
            ez = f["Ez"][0][sx][:, sy]
            ax1.pcolormesh(x, y, ez.real.T, cmap=FIELD, shading="auto", rasterized=True,
                           vmin=-np.percentile(np.abs(ez), 99.7), vmax=np.percentile(np.abs(ez), 99.7))
            _outline(ax1, x, y, f["eps"][np.ix_(sx, sy)], INK, lw=0.5)
            ax1.set_aspect("equal")
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(y.min(), y.max())
            metric = f"eval metric {summ['official_eval_metric']:+.3f}"
        _bare(ax0)
        _bare(ax1)
        ax0.set_title(f"{short_name(c)}\n{metric}", fontsize=8.5)
    fig.suptitle("invrs-gym challenges designed with Wakis (FIT time domain + exact adjoint)\n"
                 "eval metrics from the official gym backends (ceviche FDFD, fmmax)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
