"""
Figures comparing the meshes used by ceviche (FDFD), fmmax (RCWA) and Wakis
(FIT) for two invrs-gym challenges: the lightweight waveguide bend and the
metalens. Geometry only, no simulation.

Usage: uv run python scripts/mesh_figures.py --outdir <dir> [--cells-per-wavelength 20]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from wakis.templates.meshing import graded_axis

NM = 1e-9
UM = 1e-6


def faces(mask_1d_changes, h):
    """Coordinates of material interfaces from a 1D boolean change array."""
    return np.flatnonzero(mask_1d_changes) * h


def graded_mesh(points, eps_strip, lam_min, n_cells, grading):
    sizes = [lam_min / (np.sqrt(e) * n_cells) for e in eps_strip]
    return graded_axis(points, sizes, min_cells=1, grading=grading)


def cells_per_wavelength(x, y, eps_xy, lam_min):
    """lambda / (n max(hx, hy)) on each cell of a tensor mesh."""
    hx, hy = np.diff(x), np.diff(y)
    h = np.maximum(hx[:, None], hy[None, :])
    return lam_min / (np.sqrt(eps_xy) * h)


def strip_eps(points, coords, eps_line_max):
    """Max permittivity over each interval of an axis (any position across)."""
    out = []
    for a, b in zip(points[:-1], points[1:]):
        sel = (coords >= a - 1e-15) & (coords < b - 1e-15)
        out.append(eps_line_max[sel].max() if sel.any() else eps_line_max.min())
    return out


def draw_mesh(ax, x, y, color="k", lw=0.25, alpha=0.6):
    ax.vlines(x, y[0], y[-1], color=color, lw=lw, alpha=alpha)
    ax.hlines(y, x[0], x[-1], color=color, lw=lw, alpha=alpha)


# ----------------------------------------------------------------------------
# Ceviche lightweight waveguide bend
# ----------------------------------------------------------------------------


def bend_meshes(cpw, grading):
    import invrs_gym

    model = invrs_gym.challenges.ceviche_lightweight_waveguide_bend().component.ceviche_model
    dl = float(model.dl)
    nx, ny = (int(n) for n in model.shape)
    eps_clad, eps_si = float(model.cladding_permittivity), float(model.slab_permittivity)
    rho_bg = np.asarray(model.density_bg, dtype=float)
    x0, y0, x1, y1 = (int(v) for v in model.design_region_coords)
    design = np.zeros((nx, ny), bool)
    design[x0:x1, y0:y1] = True
    eps = np.where((rho_bg > 0.5) | design, eps_si, eps_clad)  # design may hold Si anywhere
    n_pml = int(model.pml_width)
    lam_min = 1270 * NM

    xu = np.arange(nx + 1) * dl
    yu = np.arange(ny + 1) * dl
    xc = (np.arange(nx) + 0.5) * dl
    yc = (np.arange(ny) + 0.5) * dl

    # Graded FIT mesh: nodes on every interface, the design box and the PML
    px = np.unique(np.r_[0, nx, x0, x1, n_pml, nx - n_pml,
                         np.flatnonzero(np.diff(eps, axis=0).any(axis=1)) + 1]) * dl
    py = np.unique(np.r_[0, ny, y0, y1, n_pml, ny - n_pml,
                         np.flatnonzero(np.diff(eps, axis=1).any(axis=0)) + 1]) * dl
    xg = graded_mesh(px, strip_eps(px, xc, eps.max(axis=1)), lam_min, cpw, grading)
    yg = graded_mesh(py, strip_eps(py, yc, eps.max(axis=0)), lam_min, cpw, grading)

    def eps_on(x, y):
        ix = np.clip(((x[:-1] + x[1:]) / 2 / dl).astype(int), 0, nx - 1)
        iy = np.clip(((y[:-1] + y[1:]) / 2 / dl).astype(int), 0, ny - 1)
        return eps[np.ix_(ix, iy)]

    return dict(dl=dl, nx=nx, ny=ny, eps=eps, design=(x0, y0, x1, y1), n_pml=n_pml,
                lam_min=lam_min, eps_si=eps_si, eps_clad=eps_clad,
                uniform40=(xu, yu), uniform10=(np.arange(4 * nx + 1) * dl / 4,
                                               np.arange(4 * ny + 1) * dl / 4),
                graded=(xg, yg), eps_on=eps_on)


def plot_bend(m, outdir, cpw):
    dl, nx, ny = m["dl"], m["nx"], m["ny"]
    ext = [0, nx * dl / UM, 0, ny * dl / UM]
    x0, y0, x1, y1 = m["design"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    meshes = [("ceviche FDFD / Wakis bridge (uniform 40 nm)", m["uniform40"]),
              ("ceviche standard resolution (uniform 10 nm)", m["uniform10"]),
              (f"Wakis FIT graded mesh ({cpw} cells/λ in each material)", m["graded"])]
    cmin = None
    for col, (title, (x, y)) in enumerate(meshes):
        ax = axes[0, col]
        ax.imshow(m["eps"].T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=16, alpha=0.55)
        step = 4 if len(x) > 300 else 1  # draw every 4th line of the 10 nm grid
        draw_mesh(ax, x[::step] / UM, y[::step] / UM, lw=0.3)
        ax.add_patch(Rectangle((x0 * dl / UM, y0 * dl / UM), (x1 - x0) * dl / UM,
                               (y1 - y0) * dl / UM, fill=False, ec="tab:blue", lw=1.5, ls="--"))
        p = m["n_pml"] * dl / UM
        for r in [(0, 0, p, ext[3]), (ext[1] - p, 0, p, ext[3]), (0, 0, ext[1], p), (0, ext[3] - p, ext[1], p)]:
            ax.add_patch(Rectangle(r[:2], r[2], r[3], fc="tab:orange", alpha=0.12, ec="none"))
        ncell = (len(x) - 1) * (len(y) - 1)
        hmin = min(np.diff(x).min(), np.diff(y).min())
        ax.set_title(f"{title}\n{ncell:,} cells, min cell {hmin / NM:.0f} nm"
                     + (" (every 4th line drawn)" if step > 1 else ""), fontsize=10)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)"); ax.set_aspect("equal")

        cpwmap = cells_per_wavelength(x, y, m["eps_on"](x, y), m["lam_min"])
        ax2 = axes[1, col]
        im = ax2.pcolormesh(x / UM, y / UM, cpwmap.T, cmap="viridis", norm=LogNorm(4, 80), shading="flat")
        ax2.contour((np.arange(nx) + 0.5) * dl / UM, (np.arange(ny) + 0.5) * dl / UM,
                    m["eps"].T, levels=[(m["eps_si"] + m["eps_clad"]) / 2], colors="w", linewidths=0.6)
        si_min = cpwmap[m["eps_on"](x, y) == m["eps_si"]].min()
        ax2.set_title(f"cells per wavelength at 1270 nm (min in Si: {si_min:.1f})", fontsize=10)
        ax2.set_xlabel("x (µm)"); ax2.set_ylabel("y (µm)"); ax2.set_aspect("equal")
    fig.colorbar(im, ax=axes[1, :], shrink=0.8, label="λ / (n · cell size)")
    fig.suptitle("Ceviche lightweight waveguide bend: simulation meshes "
                 "(grey: Si/design region; dashed: design box; orange: PML)", fontsize=12)
    path = os.path.join(outdir, "mesh_ceviche_bend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Zoom on a corner of the design region: design pixels vs mesh lines
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    zx = (x0 * dl - 0.3 * UM, x0 * dl + 0.5 * UM)
    zy = (y0 * dl - 0.3 * UM, y0 * dl + 0.5 * UM)
    for ax, (title, (x, y)) in zip(axes, [meshes[0], meshes[2]]):
        ax.imshow(m["eps"].T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=16, alpha=0.5)
        sx = x[(x >= zx[0]) & (x <= zx[1])]
        sy = y[(y >= zy[0]) & (y <= zy[1])]
        ax.vlines(sx / UM, zy[0] / UM, zy[1] / UM, color="k", lw=0.8)
        ax.hlines(sy / UM, zx[0] / UM, zx[1] / UM, color="k", lw=0.8)
        gx = np.arange(nx + 1) * dl
        gx = gx[(gx >= zx[0]) & (gx <= zx[1])]
        ax.vlines(gx / UM, zy[0] / UM, zy[1] / UM, color="tab:red", lw=1.2, ls=":", label="40 nm design pixels")
        ax.hlines(gx[:0], 0, 1)
        ax.add_patch(Rectangle((x0 * dl / UM, y0 * dl / UM), 5, 5, fill=False, ec="tab:blue", lw=2, ls="--"))
        ax.set_xlim(zx[0] / UM, zx[1] / UM); ax.set_ylim(zy[0] / UM, zy[1] / UM)
        ax.set_title(title, fontsize=10); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Zoom at the design-region corner: the FIT mesh need not coincide with the design pixels\n"
                 "(the design density is interpolated onto the mesh edges by MaterialGrid)", fontsize=11)
    zpath = os.path.join(outdir, "mesh_ceviche_bend_zoom.png")
    fig.savefig(zpath, dpi=150)
    plt.close(fig)
    return path, zpath


# ----------------------------------------------------------------------------
# Metalens
# ----------------------------------------------------------------------------


def metalens_meshes(cpw, grading):
    from invrs_gym.challenges.metalens import challenge as mc

    s = mc.METALENS_SPEC
    g = s.grid_spacing * UM
    w_lens = s.width_lens * UM
    off = s.pml_lens_offset * UM
    period = w_lens + 2 * off
    t_sub, t_lens, t_amb = s.thickness_substrate * UM, s.thickness_lens * UM, s.thickness_ambient * UM
    focus = s.focus_offset * UM
    eps_hi = float(np.real(s.permittivity_metalens))
    lam_min = 450 * NM
    # z layout (Wakis): substrate | lens | air up to the focus + margin
    z = np.array([0, t_sub, t_sub + t_lens, t_sub + t_lens + focus + 0.5 * UM])
    eps_z = [eps_hi, eps_hi, 1.0]  # lens layer may be fully TiO2-like
    x_pts = np.array([0, off, off + w_lens, period])
    xg = graded_mesh(x_pts, [eps_hi] * 3, lam_min, cpw, grading)  # substrate spans all x
    zg = graded_mesh(z, eps_z, lam_min, cpw, grading)
    xu, zu = np.arange(0, period + g / 2, g), np.arange(0, z[-1] + g / 2, g)
    return dict(g=g, period=period, z=z, eps_hi=eps_hi, lam_min=lam_min, off=off, w_lens=w_lens,
                t_sub=t_sub, t_lens=t_lens, focus=focus, uniform=(xu, zu), graded=(xg, zg),
                n_terms=s.approximate_num_terms if hasattr(s, "approximate_num_terms") else 280)


def plot_metalens(m, outdir, cpw):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), constrained_layout=True)
    P, z = m["period"] / UM, m["z"] / UM
    for ax in axes:
        ax.add_patch(Rectangle((0, z[0]), P, z[1] - z[0], fc="tab:purple", alpha=0.18, ec="none"))
        ax.add_patch(Rectangle((m["off"] / UM, z[1]), m["w_lens"] / UM, z[2] - z[1],
                               fc="tab:purple", alpha=0.35, ec="none"))
        ax.plot([P / 2], [(m["t_sub"] + m["t_lens"] + m["focus"]) / UM], "r*", ms=12)
        ax.set_xlim(0, P); ax.set_ylim(z[0], z[-1]); ax.set_xlabel("x (µm)"); ax.set_ylabel("z (µm)")

    # (a) fmmax: no in-plane mesh, Fourier harmonics; z: layer slices
    ax = axes[0]
    n_slices = int(round(m["t_lens"] / m["g"]))
    for k in range(n_slices + 1):
        ax.axhline(z[1] + k * m["g"] / UM, color="k", lw=0.3)
    ax.axhline(z[1], color="k", lw=1.2); ax.axhline(z[2], color="k", lw=1.2)
    xs = np.linspace(0, P, 2000)
    for kk, c in [(1, "tab:blue"), (5, "tab:green"), (20, "tab:red")]:
        ax.plot(xs, z[3] - 0.6 + 0.25 * np.cos(2 * np.pi * kk * xs / P) - 0.3 * (kk > 1) - 0.3 * (kk > 5), c, lw=0.8)
    ax.text(0.3, z[3] - 0.15, f"in-plane: ~{m['n_terms']} Fourier harmonics (no mesh)", fontsize=9)
    ax.text(0.3, (z[1] + z[2]) / 2 + 0.6, f"lens: {n_slices} layers of {m['g'] / NM:.0f} nm\n"
            "uniform layers: analytic propagation in z", fontsize=9)
    ax.set_title("fmmax RCWA (gym reference)", fontsize=11)

    for ax, (title, (x, zz)) in zip(axes[1:], [("Wakis bridge (uniform 20 nm)", m["uniform"]),
                                              (f"Wakis FIT graded ({cpw} cells/λ at 450 nm)", m["graded"])]):
        step = max(1, len(x) // 160)
        draw_mesh(ax, x[::step] / UM, zz[::max(1, len(zz) // 160)] / UM, lw=0.3)
        hz = np.diff(zz)
        n_hi = 450 / np.sqrt(m["eps_hi"]) / (max(np.diff(x).max(), hz[zz[:-1] < m["z"][2]].max()) / NM)
        n_air = 450 / (hz[zz[:-1] >= m["z"][2]].max() / NM)
        ncell = (len(x) - 1) * (len(zz) - 1)
        ax.set_title(f"{title}\n{ncell:,} cells; cells/λ: {n_hi:.1f} in n=2.4, {n_air:.0f} in air"
                     + (f" (every {step}th line)" if step > 1 else ""), fontsize=10)
    fig.suptitle("Metalens (10 µm lens, n = 2.4, 450–650 nm): purple = high index, red star = focus",
                 fontsize=12)
    path = os.path.join(outdir, "mesh_metalens.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def summary(bend, lens, outdir, cpw):
    rows = []
    h_si = bend["lam_min"] / (np.sqrt(bend["eps_si"]) * cpw)
    Lx, Ly = bend["nx"] * bend["dl"], bend["ny"] * bend["dl"]
    nu = int(np.ceil(Lx / h_si)), int(np.ceil(Ly / h_si))
    eq_bend = (np.linspace(0, Lx, nu[0] + 1), np.linspace(0, Ly, nu[1] + 1))
    for name, (x, y) in [("ceviche lightweight, uniform 40 nm", bend["uniform40"]),
                         ("ceviche standard, uniform 10 nm", bend["uniform10"]),
                         ("Wakis bridge as run (= 40 nm)", bend["uniform40"]),
                         (f"uniform at {cpw} cells/λ in Si", eq_bend),
                         (f"Wakis graded FIT, {cpw} cells/λ", bend["graded"])]:
        e = bend["eps_on"](x, y)
        cp = cells_per_wavelength(x, y, e, bend["lam_min"])
        ncell = (len(x) - 1) * (len(y) - 1)
        hmin = min(np.diff(x).min(), np.diff(y).min())
        rows.append(("bend", name, ncell, cp[e == bend["eps_si"]].min(), ncell / hmin))

    def hi_cpw(x, zz):
        hz = np.diff(zz)[zz[:-1] < lens["z"][2]]  # substrate + lens layers
        return 450 / np.sqrt(lens["eps_hi"]) / (max(np.diff(x).max(), hz.max()) / NM)

    h_hi = lens["lam_min"] / (np.sqrt(lens["eps_hi"]) * cpw)
    P, Z = lens["period"], lens["z"][-1]
    eq_lens = (np.linspace(0, P, int(np.ceil(P / h_hi)) + 1), np.linspace(0, Z, int(np.ceil(Z / h_hi)) + 1))
    for name, (x, zz) in [("Wakis bridge as run (uniform 20 nm)", lens["uniform"]),
                          (f"uniform at {cpw} cells/λ in n=2.4", eq_lens),
                          (f"Wakis graded FIT, {cpw} cells/λ", lens["graded"])]:
        ncell = (len(x) - 1) * (len(zz) - 1)
        hmin = min(np.diff(x).min(), np.diff(zz).min())
        rows.append(("metalens", name, ncell, hi_cpw(x, zz), ncell / hmin))
    ref = {"bend": rows[0][4], "metalens": rows[5][4]}
    lines = ["| challenge | mesh | cells | min cells/λ in high index | relative cost per run* |",
             "|---|---|---|---|---|"]
    for ch, name, n, hi, cost in rows:
        lines.append(f"| {ch} | {name} | {n:,} | {hi:.1f} | {cost / ref[ch]:.2f} |")
    lines.append("\n*cells × time steps (∝ cells / smallest cell, explicit CFL limit), relative to the "
                 "first mesh of each challenge. fmmax (metalens reference) uses ~280 Fourier terms × 50 lens "
                 "layers instead of a real-space mesh.")
    text = "\n".join(lines)
    open(os.path.join(outdir, "mesh_summary.md"), "w").write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cells-per-wavelength", type=int, default=20)
    ap.add_argument("--grading", type=float, default=1.3)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    bend = bend_meshes(a.cells_per_wavelength, a.grading)
    print(plot_bend(bend, a.outdir, a.cells_per_wavelength))
    lens = metalens_meshes(a.cells_per_wavelength, a.grading)
    print(plot_metalens(lens, a.outdir, a.cells_per_wavelength))
    print(summary(bend, lens, a.outdir, a.cells_per_wavelength))


if __name__ == "__main__":
    main()
