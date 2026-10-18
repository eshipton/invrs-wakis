"""Field maps and dense spectra of a final design (one extra forward run).

A single broadband Wakis run gives, at once, the frequency-domain fields over
the whole 2D domain at the challenge wavelengths (DFT monitors) and, for the
ceviche challenges, the S-parameter spectrum on a dense wavelength grid.
"""

import numpy as np
import wakis
from invrs_gym import utils as gym_utils
from scipy.constants import epsilon_0 as EPS0
from wakis.inverse import AdjointSimulation

from invrs_wakis.ceviche import CevicheSimulation

UM = 1e-6


def _density(params):
    return np.asarray(
        gym_utils.transforms.rescaled_density_array(params, lower_bound=0.0, upper_bound=1.0)
    )


def ceviche_fields(challenge, params, margin_nm=20.0, n_dense=61):
    """
    Ez maps at the challenge wavelengths and the dense |S_j1|^2 spectrum.

    Returns
    -------
    dict of ndarray
        'wavelengths_nm', 'Ez' (n_wl, nx, ny) complex64 normalized to unit
        incident mode amplitude, 'x_um', 'y_um' (Ez sample coordinates),
        'eps' (relative permittivity at the Ez samples), 'dense_wavelengths_nm',
        'dense_S' (n_dense, n_ports) complex, 'pml_um', 'ports' (n_ports, 5:
        x, y [um] of the port centre, width [um], axis 0/1, direction sign),
        'design_um' (x0, y0, x1, y1).
    """
    comp = challenge.component
    g = comp.geometry
    model = comp.ceviche_model
    wl = np.asarray(model.output_wavelengths, dtype=float)
    dense = np.linspace(wl.min() - margin_nm, wl.max() + margin_nm, n_dense)
    dense = np.unique(np.concatenate([dense, wl]))
    spectrum = CevicheSimulation(g, 0, dense)
    _, f_sim = g.frequencies(wl)
    nx, ny = g.shape
    lx, ly = nx * g.dl, ny * g.dl
    field = wakis.DFTMonitor(
        f_sim, components=("Ez",), center=(lx / 2 + g.h / 2, ly / 2 + g.h / 2, 0.0),
        size=(lx, ly, 0.0),
    )
    sim = AdjointSimulation(
        g.make_solver, spectrum.sim.sources, {**spectrum.sim.monitors, "field": field},
        spectrum.design, spectrum.Nt, store_dtype=np.float32,
    )  # fmt: skip
    sim.output_shapes()
    spectrum.sim = sim
    rho = _density(params)
    raw = sim.forward(rho)
    a, b = (np.asarray(v) for v in spectrum.amplitudes(raw))
    S = b / a[:, :1]
    idx = [int(np.argmin(np.abs(dense - w))) for w in wl]
    mon = sim._monitors["field"]
    ez, (xs, ys, _) = mon.get_field("Ez")
    ez = ez[..., 0] / a[idx, 0][:, None, None]
    ieps = sim._solver.ieps.array[np.asarray(mon._flat["Ez"])]
    with np.errstate(divide="ignore"):
        eps = (1 / (EPS0 * ieps)).reshape(len(xs), len(ys))
    ports = []
    for p in model.ports:
        center, size = g.port_plane(p)
        axis = 0 if p.dir.is_along_x else 1
        ports.append([center[0] / UM, center[1] / UM, p.width * g.dl / UM, axis, p.dir.sign])
    x0, y0, x1, y1 = g.design_coords
    return {
        "wavelengths_nm": wl,
        "Ez": ez.astype(np.complex64),
        "x_um": xs / UM,
        "y_um": ys / UM,
        "eps": eps.astype(np.float32),
        "dense_wavelengths_nm": dense,
        "dense_S": S.astype(np.complex64),
        "pml_um": np.float64(g.n_pml * g.h / UM),
        "ports": np.asarray(ports),
        "design_um": np.asarray([x0, y0, x1, y1], float) * g.dl / UM,
    }


def metalens_fields(challenge, params, above_um=0.4):
    """
    In-plane field maps of the final metalens (EX polarization).

    Returns
    -------
    dict of ndarray
        'wavelengths_nm', 'Ex' (n_wl, nx, ny) complex64 at the Ex samples
        (relative to the focal field without lens),
        'enhancement' (n_wl, nx, ny) = |E|^2 / |E_focus,no lens|^2 (the
        value at the focus is the challenge's enhancement), 'x_um', 'y_um',
        'eps' (pixel permittivity at the same points), 'focus_um',
        'lens_um' (x0, x1, y0, y1), 'source_y_um'.
    """
    comp = challenge.component
    rho = _density(params)
    msim = comp.simulation("ex", rho.shape)
    g = msim.geometry
    width = g.nx * g.g
    y_lo = g.y_source + 0.1 * UM
    y_hi = g.y_focus + above_um * UM
    field = wakis.DFTMonitor(
        msim.f_sim, components=("Ex", "Ey"),
        center=(width / 2 + g.h / 2, 0.5 * (y_lo + y_hi), 0.0), size=(width, y_hi - y_lo, 0.0),
    )  # fmt: skip
    sim = AdjointSimulation(
        g.make_solver, msim.sim.sources, {"field": field}, msim.design, msim.Nt,
        store_dtype=np.float32,
    )  # fmt: skip
    sim.forward(rho)
    mon = sim._monitors["field"]
    ex, (xe, ye, _) = mon.get_field("Ex")
    ey, (xy, yy, _) = mon.get_field("Ey")
    ex, ey = ex[..., 0], ey[..., 0]
    # Ey onto the Ex samples (x_{m+1/2}, y_k): mean of the 4 neighbours
    # (x_m, x_{m+1}) x (y_{k-1/2}, y_{k+1/2}); x is periodic
    n = min(ex.shape[2], ey.shape[2]) - 1
    eyc = 0.25 * (ey[:, :, :n] + ey[:, :, 1 : n + 1])
    eyc = eyc + np.roll(eyc, -1, axis=1)
    ex = ex[:, :, 1 : n + 1] if ye[0] < yy[0] else ex[:, :, :n]
    ye = ye[1 : n + 1] if ye[0] < yy[0] else ye[:n]
    intensity = (np.abs(ex) ** 2 + np.abs(eyc) ** 2) / msim.intensity_ref[:, None, None]
    density_bottom_up = rho[:, ::-1]
    eps_pix = g.pixel_eps(density_bottom_up)  # (nx, nrows + 2)
    col = np.clip((xe / g.g).astype(int), 0, g.nx - 1)
    row = np.clip(np.floor(ye / g.g).astype(int), -1, g.nrows) + 1
    eps = eps_pix[np.ix_(col, row)]
    return {
        "wavelengths_nm": g.wavelengths / 1e-9,
        "Ex": (ex / np.sqrt(msim.intensity_ref)[:, None, None]).astype(np.complex64),
        "enhancement": intensity.astype(np.float32),
        "x_um": xe / UM,
        "y_um": ye / UM,
        "eps": eps.astype(np.float32),
        "focus_um": np.asarray([g.x_focus, g.y_focus]) / UM,
        "lens_um": np.asarray(
            [g.spec.lens_offset, g.spec.width - g.spec.lens_offset, 0.0, g.nrows * g.g / UM]
        ),
        "source_y_um": np.float64(g.y_source / UM),
    }
