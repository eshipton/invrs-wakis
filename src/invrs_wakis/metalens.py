"""Wakis (FIT time-domain) backend for the invrs-gym metalens challenge.

The gym's metalens (the RGB metalens of Chen et al., photonics-opt-testbed) is
a 2D problem: a 10 um wide, 1 um thick lens on a substrate (eps = 2.4^2) in a
16 um periodic cell, illuminated from the substrate by a smoothed-top-hat beam
at 450, 550 and 650 nm. The figure of merit is the intensity at a point
2.4 um above the lens divided by the intensity there without the lens. The
gym solves it with fmmax (Fourier modal method); here the same quantity is
computed with one broadband Wakis run per polarization.

Wakis layout: x = gym x (periodic, 16 um), y = gym z reversed (height above
the density's bottom surface), one periodic cell along z (the gym's
invariant y). The gym's EX polarization (E in the plane of the lens
cross-section) is the (Ex, Ey, Hz) field family in Wakis, EY is (Ez, Hx, Hy).
Each density pixel (20 nm) is mapped onto `resolution`^2 Wakis cells; each E
edge takes the harmonic mean of the pixels it crosses (`PixelDesign`).
"""

import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wakis
from invrs_gym import challenges as gym_challenges
from invrs_gym import utils as gym_utils
from invrs_gym.challenges import base
from invrs_gym.challenges.metalens import component as gym_metalens
from scipy import ndimage
from scipy.constants import c as C0
from scipy.sparse import coo_matrix
from wakis.field_monitors import _flat_index
from wakis.inverse import AdjointSimulation, jax_simulation
from wakis.sources import CurrentSource, GaussianSource

from invrs_wakis._design import PixelDesign, check_smoothing

jax.config.update("jax_enable_x64", True)
UM = 1e-6


@dataclasses.dataclass(frozen=True)
class MetalensOptions:
    """
    Numerical settings of the Wakis metalens simulation.

    Attributes
    ----------
    resolution : int
        Wakis cells per 20 nm gym pixel.
    courant : float
        Time step as a fraction of the 2D stability limit.
    run_time : float
        Simulated time after the source peak [s].
    fwidth : float
        Spectral standard deviation of the pulse relative to its centre.
    pml : float
        CPML thickness [m].
    source_depth : float
        Depth of the source below the bottom of the density array [m] (the
        gym defines its incident field 1 um below the lens).
    margin : float
        Free space between the focal plane and the top CPML, and between the
        source and the bottom CPML [m].
    match_time_dispersion : bool
        Sample the run at the frequencies whose discrete-time angular
        frequency equals the target one (removes leapfrog temporal
        dispersion).
    polarizations : tuple of str
        Polarizations simulated ('ex', 'ey'); enhancements of the others are
        returned as NaN.
    store_dtype : numpy dtype
        Precision of the stored adjoint increments.
    smoothing : {'none', 'tensor'}
        Material smoothing of the design ('tensor': Wakis MaterialGrid
        subpixel averaging, see `invrs_wakis._design.check_smoothing`).
    use_gpu : bool
        Run the Wakis solver (forward and adjoint) on the GPU (CuPy).
    """

    resolution: int = 1
    courant: float = 0.95
    run_time: float = 150e-15
    fwidth: float = 0.25
    pml: float = 0.6 * UM
    source_depth: float = 1.0 * UM
    margin: float = 0.4 * UM
    match_time_dispersion: bool = True
    polarizations: Tuple[str, ...] = ("ex",)
    store_dtype: object = np.float32
    smoothing: str = "none"
    use_gpu: bool = False


class MetalensGeometry:
    def __init__(self, spec: gym_metalens.MetalensSpec, wavelengths_um, density_shape,
                 options: MetalensOptions):
        if spec.width_pml != 0:
            raise NotImplementedError("Only the periodic (width_pml = 0) metalens is supported.")
        check_smoothing(options.smoothing)
        if options.smoothing != "none":
            raise NotImplementedError("Tensor smoothing is wired for the ceviche bridge only.")
        self.spec, self.options = spec, options
        self.r = r = int(options.resolution)
        self.g = spec.grid_spacing * UM
        self.h = self.g / r
        self.nx, self.nrows = density_shape  # (800, 50): row 0 at the top
        assert abs(self.nx * self.g - spec.width * UM) < 1e-12
        self.eps_amb = float(np.real(spec.permittivity_ambient))
        self.eps_lens = float(np.real(spec.permittivity_metalens))
        self.eps_sub = float(np.real(spec.permittivity_substrate))
        h = self.h
        top = self.nrows * self.g
        self.y_focus = top + spec.focus_offset * UM
        self.x_focus = 0.5 * spec.width * UM
        self.y_source = -options.source_depth
        n_pml = int(round(options.pml / h))
        self.n_pml = n_pml
        # y nodes at (q + 1/2) h, q = fine row index (q = 0: bottom fine row of
        # the density); n_below fine rows under the density
        self.n_below = int(np.ceil((options.source_depth + options.margin) / h)) + n_pml
        n_above = int(np.ceil((self.y_focus - top + options.margin) / h)) + n_pml
        self.ny_cells = self.n_below + self.nrows * r + n_above
        self.ymin = -self.n_below * h + h / 2
        self.wavelengths = np.asarray(wavelengths_um, float) * UM
        self.dt = options.courant * h / (C0 * np.sqrt(2.0))
        # Coarse column of each x node (periodic: N nodes) and height row of
        # each y node (-1: substrate, nrows: ambient; else density row index
        # counted from the bottom)
        self.col = np.arange(self.nx * r) // r
        q = np.arange(self.ny_cells + 1) - self.n_below
        self.hrow = np.clip(np.floor_divide(q, r), -1, self.nrows)

    def make_grid(self):
        h = self.h
        return wakis.GridFIT3D(
            h / 2, self.nx * self.g + h / 2, self.ymin, self.ymin + self.ny_cells * h,
            -h / 2, h / 2, self.nx * self.r, self.ny_cells, 1, verbose=0,
        )  # fmt: skip

    def pixel_eps(self, density_rows_bottom_up):
        """eps on the (column, height row) grid incl. substrate/ambient rows."""
        d = density_rows_bottom_up
        eps = np.empty((self.nx, self.nrows + 2))
        eps[:, 0] = self.eps_sub
        eps[:, -1] = self.eps_amb
        eps[:, 1:-1] = self.eps_amb + (self.eps_lens - self.eps_amb) * d
        return eps

    def _edge_pixels(self, solver, comp):
        """Field.array indices of all `comp` edges and the (col, hrow+1) pixel
        pairs they cross (equal pairs for edges inside one pixel)."""
        Nx, Ny = solver.Nx, solver.Ny
        I, J = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        flat = _flat_index(solver, comp, I, J, np.zeros_like(I))
        c0 = self.col[I]
        r0 = self.hrow[J] + 1
        if comp == "x":
            c1, r1 = self.col[(I + 1) % Nx], r0
        elif comp == "y":
            c1, r1 = c0, self.hrow[np.minimum(J + 1, Ny - 1)] + 1
        else:
            c1, r1 = c0, r0
        return flat.ravel(), (c0.ravel(), r0.ravel()), (c1.ravel(), r1.ravel())

    def make_solver(self, density_rows_bottom_up=None):
        solver = wakis.SolverFIT3D(
            self.make_grid(),
            bc_low=["periodic", "cpml", "periodic"],
            bc_high=["periodic", "cpml", "periodic"],
            n_pml=self.n_pml,
            dt=self.dt,
            use_gpu=self.options.use_gpu,
            verbose=0,
        )
        xp = solver.E.xp
        d = np.zeros((self.nx, self.nrows)) if density_rows_bottom_up is None else density_rows_bottom_up
        eps = self.pixel_eps(d)
        for comp in "xyz":
            flat, p0, p1 = self._edge_pixels(solver, comp)
            ieps_rel = 0.5 / eps[p0] + 0.5 / eps[p1]
            flat = xp.asarray(flat)
            solver.ieps.array[flat] = solver.ieps.array[flat] * xp.asarray(ieps_rel)
        return solver

    def design(self, comps):
        """`PixelDesign` for the density (gym orientation, row 0 at the top)."""
        nx, nrows = self.nx, self.nrows

        def build(solver):
            flats, rows, cols, vals, consts = [], [], [], [], []
            offset = 0
            for comp in comps:
                flat, p0, p1 = self._edge_pixels(solver, comp)
                touch = np.zeros(flat.size, bool)
                c_e = np.zeros(flat.size)
                for (cc, rr) in (p0, p1):
                    inside = (rr >= 1) & (rr <= nrows)
                    touch |= inside
                for (cc, rr) in (p0, p1):
                    inside = (rr >= 1) & (rr <= nrows)
                    # gym pixel index: column cc, row from the top = nrows - rr
                    pix = cc * nrows + (nrows - rr)
                    sel = touch & inside
                    rows.append(offset + np.flatnonzero(sel[touch]))
                    cols.append(pix[sel])
                    vals.append(np.full(sel.sum(), 0.5))
                    out = touch & ~inside
                    eps_out = np.where(rr == 0, self.eps_sub, self.eps_amb)
                    c_e[out] += 0.5 / eps_out[out]
                flats.append(flat[touch])
                consts.append(c_e[touch])
                offset += touch.sum()
            W = coo_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(offset, nx * nrows),
            ).tocsr()
            return np.concatenate(flats), W, np.concatenate(consts)

        return PixelDesign((nx, nrows), self.eps_amb, self.eps_lens, build)

    def frequencies(self):
        f = C0 / self.wavelengths
        if not self.options.match_time_dispersion:
            return f
        return np.arcsin(np.pi * f * self.dt) / (np.pi * self.dt)

    def source_profile(self, x):
        """The gym's smoothed top-hat incident profile, interpolated at x [m]."""
        spec = self.spec
        xs = (np.arange(0.5, self.nx) / self.nx) * spec.width
        prof = ((xs > spec.source_offset) & (xs < spec.width - spec.source_offset)).astype(float)
        sigma = spec.source_smoothing_fwhm / (2 * np.sqrt(2 * np.log(2)))
        prof = ndimage.gaussian_filter1d(prof, sigma=sigma / spec.grid_spacing)
        return np.interp(np.asarray(x) / UM, xs, prof, period=spec.width)


class MetalensSimulation:
    def __init__(self, geometry: MetalensGeometry, polarization: str):
        """Differentiable focal-intensity spectrum for one polarization."""
        g = self.geometry = geometry
        self.polarization = polarization
        self.f_sim = g.frequencies()
        fc = 0.5 * (self.f_sim.min() + self.f_sim.max())
        self.pulse = GaussianSource(fc, g.options.fwidth * fc)
        comp = "x" if polarization == "ex" else "z"
        self.comps = ("Ex", "Ey") if polarization == "ex" else ("Ez",)
        source = CurrentSource(
            self.pulse,
            component=comp,
            center=(g.nx * g.g / 2 + g.h / 2, g.y_source, 0.0),
            size=(2 * g.nx * g.g, 0.0, 0.0),
            amp_func=lambda x, y, z: g.source_profile(x),
        )
        monitor = wakis.DFTMonitor(
            self.f_sim,
            components=self.comps,
            center=(g.x_focus, g.y_focus, 0.0),
            size=(1.02 * g.h, 1.02 * g.h, 0.0),
        )
        self.Nt = int(np.ceil((self.pulse.t0 + g.options.run_time) / g.dt))
        design_comps = ("x", "y") if polarization == "ex" else ("z",)
        self.design = g.design(design_comps)
        self.sim = AdjointSimulation(
            g.make_solver, [source], {"focus": monitor}, self.design, self.Nt,
            store_dtype=g.options.store_dtype,
        )
        self.simulate = jax_simulation(self.sim)
        # Reference: substrate up to the top of density row 1 (the gym's
        # no-lens stack puts the interface focus_offset + one pixel below the
        # focus), same source
        ref = np.zeros((g.nx, g.nrows))
        ref[:, 1:] = 1.0
        out = self.sim.forward(ref)
        self.intensity_ref = np.asarray(self.intensity(out["focus"]))

    def intensity(self, raw):
        """|E|^2 at the focal point: each component averaged over the two
        samples straddling the point."""
        total = 0.0
        for c in self.comps:
            v = raw[c].reshape(raw[c].shape[0], -1)
            total = total + jnp.abs(jnp.mean(v, axis=1)) ** 2
        return total

    def __call__(self, density_gym):
        return self.intensity(self.simulate(density_gym)["focus"]) / self.intensity_ref


class WakisMetalensComponent(gym_metalens.MetalensComponent):
    def __init__(self, gym_component, options: MetalensOptions = MetalensOptions()):
        """Wakis-backed drop-in replacement of the gym's `MetalensComponent`."""
        self.spec = gym_component.spec
        self.sim_params = gym_component.sim_params
        self.seed_density = gym_component.seed_density
        self.density_initializer = gym_component.density_initializer
        self.expansion = gym_component.expansion
        self.options = options
        self._sims = {}

    def simulation(self, polarization, density_shape=None):
        """Simulation for a polarization and density shape (the gym accepts
        lenses thicker or thinner than the spec; the focus stays
        `focus_offset` above the top of the density)."""
        shape = tuple(density_shape or self.seed_density.shape)
        key = (polarization, shape)
        if key not in self._sims:
            geometry = MetalensGeometry(
                self.spec, np.asarray(self.sim_params.wavelength), shape, self.options
            )
            self._sims[key] = MetalensSimulation(geometry, polarization)
        return self._sims[key]

    def response(self, params, *, wavelength=None, expansion=None, compute_fields=False):
        """Response with the gym's signature (default wavelengths only)."""
        if wavelength is not None and not np.allclose(wavelength, self.sim_params.wavelength):
            raise NotImplementedError("Only the challenge wavelengths are supported.")
        density = gym_utils.transforms.rescaled_density_array(
            params,
            lower_bound=gym_metalens.DENSITY_LOWER_BOUND,
            upper_bound=gym_metalens.DENSITY_UPPER_BOUND,
        )
        nan = jnp.full(len(self.sim_params.wavelength), jnp.nan)
        enh = {
            p: (
                self.simulation(p, density.shape)(density)
                if p in self.options.polarizations
                else nan
            )
            for p in ("ex", "ey")
        }
        response = gym_metalens.MetalensResponse(
            wavelength=jnp.asarray(self.sim_params.wavelength),
            enhancement_ex=enh["ex"],
            enhancement_ey=enh["ey"],
        )
        return response, {}


def wakis_metalens_challenge(options: MetalensOptions = MetalensOptions(), **kwargs):
    """(wakis_challenge, gym_challenge) for the gym's metalens challenge."""
    gym_challenge = gym_challenges.metalens(**kwargs)
    component = WakisMetalensComponent(gym_challenge.component, options)
    return dataclasses.replace(gym_challenge, component=component), gym_challenge
