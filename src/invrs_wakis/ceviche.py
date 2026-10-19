"""Wakis (FIT time-domain) backend for the invrs-gym ceviche challenges.

The gym's ceviche components solve the 2D Ez problem by FDFD (ceviche) on the
pixel grid of the design, once per wavelength. `WakisCevicheComponent` solves
the same problem with the Wakis FIT time-domain solver: one broadband run per
excited port gives all wavelengths, and its exact discrete adjoint gives the
gradient with respect to every design pixel (`jax.custom_vjp` through
`wakis.inverse.jax_simulation`).

Geometry, permittivities, ports, mode orders and wavelengths are read from the
gym's own `ceviche_challenges` model object, so the component is a drop-in
replacement: same parameters (`totypes.Density2DArray`), same
`CevicheResponse`, and the gym's `loss`, `eval_metric` and `metrics` apply
unchanged.

Discretization
--------------
The gym pixel i covers [i dl, (i+1) dl]. The Wakis mesh has `resolution` = r
square cells per pixel, with the Ez samples (grid nodes) at the fine-pixel
centres (m + 1/2) dl / r; every node takes the permittivity of the gym pixel
that contains it. For r = 1 the Ez/Hx/Hy staggering is exactly that of
ceviche's Yee FDFD grid, so the two spatial operators coincide away from the
absorbing layers. The leapfrog time step then only replaces omega by
omega_t = 2/dt sin(omega dt / 2) (temporal dispersion). With
`match_fdfd=True` the run is sampled at the frequency whose omega_t equals the
target omega, which makes the time-domain result the solution of ceviche's
discrete frequency-domain problem; with r > 1 the mesh is finer than the gym's
and the result converges to the continuum 2D problem instead.

S-parameters are ratios of unit-power mode amplitudes (Wakis `ModeMonitor`):
S_ji = b_j / a_i with a_i the incident amplitude measured at the excited port
and b_j the outgoing amplitude at port j, as in ceviche (|S|^2 is the power
fraction; the phase reference planes differ from ceviche's).
"""

import dataclasses
import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wakis
from ceviche_challenges import defs
from invrs_gym import challenges as gym_challenges
from invrs_gym import utils as gym_utils
from invrs_gym.challenges import base
from invrs_gym.challenges.ceviche import challenge as gym_ceviche
from scipy.constants import c as C0
from scipy.constants import epsilon_0 as EPS0
from scipy.sparse import csr_matrix
from wakis.field_monitors import _flat_index
from wakis.inverse import AdjointSimulation, jax_simulation, mode_amplitudes
from wakis.sources import GaussianSource

from invrs_wakis._design import PaddedMaterialGrid, PixelDesign, check_smoothing, host
from invrs_wakis._modes import PolarizedModeMonitor, PolarizedModeSource

jax.config.update("jax_enable_x64", True)


@dataclasses.dataclass(frozen=True)
class SimOptions:
    """
    Numerical settings of the Wakis simulation.

    Attributes
    ----------
    resolution : int
        Wakis cells per gym pixel along x and y.
    courant : float
        Time step as a fraction of the 2D stability limit
        h / (c sqrt(2)).
    match_fdfd : bool
        Sample the run at the frequencies whose discrete-time angular
        frequency equals the target omega (removes the leapfrog temporal
        dispersion, so that r = 1 reproduces ceviche's discrete problem).
    run_time : float
        Simulated time after the source peak [s].
    fwidth : float
        Spectral standard deviation of the Gaussian pulse, relative to the
        centre frequency.
    pml_cells : int or None
        CPML thickness in Wakis cells; None uses the gym's PML thickness
        (pml_width pixels).
    store_dtype : numpy dtype
        Precision of the stored adjoint increments.
    port_estimator : {'modal', 'ceviche'}
        'modal': mode amplitudes from the Wakis discrete eigenmodes and the
        exact discrete orthogonality relation (`ModeMonitor`); a uniform
        guide shows no spurious reflection. 'ceviche': the estimator of
        `ceviche_challenges.scattering.calculate_amplitudes` (continuum-form
        overlaps of ceviche's own port modes with Ez averaged over two pixel
        columns and the H column between them), evaluated on the Wakis DFT
        fields; resolution 1 only. It reproduces the official evaluator,
        including its reflection floor (about -31 dB at 40 nm pixels).
    smoothing : {'none', 'tensor'}
        Material smoothing of the design ('tensor': Wakis MaterialGrid
        subpixel averaging, see `invrs_wakis._design.check_smoothing`).
    use_gpu : bool
        Run the Wakis solver (forward and adjoint) on the GPU (CuPy).
    fused : bool
        GPU only: Wakis fused time step (CPML, sources, DFT monitors and the
        adjoint in a few kernels with CUDA graphs; `SolverFIT3D(fused=True)`).
    adjoint_memory : {'store', 'checkpoint', 'reverse', 'auto'}
        How the Wakis adjoint obtains the per-step design rows
        (`AdjointSimulation(memory=...)`); 'auto' keeps them if they fit in
        `adjoint_memory_budget_gb`, else checkpoints and replays.
    adjoint_memory_budget_gb : float or None
        Budget for 'auto'/'checkpoint' [GB] (None: the Wakis default).
    """

    resolution: int = 1
    courant: float = 0.95
    match_fdfd: bool = True
    run_time: float = 1.2e-12
    fwidth: float = 0.05
    pml_cells: Optional[int] = None
    store_dtype: Any = np.float32
    port_estimator: str = "modal"
    smoothing: str = "none"
    use_gpu: bool = False
    fused: bool = False
    adjoint_memory: str = "store"
    adjoint_memory_budget_gb: Optional[float] = None


def adjoint_memory_kwargs(options):
    """AdjointSimulation memory arguments of the options (none for 'store',
    so older Wakis versions keep working)."""
    if options.adjoint_memory == "store":
        return {}
    kw = {"memory": options.adjoint_memory}
    if options.adjoint_memory_budget_gb is not None:
        kw["memory_budget_gb"] = options.adjoint_memory_budget_gb
    return kw


class CevicheGeometry:
    def __init__(self, model, options: SimOptions):
        """
        Wakis mesh, background permittivity and ports of a ceviche model.

        Parameters
        ----------
        model : ceviche_challenges.model_base.Model
            The gym's ceviche model (e.g. `component.ceviche_model`).
        options : SimOptions
            Numerical settings.
        """
        if model.design_variable_shape != model.design_region_shape:
            raise NotImplementedError("Design variable transforms are not supported.")
        check_smoothing(options.smoothing)
        self.model = model
        self.options = options
        self.r = r = int(options.resolution)
        self.dl = float(model.dl)
        self.h = self.dl / r
        self.shape = tuple(int(n) for n in model.shape)
        self.eps_clad = float(model.cladding_permittivity)
        self.eps_slab = float(model.slab_permittivity)
        x0, y0, x1, y1 = (int(v) for v in model.design_region_coords)
        self.design_coords = (x0, y0, x1, y1)
        self.design_shape = (x1 - x0, y1 - y0)
        self.n_pml = (
            int(model.pml_width) * r
            if options.pml_cells is None
            else int(options.pml_cells)
        )
        # 2D stability limit of the leapfrog scheme (no z derivatives)
        self.dt_2d = self.h / (C0 * np.sqrt(2.0))
        courant = options.courant
        if options.smoothing == "tensor":
            # The off-diagonal tensor couplings lower the stable step
            # to dt_2d / sqrt(1 + mu_max) (wakis.tensor_averaging.
            # bound_stable_dt); mu_max ~ 0.31 for Si/SiO2 pixel designs, and
            # Wakis raises if a design needs a smaller step
            courant = min(courant, 0.7)
        self.dt = courant * self.dt_2d
        # Pixel of each grid node (N + 1 nodes per axis, the last on the wall)
        self.node_pixel = [
            np.minimum((np.arange(r * n + 1) + 0.5) // r, n - 1).astype(int)
            for n in self.shape
        ]
        self._eps_bg_nodes = self._node_permittivity(np.asarray(model.density_bg))

    def _node_permittivity(self, density):
        eps = self.eps_clad + (self.eps_slab - self.eps_clad) * density
        return eps[np.ix_(self.node_pixel[0], self.node_pixel[1])]

    def make_grid(self):
        """Wakis grid: r cells per pixel, one periodic cell along z."""
        h, (nx, ny) = self.h, self.shape
        return wakis.GridFIT3D(
            h / 2, nx * self.dl + h / 2, h / 2, ny * self.dl + h / 2, -h / 2, h / 2,
            self.r * nx, self.r * ny, 1, verbose=0,
        )  # fmt: skip

    def make_solver(self):
        """Fresh solver with the background permittivity (design region = cladding)."""
        solver = wakis.SolverFIT3D(
            self.make_grid(),
            bc_low=["cpml", "cpml", "periodic"],
            bc_high=["cpml", "cpml", "periodic"],
            n_pml=self.n_pml,
            dt=self.dt,
            use_gpu=self.options.use_gpu,
            verbose=0,
            **({"fused": True} if self.options.fused else {}),
        )
        if self.options.smoothing == "tensor":
            # Stability bound of the tensor step relative to the 2D limit (the
            # single periodic z cell adds no z derivative)
            solver.conformal_dt_max = self.dt_2d
        eps = self._eps_bg_nodes
        for comp in "xyz":
            # Scaling keeps the zero ieps of PEC walls and ghost samples; with
            # CPML the time step reads ieps directly (no operator to rebuild).
            # On the host: Field.__getitem__ may return a NumPy copy on GPU
            # solvers, and __setitem__ moves the result to the device
            solver.ieps[:, :, 0, comp] = host(solver.ieps[:, :, 0, comp]) / eps
        return solver

    def frequencies(self, wavelengths_nm):
        """Target frequencies [Hz] and the frequencies at which the run is sampled."""
        f = C0 / (np.asarray(wavelengths_nm, dtype=float) * 1e-9)
        if not self.options.match_fdfd:
            return f, f
        # omega_t(f_sim) = 2/dt sin(pi f_sim dt) = 2 pi f
        return f, np.arcsin(np.pi * f * self.dt) / (np.pi * self.dt)

    def port_plane(self, port, offset=0):
        """Centre and size [m] of the plane of a ceviche port, shifted by
        `offset` pixels along its direction."""
        d = port.dir
        shift = offset * d.sign
        dl = self.dl
        # A hair off the pixel centre so that the nearest node is unambiguous
        eps = 1e-3 * self.h
        if d.is_along_x:
            center = ((port.x + shift + 0.5) * dl + eps, port.y * dl, 0.0)
            size = (0.0, port.width * dl, np.inf)
        else:
            center = (port.x * dl, (port.y + shift + 0.5) * dl + eps, 0.0)
            size = (port.width * dl, 0.0, np.inf)
        return center, size

    def num_steps(self, source):
        return int(np.ceil((source.t0 + self.options.run_time) / self.dt))

    def design(self):
        """
        Design map: every Ez edge (grid node) inside the design region takes
        the permittivity of the gym pixel containing it (one weight 1 per
        edge); Ex/Ey stay zero in the Ez problem and keep the cladding.
        """
        x0, y0, x1, y1 = self.design_coords
        shape = self.design_shape
        px, py = self.node_pixel
        if self.options.smoothing == "tensor":
            dl = self.dl
            frame = np.asarray(self.model.density_bg, float)[x0 - 1 : x1 + 1, y0 - 1 : y1 + 1]
            center = (0.5 * (x0 + x1) * dl, 0.5 * (y0 + y1) * dl, 0.0)
            size = ((x1 - x0 + 1) * dl, (y1 - y0 + 1) * dl, 0.0)
            return PaddedMaterialGrid(center, size, frame, self.eps_clad, self.eps_slab)

        def build(solver):
            ii = np.flatnonzero((px >= x0) & (px < x1))
            jj = np.flatnonzero((py >= y0) & (py < y1))
            I, J = np.meshgrid(ii, jj, indexing="ij")
            flat = _flat_index(solver, "z", I, J, np.zeros_like(I)).ravel()
            cols = ((px[I] - x0) * shape[1] + (py[J] - y0)).ravel()
            W = csr_matrix(
                (np.ones(flat.size), (np.arange(flat.size), cols)),
                shape=(flat.size, int(np.prod(shape))),
            )
            return flat, W, np.zeros(flat.size)

        return PixelDesign(shape, self.eps_clad, self.eps_slab, build)


class CevicheSimulation:
    def __init__(self, geometry: CevicheGeometry, excite_port_idx: int, wavelengths_nm):
        """
        Differentiable broadband simulation of one excited port.

        Calling the object with a density rho (design shape) returns the
        complex S-parameter column S[:, j] = b_j / a_excite, shape
        (n_wavelengths, n_ports), as a JAX array differentiable through the
        Wakis adjoint.
        """
        g = geometry
        self.geometry = g
        self.ports = list(g.model.ports)
        self.excite = excite_port_idx
        self.f_target, self.f_sim = g.frequencies(wavelengths_nm)
        fc = 0.5 * (self.f_sim.min() + self.f_sim.max())
        self.pulse = GaussianSource(fc, g.options.fwidth * fc)

        port = self.ports[excite_port_idx]
        center, size = g.port_plane(port)
        sources = [
            PolarizedModeSource(
                self.pulse,
                center=center,
                size=size,
                direction="+" if port.dir.sign > 0 else "-",
                mode_index=port.order - 1,
            )
        ]
        self.estimator = g.options.port_estimator
        if self.estimator not in ("modal", "ceviche"):
            raise ValueError(f"Unknown port_estimator {self.estimator!r}.")
        if self.estimator == "ceviche" and g.r != 1:
            raise ValueError("The 'ceviche' port estimator needs resolution=1.")
        monitors = {}
        self._ceviche_modes = []
        for j, p in enumerate(self.ports):
            if self.estimator == "modal":
                # Mode decomposition `offset` pixels from the port into the device
                center, size = g.port_plane(p, offset=int(p.offset))
                monitors[f"p{j}"] = PolarizedModeMonitor(
                    self.f_sim, center=center, size=size, n_modes=p.order
                )
            else:
                monitors[f"p{j}"] = self._ceviche_monitor(p)
                self._ceviche_modes.append(self._ceviche_port_modes(p))
        self.design = g.design()
        self.Nt = g.num_steps(self.pulse)
        self.sim = AdjointSimulation(
            g.make_solver,
            sources,
            monitors,
            self.design,
            self.Nt,
            store_dtype=g.options.store_dtype,
            **adjoint_memory_kwargs(g.options),
        )
        self.simulate = jax_simulation(self.sim)

    def _ceviche_monitor(self, port):
        """
        DFT of Ez on the two pixel columns c - 1, c and of the H component
        between them (ceviche's H[c], located at c - 1/2), where c is the
        port column shifted by its offset; rows are the port slice pixels.
        """
        dl = self.geometry.dl
        c = (port.x if port.dir.is_along_x else port.y) + port.signed_offset()
        t = port.y if port.dir.is_along_x else port.x  # transverse centre
        along, across = c * dl, t * dl
        if port.dir.is_along_x:
            center, size, comps = (along, across, 0.0), (dl, (port.width - 1) * dl, 0.0), ("Ez", "Hy")
        else:
            center, size, comps = (across, along, 0.0), ((port.width - 1) * dl, dl, 0.0), ("Ez", "Hx")
        return wakis.DFTMonitor(self.f_sim, components=comps, center=center, size=size)

    def _ceviche_port_modes(self, port):
        """ceviche's own (e, h) port mode profiles at each target frequency."""
        eps_bg = self.geometry.model.epsilon_r_bg()
        profiles = [
            port.field_profiles(eps_bg[port.coords()], 2 * np.pi * f, self.geometry.dl)
            for f in self.f_target
        ]
        e = np.stack([np.asarray(p[0]) for p in profiles])
        h = np.stack([np.asarray(p[1]) for p in profiles])
        return e, h

    def _ceviche_amplitudes(self, raw, j):
        """`ceviche_challenges.scattering.calculate_amplitudes`, in JAX."""
        port = self.ports[j]
        e_m, h_m = self._ceviche_modes[j]
        if port.dir.is_along_x:
            ez = raw["Ez"][..., 0]  # (f, 2, w)
            E = 0.5 * (ez[:, 0, :] + ez[:, 1, :])
            H = raw["Hy"][:, 0, :, 0]
            o1 = -jnp.sum(np.conj(e_m) * H, axis=-1)
        else:
            ez = raw["Ez"][..., 0]  # (f, w, 2)
            E = 0.5 * (ez[:, :, 0] + ez[:, :, 1])
            H = raw["Hx"][:, :, 0, 0]
            o1 = jnp.sum(np.conj(e_m) * H, axis=-1)
        o2 = jnp.sum(np.conj(h_m) * E, axis=-1)
        norm = np.sqrt(2 * -np.sum(np.conj(e_m) * h_m, axis=-1) + 0j)
        s_plus, s_minus = (o1 + o2) / 2 / norm, (o1 - o2) / 2 / norm
        return (s_plus, s_minus) if port.dir.sign > 0 else (s_minus, s_plus)

    def amplitudes(self, raw):
        """Incident and outgoing mode amplitudes (a_j, b_j) of all ports."""
        if self.estimator == "ceviche":
            ab = [self._ceviche_amplitudes(raw[f"p{j}"], j) for j in range(len(self.ports))]
            return (
                jnp.stack([x[0] for x in ab], axis=-1),
                jnp.stack([x[1] for x in ab], axis=-1),
            )
        a, b = [], []
        for j, p in enumerate(self.ports):
            plus, minus = mode_amplitudes(
                raw[f"p{j}"], self.sim.readouts[f"p{j}"], mode_index=p.order - 1
            )
            # Ports point into the device: incident along dir, outgoing against
            inc, out = (plus, minus) if p.dir.sign > 0 else (minus, plus)
            a.append(inc)
            b.append(out)
        return jnp.stack(a, axis=-1), jnp.stack(b, axis=-1)

    def __call__(self, rho):
        a, b = self.amplitudes(self.simulate(rho))
        return b / a[:, self.excite : self.excite + 1]


class WakisCevicheComponent(gym_ceviche.CevicheComponent):
    def __init__(self, gym_component, options: SimOptions = SimOptions()):
        """
        Wakis-backed drop-in replacement of a gym `CevicheComponent`.

        Parameters
        ----------
        gym_component : invrs_gym.challenges.ceviche.challenge.CevicheComponent
            The gym component; its ceviche model, seed density (shape, fixed
            pixels, symmetries, length-scale targets) and initializer are
            reused, so `init` returns exactly the gym's parameters.
        options : SimOptions
            Numerical settings of the Wakis simulation.
        """
        self.ceviche_model = gym_component.ceviche_model
        self.seed_density = gym_component.seed_density
        self.density_initializer = gym_component.density_initializer
        self.options = options
        self.geometry = CevicheGeometry(self.ceviche_model, options)
        self._sims: Dict[Tuple, CevicheSimulation] = {}

    def simulation(self, excite_port_idx, wavelengths_nm):
        key = (int(excite_port_idx), tuple(float(w) for w in wavelengths_nm))
        if key not in self._sims:
            self._sims[key] = CevicheSimulation(self.geometry, *key[:1], key[1])
        return self._sims[key]

    def response(
        self,
        params,
        *,
        excite_port_idxs: Sequence[int] = (0,),
        wavelengths_nm: Optional[jnp.ndarray] = None,
        max_parallelizm: Optional[int] = None,
    ) -> Tuple[gym_ceviche.CevicheResponse, base.AuxDict]:
        """Response with the gym's signature; `max_parallelizm` is ignored."""
        del max_parallelizm
        with jax.ensure_compile_time_eval():
            if wavelengths_nm is None:
                wavelengths_nm = jnp.asarray(self.ceviche_model.output_wavelengths)
        wl = np.asarray(wavelengths_nm, dtype=float)
        density = gym_utils.transforms.rescaled_density_array(
            params,
            lower_bound=gym_ceviche.DENSITY_LOWER_BOUND,
            upper_bound=gym_ceviche.DENSITY_UPPER_BOUND,
        )
        columns = [self.simulation(i, wl)(density) for i in excite_port_idxs]
        sparams = jnp.stack(columns, axis=1)  # (wavelength, excite, port)
        response = gym_ceviche.CevicheResponse(
            s_parameters=sparams,
            wavelengths_nm=jnp.asarray(wl),
            excite_port_idxs=jnp.asarray(excite_port_idxs),
        )
        return response, {gym_ceviche.SPARAMS: sparams}


def wakis_ceviche_challenge(name: str, options: SimOptions = SimOptions(), **kwargs):
    """
    A gym ceviche challenge with its component replaced by the Wakis backend.

    Parameters
    ----------
    name : str
        Gym challenge name, e.g. 'ceviche_lightweight_waveguide_bend'.
    options : SimOptions
        Wakis numerical settings.
    **kwargs
        Passed to the gym's challenge constructor.

    Returns
    -------
    tuple
        (wakis_challenge, gym_challenge): the two share the loss, eval
        metric, bounds and parameters; only the component differs.
    """
    gym_challenge = gym_challenges.BY_NAME[name](**kwargs)
    component = WakisCevicheComponent(gym_challenge.component, options)
    return dataclasses.replace(gym_challenge, component=component), gym_challenge
