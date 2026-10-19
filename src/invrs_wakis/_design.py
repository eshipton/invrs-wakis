"""Map of gym design pixels onto Wakis E edges (design interface of
`wakis.inverse.AdjointSimulation`)."""

import numpy as np
from scipy.constants import epsilon_0 as EPS0


def host(array):
    """NumPy copy of a NumPy or CuPy array."""
    return array.get() if hasattr(array, "get") else np.asarray(array)
from scipy.sparse import csr_matrix


class PixelDesign:
    def __init__(self, shape, eps_min, eps_max, build):
        """
        Permittivity of Wakis E edges from a pixelated density rho.

        Each pixel p has eps_p = eps_min + (eps_max - eps_min) rho_p. An E edge
        e takes the harmonic mean of the pixels it crosses (series
        capacitance along the edge, the FIT average for a field component
        normal to pixel interfaces):

            ieps_e = 1 / eps_0 * (sum_p W_ep / eps_p + c_e),

        with weights W (rows summing with c_e to the inverse of a mean) and a
        constant c_e for the part of the edge outside the design region. An
        edge inside one pixel has a single weight 1 and ieps_e = 1/(eps_0
        eps_p). The gradient follows exactly from
        d ieps_e / d rho_p = -W_ep (eps_max - eps_min) / (eps_0 eps_p^2).

        Parameters
        ----------
        shape : tuple of int
            Shape of rho.
        eps_min, eps_max : float
            Relative permittivity at rho = 0 and rho = 1.
        build : callable
            build(solver) -> (flat_idx, W, c): Field.array indices of the
            design edges, the sparse weight matrix (edges x pixels) and the
            constant terms (per edge).
        """
        self.shape = tuple(shape)
        self.eps_min, self.eps_max = float(eps_min), float(eps_max)
        self._build_fn = build
        self.flat_idx = None
        self.W = None
        self.c = None

    def _build(self, solver):
        flat, W, c = self._build_fn(solver)
        keep = host(solver.ieps.array)[flat] != 0  # PEC walls and ghost samples stay
        self.flat_idx = flat[keep]
        self.W = csr_matrix(W)[keep]
        self.c = np.asarray(c, dtype=float)[keep]

    def eps_pixels(self, rho):
        rho = np.asarray(rho, dtype=float).reshape(-1)
        return self.eps_min + (self.eps_max - self.eps_min) * rho

    def ieps(self, rho):
        """1/(eps_0 eps) on the design edges [m/F], ordered as `flat_idx`."""
        return (self.W @ (1.0 / self.eps_pixels(rho)) + self.c) / EPS0

    def apply(self, solver, rho):
        """Write the design into a solver that has not started stepping (CPML)."""
        if not solver.step_0:
            raise RuntimeError("Apply the design before the first time step.")
        if not solver.activate_cpml:
            raise NotImplementedError("PixelDesign writes ieps directly (CPML solvers).")
        if self.W is None:
            self._build(solver)
        xp = solver.E.xp  # NumPy or CuPy (use_gpu)
        solver.ieps.array[xp.asarray(self.flat_idx)] = xp.asarray(self.ieps(rho))

    def dispersive_group(self, solver):
        return None

    def vjp(self, rho, grad_ieps):
        """dF/drho from dF/dieps on the design edges."""
        eps = self.eps_pixels(rho)
        g = self.W.T @ np.asarray(grad_ieps, dtype=float)
        return (-(self.eps_max - self.eps_min) * g / (EPS0 * eps**2)).reshape(self.shape)


SMOOTHING_MODES = ("none", "tensor")

# Total number of solver setups in this process whose tensor operator clamped
# couplings (n_clamped > 0: the gradient is then not exact)
TENSOR_CLAMPED = {"setups": 0, "clamped_setups": 0, "max_n_clamped": 0}


def check_smoothing(smoothing: str) -> None:
    """
    Validate the material-smoothing option of the bridges.

    'none' (default): every Wakis E sample takes the permittivity of the gym
    pixel(s) it lies in (`PixelDesign`). 'tensor': Meep-style anisotropic
    subpixel averaging of the density by `wakis.inverse.MaterialGrid(...,
    smoothing='tensor')` (fill fraction = linearly interpolated density,
    interface normal = density gradient; exact adjoint), see
    `PaddedMaterialGrid`.

    Note: in the ceviche (Ez) problem E is parallel to every interface in
    the plane, so the averaged inverse permittivity of an Ez edge is
    1/<eps> (arithmetic mean over the fill fraction). At resolution 1 the Ez
    samples sit at the pixel centres (fill fraction = pixel density), so
    'tensor' only changes the r >= 2 simulations.
    """
    if smoothing not in SMOOTHING_MODES:
        raise ValueError(f"smoothing must be one of {SMOOTHING_MODES}, got {smoothing!r}")
    if smoothing == "tensor":
        import inspect

        from wakis.inverse import MaterialGrid

        if "smoothing" not in inspect.signature(MaterialGrid.__init__).parameters:
            raise NotImplementedError(
                "smoothing='tensor' needs Wakis with MaterialGrid(smoothing=...) "
                "(develop ac275ba or later)."
            )


class PaddedMaterialGrid:
    """
    Wakis `MaterialGrid(smoothing='tensor')` over the gym's pixel densities.

    The MaterialGrid nodes are the gym pixel centres of the design region
    plus a one-pixel frame taken from the (fixed) background density, so
    that edges between the outermost pixel centres and the design-region
    boundary interpolate towards the real surroundings (waveguide stubs,
    cladding) instead of being left out. The frame is constant, so the
    gradient is the interior block of the MaterialGrid gradient.

    Parameters
    ----------
    center, size : tuple of float
        Box spanned by the frame pixel centres [m] (z size 0).
    frame : ndarray
        Background density on the (nx + 2, ny + 2) padded pixel grid
        (interior values are ignored).
    eps_min, eps_max : float
        Relative permittivity at density 0 and 1.
    """

    def __init__(self, center, size, frame, eps_min, eps_max):
        from wakis.inverse import MaterialGrid

        self.frame = np.asarray(frame, dtype=float)
        self.shape = (self.frame.shape[0] - 2, self.frame.shape[1] - 2)
        self.grid = MaterialGrid(center, size, self.frame.shape + (1,), eps_min, eps_max,
                                 smoothing="tensor")
        self.smoothing = "tensor"

    def _pad(self, rho):
        full = self.frame.copy()
        full[1:-1, 1:-1] = np.asarray(rho, dtype=float).reshape(self.shape)
        return full[:, :, None]

    @property
    def flat_idx(self):
        return self.grid.flat_idx

    def apply(self, solver, rho):
        self.grid.apply(solver, self._pad(rho))
        op = getattr(solver, "tensor_operator", None)
        n = int(getattr(op, "n_clamped", 0)) if op is not None else 0
        TENSOR_CLAMPED["setups"] += 1
        if n:
            TENSOR_CLAMPED["clamped_setups"] += 1
            TENSOR_CLAMPED["max_n_clamped"] = max(TENSOR_CLAMPED["max_n_clamped"], n)

    def dispersive_group(self, solver):
        return None

    def vjp(self, rho, grad_ieps, pair_grad=None):
        g = self.grid.vjp(self._pad(rho), grad_ieps, pair_grad)
        return np.asarray(g)[1:-1, 1:-1, 0]

    def __getattr__(self, name):  # remaining MaterialGrid interface
        return getattr(self.__dict__["grid"], name)
