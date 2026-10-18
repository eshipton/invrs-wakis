"""Map of gym design pixels onto Wakis E edges (design interface of
`wakis.inverse.AdjointSimulation`)."""

import numpy as np
from scipy.constants import epsilon_0 as EPS0
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
        keep = solver.ieps.array[flat] != 0  # PEC walls and ghost samples stay
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
        solver.ieps.array[self.flat_idx] = self.ieps(rho)

    def dispersive_group(self, solver):
        return None

    def vjp(self, rho, grad_ieps):
        """dF/drho from dF/dieps on the design edges."""
        eps = self.eps_pixels(rho)
        g = self.W.T @ np.asarray(grad_ieps, dtype=float)
        return (-(self.eps_max - self.eps_min) * g / (EPS0 * eps**2)).reshape(self.shape)


SMOOTHING_MODES = ("none", "tensor")


def check_smoothing(smoothing: str) -> None:
    """
    Validate the material-smoothing option of the bridges.

    'none' (default): every Wakis E sample takes the permittivity of the gym
    pixel(s) it lies in (`PixelDesign`). 'tensor': Meep-style anisotropic
    subpixel averaging of the density (arithmetic mean for E parallel to an
    interface, harmonic mean for E normal to it), to be provided by the Wakis
    `MaterialGrid` (Wakis branch feature/tensor-averaging, not yet on
    develop). Until the installed Wakis exposes it this raises
    NotImplementedError, so a run cannot silently fall back to 'none'.

    Note: in the ceviche (Ez) problem with an integer `resolution`, every Ez
    sample lies inside a single gym pixel and E is parallel to all pixel
    interfaces, so tensor averaging of a *binary pixel* design reproduces
    the 'none' permittivities; it matters for gray densities sampled
    off-centre, graded meshes and the metalens (in-plane E).
    """
    if smoothing not in SMOOTHING_MODES:
        raise ValueError(f"smoothing must be one of {SMOOTHING_MODES}, got {smoothing!r}")
    if smoothing == "tensor":
        import inspect

        from wakis.inverse import MaterialGrid

        if "smoothing" not in inspect.signature(MaterialGrid.__init__).parameters:
            raise NotImplementedError(
                "smoothing='tensor' needs Wakis MaterialGrid(smoothing=...) "
                "(feature/tensor-averaging); the installed Wakis does not provide it."
            )
        raise NotImplementedError(
            "Wakis provides MaterialGrid smoothing; wire it into the invrs-wakis "
            "design map (PixelDesign -> MaterialGrid) before using smoothing='tensor'."
        )
