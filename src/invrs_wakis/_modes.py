"""Mode ports restricted to one polarization of a 2D (one periodic cell) Wakis grid.

In a 2D simulation (a single periodic cell along z) the TM set (Ez, Hx, Hy) and
the TE set (Hz, Ex, Ey) decouple exactly. `wakis.mode_solver.solve_modes` returns
the guided modes of both sets sorted by effective index, so "the n-th mode" of a
port depends on how the two families interleave. The ceviche challenges are
defined for the Ez polarization only, and number its modes 1, 2, ... by
decreasing effective index; the classes below select modes the same way.
"""

import numpy as np
from wakis.mode_solver import ModeMonitor, ModeSource, solve_modes


def _z_fraction(mode):
    """Fraction of |E_t|^2 carried by the z component of a mode profile."""
    axis = {"b": mode.axes[1], "c": mode.axes[2]}
    ez = sum(np.sum(np.abs(v) ** 2) for k, v in mode.e.items() if axis[k] == "z")
    total = sum(np.sum(np.abs(v) ** 2) for v in mode.e.values())
    return ez / total


def polarized_modes(solver, frequency, center, size, n_modes, polarization="Ez"):
    """
    Guided modes of one polarization family, by decreasing effective index.

    Parameters
    ----------
    solver : wakis.SolverFIT3D
        Solver of a 2D grid (one periodic cell along z).
    frequency : float
        Frequency [Hz].
    center, size : tuple of float
        Port plane, as in `wakis.mode_solver.solve_modes` [m].
    n_modes : int
        Number of modes of the family to return (fewer if fewer propagate).
    polarization : {'Ez', 'Hz'}
        'Ez' keeps the TM family (E out of plane, ceviche's `fdfd_ez`), 'Hz'
        the TE family (E in plane).

    Returns
    -------
    list of wakis.mode_solver.Mode
    """
    want_ez = polarization == "Ez"
    modes = solve_modes(solver, frequency, center, size, n_modes=2 * n_modes + 2)
    picked = [m for m in modes if (_z_fraction(m) > 0.5) == want_ez]
    return picked[:n_modes]


class PolarizedModeSource(ModeSource):
    """`ModeSource` whose `mode_index` counts modes of one polarization only."""

    def __init__(self, *args, polarization="Ez", **kwargs):
        super().__init__(*args, **kwargs)
        self.polarization = polarization
        self.family_index = self.mode_index

    def _initialize(self, solver):
        n = self.family_index + 1
        family = polarized_modes(
            solver, self.mode_frequency, self.center, self.size, n, self.polarization
        )
        if len(family) < n:
            raise ValueError(f"{self.polarization} mode {n} does not propagate.")
        target = family[self.family_index]
        modes = solve_modes(
            solver, self.mode_frequency, self.center, self.size, n_modes=2 * n + 2
        )
        # Position of the selected mode in the unfiltered list used by ModeSource
        self.mode_index = int(np.argmin([abs(m.beta - target.beta) for m in modes]))
        super()._initialize(solver)


class PolarizedModeMonitor(ModeMonitor):
    """`ModeMonitor` whose mode indices count modes of one polarization only."""

    def __init__(self, *args, polarization="Ez", **kwargs):
        super().__init__(*args, **kwargs)
        self.polarization = polarization

    def _solve_modes(self):
        def family(f):
            return polarized_modes(
                self.solver, f, self.center, self.size, self.n_modes, self.polarization
            )

        if self.mode_frequency is not None:
            self.modes = [family(self.mode_frequency)] * self.frequencies.size
        else:
            self.modes = [family(f) for f in self.frequencies]
