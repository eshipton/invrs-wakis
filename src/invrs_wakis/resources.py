"""Memory preflight of a run with `wakis.estimate` (calibrated peak RSS)."""

import numpy as np
from wakis.estimate import Estimate, estimate, preflight
from wakis.sources import GaussianSource


def simulation_estimate(name, challenge) -> Estimate:
    """
    Peak memory of one forward + adjoint run of a Wakis challenge (the
    simulation its loss runs; ceviche: excited port 0, metalens: ex).
    """
    comp = challenge.component
    if name == "metalens":
        from invrs_wakis.metalens import MetalensGeometry

        # Geometry only (building the simulation would allocate the solver)
        g = MetalensGeometry(comp.spec, np.asarray(comp.sim_params.wavelength),
                             tuple(comp.seed_density.shape), comp.options)
        f_sim = g.frequencies()
        fc = 0.5 * (f_sim.min() + f_sim.max())
        steps = int(np.ceil((GaussianSource(fc, g.options.fwidth * fc).t0
                             + g.options.run_time) / g.dt))
        cells = (g.nx * g.r, g.ny_cells + 1, 1)
        design_edges = 2 * (g.nx * g.r) * (g.nrows * g.r)  # Ex and Ey
        monitors = [(len(f_sim), 4, 2)]
    else:
        g = comp.geometry
        wl = np.asarray(comp.ceviche_model.output_wavelengths, float)
        _, f_sim = g.frequencies(wl)
        fc = 0.5 * (f_sim.min() + f_sim.max())
        steps = g.num_steps(GaussianSource(fc, g.options.fwidth * fc))
        cells = (g.r * g.shape[0] + 1, g.r * g.shape[1] + 1, 1)
        design_edges = g.r**2 * int(np.prod(g.design_shape))
        ports = list(g.model.ports)
        monitors = [(len(wl), 2 * g.r * int(p.width), 3) for p in ports]
    store = np.dtype(comp.options.store_dtype).itemsize
    est = estimate(cells, steps=steps, dtype="float64", backend="cpu", cpml=True,
                   monitors=monitors, adjoint={"design_edges": design_edges,
                                               "store_bytes": store})
    if getattr(comp.options, "smoothing", "none") == "tensor":
        # Measured: bend r = 2 tensor run peaks at 4.8 GB RSS vs 1.44 GB
        # estimated without the tensor operator (profile_iteration.py)
        est.host_bytes *= TENSOR_MEMORY_FACTOR
    return est


TENSOR_MEMORY_FACTOR = 3.4


def run_preflight(name, challenges_by_resolution, machine, extra_sims=1, log=print):
    """
    Estimate the run's peak memory (all resolutions' simulations held at
    once, plus `extra_sims` copies of the r = 1 one for post-processing, plus
    about 1.5 GB for Python/JAX/ceviche) and check it against `machine`
    (a `wakis.estimate.MACHINES` key). Raises MemoryError if it does not fit.

    Returns a dict for summary.json.
    """
    ests = {r: simulation_estimate(name, ch) for r, ch in challenges_by_resolution.items()}
    total_host = sum(e.host_bytes for e in ests.values())
    total_host += extra_sims * ests[min(ests)].host_bytes + 1.5e9
    total = Estimate(sum(e.cells for e in ests.values()), total_host, 0.0,
                     sum(e.seconds for e in ests.values()))
    info = {
        "machine": machine,
        "per_resolution": {int(r): {"cells": e.cells, "host_GB": e.host_bytes / 1e9,
                                    "forward_adjoint_s_est": e.seconds}
                           for r, e in ests.items()},
        "total_host_GB": total_host / 1e9,
    }
    log({"preflight": info})
    if machine:
        preflight(total, machine)  # raises MemoryError
    return info
