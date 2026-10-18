"""Cost profile of the round-2 driver (one Batch task; independent
measurements run concurrently in a process pool).

  A  Nt, time per step and a cProfile of one value_and_grad at r = 1
     (ceviche estimator), r = 2 (modal) and r = 2 tensor, waveguide bend
  B  |S|^2 / eval-metric error vs the official ceviche (40 nm) of shorter
     run times and wider pulses, for the round-1 designs (tests/data)
  C  ceviche 10 nm re-check: wall time and peak RSS with the default SciPy
     SuperLU (COLAMD) solve, with MMD_AT_PLUS_A ordering, and with the two
     wavelengths in parallel (max_parallelizm=2); 40 nm official eval times

  uv run python scripts/profile_iteration.py --outdir out [--shard 0 --nshards 1]
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import cProfile  # noqa: E402
import dataclasses  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pathlib  # noqa: E402
import pstats  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data"
BEND = "ceviche_lightweight_waveguide_bend"


def _params(name, design=None, key=0):
    import jax
    import numpy as np
    from invrs_gym import challenges

    gym = challenges.BY_NAME[name]()
    p = gym.component.init(jax.random.PRNGKey(key))
    if design is not None:
        p = dataclasses.replace(p, array=jax.numpy.asarray(np.asarray(design, float)))
    return gym, p


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kB


def job_iteration(resolution, estimator, smoothing):
    import jax
    import numpy as np

    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

    ch, _ = wakis_ceviche_challenge(BEND, SimOptions(
        resolution=resolution, port_estimator=estimator, run_time=0.8e-12,
        smoothing=smoothing))
    _, p = _params(BEND, np.load(DATA / "round1_waveguide_bend.npy") * 0.8 + 0.1)

    def loss(x):
        return ch.loss(ch.component.response(dataclasses.replace(p, array=x))[0])

    vg = jax.value_and_grad(loss)
    vg(p.array)  # warm-up (simulation set-up, JAX tracing)
    prof = cProfile.Profile()
    t = time.time()
    prof.enable()
    vg(p.array)
    prof.disable()
    wall = time.time() - t
    sim = next(iter(ch.component._sims.values()))
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(18)
    s2 = io.StringIO()
    pstats.Stats(prof, stream=s2).sort_stats("tottime").print_stats(12)
    return {"resolution": resolution, "estimator": estimator, "smoothing": smoothing,
            "Nt": int(sim.Nt), "cells": int(np.prod(ch.component.geometry.shape)) * resolution**2,
            "value_and_grad_s": wall, "per_step_us": 1e6 * wall / sim.Nt,
            "peak_rss_GB": rss_gb(), "profile_cumulative": s.getvalue()[-6000:],
            "profile_tottime": s2.getvalue()[-4000:]}


def job_runtime(name, run_time, fwidth):
    import numpy as np

    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

    design = np.load(DATA / f"round1_{name.replace('ceviche_lightweight_', '')}.npy")
    gym, p = _params(name, design)
    ref, _ = gym.component.response(p)
    ch, _ = wakis_ceviche_challenge(name, SimOptions(
        port_estimator="ceviche", run_time=run_time, fwidth=fwidth))
    t = time.time()
    r, _ = ch.component.response(p)
    s_ref = np.abs(np.asarray(ref.s_parameters)) ** 2
    s = np.abs(np.asarray(r.s_parameters)) ** 2
    sim = next(iter(ch.component._sims.values()))
    return {"name": name, "run_time_ps": run_time * 1e12, "fwidth": fwidth,
            "Nt": int(sim.Nt), "forward_s": time.time() - t,
            "max_abs_dS2": float(np.max(np.abs(s - s_ref))),
            "eval_metric_wakis": float(ch.eval_metric(r)),
            "eval_metric_official": float(gym.eval_metric(ref))}


def job_10nm(variant):
    import jax
    import numpy as np
    import scipy.sparse.linalg as spl
    from ceviche import solvers
    from invrs_gym import challenges

    if variant == "mmd":
        solvers._solve_direct = lambda A, b: spl.spsolve(A.tocsc(), b, permc_spec="MMD_AT_PLUS_A")
    design = np.load(DATA / "round1_waveguide_bend.npy")
    gym10 = challenges.BY_NAME[BEND](grid_spacing_nm=10)
    p10 = gym10.component.init(jax.random.PRNGKey(0))
    p10 = dataclasses.replace(p10, array=np.kron(design, np.ones((4, 4))))
    kw = {"max_parallelizm": 2} if variant == "parallel2" else {}
    t = time.time()
    r, _ = gym10.component.response(p10, **kw)
    first = time.time() - t
    t = time.time()
    gym10.component.response(p10, **kw)
    return {"variant": variant, "first_call_s": first, "second_call_s": time.time() - t,
            "eval_metric": float(gym10.eval_metric(r)), "peak_rss_GB": rss_gb()}


def job_40nm_official():
    import numpy as np

    gym, p = _params(BEND, np.load(DATA / "round1_waveguide_bend.npy"))
    times = []
    for _ in range(3):
        t = time.time()
        gym.component.response(p)
        times.append(time.time() - t)
    return {"official_40nm_call_s": times, "peak_rss_GB": rss_gb()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [("A", job_iteration, (1, "ceviche", "none")),
            ("A", job_iteration, (2, "modal", "none")),
            ("A", job_iteration, (2, "modal", "tensor")),
            ("C", job_10nm, ("default",)), ("C", job_10nm, ("mmd",)),
            ("C", job_10nm, ("parallel2",)), ("C", job_40nm_official, ())]
    for name in ("ceviche_lightweight_waveguide_bend", "ceviche_lightweight_mode_converter",
                 "ceviche_lightweight_power_splitter"):
        for rt in (0.8e-12, 0.6e-12, 0.4e-12, 0.3e-12):
            jobs.append(("B", job_runtime, (name, rt, 0.05)))
        for fw in (0.1, 0.2):
            jobs.append(("B", job_runtime, (name, 0.8e-12, fw)))
    results = {"A": [], "B": [], "C": []}
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(os.cpu_count(), mp_context=ctx) as pool:
        futs = {pool.submit(fn, *args): (sec, fn.__name__, args) for sec, fn, args in jobs}
        for f in concurrent.futures.as_completed(futs):
            sec, fname, args = futs[f]
            try:
                res = f.result()
            except Exception as err:  # noqa: BLE001
                res = {"job": fname, "args": repr(args), "error": repr(err)}
            results[sec].append(res)
            print(json.dumps({k: v for k, v in res.items() if not k.startswith("profile")}),
                  flush=True)
            (out / "profile.json").write_text(json.dumps(results, indent=1))
    results["wall_s"] = time.time() - t0
    (out / "profile.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
