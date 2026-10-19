"""Optimization loop with the gym's recommended optimizers (invrs-opt).

Stages:
  1. `invrs_opt.density_lbfgsb` (Gaussian filter, kernel set by the density's
     minimum width/spacing targets, then tanh projection) over a beta
     continuation schedule;
  2. `invrs_opt.levelset_lbfgsb` (Gaussian-levelset parameterization with
     length-scale, curvature and fixed-pixel penalties), started from the
     thresholded stage-1 design.
The loss is the challenge's own `loss` of the (Wakis) response.
"""

import json
import time
from typing import Callable, List, Optional, Sequence, Tuple

import invrs_opt
import jax
import numpy as np

from invrs_wakis.analysis import binarize, evaluate


def optimize(
    challenge,
    params,
    betas: Sequence[float] = (2.0, 4.0, 8.0),
    density_steps: int = 40,
    levelset_steps: int = 80,
    penalty: float = 1.0,
    log: Callable[[dict], None] = print,
    snapshot: Optional[Callable[[str, int, np.ndarray, float, float], None]] = None,
) -> Tuple[object, List[Tuple[str, object]], float, str]:
    """
    Run the stages; return (selected binary params, stage checkpoints,
    optimization time [s], name of the selected stage).

    The selected design is the thresholded stage result with the best eval
    metric according to `challenge` itself (no official-backend selection).
    `snapshot(stage, step, density, loss, eval_metric)` is called with the
    density simulated at every iteration (after the optimizer's filter,
    projection or levelset transform).
    """
    t0 = time.time()

    def loss_fn(p):
        response, _ = challenge.component.response(p)
        return challenge.loss(response), challenge.eval_metric(response)

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    checkpoints = []

    def stage(name, opt, p, steps):
        state = opt.init(p)
        for i in range(steps):
            p = opt.params(state)
            (value, metric), grad = value_and_grad(p)
            state = opt.update(grad=grad, value=value, params=p, state=state)
            log({"stage": name, "step": i, "loss": float(value),
                 "eval_metric": float(metric), "time": time.time() - t0})
            if snapshot is not None:
                snapshot(name, i, np.asarray(p.array), float(value), float(metric))
        p = opt.params(state)
        checkpoints.append((name, p))
        return p

    for beta in betas:
        params = stage(f"density_beta{beta:g}", invrs_opt.density_lbfgsb(beta=beta),
                       params, density_steps)
    if levelset_steps > 0:
        params = stage("levelset", invrs_opt.levelset_lbfgsb(penalty=penalty),
                       binarize(params), levelset_steps)
    opt_time = time.time() - t0

    scored = []
    for name, p in checkpoints:
        b = binarize(p)
        e = evaluate(challenge, b)["eval_metric"]
        log({"stage": name, "binarized_eval_metric": e})
        scored.append((e, name, b))
    best = max(scored, key=lambda s: s[0])
    return best[2], checkpoints, opt_time, best[1]


class SnapshotRecorder:
    """Collects the per-iteration densities (float16) and metrics; `save`
    writes them to an npz (designs, loss, eval_metric, stage, step)."""

    def __init__(self):
        self.designs, self.loss, self.metric, self.stage, self.step = [], [], [], [], []

    def __call__(self, stage, step, density, loss, metric):
        self.designs.append(np.asarray(density, dtype=np.float16))
        self.loss.append(loss)
        self.metric.append(metric)
        self.stage.append(stage)
        self.step.append(step)

    def save(self, path):
        np.savez_compressed(
            path,
            designs=np.stack(self.designs),
            loss=np.asarray(self.loss),
            eval_metric=np.asarray(self.metric),
            stage=np.asarray(self.stage),
            step=np.asarray(self.step),
        )


def jsonl_logger(path, append=False):
    f = open(path, "a" if append else "w")

    def log(rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(rec, flush=True)

    return log


# ---------------------------------------------------------------------------
# Round 2: length-scale-feasible parameterization, beta continuation,
# straight-through stages, multi-resolution objective


def multi_resolution_loss(challenges, weights):
    """
    loss_fn(rho_params) -> (weighted mean loss, {resolution: eval metric}).

    Parameters
    ----------
    challenges : dict
        {resolution: Wakis challenge}; all share the gym's loss/eval metric.
    weights : dict
        {resolution: weight}; resolutions with weight 0 are not simulated.
    """
    active = [r for r in challenges if weights.get(r, 0.0) > 0.0]
    wsum = sum(weights[r] for r in active)

    def loss_fn(p):
        total, metrics = 0.0, {}
        for r in active:
            response, _ = challenges[r].component.response(p)
            total = total + weights[r] * challenges[r].loss(response)
            metrics[r] = challenges[r].eval_metric(response)
        return total / wsum, metrics

    return loss_fn


class CandidatePool:
    """Feasible (length-scale repaired) designs seen during the run, ranked
    by the Wakis r = 1 eval metric (which tracks the official metric)."""

    def __init__(self, size):
        self.size = size
        self.items = {}  # design bytes -> record

    def add(self, binary, metrics, stage, step):
        key = np.asarray(binary, bool).tobytes()
        score = float(metrics.get(1, min(metrics.values())))
        rec = {"design": np.asarray(binary, bool), "stage": stage, "step": step,
               "wakis_metrics": {int(k): float(v) for k, v in metrics.items()}, "score": score}
        if key not in self.items or score > self.items[key]["score"]:
            self.items[key] = rec

    def top(self):
        return sorted(self.items.values(), key=lambda r: -r["score"])[: self.size]


def optimize_feasible(
    challenges,
    feasible,
    latent0,
    stages,
    log: Callable[[dict], None] = print,
    snapshot: Optional[Callable] = None,
    n_candidates: int = 4,
    checkpoint=None,
):
    """
    Staged optimization over the latent of a `FeasibleDensity`.

    Parameters
    ----------
    challenges : dict
        {resolution: Wakis challenge}; resolution 1 must be present (it is
        the discretization the official evaluator uses).
    feasible : invrs_wakis.lengthscale.FeasibleDensity
        Parameterization (filter, projection, length-scale repair).
    latent0 : array
        Initial latent (pixels, gym bounds).
    stages : list of dict
        Each with keys 'beta' (number or "inf"), 'steps', and optionally
        'optimizer' ('lbfgsb' | 'adam'), 'lr' (Adam), 'ste' (bool: simulate
        the repaired binary design, straight-through gradient),
        'projection' ('tanh' (default) | 'ssp': subpixel-smoothed
        projection, also the STE backward pass), 'weights'
        ({resolution: weight}, default {1: 1}), 'name'.
    n_candidates : int
        Number of best feasible designs returned.
    checkpoint : Checkpointer, optional
        Saves (stage, step, latent, candidate pool) periodically and resumes
        from the last save (spot preemption). A resumed L-BFGS-B or Adam
        stage restarts its optimizer state from the saved latent.

    Returns
    -------
    (candidates, latent, optimization time [s]) where candidates is a list
    of dicts (design (bool), stage, step, wakis_metrics, score), best first.
    Every candidate meets the length-scale targets (imageruler).
    """
    import optax
    from totypes import types

    t0 = time.time()
    seed = feasible.seed
    pool = CandidatePool(n_candidates)
    latent = jnp_array(latent0)
    k0, i0 = 0, 0
    if checkpoint is not None:
        state = checkpoint.load()
        if state is not None:
            k0, i0 = int(state["stage"]), int(state["step"])
            latent = jnp_array(state["latent"])
            pool.items = state["pool"]
            t0 -= float(state["elapsed"])
            log({"resumed": True, "stage_index": k0, "step": i0})

    def save(k, i, lat, force=False):
        if checkpoint is not None:
            checkpoint.maybe_save({"stage": k, "step": i, "latent": np.asarray(lat),
                                   "pool": pool.items, "elapsed": time.time() - t0},
                                  force=force)

    def as_params(rho):
        return dataclasses_replace(seed, array=rho)

    for k, st in enumerate(stages):
        if k < k0:
            continue
        start = i0 if k == k0 else 0
        beta = float(st["beta"])
        ste = bool(st.get("ste", False))
        proj = st.get("projection", "tanh")
        weights = {int(r): float(w) for r, w in st.get("weights", {1: 1.0}).items()}
        name = st.get("name") or (f"{'ste' if ste else 'density'}"
                                  f"{'_ssp' if proj == 'ssp' else ''}_beta{beta:g}")
        loss_r = multi_resolution_loss(challenges, weights)
        to_rho = (lambda l: feasible.ste_density(l, beta, proj)) if ste else (
            lambda l: feasible.projected(l, beta, proj))

        def loss_fn(l, to_rho=to_rho, loss_r=loss_r):
            rho = to_rho(l)
            value, metrics = loss_r(as_params(rho))
            return value, (metrics, rho)

        vg = jax.value_and_grad(loss_fn, has_aux=True)

        def record(i, value, metrics, rho):
            m = {int(r): float(v) for r, v in metrics.items()}
            log({"stage": name, "step": i, "loss": float(value),
                 "eval_metric": m.get(1, min(m.values())),
                 "eval_metric_by_resolution": m, "time": time.time() - t0})
            if snapshot is not None:
                snapshot(name, i, np.asarray(rho), float(value), m.get(1, min(m.values())))
            if ste:  # the simulated design is feasible: a candidate
                pool.add(feasible.binary(rho), m, name, i)

        if st.get("optimizer", "lbfgsb") == "lbfgsb":
            opt = invrs_opt.lbfgsb()
            p = types.Density2DArray(array=latent, lower_bound=seed.lower_bound,
                                     upper_bound=seed.upper_bound)
            state = opt.init(p)
            for i in range(start, int(st["steps"])):
                p = opt.params(state)
                (value, (metrics, rho)), grad = vg(p.array)
                grad = dataclasses_replace(p, array=grad)
                state = opt.update(grad=grad, value=value, params=p, state=state)
                record(i, value, metrics, rho)
                save(k, i + 1, opt.params(state).array)
            latent = opt.params(state).array
        else:
            tx = optax.adam(float(st.get("lr", 0.02)))
            state = tx.init(latent)
            for i in range(start, int(st["steps"])):
                (value, (metrics, rho)), grad = vg(latent)
                record(i, value, metrics, rho)
                updates, state = tx.update(grad, state)
                latent = jax.numpy.clip(optax.apply_updates(latent, updates),
                                        seed.lower_bound, seed.upper_bound)
                save(k, i + 1, latent)

        # End of stage: the repaired design of the current latent (r = 1 only)
        binary = feasible.repair(np.asarray(feasible.projected(latent, beta)))
        rho = jax.numpy.asarray(binary, float) * (seed.upper_bound - seed.lower_bound) + seed.lower_bound
        _, metrics = multi_resolution_loss(challenges, {1: 1.0})(as_params(rho))
        m = {int(r): float(v) for r, v in metrics.items()}
        pool.add(binary, m, name, int(st["steps"]))
        log({"stage": name, "repaired_eval_metric": m[1],
             "length_scale": feasible.length_scale(binary), "time": time.time() - t0})
        save(k + 1, 0, latent, force=True)

    return pool.top(), latent, time.time() - t0


def jnp_array(x):
    return jax.numpy.asarray(np.asarray(x, dtype=float))


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


class Checkpointer:
    """
    Periodic optimization checkpoints, local and (on Cloud Batch) in GCS.

    The state (stage index, step, latent, candidate pool, elapsed time) and
    the snapshot recorder are pickled to `path`; if `gcs_uri` is set the file
    is copied there with `gcloud storage cp` (the task environment has the
    gcloud CLI) and fetched from there on start, so that a Batch task retried
    after a spot preemption resumes the run.
    """

    def __init__(self, path, gcs_uri=None, every_s=600.0, recorder=None):
        self.path = path
        self.gcs_uri = gcs_uri
        self.every_s = every_s
        self.recorder = recorder
        self._last = time.time()

    def _gcloud(self, src, dst):
        import subprocess

        r = subprocess.run(["gcloud", "storage", "cp", str(src), str(dst), "--quiet"],
                           capture_output=True, text=True)
        return r.returncode == 0

    def load(self):
        import pickle

        if self.gcs_uri and not self.path.exists():
            self._gcloud(self.gcs_uri, self.path)
        if not self.path.exists():
            return None
        state = pickle.loads(self.path.read_bytes())
        if self.recorder is not None and "snapshots" in state:
            self.recorder.__dict__.update(state["snapshots"])
        return state

    def maybe_save(self, state, force=False):
        import pickle

        if not force and time.time() - self._last < self.every_s:
            return
        if self.recorder is not None:
            state = dict(state, snapshots=dict(self.recorder.__dict__))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(self.path)
        if self.gcs_uri:
            self._gcloud(self.path, self.gcs_uri)
        self._last = time.time()
