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
from typing import Callable, List, Sequence, Tuple

import invrs_opt
import jax

from invrs_wakis.analysis import binarize, evaluate


def optimize(
    challenge,
    params,
    betas: Sequence[float] = (2.0, 4.0, 8.0),
    density_steps: int = 40,
    levelset_steps: int = 80,
    penalty: float = 1.0,
    log: Callable[[dict], None] = print,
) -> Tuple[object, List[Tuple[str, object]], float, str]:
    """
    Run the stages; return (selected binary params, stage checkpoints,
    optimization time [s], name of the selected stage).

    The selected design is the thresholded stage result with the best eval
    metric according to `challenge` itself (no official-backend selection).
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


def jsonl_logger(path):
    f = open(path, "w")

    def log(rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(rec, flush=True)

    return log
