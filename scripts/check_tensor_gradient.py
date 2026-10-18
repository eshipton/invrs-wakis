"""Directional finite-difference check of the gym loss gradient through the
Wakis adjoint with smoothing='tensor' (r = 2), for a smooth gray design.

  uv run python scripts/check_tensor_gradient.py --outdir out [--resolution 2]
(--shard/--nshards accepted for the Batch sweep driver; one shard)."""

import argparse
import dataclasses
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from invrs_wakis._design import TENSOR_CLAMPED
from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge
from invrs_wakis.lengthscale import random_latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--name", default="ceviche_lightweight_waveguide_bend")
    ap.add_argument("--h", type=float, default=1e-3)
    a = ap.parse_args()
    ch, _ = wakis_ceviche_challenge(a.name, SimOptions(
        resolution=a.resolution, smoothing="tensor", run_time=0.8e-12))
    p0 = ch.component.init(jax.random.PRNGKey(0))
    rho = jnp.asarray(random_latent(p0, jax.random.PRNGKey(1), 4.0))

    def f(x):
        return ch.loss(ch.component.response(dataclasses.replace(p0, array=x))[0])

    t = time.time()
    val, grad = jax.value_and_grad(f)(rho)
    res = {"loss": float(val), "grad_time_s": time.time() - t, "checks": []}
    rng = np.random.default_rng(0)
    for k in range(2):
        v = rng.standard_normal(rho.shape)
        v /= np.linalg.norm(v)
        fd = (float(f(rho + a.h * v)) - float(f(rho - a.h * v))) / (2 * a.h)
        ad = float(jnp.sum(grad * v))
        res["checks"].append({"adjoint": ad, "fd": fd, "rel_err": abs(ad - fd) / abs(fd)})
        print(res["checks"][-1], flush=True)
    res["tensor_clamped"] = dict(TENSOR_CLAMPED)
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tensor_gradient_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
