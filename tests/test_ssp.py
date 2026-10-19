"""Subpixel-smoothed projection (SSP) in the round-2 parameterization
(fast, no simulation)."""

import types as pytypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from invrs_gym import challenges

from invrs_wakis.lengthscale import (
    FeasibleDensity,
    random_latent,
    smoothed_projection,
    tanh_projection,
)
from invrs_wakis.optimize import optimize_feasible

jax.config.update("jax_enable_x64", True)

NAMES = ["ceviche_lightweight_waveguide_bend", "ceviche_lightweight_mode_converter"]


def _density(name, key=0):
    params = challenges.BY_NAME[name]().component.init(jax.random.PRNGKey(key))
    return params, FeasibleDensity(params)


@pytest.mark.parametrize("name", NAMES)
def test_threshold_repair_and_imageruler_unchanged(name):
    """SSP has the threshold of the tanh projection (SSP > 1/2 iff filtered >
    1/2, eta = 1/2), so the repaired design and the imageruler length scales
    are identical to the round-2 ones; repair meets the targets; symmetry
    and fixed pixels are kept. Not vacuous: SSP differs from tanh at the
    interfaces, and some raw designs violate the targets before repair."""
    params, fd = _density(name)
    raw_bad = 0
    for k in range(6):
        lat = random_latent(params, jax.random.PRNGKey(k), radius=1.5 + 2 * (k % 3))
        for beta in (8.0, 32.0, np.inf):
            r_ssp = np.asarray(fd.projected(lat, beta, "ssp"))
            r_tanh = np.asarray(fd.projected(lat, beta, "tanh"))
            if beta == np.inf:  # at small beta both are smooth on the pixel scale
                assert np.abs(r_ssp - r_tanh).max() > 0.1
            b = fd.binary(r_ssp)
            assert np.array_equal(b, fd.binary(r_tanh))
            assert fd.length_scale(b) == fd.length_scale(fd.binary(r_tanh))
            assert np.all(r_ssp[fd.fixed_solid] == params.upper_bound)
            assert np.all(r_ssp[fd.fixed_void] == params.lower_bound)
            sym = np.asarray(fd.symmetrize(jnp.asarray(r_ssp)))
            np.testing.assert_allclose(sym, r_ssp, atol=1e-12)
        raw_bad += not fd.meets_targets(b)
        x = fd.repair(r_ssp)
        assert fd.meets_targets(x)
    assert raw_bad > 0


def test_ssp_beta_inf_binary_except_interface_layer():
    """At beta = inf SSP is exactly 0/1 except on pixels within R = 0.55 px
    of the level set (|filtered - 1/2| < R |grad|), and those are gray."""
    params, fd = _density(NAMES[0])
    lat = random_latent(params, jax.random.PRNGKey(3), 4.0)
    y, q = fd.filtered_gradient(lat)
    out = np.asarray(smoothed_projection(y, q, np.inf))
    band = np.abs(np.asarray(y) - 0.5) < 0.55 * np.sqrt(np.asarray(q))
    assert np.all(np.isin(out[~band], (0.0, 1.0)))
    assert np.all((out[band] > 0) & (out[band] < 1))
    assert 0 < band.mean() < 0.3


def test_gradient_at_beta_inf_and_finite_differences():
    """d(objective)/d(latent) through filter + SSP is nonzero at beta = inf
    (the tanh gradient is exactly zero there) and matches central finite
    differences along random directions (beta = 32 and inf)."""
    params, fd = _density(NAMES[1])
    lat = jnp.asarray(random_latent(params, jax.random.PRNGKey(5), 4.0))
    w = jnp.asarray(np.random.default_rng(0).normal(size=lat.shape))

    def obj(l, beta, proj):
        return jnp.sum(w * fd.projected(l, beta, proj))

    g_tanh = jax.grad(obj)(lat, np.inf, "tanh")
    assert float(jnp.abs(g_tanh).max()) == 0.0
    rng = np.random.default_rng(1)
    for beta in (32.0, np.inf):
        g = jax.grad(obj)(lat, beta, "ssp")
        assert float(jnp.abs(g).max()) > 0
        for _ in range(3):
            v = jnp.asarray(rng.normal(size=lat.shape))
            h = 1e-6
            fd_val = (obj(lat + h * v, beta, "ssp") - obj(lat - h * v, beta, "ssp")) / (2 * h)
            ad = jnp.sum(g * v)
            assert abs(float(fd_val - ad)) < 1e-5 * abs(float(ad)), (beta, fd_val, ad)


def test_interface_motion_is_smooth_at_beta_inf():
    """A straight level set x = x0 moved through the grid in 1/40 px steps:
    the SSP(beta = inf) material sum M(x0) is continuous and increasing,
    with slope dM/dx0 = sum_i 15/(16 R) (1 - u_i^2)^2, u_i = (x_i - x0)/R
    (analytic), whose mean over a pixel is exactly 1 (the area rate of the
    true interface) but which ripples between 0.10 (interface midway
    between pixel centres) and 1.70 (on a centre) for R = 0.55 px. The step
    projection instead jumps by whole pixels with zero slope. Mutation:
    R = 0.45 px < 1/2 px leaves interface positions with zero slope (frozen
    gradient)."""
    xs = np.arange(16.0)

    def material(x0, beta, radius=0.55):
        y = 0.5 + 0.1 * (x0 - xs)  # filtered density, solid for x < x0
        q = jnp.full(xs.shape, 0.01)
        return jnp.sum(smoothed_projection(jnp.asarray(y), q, beta, radius=radius))

    x0s = np.linspace(7.0, 9.0, 81) + 1 / 80  # no sample exactly on a pixel centre
    slope = np.array([float(jax.grad(material)(x0, np.inf)) for x0 in x0s])
    M = np.array([float(material(x0, np.inf)) for x0 in x0s])
    M_step = np.array([float(material(x0, np.inf, radius=1e-9)) for x0 in x0s])
    u = (xs[None, :] - x0s[:, None]) / 0.55
    expected = np.sum(np.where(np.abs(u) < 1, 15 / 16 / 0.55 * (1 - u**2) ** 2, 0), axis=1)
    np.testing.assert_allclose(slope, expected, rtol=1e-12)
    assert np.all(np.diff(M) > 0) and np.diff(M).max() < 1.75 * (x0s[1] - x0s[0])
    assert abs(np.mean(slope[:-1]) - 1) < 0.01  # one period of 1 px per 40 samples x 2
    assert 0.09 < slope.min() < 0.12 and 1.69 < slope.max() < 1.71
    assert np.abs(np.diff(M_step)).max() == 1.0  # whole-pixel jumps
    slope_frozen = np.array([float(jax.grad(material)(x0, np.inf, 0.45)) for x0 in x0s])
    assert slope_frozen.min() == 0.0


def test_optimizer_ssp_stages():
    """Stages with projection 'ssp' and beta 'inf' (L-BFGS-B and STE) run,
    decrease a mock loss and return only feasible candidates."""
    params, fd = _density(NAMES[0])
    target = jnp.asarray(random_latent(params, jax.random.PRNGKey(7), 4.0))

    def response(p):
        return jnp.mean((p.array - target) ** 2), {}

    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=response),
        loss=lambda r: r,
        eval_metric=lambda r: -r,
    )
    logs = []
    stages = [
        {"beta": 8, "steps": 3, "projection": "ssp"},
        {"beta": "inf", "steps": 4, "projection": "ssp"},
        {"beta": "inf", "steps": 3, "projection": "ssp", "ste": True,
         "optimizer": "adam", "lr": 0.05},
    ]
    cands, _, _ = optimize_feasible({1: mock}, fd, np.asarray(params.array), stages,
                                    log=logs.append)
    assert cands and all(fd.meets_targets(c["design"]) for c in cands)
    inf_stage = [r["loss"] for r in logs if r.get("stage") == "density_ssp_betainf"
                 and "loss" in r]
    assert len(inf_stage) == 4 and inf_stage[-1] < inf_stage[0]


def test_tanh_limit():
    """SSP -> identity for beta -> 0+ (no projection) and -> F(d) for beta
    -> inf, as in Hammond et al. (2025), Fig. 4."""
    y = jnp.linspace(0.3, 0.7, 41)
    q = jnp.full(y.shape, 0.04 ** 2)
    np.testing.assert_allclose(smoothed_projection(y, q, 1e-7), y, atol=1e-6)
    u = (0.5 - y) / 0.04 / 0.55
    F = np.where(np.abs(u) < 1, 0.5 - 15 / 16 * u + 5 / 8 * u**3 - 3 / 16 * u**5,
                 (u < 0).astype(float))
    np.testing.assert_allclose(smoothed_projection(y, q, np.inf), F, atol=1e-12)
    assert float(jnp.abs(smoothed_projection(y, q, 400.0) - F).max()) < 1e-3
    np.testing.assert_allclose(tanh_projection(y, np.inf), (y > 0.5).astype(float))
