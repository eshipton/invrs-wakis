"""Length-scale guarantee of the round-2 parameterization (fast, no simulation)."""

import types as pytypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from invrs_gym import challenges

from invrs_wakis.lengthscale import FeasibleDensity, random_latent
from invrs_wakis.optimize import optimize_feasible

NAMES = [
    "ceviche_lightweight_waveguide_bend",
    "ceviche_lightweight_mode_converter",
    "ceviche_lightweight_power_splitter",
    "ceviche_lightweight_beam_splitter",
    "ceviche_lightweight_wdm",
]


@pytest.mark.parametrize("name", NAMES)
def test_repair_meets_targets(name):
    """Random projected designs (1-px features before repair) always meet the
    gym's minimum width/spacing after repair, keep fixed pixels and symmetry."""
    params = challenges.BY_NAME[name]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    raw_ok = 0
    for k in range(8):
        lat = random_latent(params, jax.random.PRNGKey(k), radius=1.5 + k % 3 * 2)
        rho = np.asarray(fd.projected(lat, 32.0))
        raw_ok += fd.meets_targets(fd.binary(rho))
        x = fd.repair(rho)
        w, s = fd.length_scale(x)
        assert w >= params.minimum_width and s >= params.minimum_spacing
        assert np.all(x[fd.fixed_solid]) and not np.any(x[fd.fixed_void])
        sym = np.asarray(fd.symmetrize(jnp.asarray(x, float)))
        assert np.array_equal(sym > 0.5, x)
    assert raw_ok < 8  # the check is not vacuous


def test_ste_gradient_and_optimizer_plumbing():
    """STE forward = repaired binary design; gradient = projected path; the
    staged optimizer returns only feasible candidates (mock objective)."""
    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    target = jnp.asarray(random_latent(params, jax.random.PRNGKey(7), 4.0))

    def response(p):
        return jnp.mean((p.array - target) ** 2), {}

    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=response),
        loss=lambda r: r,
        eval_metric=lambda r: -r,
    )
    lat = np.asarray(params.array)
    rho = fd.ste_density(jnp.asarray(lat), 64.0)
    assert set(np.unique(np.asarray(rho))) <= {0.0, 1.0}
    g = jax.grad(lambda l: jnp.sum(fd.ste_density(l, 8.0) * target))(jnp.asarray(lat))
    assert float(jnp.abs(g).sum()) > 0
    stages = [
        {"beta": 4, "steps": 3},
        {"beta": 64, "steps": 3, "ste": True, "optimizer": "adam", "lr": 0.05},
    ]
    cands, _, _ = optimize_feasible({1: mock}, fd, lat, stages, log=lambda r: None)
    assert cands
    for c in cands:
        assert fd.meets_targets(c["design"])
