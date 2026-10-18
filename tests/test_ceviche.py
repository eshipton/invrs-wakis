"""Wakis ceviche component vs the gym's ceviche (FDFD) backend.

At resolution 1 the Wakis Ez/Hx/Hy staggering is that of ceviche's Yee FDFD
grid and the run is sampled at the frequency whose discrete-time omega equals
ceviche's omega, so with ceviche's own port estimator the two backends solve
the same discrete problem; they differ only in the absorbing layers (CPML vs
SC-PML) and in the time-domain DFT truncation. Observed |S|^2 differences are
~1e-4 (absolute); the tolerance 2e-3 leaves margin for design dependence.
"""

import dataclasses
import warnings

import jax
import numpy as np
import pytest

from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

warnings.filterwarnings("ignore")
FAST = dict(run_time=0.6e-12)


@pytest.mark.parametrize(
    "name",
    [
        "ceviche_lightweight_waveguide_bend",
        "ceviche_lightweight_mode_converter",  # second-order output mode
        "ceviche_lightweight_power_splitter",  # three ports
    ],
)
def test_matches_gym_backend(name):
    """|S|^2 of a random gray design agrees with ceviche within 2e-3, and the
    gym's eval metric within 1% (it divides |S|^2 distances by the target
    window width, down to 0.01 for reflections, so it amplifies them 100x)."""
    wakis_ch, gym_ch = wakis_ceviche_challenge(
        name, SimOptions(port_estimator="ceviche", **FAST)
    )
    params = gym_ch.component.init(jax.random.PRNGKey(0))
    r_gym, _ = gym_ch.component.response(params)
    r_wakis, _ = wakis_ch.component.response(params)
    assert r_wakis.s_parameters.shape == r_gym.s_parameters.shape
    t_gym = np.abs(np.asarray(r_gym.s_parameters)) ** 2
    t_wakis = np.abs(np.asarray(r_wakis.s_parameters)) ** 2
    print(name, "max |dS2| =", np.max(np.abs(t_gym - t_wakis)))
    np.testing.assert_allclose(t_wakis, t_gym, atol=2e-3)
    e_w, e_g = float(wakis_ch.eval_metric(r_wakis)), float(gym_ch.eval_metric(r_gym))
    assert abs(e_w - e_g) < 1e-2 * abs(e_g)


def test_init_is_gym_init():
    """The component returns exactly the gym's parameters (shape, fixed
    pixels, symmetries, length-scale targets)."""
    wakis_ch, gym_ch = wakis_ceviche_challenge("ceviche_lightweight_waveguide_bend")
    key = jax.random.PRNGKey(3)
    p_w, p_g = wakis_ch.component.init(key), gym_ch.component.init(key)
    np.testing.assert_array_equal(p_w.array, p_g.array)
    assert p_w.symmetries == p_g.symmetries
    assert (p_w.minimum_width, p_w.minimum_spacing) == (p_g.minimum_width, p_g.minimum_spacing)
    np.testing.assert_array_equal(p_w.fixed_solid, p_g.fixed_solid)


def _straight_guide(gym_ch):
    """Mode-converter design continuing the input guide straight across."""
    m = gym_ch.component.ceviche_model
    params = gym_ch.component.init(jax.random.PRNGKey(0))
    x0, y0, x1, y1 = m.design_region_coords
    row = np.asarray(m.density_bg)[x0 - 1, y0:y1]
    return dataclasses.replace(params, array=np.tile(row, (x1 - x0, 1)))


def test_uniform_guide_has_no_reflection():
    """A uniform waveguide reflects nothing. The Wakis modal estimator (exact
    discrete orthogonality) gives |S11|^2 < 1e-8, whereas ceviche's
    continuum-form estimator shows a floor of ~8e-4 (-31 dB) at 40 nm
    pixels, which the 'ceviche' estimator reproduces."""
    name = "ceviche_lightweight_mode_converter"
    modal, gym_ch = wakis_ceviche_challenge(name, SimOptions(**FAST))
    cev = wakis_ceviche_challenge(name, SimOptions(port_estimator="ceviche", **FAST))[0]
    params = _straight_guide(gym_ch)
    s11 = {
        k: np.abs(np.asarray(ch.component.response(params)[0].s_parameters[:, 0, 0])) ** 2
        for k, ch in (("gym", gym_ch), ("modal", modal), ("ceviche", cev))
    }
    print(s11)
    assert np.all(s11["modal"] < 1e-8)
    assert np.all(s11["gym"] > 5e-4)
    np.testing.assert_allclose(s11["ceviche"], s11["gym"], rtol=0.02)


@pytest.mark.parametrize("estimator", ["modal", "ceviche"])
def test_gradient_matches_finite_differences(estimator):
    """Directional derivative of the gym's loss through the Wakis component:
    exact discrete adjoint (jax.custom_vjp) vs central differences
    (step 1e-3, O(h^2) truncation ~1e-6 relative)."""
    wakis_ch, _ = wakis_ceviche_challenge(
        "ceviche_lightweight_waveguide_bend",
        SimOptions(port_estimator=estimator, store_dtype=np.float64, **FAST),
    )
    params = wakis_ch.component.init(jax.random.PRNGKey(1))

    def loss(p):
        return wakis_ch.loss(wakis_ch.component.response(p)[0])

    _, grad = jax.value_and_grad(loss)(params)
    v = np.random.default_rng(0).normal(size=params.shape)
    h = 1e-3
    fd = (
        float(loss(dataclasses.replace(params, array=params.array + h * v)))
        - float(loss(dataclasses.replace(params, array=params.array - h * v)))
    ) / (2 * h)
    ad = float(np.sum(np.asarray(grad.array) * v))
    print(estimator, ad, fd)
    assert abs(ad - fd) < 1e-4 * abs(fd)
