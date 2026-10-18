"""Wakis metalens component vs the gym's fmmax backend (slow: ~1-2 min)."""

import dataclasses
import pathlib
import warnings

import jax
import numpy as np
import pytest

from invrs_wakis.metalens import MetalensOptions, wakis_metalens_challenge

warnings.filterwarnings("ignore")
REFERENCE = pathlib.Path(__file__).parents[2] / "invrs-gym/reference_designs/metalens/Rasmus123nm.csv"


@pytest.mark.slow
@pytest.mark.skipif(not REFERENCE.exists(), reason="invrs-gym reference designs not found")
def test_reference_design_matches_fmmax():
    """Focal intensity enhancement of a published binary design (EX).

    At resolution 1 (20 nm, ~9 cells per wavelength in the n = 2.4 lens at
    450 nm) staircasing and numerical dispersion limit the agreement; the
    observed deviation from fmmax (itself truncated at 280 Fourier orders) is
    <= 5 %, the tolerance is 10 %.
    """
    wakis_ch, gym_ch = wakis_metalens_challenge(MetalensOptions())
    design = np.genfromtxt(REFERENCE, delimiter=",")
    params = gym_ch.component.init(jax.random.PRNGKey(0))
    params = dataclasses.replace(params, array=design, fixed_solid=None, fixed_void=None)
    e_gym = np.asarray(gym_ch.component.response(params)[0].enhancement_ex)
    e_wakis = np.asarray(wakis_ch.component.response(params)[0].enhancement_ex)
    print(e_gym, e_wakis)
    np.testing.assert_allclose(e_wakis, e_gym, rtol=0.10)
