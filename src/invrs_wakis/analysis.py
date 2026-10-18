"""Evaluation helpers: binarization, length scale (imageruler), official metrics."""

import dataclasses
from typing import Any, Dict, Tuple

import jax
import numpy as np
from invrs_gym.utils import metrics as gym_metrics
from totypes import types


def binarize(params: types.Density2DArray) -> types.Density2DArray:
    """Threshold a density at the midpoint of its bounds (fixed pixels kept)."""
    mid = 0.5 * (params.lower_bound + params.upper_bound)
    array = np.where(np.asarray(params.array) > mid, params.upper_bound, params.lower_bound)
    if params.fixed_solid is not None:
        array = np.where(params.fixed_solid, params.upper_bound, array)
    if params.fixed_void is not None:
        array = np.where(params.fixed_void, params.lower_bound, array)
    return dataclasses.replace(params, array=jax.numpy.asarray(array, float))


def length_scale_pixels(binary: np.ndarray, pad: Any = None) -> Tuple[int, int]:
    """
    Minimum solid and void length scales [pixels] measured by imageruler.

    Parameters
    ----------
    binary : ndarray of bool
        Binary pattern.
    pad : ndarray of bool, optional
        Larger pattern to measure instead (e.g. the design with its
        surrounding waveguides), so that features touching the design border
        are measured together with what they connect to.
    """
    from imageruler import imageruler

    x = np.asarray(binary if pad is None else pad, dtype=bool)
    return tuple(int(v) for v in imageruler.minimum_length_scale(x))


def ceviche_length_scale(model, params, border: int = 4) -> Tuple[int, int]:
    """Length scale of a ceviche design, measured with `border` pixels of its
    surroundings (input/output waveguides, cladding) included."""
    density = np.asarray(model.density(np.asarray(params.array))) > 0.5
    x0, y0, x1, y1 = model.design_region_coords
    window = density[x0 - border : x1 + border, y0 - border : y1 + border]
    return length_scale_pixels(window)


def evaluate(challenge, params, **response_kwargs) -> Dict[str, Any]:
    """Response, loss, eval metric and binarization degree of `params`."""
    response, aux = challenge.component.response(params, **response_kwargs)
    return {
        "response": response,
        "loss": float(challenge.loss(response)),
        "eval_metric": float(challenge.eval_metric(response)),
        "binarization_degree": float(gym_metrics.binarization_degree(params)),
    }
