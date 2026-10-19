"""Length-scale-feasible density parameterization.

latent (pixels, unbounded in principle, kept in [0, 1])
  -> symmetrize (the gym density's symmetries)
  -> conic filter of radius R pixels (weights max(0, 1 - r/R), normalized)
  -> tanh projection at threshold 1/2 with sharpness beta
     (Wang, Lazarov & Sigmund, Struct. Multidiscip. Optim. 43, 767 (2011))
  -> fixed pixels.

The filter and projection only make small features unlikely. The minimum
length scale is *guaranteed* by `repair`: the thresholded design is opened
with the imageruler brush of the target minimum width and closed with the
brush of the target minimum spacing (imageruler's own morphology, so the
result is feasible by the very definition used by imageruler and the
invrs leaderboard), alternately, with the fixed pixels re-imposed, until
imageruler measures both targets. In the straight-through (STE) stages the
simulated design *is* the repaired binary design, while the gradient flows
through the filter and projection (`ste_density`); every design the
optimizer evaluates is therefore fabricable, and any of them can be
submitted.

Subpixel-smoothed projection (SSP, `smoothed_projection`; Hammond et al.,
Opt. Express 33, 33620 (2025), arXiv:2503.20189) is an alternative to the
tanh projection: it also uses the norm of the gradient of the filtered
density, so that the projected design stays a differentiable function of
the interface positions even at beta = inf (binary except for a ~1 px layer
at the interfaces). Its threshold (SSP > 1/2 <=> filtered > 1/2) is that of
the tanh projection, so `repair` and imageruler apply unchanged.
"""

import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from imageruler import imageruler as ir
from totypes import types


def conic_kernel(radius: float) -> np.ndarray:
    """Normalized conic ("hat") filter kernel, weights max(0, 1 - r/R) [px]."""
    n = int(np.ceil(radius)) - 1
    d = np.arange(-n, n + 1)
    r = np.hypot(d[:, None], d[None, :])
    w = np.maximum(0.0, 1.0 - r / radius)
    return w / w.sum()


def tanh_projection(x, beta: float, eta: float = 0.5):
    """Smoothed Heaviside of Wang et al. (2011), maps [0, 1] -> [0, 1]."""
    if not np.isfinite(beta):
        return jnp.where(x > eta, 1.0, 0.0)
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))
    return num / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta)))


# Smoothing radius of SSP [px]: 2R slightly above the pixel pitch, so that
# every interface position has a pixel within R (else the gradient vanishes)
SSP_RADIUS_PX = 0.55


def smoothed_projection(x, grad_sq, beta: float, eta: float = 0.5,
                        radius: float = SSP_RADIUS_PX):
    """
    Subpixel-smoothed projection of Hammond et al. (2025), Eqs. (7), (10)-(12).

    With g = |grad x| (grad_sq = g^2, per pixel) and the signed distance
    d = (eta - x) / g to the planar approximation of the level set x = eta:
    for |d| < radius,
        out = (1 - F) P(x - R g F) + F P(x + R g (1 - F)),
        F = 1/2 - 15/16 u + 5/8 u^3 - 3/16 u^5,  u = d / R
    (the C^2 fill fraction of a parabolic smoothing kernel), else
    out = P(x), with P = `tanh_projection`. beta = inf gives F(d) near the
    level set and exactly 0 / 1 elsewhere. Same formula as
    `wakis.inverse.jax_tools.smoothed_projection` (Wakis fork).
    """
    nonzero = grad_sq > 0
    g = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, grad_sq, 1.0)), 1.0)
    d = (eta - x) / g
    smooth = nonzero & (jnp.abs(d) < radius)
    u = jnp.where(smooth, d / radius, 0.0)
    F = 0.5 - 15 / 16 * u + 5 / 8 * u**3 - 3 / 16 * u**5
    minus = tanh_projection(x - radius * g * F, beta, eta)
    plus = tanh_projection(x + radius * g * (1 - F), beta, eta)
    return jnp.where(smooth, (1 - F) * minus + F * plus, tanh_projection(x, beta, eta))


PROJECTIONS = ("tanh", "ssp")


@dataclasses.dataclass
class FeasibleDensity:
    """
    Filter-project parameterization of a gym `Density2DArray` with a
    guaranteed minimum length scale after `repair`.

    Parameters
    ----------
    seed : totypes.types.Density2DArray
        The gym's parameters (bounds, symmetries, fixed pixels and the
        minimum_width / minimum_spacing targets [px]).
    filter_radius : float, optional
        Conic filter radius [px]; default: the mean of the two targets.
    min_width, min_spacing : int, optional
        Length-scale targets [px] guaranteed by `repair`; default: the
        seed's minimum_width / minimum_spacing.
    """

    seed: types.Density2DArray
    filter_radius: Optional[float] = None
    min_width: Optional[int] = None
    min_spacing: Optional[int] = None

    def __post_init__(self):
        if self.min_width is None:
            self.min_width = int(self.seed.minimum_width)
        if self.min_spacing is None:
            self.min_spacing = int(self.seed.minimum_spacing)
        if self.filter_radius is None:
            self.filter_radius = 0.5 * (self.min_width + self.min_spacing)
        self.kernel = conic_kernel(self.filter_radius)
        shape = self.seed.shape[-2:]
        fs = self.seed.fixed_solid
        fv = self.seed.fixed_void
        self.fixed_solid = np.zeros(shape, bool) if fs is None else np.asarray(fs, bool)
        self.fixed_void = np.zeros(shape, bool) if fv is None else np.asarray(fv, bool)
        self.periodic = tuple(bool(p) for p in self.seed.periodic)

    # -- differentiable part -------------------------------------------------
    def _to_unit(self, array):
        lo, hi = self.seed.lower_bound, self.seed.upper_bound
        return (array - lo) / (hi - lo)

    def _from_unit(self, x):
        lo, hi = self.seed.lower_bound, self.seed.upper_bound
        return lo + (hi - lo) * x

    def symmetrize(self, array):
        d = dataclasses.replace(self.seed, array=array)
        return types.symmetrize_density(d).array

    def _filtered_extended(self, latent, extra):
        """Filtered latent on the design grid grown by `extra` pixels per
        side (the filter of the padded latent, as the interior)."""
        x = self._to_unit(self.symmetrize(jnp.asarray(latent)))
        n = self.kernel.shape[0] // 2 + extra
        mode = ["wrap" if p else "edge" for p in self.periodic]
        x = jnp.pad(x, ((n, n), (0, 0)), mode=mode[0])
        x = jnp.pad(x, ((0, 0), (n, n)), mode=mode[1])
        k = jnp.asarray(self.kernel)[None, None]
        return jax.lax.conv_general_dilated(x[None, None], k, (1, 1), "VALID")[0, 0]

    def filtered(self, latent):
        """Symmetrized, conic-filtered latent in [0, 1] (unit bounds)."""
        return self._filtered_extended(latent, 0)

    def filtered_gradient(self, latent):
        """
        (filtered density, |grad|^2 [1/px^2]): central differences of the
        filtered field computed one pixel beyond the design grid, so the
        boundary pixels use the same (padded) surroundings as the filter.
        """
        y = self._filtered_extended(latent, 1)
        gx = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
        gy = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
        return y[1:-1, 1:-1], gx**2 + gy**2

    def _with_fixed(self, y):
        y = jnp.where(self.fixed_solid, 1.0, jnp.where(self.fixed_void, 0.0, y))
        return self._from_unit(y)

    def projected(self, latent, beta, projection: str = "tanh"):
        """
        Filtered, projected density (gym bounds) with fixed pixels set.
        projection: 'tanh' (Wang et al. 2011) or 'ssp' (subpixel-smoothed
        projection, `smoothed_projection`).
        """
        if projection == "ssp":
            y, q = self.filtered_gradient(latent)
            return self._with_fixed(smoothed_projection(y, q, beta))
        if projection != "tanh":
            raise ValueError(f"projection must be one of {PROJECTIONS}")
        return self._with_fixed(tanh_projection(self.filtered(latent), beta))

    def ste_density(self, latent, beta, projection: str = "tanh"):
        """Repaired binary design in the forward pass, gradient of
        `projected` in the backward pass (straight-through estimator).
        With projection='ssp' the backward pass is the SSP gradient, which
        stays finite at beta = inf (the tanh one vanishes there)."""
        rho = self.projected(latent, beta, projection)
        feasible = jax.pure_callback(
            lambda r: np.asarray(self.repair(np.asarray(r)), dtype=r.dtype),
            jax.ShapeDtypeStruct(rho.shape, rho.dtype),
            jax.lax.stop_gradient(rho),
        )
        feasible = self._from_unit(feasible)
        return rho + jax.lax.stop_gradient(feasible - rho)

    # -- length-scale guarantee (host, numpy) ---------------------------------
    def binary(self, rho) -> np.ndarray:
        mid = 0.5 * (self.seed.lower_bound + self.seed.upper_bound)
        return np.asarray(rho) > mid

    def _impose(self, x):
        return (x | self.fixed_solid) & ~self.fixed_void

    def repair(self, rho, max_iter: int = 30) -> np.ndarray:
        """
        Binary design (bool) meeting the length-scale targets.

        1. Alternating opening with the imageruler brush of min_width
           (removes narrower solid features) and closing with the brush of
           min_spacing (fills narrower gaps), re-imposing the fixed pixels
           and the symmetries, until imageruler accepts the design or a fixed
           point is reached.
        2. Closing can leave narrow solid necks that opening does not
           remove (the open/close fixed point is not always feasible). The
           remaining flagged pixels are then grown: solid violations are
           dilated by a (2 min_width - 1) square, void violations filled, and the
           result closed again. The solid set only grows in this phase, so
           it terminates (in the limit: everything not fixed void is solid,
           which is feasible).
        """
        x = self._impose(self.binary(rho))
        if self.meets_targets(x):
            return x
        kw = ir.kernel_for_length_scale(self.min_width)
        ks = ir.kernel_for_length_scale(self.min_spacing)
        pad = ir.PaddingMode.SOLID
        # Growth brush: a square of side 2 min_width - 1 around a flagged
        # pixel contains a min_width brush placement that covers it
        square = np.ones((2 * self.min_width - 1,) * 2, bool)

        def close(y):
            return ~ir.binary_opening(~y, ks, self.periodic, pad)

        for _ in range(max_iter):
            prev = x
            x = close(ir.binary_opening(x, kw, self.periodic, pad))
            x = self._impose(self._symmetrize_binary(x))
            if self.meets_targets(x):
                return x
            if np.array_equal(x, prev):
                break
        for _ in range(4 * max_iter):
            vs = ir.length_scale_violations_solid(x, self.min_width, self.periodic)
            vv = ir.length_scale_violations_solid(~x, self.min_spacing, self.periodic)
            grow = ir.binary_dilation(vs, square, self.periodic, ir.PaddingMode.VOID)
            x = self._impose(self._symmetrize_binary(close(x | grow | vv)))
            if self.meets_targets(x):
                return x
        raise RuntimeError("length-scale repair did not converge")

    def _symmetrize_binary(self, x):
        if not self.seed.symmetries:
            return x
        s = np.asarray(self.symmetrize(jnp.asarray(x, float)))
        return s > 0.5 - 1e-9  # ties (pixel pairs that disagree) become solid

    def length_scale(self, x) -> Tuple[int, int]:
        """(solid, void) minimum length scale [px] measured as the invrs
        leaderboard does (design array alone, imageruler defaults)."""
        return tuple(int(v) for v in ir.minimum_length_scale(np.asarray(x, bool),
                                                             periodic=self.periodic))

    def meets_targets(self, x) -> bool:
        w, s = self.length_scale(x)
        return w >= self.min_width and s >= self.min_spacing


def random_latent(seed: types.Density2DArray, key, radius: float = 4.0) -> np.ndarray:
    """Smooth random start: uniform noise blurred by a conic kernel of
    `radius` px, rescaled to span the density bounds."""
    shape = seed.shape[-2:]
    noise = np.asarray(jax.random.uniform(key, shape))
    k = conic_kernel(radius)
    n = k.shape[0] // 2
    p = np.pad(noise, n, mode="reflect")
    from scipy.signal import convolve2d

    y = convolve2d(p, k, mode="valid")
    y = (y - y.min()) / (np.ptp(y) + 1e-12)
    return seed.lower_bound + (seed.upper_bound - seed.lower_bound) * y
