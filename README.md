# invrs-wakis

Wakis backend for the [invrs-gym](https://github.com/invrs-io/gym) photonic
inverse-design challenges (Schubert et al., arXiv:2410.24132). The time-domain
solver is [Wakis](https://github.com/ImpedanCEi/wakis), a finite integration
technique (FIT) code. This fork adds an exact discrete adjoint and a JAX
wrapper (`wakis.inverse`).

The components here take exactly the gym's parameters (`totypes.Density2DArray`
with the gym's shapes, fixed pixels, symmetries and length-scale targets). They
return the gym's response types (`CevicheResponse`, `MetalensResponse`), so the
gym's own `loss`, `eval_metric` and `metrics` apply unchanged, and
`jax.value_and_grad(loss_fn)` differentiates through the Wakis adjoint
(`jax.custom_vjp`).

```python
from invrs_wakis import SimOptions, wakis_ceviche_challenge

challenge, gym_challenge = wakis_ceviche_challenge(
    "ceviche_lightweight_waveguide_bend", SimOptions(port_estimator="ceviche"))
params = challenge.component.init(jax.random.PRNGKey(0))
loss, grad = jax.value_and_grad(
    lambda p: challenge.loss(challenge.component.response(p)[0]))(params)
official = gym_challenge.eval_metric(gym_challenge.component.response(params)[0])
```

## What is simulated

**Ceviche challenges** (`invrs_wakis.ceviche`). This is the 2D Ez problem. The
mesh has a single periodic cell along z. CPML replaces ceviche's SC-PML and has
the same thickness. Geometry, permittivities (2.25 and 12.25), ports, mode
orders and wavelengths are read from the gym's `ceviche_challenges` model
object.
- `resolution` r gives r×r Wakis cells per gym pixel. The Ez samples sit at the
  centres of the fine pixels. At r = 1 the Ez/Hx/Hy staggering is identical to
  ceviche's Yee FDFD grid.
- A single broadband Gaussian pulse per excited port covers all wavelengths.
- `match_fdfd`: the run is sampled at the frequency whose discrete-time
  angular frequency 2/dt sin(ω dt/2) equals ceviche's ω. This removes the
  leapfrog temporal dispersion, so at r = 1 FIT and FDFD solve the same
  discrete problem.
- Port estimators:
  - `modal` uses Wakis discrete eigenmodes and the exact discrete
    orthogonality relation.
  - `ceviche` re-implements `ceviche_challenges.scattering.calculate_amplitudes`
    on the Wakis DFT fields, using ceviche's own port modes. It reproduces the
    official evaluator.

**Metalens** (`invrs_wakis.metalens`). This is the 2D x–z cross-section,
periodic over 16 µm, with CPML above and below. The gym's EX polarization is
the Wakis (Ex, Ey, Hz) family.
- The source is a current sheet with the gym's smoothed top-hat profile, 1 µm
  below the lens.
- The enhancement is |E|² at the focus divided by the same quantity for the
  gym's no-lens reference stack. Both come from one broadband run each, at
  450, 550 and 650 nm.
- E edges take the harmonic mean of the pixels they cross.

## Validation (local, CPU)

| check | result |
|---|---|
| bend / mode converter / power splitter, random gray design, Wakis r=1 + `ceviche` estimator vs gym ceviche | max abs Δ\|S\|² = 7.5e-5 / 3.6e-4 / 5.3e-4 |
| bend, gym eval_metric | −8.1400 (ceviche) vs −8.1402 (Wakis) |
| wdm, random design | \|S\|² agrees to ≤1e-4 (run time 0.4–0.8 ps) |
| uniform straight guide through the design region, \|S11\|² | ceviche **7.9e-4 (−31 dB floor)**; Wakis modal 1.6e-12; Wakis `ceviche` estimator 7.9e-4 |
| gradient of the gym loss (bend) vs central FD, random directions | relative error 4e-7 to 2e-4 (modal and ceviche estimators) |
| metalens, reference design Rasmus123nm, enhancement at 450/550/650 nm | fmmax 16.39 / 14.30 / 15.45; Wakis r=1 (20 nm) 15.55 / 14.27 / 16.09 |

**Finding: ceviche's reflection estimate has a floor.** At the 40 nm
lightweight resolution, `calculate_amplitudes` combines discrete Yee fields
with continuum-form mode overlaps: it uses H = β/(ωμ) e and averages Ez over
two columns. A purely forward wave therefore leaks about 2.8 % in amplitude
(−31 dB in power) into the "backward" amplitude. This floor is only 11 dB
below the −20 dB reflection target. Optimization therefore uses the `ceviche`
estimator at r = 1, which matches what the official metric measures. The
`modal` estimator gives the physically consistent value.

## Running

Local quick checks (each under about 1 min):

```bash
uv sync
uv run pytest -q tests/
```

Optimizations and sweeps are sharded and meant for cloud CPU batch jobs:

```bash
uv run python scripts/run_challenge.py --shard i --nshards 6 --outdir out \
    [--challenges ...] [--seeds 0 1] [--density-steps 40 --levelset-steps 80]
uv run python scripts/crossval_ceviche.py --shard i --nshards 5 --outdir out [--fine]
```

Rendering (matplotlib Agg + pillow, headless):

```bash
uv run python scripts/render.py --results <collected dir> [--runs]   # gallery.png (+ re-render runs)
```

Each run directory contains, besides `summary.json`/`params.json`/`design.npy`:
`snapshots.npz` (density simulated at every iteration, float16, with loss and
eval metric), `fields.npz` (Wakis DFT fields of the final design at the
challenge wavelengths and, for ceviche, the |S|² spectrum on a dense grid —
all from one extra broadband run), and the renders `design.png`,
`optimization.gif` (design evolving with the eval-metric trace),
`fields.png` (Re E and |E|² per wavelength with design outline and ports;
metalens: intensity enhancement map), `wave.gif` (phase sweep
Re(E e^{iφ}) of the DFT fields) and `spectrum.png` (ceviche: broadband
|S_j1|² vs the target windows, official values marked).

`run_challenge.py` maps shard i to `specs[i::N]`, where specs are the
(challenge, seed) pairs. The optimizer stack is the gym's recommendation from
invrs-opt (MIT):
1. `density_lbfgsb`, a Gaussian filter sized by the minimum width and spacing
   followed by tanh projection, run with β = 2, 4, 8;
2. `levelset_lbfgsb`, with length-scale, curvature and fixed-pixel penalties.

The best thresholded stage result is selected by its Wakis eval metric. That
design is then evaluated with the official gym backend: ceviche FDFD for the
ceviche challenges, fmmax for the metalens. The script records the
binarization degree, the imageruler length scales and the runtime. It writes
`summary.json`, `params.json`, `design.npy`, `design.png` and `log.jsonl`.

## Results

To be filled in from the cloud sweep.

## License

MIT (see LICENSE). Wakis is Apache-2.0. invrs-gym, invrs-opt and imageruler
are MIT. ceviche_challenges is Apache-2.0 and is called, not copied.
