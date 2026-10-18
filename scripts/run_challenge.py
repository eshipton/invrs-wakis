"""Sharded driver: optimize gym challenges with the Wakis backend, evaluate
the final design with the gym's official backend.

Shard i of N runs specs[i::N] with specs = [(challenge, seed) for challenge in
--challenges for seed in --seeds]. For each spec it writes into
<outdir>/<challenge>_seed<seed>/:
  log.jsonl     optimization trace (loss, Wakis eval metric per step)
  snapshots.npz density simulated at every iteration (float16) + metrics
  params.json   final binary params (totypes JSON), design.npy
                (written right after the optimization, before any
                post-processing)
  summary.json  official eval metric (ceviche FDFD / fmmax), Wakis eval
                metrics (several discretizations), |S|^2 or enhancements,
                binarization degree, imageruler length scales, runtimes;
                'errors' lists post-processing steps that failed
  fields.npz    Wakis DFT fields of the final design at the challenge
                wavelengths (+ dense S spectrum for ceviche), one extra run
  design.png, optimization.gif, fields.png, wave.gif, spectrum.png (ceviche)

Example (cloud, one task per challenge):
  uv run python scripts/run_challenge.py --shard 0 --nshards 6 --outdir out
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless batch containers

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from totypes import json_utils  # noqa: E402

from invrs_wakis.analysis import (  # noqa: E402
    ceviche_length_scale,
    evaluate,
    length_scale_pixels,
)
from invrs_wakis.optimize import SnapshotRecorder, jsonl_logger, optimize  # noqa: E402

DEFAULT_CHALLENGES = [
    "ceviche_lightweight_waveguide_bend",
    "ceviche_lightweight_mode_converter",
    "ceviche_lightweight_power_splitter",
    "ceviche_lightweight_beam_splitter",
    "ceviche_lightweight_wdm",
    "metalens",
]


def s2(response):
    return (np.abs(np.asarray(response.s_parameters)) ** 2).tolist()


class Summary(dict):
    """summary.json contents; `step` runs one post-processing step, records
    its runtime and keeps going (traceback in 'errors') if it fails."""

    def __init__(self, path, **kw):
        super().__init__(errors=[], **kw)
        self.path = path

    def write(self):
        self.path.write_text(json.dumps(self, indent=2, default=str))

    def step(self, label, fn):
        t = time.time()
        try:
            result = fn()
            self[f"{label}_time_s"] = time.time() - t
            return result
        except Exception:  # noqa: BLE001 - keep the optimized design and other steps
            self["errors"].append({"step": label, "traceback": traceback.format_exc()})
            print(f"[{label}] failed:\n{traceback.format_exc()}", flush=True)
            return None
        finally:
            self.write()


def build(name, args, resolution=1):
    """(Wakis challenge, gym challenge). Resolution 1 uses the --estimator
    (default 'ceviche', which reproduces the official evaluator); finer
    resolutions use the 'modal' estimator (the 'ceviche' one needs r = 1)."""
    if name == "metalens":
        from invrs_wakis.metalens import MetalensOptions, wakis_metalens_challenge

        return wakis_metalens_challenge(MetalensOptions(
            resolution=args.metalens_resolution * resolution, smoothing=args.smoothing))
    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

    estimator = args.estimator if resolution == 1 else "modal"
    return wakis_ceviche_challenge(name, SimOptions(
        resolution=resolution, port_estimator=estimator, run_time=args.run_time,
        smoothing=args.smoothing))


def ceviche_post(name, challenge, gym, best, summary):
    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge
    from invrs_wakis.fields import ceviche_fields

    def official():
        r = evaluate(gym, best)
        summary.update(
            official_backend="ceviche FDFD (invrs_gym component.response)",
            official_eval_metric=r["eval_metric"],
            official_loss=r["loss"],
            official_S2=s2(r["response"]),
            binarization_degree=r["binarization_degree"],
        )

    summary.step("official_eval", official)

    def length_scale():
        seed_density = gym.component.seed_density
        summary.update(
            # As measured by the invrs leaderboard: the design array alone
            length_scale_leaderboard_px=length_scale_pixels(np.asarray(best.array) > 0.5),
            # With 4 pixels of the surrounding waveguides/cladding included
            length_scale_solid_void_px=ceviche_length_scale(gym.component.ceviche_model, best),
            target_min_width_spacing_px=[
                int(seed_density.minimum_width),
                int(seed_density.minimum_spacing),
            ],
        )

    summary.step("length_scale", length_scale)

    # The same design on other Wakis discretizations / estimators
    variants = {
        "wakis_r1_ceviche_estimator": SimOptions(port_estimator="ceviche"),
        "wakis_r1_modal": SimOptions(),
        "wakis_r2_modal": SimOptions(resolution=2),
    }
    for key, opts in variants.items():

        def variant(key=key, opts=opts):
            r = evaluate(wakis_ceviche_challenge(name, opts)[0], best)
            summary[f"{key}_eval_metric"] = r["eval_metric"]
            summary[f"{key}_S2"] = s2(r["response"])

        summary.step(key, variant)

    def ceviche_10nm():
        r = eval_10nm(name, best)
        summary["ceviche_10nm_eval_metric"] = r["eval_metric"]
        summary["ceviche_10nm_S2"] = s2(r["response"])

    summary.step("ceviche_10nm", ceviche_10nm)
    return lambda: ceviche_fields(challenge, best)


def metalens_post(challenge, gym, best, summary):
    from invrs_wakis.fields import metalens_fields

    def official():
        r = evaluate(gym, best)
        summary.update(
            official_backend="fmmax FMM (invrs_gym component.response)",
            official_eval_metric=r["eval_metric"],
            official_enhancement_ex=np.asarray(r["response"].enhancement_ex).tolist(),
            binarization_degree=r["binarization_degree"],
        )

    summary.step("official_eval", official)

    def wakis_eval():
        r = evaluate(challenge, best)
        summary["wakis_eval_metric"] = r["eval_metric"]
        summary["wakis_enhancement_ex"] = np.asarray(r["response"].enhancement_ex).tolist()

    summary.step("wakis_eval", wakis_eval)

    def length_scale():
        summary["length_scale_solid_void_px"] = length_scale_pixels(np.asarray(best.array) > 0.5)
        summary["target_min_width_spacing_px"] = [
            int(gym.component.seed_density.minimum_width),
            int(gym.component.seed_density.minimum_spacing),
        ]

    summary.step("length_scale", length_scale)
    return lambda: metalens_fields(challenge, best)


def run(name, seed, out, args):
    t0 = time.time()
    summary = Summary(out / "summary.json", challenge=name, seed=seed, args=vars(args))
    challenge, gym = build(name, args)
    summary["preflight"] = preflight(name, {1: challenge}, args)
    if args.preflight_only:
        return
    params = challenge.component.init(jax.random.PRNGKey(seed))
    snapshots = SnapshotRecorder()
    best, _, opt_time, stage = optimize(
        challenge, params, args.betas, args.density_steps, args.levelset_steps,
        args.penalty, jsonl_logger(out / "log.jsonl"), snapshots,
    )  # fmt: skip
    # Persist the optimized design before any post-processing
    np.save(out / "design.npy", np.asarray(best.array))
    (out / "params.json").write_text(json_utils.json_from_pytree(best))
    summary.update(optimization_time_s=opt_time, selected_stage=stage)
    summary.write()
    summary.step("snapshots", lambda: snapshots.save(out / "snapshots.npz"))

    post(name, challenge, gym, best, summary, out, args, t0)


def preflight(name, challenges_by_r, args):
    """Log the wakis.estimate memory estimate; abort (MemoryError) if the run
    does not fit --machine."""
    from invrs_wakis.resources import run_preflight

    return run_preflight(name, challenges_by_r, args.machine)


def load_config(path, name):
    cfg = json.loads(pathlib.Path(path).read_text())
    if name not in cfg:
        raise KeyError(f"{name} not in {path}")
    out = dict(cfg.get("defaults", {}))
    out.update(cfg[name])
    return out


def apply_overrides(stages, args):
    """--r2-weight W sets the weight of resolution 2 in every stage that
    simulates more than r = 1 (0 removes it); --steps-scale scales steps."""
    stages = [dict(st) for st in stages]
    for st in stages:
        w = {int(r): float(v) for r, v in st.get("weights", {"1": 1.0}).items()}
        if args.r2_weight is not None and len(w) > 1:
            w[2] = args.r2_weight
        st["weights"] = {r: v for r, v in w.items() if v > 0}
        st["steps"] = max(1, int(round(st["steps"] * args.steps_scale)))
    return stages


def initial_latent(cfg, params, seed):
    from invrs_wakis.lengthscale import random_latent

    init = cfg.get("init", "gym")
    if init == "gym":  # the gym's initializer (0.5 + 10 % noise), seeded
        return np.asarray(params.array)
    if init == "random":  # smooth random start, conic blur of 2 filter radii
        r = cfg.get("filter_radius") or 0.5 * (
            params.minimum_width + params.minimum_spacing)
        return random_latent(params, jax.random.PRNGKey(1000 + seed), 2.0 * r)
    raise ValueError(f"unknown init {init!r}")


def run_feasible(name, seed, out, args):
    """Round-2 run: length-scale-feasible parameterization (config file)."""
    from invrs_wakis.lengthscale import FeasibleDensity
    from invrs_wakis.optimize import optimize_feasible

    t0 = time.time()
    cfg = load_config(args.config, name)
    stages = apply_overrides(cfg["stages"], args)
    summary = Summary(out / "summary.json", challenge=name, seed=seed, args=vars(args),
                      method="feasible (conic filter + projection + length-scale repair, STE)",
                      stages=stages)
    resolutions = sorted({r for st in stages for r in st["weights"]} | {1})
    built = {r: build(name, args, resolution=r) for r in resolutions}
    challenges_by_r = {r: b[0] for r, b in built.items()}
    gym = built[1][1]
    summary["preflight"] = preflight(name, challenges_by_r, args)
    if args.preflight_only:
        return
    params = gym.component.init(jax.random.PRNGKey(seed))
    feasible = FeasibleDensity(params, filter_radius=cfg.get("filter_radius"))
    summary.update(filter_radius_px=feasible.filter_radius,
                   target_min_width_spacing_px=[feasible.min_width, feasible.min_spacing],
                   resolutions=resolutions)
    snapshots = SnapshotRecorder()
    candidates, latent, opt_time = optimize_feasible(
        challenges_by_r, feasible, initial_latent(cfg, params, seed), stages,
        jsonl_logger(out / "log.jsonl"), snapshots,
        n_candidates=int(args.candidates or cfg.get("n_candidates", 4)),
    )
    summary.update(optimization_time_s=opt_time)
    np.save(out / "latent.npy", np.asarray(latent))
    summary.step("snapshots", lambda: snapshots.save(out / "snapshots.npz"))

    # Score the candidates with the official backend (+ 10 nm re-check)
    def as_params(binary):
        lo, hi = params.lower_bound, params.upper_bound
        return dataclasses.replace(params, array=jax.numpy.asarray(np.where(binary, hi, lo), float))

    scored = []
    for k, c in enumerate(candidates):
        rec = {key: c[key] for key in ("stage", "step", "wakis_metrics", "score")}
        rec["length_scale_leaderboard_px"] = feasible.length_scale(c["design"])
        p = as_params(c["design"])

        def official(p=p, rec=rec):
            rec["official_eval_metric"] = evaluate(gym, p)["eval_metric"]

        summary.step(f"candidate{k}_official", official)
        if name != "metalens":
            def c10(p=p, rec=rec):
                rec["ceviche_10nm_eval_metric"] = eval_10nm(name, p)["eval_metric"]

            summary.step(f"candidate{k}_10nm", c10)
        scored.append((rec.get("official_eval_metric", -np.inf), k, p, rec))
        summary["candidates"] = [s[3] for s in scored]
        summary.write()
    _, kbest, best, rec = max(scored, key=lambda s: (s[0], -s[1]))
    summary.update(selected_candidate=kbest, selected_stage=f"{rec['stage']}:{rec['step']}")
    np.save(out / "design.npy", np.asarray(best.array))
    (out / "params.json").write_text(json_utils.json_from_pytree(best))
    summary.write()
    post(name, built[1][0], gym, best, summary, out, args, t0)


def eval_10nm(name, best):
    from invrs_gym import challenges

    gym10 = challenges.BY_NAME[name](grid_spacing_nm=10)
    p10 = gym10.component.init(jax.random.PRNGKey(0))
    p10 = dataclasses.replace(p10, array=np.kron(np.asarray(best.array), np.ones((4, 4))))
    return evaluate(gym10, p10)


def post(name, challenge, gym, best, summary, out, args, t0):
    if name == "metalens":
        fields_fn = metalens_post(challenge, gym, best, summary)
    else:
        fields_fn = ceviche_post(name, challenge, gym, best, summary)

    def fields():
        np.savez_compressed(out / "fields.npz", **fields_fn())

    summary.step("fields", fields)
    if not args.no_render:
        from invrs_wakis.visualize import render_run

        summary["rendered"] = summary.step("render", lambda: render_run(out, name))
    summary["total_time_s"] = time.time() - t0
    summary.write()
    keys = ("official_eval_metric", "ceviche_10nm_eval_metric", "binarization_degree",
            "length_scale_leaderboard_px", "length_scale_solid_void_px",
            "optimization_time_s", "total_time_s", "rendered")
    print(json.dumps({k: summary.get(k) for k in keys}, default=str), flush=True)
    print(f"{len(summary['errors'])} post-processing errors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--challenges", nargs="+", default=DEFAULT_CHALLENGES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--betas", type=lambda s: [float(b) for b in s.split(",")],
                    default=[2.0, 4.0, 8.0])
    ap.add_argument("--density-steps", type=int, default=40)
    ap.add_argument("--levelset-steps", type=int, default=80)
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--estimator", default="ceviche", choices=["ceviche", "modal"])
    ap.add_argument("--run-time", type=float, default=0.8e-12)
    ap.add_argument("--metalens-resolution", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    # Round 2
    ap.add_argument("--config", default=None,
                    help="JSON config (configs/round2.json): use the length-scale-feasible "
                         "optimizer with the per-challenge stages")
    ap.add_argument("--r2-weight", type=float, default=None,
                    help="weight of the r=2 (20 nm) objective in multi-resolution stages "
                         "(0 = optimize on the 40 nm grid only)")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--smoothing", default="none", choices=["none", "tensor"],
                    help="material smoothing of the design in Wakis ('tensor' needs the "
                         "Wakis tensor-averaging feature)")
    ap.add_argument("--machine", default=None,
                    help="machine type the run executes on (wakis.estimate.MACHINES, e.g. "
                         "n2-standard-8): abort before simulating if the estimated peak memory "
                         "does not fit")
    ap.add_argument("--preflight-only", action="store_true",
                    help="only log the memory estimate (no simulation)")
    args = ap.parse_args()

    specs = [(c, s) for c in args.challenges for s in args.seeds]
    for name, seed in specs[args.shard :: args.nshards]:
        out = pathlib.Path(args.outdir) / f"{name}_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        if args.config:
            run_feasible(name, seed, out, args)
        else:
            run(name, seed, out, args)


if __name__ == "__main__":
    main()
