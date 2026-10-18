"""Sharded driver: optimize gym challenges with the Wakis backend, evaluate
the final design with the gym's official backend.

Shard i of N runs specs[i::N] with specs = [(challenge, seed) for challenge in
--challenges for seed in --seeds]. For each spec it writes into
<outdir>/<challenge>_seed<seed>/:
  log.jsonl     optimization trace (loss, Wakis eval metric per step)
  summary.json  official eval metric (ceviche FDFD / fmmax), Wakis eval
                metrics (several discretizations), |S|^2 or enhancements,
                binarization degree, imageruler length scales, runtime
  params.json   final binary params (totypes JSON), design.npy
  design.png    final design (with surroundings for ceviche)

Example (cloud, one task per challenge):
  uv run python scripts/run_challenge.py --shard 0 --nshards 6 --outdir out
"""

import argparse
import dataclasses
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import jax
import numpy as np
from totypes import json_utils

from invrs_wakis.analysis import binarize, ceviche_length_scale, evaluate, length_scale_pixels
from invrs_wakis.optimize import jsonl_logger, optimize

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


def run_ceviche(name, seed, out, args):
    from invrs_gym import challenges

    from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

    challenge, gym = wakis_ceviche_challenge(
        name, SimOptions(port_estimator=args.estimator, run_time=args.run_time)
    )
    params = challenge.component.init(jax.random.PRNGKey(seed))
    best, _, opt_time, stage = optimize(
        challenge, params, args.betas, args.density_steps, args.levelset_steps,
        args.penalty, jsonl_logger(out / "log.jsonl"),
    )  # fmt: skip
    t = time.time()
    official = evaluate(gym, best)
    summary = {
        "official_backend": "ceviche FDFD (invrs_gym component.response)",
        "official_eval_metric": official["eval_metric"],
        "official_loss": official["loss"],
        "official_S2": s2(official["response"]),
        "official_eval_time_s": time.time() - t,
    }
    # Same design on other discretizations / estimators
    variants = {
        "wakis_r1_ceviche_estimator": SimOptions(port_estimator="ceviche"),
        "wakis_r1_modal": SimOptions(),
        "wakis_r2_modal": SimOptions(resolution=2),
    }
    for key, opts in variants.items():
        r = evaluate(wakis_ceviche_challenge(name, opts)[0], best)
        summary[f"{key}_eval_metric"] = r["eval_metric"]
        summary[f"{key}_S2"] = s2(r["response"])
    gym10 = challenges.BY_NAME[name](grid_spacing_nm=10)
    p10 = gym10.component.init(jax.random.PRNGKey(0))
    p10 = dataclasses.replace(p10, array=np.kron(np.asarray(best.array), np.ones((4, 4))))
    r10 = evaluate(gym10, p10)
    summary["ceviche_10nm_eval_metric"] = r10["eval_metric"]
    summary["ceviche_10nm_S2"] = s2(r10["response"])
    seed_density = gym.component.seed_density
    summary.update(
        binarization_degree=official["binarization_degree"],
        length_scale_solid_void_px=ceviche_length_scale(gym.component.ceviche_model, best),
        target_min_width_spacing_px=[int(seed_density.minimum_width), int(seed_density.minimum_spacing)],
        optimization_time_s=opt_time,
        selected_stage=stage,
    )
    plot_ceviche(gym.component.ceviche_model, best, out / "design.png", name, summary)
    return best, summary


def run_metalens(seed, out, args):
    from invrs_wakis.metalens import MetalensOptions, wakis_metalens_challenge

    challenge, gym = wakis_metalens_challenge(
        MetalensOptions(resolution=args.metalens_resolution)
    )
    params = challenge.component.init(jax.random.PRNGKey(seed))
    best, _, opt_time, stage = optimize(
        challenge, params, args.betas, args.density_steps, args.levelset_steps,
        args.penalty, jsonl_logger(out / "log.jsonl"),
    )  # fmt: skip
    t = time.time()
    official = evaluate(gym, best)
    wakis = evaluate(challenge, best)
    summary = {
        "official_backend": "fmmax FMM (invrs_gym component.response)",
        "official_eval_metric": official["eval_metric"],
        "official_enhancement_ex": np.asarray(official["response"].enhancement_ex).tolist(),
        "official_eval_time_s": time.time() - t,
        "wakis_eval_metric": wakis["eval_metric"],
        "wakis_enhancement_ex": np.asarray(wakis["response"].enhancement_ex).tolist(),
        "binarization_degree": official["binarization_degree"],
        "length_scale_solid_void_px": length_scale_pixels(np.asarray(best.array) > 0.5),
        "target_min_width_spacing_px": [5, 5],
        "optimization_time_s": opt_time,
        "selected_stage": stage,
    }
    plot_metalens(best, out / "design.png", summary)
    return best, summary


def plot_ceviche(model, params, path, name, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density = np.asarray(model.density(np.asarray(params.array)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(1 - density.T, cmap="gray", origin="lower", vmin=-1, vmax=1)
    x0, y0, x1, y1 = model.design_region_coords
    ax.add_patch(matplotlib.patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                              fill=False, ec="tab:red", lw=0.8, ls="--"))
    ax.set_title(f"{name}\nofficial eval_metric = {summary['official_eval_metric']:.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metalens(params, path, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(1 - np.asarray(params.array).T, cmap="gray", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"metalens: official eval_metric (min enhancement) = "
                 f"{summary['official_eval_metric']:.2f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--challenges", nargs="+", default=DEFAULT_CHALLENGES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--betas", type=lambda s: [float(b) for b in s.split(",")], default=[2.0, 4.0, 8.0])
    ap.add_argument("--density-steps", type=int, default=40)
    ap.add_argument("--levelset-steps", type=int, default=80)
    ap.add_argument("--penalty", type=float, default=1.0)
    ap.add_argument("--estimator", default="ceviche", choices=["ceviche", "modal"])
    ap.add_argument("--run-time", type=float, default=0.8e-12)
    ap.add_argument("--metalens-resolution", type=int, default=1)
    args = ap.parse_args()

    specs = [(c, s) for c in args.challenges for s in args.seeds]
    for name, seed in specs[args.shard :: args.nshards]:
        out = pathlib.Path(args.outdir) / f"{name}_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if name == "metalens":
            best, summary = run_metalens(seed, out, args)
        else:
            best, summary = run_ceviche(name, seed, out, args)
        summary.update(challenge=name, seed=seed, total_time_s=time.time() - t0, args=vars(args))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        (out / "params.json").write_text(json_utils.json_from_pytree(best))
        np.save(out / "design.npy", np.asarray(best.array))
        print(json.dumps({k: v for k, v in summary.items() if not k.endswith(("S2", "_ex"))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
