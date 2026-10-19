"""Re-score leaderboard solutions and our designs with the installed gym,
exactly as invrs-io/leaderboard `utils.py` does (csv: density array replacing
the example solution's array, fixed_solid/fixed_void = None; npy: ours;
challenge.component.response + eval_metric in float64; imageruler length
scales of the design array). Specs run in parallel (one process each).

  uv run python scripts/rescore_leaderboard.py --outdir out
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pathlib  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "leaderboard"
JOBS = [
    ("ceviche_waveguide_bend", "240630_rahulkpadhy_waveguide_bend_Aadi_Rahul.csv", -1.2504255057635416),
    ("ceviche_power_splitter", "240812_mfschubert_gym_levelset_example.csv", 0.009036790840639897),
    ("ceviche_waveguide_bend", "ours_full_bend_seed1.npy", None),
    ("ceviche_power_splitter", "ours_full_power_splitter_seed1.npy", None),
]


def score(name, fname, published):
    import dataclasses

    import jax
    import numpy as onp
    from imageruler import imageruler
    from invrs_gym import challenges

    jax.config.update("jax_enable_x64", True)
    import invrs_gym

    challenge = challenges.BY_NAME[name]()
    example = challenge.component.init(jax.random.PRNGKey(0))
    path = DATA / fname
    if fname.endswith(".csv"):
        array = onp.genfromtxt(path, delimiter=",")
        params = dataclasses.replace(example, array=array, fixed_solid=None, fixed_void=None)
    else:
        params = dataclasses.replace(example, array=onp.load(path).astype(float))
    params = jax.tree_util.tree_map(lambda x: jax.numpy.array(x, x.dtype), params)
    response, aux = challenge.component.response(params)
    metric = float(challenge.eval_metric(response))
    arr = onp.asarray(params.array) > (params.lower_bound + params.upper_bound) / 2
    w, s = imageruler.minimum_length_scale(arr, periodic=params.periodic)
    return {"challenge": name, "file": fname, "eval_metric": metric,
            "published": published, "minimum_width": int(w), "minimum_spacing": int(s),
            "s2": (onp.abs(onp.asarray(response.s_parameters)) ** 2).tolist(),
            "invrs_gym_version": getattr(invrs_gym, "__version__", "?")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(len(JOBS), mp_context=ctx) as pool:
        res = list(pool.map(score, *zip(*JOBS)))
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != "s2"}), flush=True)
    pathlib.Path(a.outdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.outdir, "rescore.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
