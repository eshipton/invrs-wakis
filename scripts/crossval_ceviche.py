"""Cross-validate the Wakis ceviche component against the gym's ceviche backend.

For each lightweight challenge and a few designs (the gym's random initial
density, a random binary pattern with ~120 nm features, and the Wakis-optimized
design if results/<challenge>/design.npy exists), compute |S|^2 with

  ceviche@40nm      the gym's official backend (FDFD on the 40 nm pixels)
  ceviche@10nm      the same FDFD with the design upsampled to 10 nm pixels
  wakis r1/ceviche  Wakis on the 40 nm pixels with ceviche's port estimator
  wakis r1/modal    Wakis on the 40 nm pixels, exact discrete mode decomposition
  wakis r2/modal    Wakis on a 20 nm mesh (each pixel = 2 x 2 cells)
  wakis r4/modal    Wakis on a 10 nm mesh (--fine)

and write <outdir>/crossval_shard<i>.json.

Usage: python scripts/crossval_ceviche.py [--fine] [--shard i --nshards N --outdir D] [challenge ...]
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
from invrs_gym import challenges
from scipy import ndimage

from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge

LIGHTWEIGHT = [
    "ceviche_lightweight_waveguide_bend",
    "ceviche_lightweight_mode_converter",
    "ceviche_lightweight_power_splitter",
    "ceviche_lightweight_beam_splitter",
    "ceviche_lightweight_wdm",
]


def random_binary(params, seed, sigma=1.5):
    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.normal(size=params.shape), sigma)
    array = (noise > 0).astype(float)
    array[params.fixed_solid] = 1.0
    array[params.fixed_void] = 0.0
    return dataclasses.replace(params, array=array)


def transmission(challenge, params):
    t = time.time()
    response, _ = challenge.component.response(params)
    s2 = np.abs(np.asarray(response.s_parameters)) ** 2
    return s2, float(challenge.eval_metric(response)), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=LIGHTWEIGHT)
    ap.add_argument("--fine", action="store_true", help="also run Wakis r=4")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--designs", default="results", help="dir with <challenge>_seed0/design.npy")
    args = ap.parse_args()

    results = {}
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.names[args.shard :: args.nshards]:
        gym = challenges.BY_NAME[name]()
        gym10 = challenges.BY_NAME[name](grid_spacing_nm=10)
        backends = {
            "wakis r1/ceviche": wakis_ceviche_challenge(name, SimOptions(port_estimator="ceviche"))[0],
            "wakis r1/modal": wakis_ceviche_challenge(name, SimOptions())[0],
            "wakis r2/modal": wakis_ceviche_challenge(name, SimOptions(resolution=2))[0],
        }
        if args.fine:
            backends["wakis r4/modal"] = wakis_ceviche_challenge(name, SimOptions(resolution=4))[0]
        params0 = gym.component.init(jax.random.PRNGKey(0))
        designs = {"random gray (gym init)": params0, "random binary": random_binary(params0, 1)}
        opt = pathlib.Path(args.designs) / f"{name}_seed0" / "design.npy"
        if opt.exists():
            designs["wakis-optimized"] = dataclasses.replace(params0, array=np.load(opt))
        results[name] = {}
        for dname, params in designs.items():
            row = {}
            s_ref, e_ref, t = transmission(gym, params)
            row["ceviche@40nm"] = {"S2": s_ref.tolist(), "eval": e_ref, "time": t}
            p10 = dataclasses.replace(
                gym10.component.init(jax.random.PRNGKey(0)),
                array=np.kron(np.asarray(params.array), np.ones((4, 4))),
            )
            s, e, t = transmission(gym10, p10)
            row["ceviche@10nm"] = {"S2": s.tolist(), "eval": e, "time": t}
            for bname, ch in backends.items():
                s, e, t = transmission(ch, params)
                row[bname] = {"S2": s.tolist(), "eval": e, "time": t}
            for b, v in row.items():
                s = np.asarray(v["S2"])
                v["max_abs_dS2_vs_ceviche40"] = float(np.max(np.abs(s - s_ref)))
                print(f"{name:40s} {dname:24s} {b:18s} eval {v['eval']:+.4f} "
                      f"max|dS2| {v['max_abs_dS2_vs_ceviche40']:.2e} ({v['time']:.1f}s)", flush=True)
            results[name][dname] = row
        (out / f"crossval_shard{args.shard}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
