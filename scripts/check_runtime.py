"""Run-time convergence of the Wakis |S|^2 vs the official ceviche for one
stored design (GPU, fused). One Batch task:

  uv run python scripts/check_runtime.py --outdir out --name ceviche_wdm \
      --design tests/data/full_wdm_smoke.npz --run-times 0.4 0.8 1.2 1.6
"""

import argparse
import dataclasses
import json
import pathlib
import time

import jax
import numpy as np

from invrs_wakis.ceviche import SimOptions, wakis_ceviche_challenge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--name", default="ceviche_wdm")
    ap.add_argument("--design", default="tests/data/full_wdm_smoke.npz")
    ap.add_argument("--run-times", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    design = np.load(a.design)["design"].astype(float)
    res = []
    gym = None
    for rt in a.run_times:
        ch, gym = wakis_ceviche_challenge(a.name, SimOptions(
            port_estimator="ceviche", run_time=rt * 1e-12, use_gpu=not a.cpu,
            fused=not a.cpu))
        p = dataclasses.replace(gym.component.init(jax.random.PRNGKey(0)),
                                array=jax.numpy.asarray(design))
        t = time.time()
        r, _ = ch.component.response(p)
        res.append({"run_time_ps": rt, "s2": (np.abs(np.asarray(r.s_parameters)) ** 2).tolist(),
                    "eval_metric": float(ch.eval_metric(r)), "time_s": time.time() - t})
        print(res[-1]["run_time_ps"], res[-1]["eval_metric"], res[-1]["time_s"], flush=True)
        ch.component._sims.clear()  # free the stored adjoint increments
        del ch, r
        import gc

        gc.collect()
    ref, _ = gym.component.response(p, max_parallelizm=6)
    s_ref = np.abs(np.asarray(ref.s_parameters)) ** 2
    for x in res:
        x["max_abs_dS2"] = float(np.max(np.abs(np.asarray(x["s2"]) - s_ref)))
    out = {"official_eval_metric": float(gym.eval_metric(ref)), "official_s2": s_ref.tolist(),
           "runs": res}
    pathlib.Path(a.outdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.outdir, "runtime_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"official": out["official_eval_metric"],
                      "runs": [(x["run_time_ps"], x["eval_metric"], x["max_abs_dS2"]) for x in res]}))


if __name__ == "__main__":
    main()
