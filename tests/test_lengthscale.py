"""Length-scale guarantee of the round-2 parameterization (fast, no simulation)."""

import types as pytypes

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from invrs_gym import challenges

from invrs_wakis.lengthscale import FeasibleDensity, random_latent
from invrs_wakis.optimize import optimize_feasible

NAMES = [
    "ceviche_lightweight_waveguide_bend",
    "ceviche_lightweight_mode_converter",
    "ceviche_lightweight_power_splitter",
    "ceviche_lightweight_beam_splitter",
    "ceviche_lightweight_wdm",
]


@pytest.mark.parametrize("name", NAMES)
def test_repair_meets_targets(name):
    """Random projected designs (1-px features before repair) always meet the
    gym's minimum width/spacing after repair, keep fixed pixels and symmetry."""
    params = challenges.BY_NAME[name]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    raw_ok = 0
    for k in range(8):
        lat = random_latent(params, jax.random.PRNGKey(k), radius=1.5 + k % 3 * 2)
        rho = np.asarray(fd.projected(lat, 32.0))
        raw_ok += fd.meets_targets(fd.binary(rho))
        x = fd.repair(rho)
        w, s = fd.length_scale(x)
        assert w >= params.minimum_width and s >= params.minimum_spacing
        assert np.all(x[fd.fixed_solid]) and not np.any(x[fd.fixed_void])
        sym = np.asarray(fd.symmetrize(jnp.asarray(x, float)))
        assert np.array_equal(sym > 0.5, x)
    assert raw_ok < 8  # the check is not vacuous


def test_ste_gradient_and_optimizer_plumbing():
    """STE forward = repaired binary design; gradient = projected path; the
    staged optimizer returns only feasible candidates (mock objective)."""
    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    target = jnp.asarray(random_latent(params, jax.random.PRNGKey(7), 4.0))

    def response(p):
        return jnp.mean((p.array - target) ** 2), {}

    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=response),
        loss=lambda r: r,
        eval_metric=lambda r: -r,
    )
    lat = np.asarray(params.array)
    rho = fd.ste_density(jnp.asarray(lat), 64.0)
    assert set(np.unique(np.asarray(rho))) <= {0.0, 1.0}
    g = jax.grad(lambda l: jnp.sum(fd.ste_density(l, 8.0) * target))(jnp.asarray(lat))
    assert float(jnp.abs(g).sum()) > 0
    stages = [
        {"beta": 4, "steps": 3},
        {"beta": 64, "steps": 3, "ste": True, "optimizer": "adam", "lr": 0.05},
    ]
    cands, _, _ = optimize_feasible({1: mock}, fd, lat, stages, log=lambda r: None)
    assert cands
    for c in cands:
        assert fd.meets_targets(c["design"])


def test_checkpoint_resume(tmp_path):
    """A run resumed from its last checkpoint skips the finished stages and
    returns the same candidate pool (mock objective, no simulation)."""
    from invrs_wakis.optimize import Checkpointer, SnapshotRecorder

    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    target = jnp.asarray(random_latent(params, jax.random.PRNGKey(3), 4.0))
    calls = []

    def response(p):
        calls.append(1)
        return jnp.mean((p.array - target) ** 2), {}

    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=response),
        loss=lambda r: r, eval_metric=lambda r: -r,
    )
    stages = [{"beta": 4, "steps": 2},
              {"beta": 64, "steps": 2, "ste": True, "optimizer": "adam", "lr": 0.05}]
    lat = np.asarray(params.array)
    rec = SnapshotRecorder()
    ck = Checkpointer(tmp_path / "c.pkl", every_s=0.0, recorder=rec)
    c1, _, _ = optimize_feasible({1: mock}, fd, lat, stages, log=lambda r: None,
                                 snapshot=rec, checkpoint=ck)
    n_first = len(calls)
    rec2 = SnapshotRecorder()
    ck2 = Checkpointer(tmp_path / "c.pkl", every_s=0.0, recorder=rec2)
    c2, _, _ = optimize_feasible({1: mock}, fd, lat, stages, log=lambda r: None,
                                 snapshot=rec2, checkpoint=ck2)
    assert len(calls) == n_first  # nothing re-simulated
    assert len(rec2.designs) == len(rec.designs)
    assert [c["score"] for c in c1] == [c["score"] for c in c2]


def test_tanh_projection_eta_endpoints():
    """tanh_projection(x, beta, eta) maps 0 -> 0 and 1 -> 1 for any eta in
    (0, 1), and FeasibleDensity.projected passes eta through."""
    from invrs_wakis.lengthscale import tanh_projection

    for beta in (1.0, 4.0, 8.0, 16.0, 64.0, np.inf):
        for eta in (0.1, 0.35, 0.5, 0.65, 0.9):
            assert float(tanh_projection(0.0, beta, eta)) == pytest.approx(0.0, abs=1e-12)
            assert float(tanh_projection(1.0, beta, eta)) == pytest.approx(1.0, abs=1e-12)
    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    lat = jnp.asarray(random_latent(params, jax.random.PRNGKey(1), 4.0))
    rho_e = np.asarray(fd.projected(lat, 8.0, eta=0.65))
    rho_n = np.asarray(fd.projected(lat, 8.0))
    rho_d = np.asarray(fd.projected(lat, 8.0, eta=0.35))
    assert np.allclose(rho_n, np.asarray(fd.projected(lat, 8.0, eta=0.5)))
    assert np.all(rho_e <= rho_n + 1e-12) and np.all(rho_n <= rho_d + 1e-12)
    assert float(rho_e.mean()) < float(rho_n.mean()) < float(rho_d.mean())


def test_robust_stage_is_max_of_three():
    """A robust stage evaluates eroded, nominal, dilated projections, logs
    robust_losses, and optimizes the maximum of the three (not the nominal)."""
    from invrs_wakis.optimize import SnapshotRecorder

    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    lat = np.asarray(random_latent(params, jax.random.PRNGKey(2), 4.0))
    # Target equals the initial nominal projection so nominal loss is 0 while
    # eroded and dilated losses are strictly positive.
    rho_nom0 = fd.projected(jnp.asarray(lat), 8.0, eta=0.5)

    def response(p):
        return jnp.mean((p.array - rho_nom0) ** 2), {}

    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=response),
        loss=lambda r: r, eval_metric=lambda r: -r,
    )
    logs = []
    rec = SnapshotRecorder()
    stages = [{"beta": 8, "steps": 1, "robust": 0.15}]
    _, lat_out, _ = optimize_feasible(
        {1: mock}, fd, lat, stages, log=logs.append, snapshot=rec,
    )
    step_logs = [r for r in logs if "step" in r]
    assert len(step_logs) == 1
    rl = step_logs[0]["robust_losses"]
    assert len(rl) == 3
    expected = [
        float(jnp.mean((fd.projected(jnp.asarray(lat), 8.0, eta=eta) - rho_nom0) ** 2))
        for eta in (0.65, 0.5, 0.35)
    ]
    assert np.allclose(rl, expected, rtol=0, atol=1e-12)
    assert abs(step_logs[0]["loss"] - max(rl)) < 1e-12
    assert step_logs[0]["loss"] > rl[1] + 1e-3  # nominal loss is 0, max is > 0
    # Aux (eval_metric, snapshot density) is the nominal design's
    assert abs(step_logs[0]["eval_metric"] - (-rl[1])) < 1e-12
    assert np.allclose(rec.designs[0], np.asarray(rho_nom0, dtype=np.float16))
    # L-BFGS-B moved the latent along the worst-case branch (nominal grad = 0)
    assert not np.allclose(np.asarray(lat_out), lat)
    # Without 'robust' (nominal loss only), loss is 0 and latent does not move
    nom_logs = []
    _, lat_nom, _ = optimize_feasible(
        {1: mock}, fd, lat, [{"beta": 8, "steps": 1}], log=nom_logs.append,
    )
    nom_step = [r for r in nom_logs if "step" in r][0]
    assert "robust_losses" not in nom_step
    assert abs(nom_step["loss"] - rl[1]) < 1e-12
    assert np.allclose(np.asarray(lat_nom), lat)


def test_robust_with_ste_raises():
    """Combining 'robust' with 'ste': True raises ValueError."""
    params = challenges.BY_NAME[NAMES[0]]().component.init(jax.random.PRNGKey(0))
    fd = FeasibleDensity(params)
    mock = pytypes.SimpleNamespace(
        component=pytypes.SimpleNamespace(response=lambda p: (jnp.mean(p.array), {})),
        loss=lambda r: r, eval_metric=lambda r: -r,
    )
    stages = [{"beta": 16, "steps": 1, "ste": True, "robust": 0.15}]
    with pytest.raises(ValueError, match="robust"):
        optimize_feasible({1: mock}, fd, np.asarray(params.array), stages, log=lambda r: None)


def test_schedule_v2_replacement_keeps_ste_weights():
    """--schedule v2 / apply_overrides replaces stages with schedule_v2 while
    keeping multi-resolution STE weights when the challenge has them."""
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.run_challenge import apply_overrides, load_config, replace_schedule

    cfg_path = root / "configs" / "round2.json"
    v2 = load_config(cfg_path, "schedule_v2")["stages"]
    assert [s["beta"] for s in v2] == [4, 8, 16, 16]
    assert [s["steps"] for s in v2] == [25, 25, 25, 60]
    assert [s.get("robust", 0.0) for s in v2] == [0.0, 0.15, 0.15, 0.0]
    assert v2[-1]["ste"] is True and v2[-1]["lr"] == 0.005

    # metalens has multi-resolution STE weights {"1": 1.0, "2": 1.0}
    ml_raw = load_config(cfg_path, "metalens")["stages"]
    ml_rep = replace_schedule(ml_raw, v2)
    assert ml_rep[-1]["weights"] == {"1": 1.0, "2": 1.0}
    assert "weights" not in ml_rep[0]
    ml_ov = apply_overrides(ml_raw, schedule="v2", config=cfg_path)
    assert len(ml_ov) == 4
    assert ml_ov[-1]["weights"] == {1: 1.0, 2: 1.0}
    assert ml_ov[0]["weights"] == {1: 1.0}

    # full ceviche challenge has single-resolution STE stage -> stays r=1 only
    wb_raw = load_config(cfg_path, "ceviche_waveguide_bend")["stages"]
    wb_ov = apply_overrides(wb_raw, schedule="v2", config=cfg_path)
    assert len(wb_ov) == 4
    assert wb_ov[-1]["weights"] == {1: 1.0}

