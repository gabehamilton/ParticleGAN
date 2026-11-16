"""Energy transport over seen-data memory: formula, off path, FIFO, checkpoints and controls.

Everything runs through public ParticleGAN APIs (Recipe, GANTrainer,
particlegan.energy_transport). The fresh-process continuation test re-imports
this module in subprocesses rooted at this checkout.
"""
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

import particlegan.energy_transport as energy_module
from particlegan import GANTrainer, Recipe, get_recipe, init
from particlegan.energy_transport import (SMOOTHING, SeenTargetMemory, energy_transport_loss,
                                          energy_transport_terms)
from particlegan.priors import StratifiedSamplingManipulator
from particlegan.training import ENERGY_STREAM_OFFSET

ROOT = Path(__file__).resolve().parents[1]
N, BATCH, MEMORY = 16, 8, 20
ENERGY_FIELDS = ("energy_transport_weight", "energy_transport_memory", "energy_transport_population")


def ring(n, generator, *, sigma=.07, shift=0.):
    angle = torch.randint(8, (n,), generator=generator).double() * math.pi / 4
    points = 3 * torch.stack((angle.cos(), angle.sin()), 1) + sigma * torch.randn(n, 2, generator=generator).double()
    return (points + torch.tensor([shift, 0.], dtype=torch.float64)).float()


def batches(step):
    """Caller-owned deterministic (critic, generator) target batches for update ``step``."""
    generator = torch.Generator().manual_seed(1000 + step)
    return ring(BATCH, generator), ring(BATCH, generator)


def make_recipe(*, weight=1., memory=MEMORY, population=True, mog=True, output_noise=.05, **extra):
    fields = dict(num_particles=N, z_dim=2, batch_size=BATCH, total_steps=12,
                  output_noise_std=output_noise, output_noise_warmup=0.)
    if mog:
        fields.update(prior_kind="mog", sigma_rel=0., standardize=False)
    if weight:
        fields.update(energy_transport_weight=weight, energy_transport_memory=memory,
                      energy_transport_population=population)
    return get_recipe("k3p", **fields, **extra)


def make_trainer(recipe=None, *, manipulators=None, **options):
    recipe = make_recipe(**options) if recipe is None else recipe
    mog = recipe.prior_kind == "mog"
    prior = recipe.make_prior(generator=torch.Generator().manual_seed(0), init_std=.5,
                              manipulators=manipulators, **({"sigma": .05} if mog else {}))
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
    init.deterministic_orthogonal_(G, seed=0)
    init.deterministic_orthogonal_(D, seed=1)
    init.deterministic_orthogonal_(prior if manipulators is None else prior.base, seed=2)
    return GANTrainer(recipe, G, D, prior=prior, seed=0)


def run(trainer, start, stop, record=None):
    for step in range(start, stop):
        real, real_g = batches(step)
        result = trainer.step(real, generator_real=real_g)
        if record is not None:
            record.append(result)
    return trainer


def assert_same(left, right, path="state"):
    """Exact equality; a NaN diagnostic matches only a NaN in the same place."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right) or (left.is_floating_point() and torch.equal(left.isnan(), right.isnan())
                                            and torch.equal(left.nan_to_num(), right.nan_to_num())), path
    elif isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


# (a) The objective equals a brute-force O(n m) reference, value and autograd derivative.

def brute_force(x, y):
    xf, yf = x.flatten(1), y.detach().flatten(1)
    n, m = len(xf), len(yf)
    mean = sum(yf[j] for j in range(m)) / m
    scale = max(math.sqrt(sum(float(((yf[j] - mean) ** 2).sum()) for j in range(m)) / m),
                torch.finfo(yf.dtype).eps)
    delta2 = (SMOOTHING * scale) ** 2
    cross = sum(torch.sqrt(((xf[i] - yf[j]) ** 2).sum() + delta2) for i in range(n) for j in range(m)) / (n * m)
    self_term = sum(torch.sqrt(((xf[i] - xf[k]) ** 2).sum() + delta2) for i in range(n) for k in range(n)) / (n * n)
    return (2 * cross - self_term) / scale


@pytest.mark.parametrize("shape", [(7, 3), (6, 2, 3), (5, 1)])
def test_loss_and_gradient_match_brute_force_reference(shape):
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    x[1] = x[0]  # coincident generated points: zero subgradient, no NaN
    y = torch.randn((11, *shape[1:]), generator=generator, dtype=torch.float64) * 2 + .5
    y.requires_grad_(True)
    ours = x.clone().requires_grad_(True)
    reference = x.clone().requires_grad_(True)
    value = energy_transport_loss(ours, y)
    expected = brute_force(reference, y)
    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-12)
    value.backward()
    expected.backward()
    assert torch.isfinite(ours.grad).all()
    torch.testing.assert_close(ours.grad, reference.grad, rtol=1e-10, atol=1e-12)
    assert y.grad is None  # targets are detached


def test_terms_scale_invariance_and_weighted_recipe_method(monkeypatch):
    generator = torch.Generator().manual_seed(4)
    x, y = torch.randn(9, 2, generator=generator, dtype=torch.float64), torch.randn(30, 2, generator=generator, dtype=torch.float64)
    terms = energy_transport_terms(x, y)
    torch.testing.assert_close(terms.loss, 2 * terms.cross - terms.fake_self)
    torch.testing.assert_close(terms.scale, (y - y.mean(0)).square().sum(1).mean().sqrt())
    torch.testing.assert_close(energy_transport_loss(7 * x, 7 * y), terms.loss, rtol=1e-12, atol=1e-12)
    recipe = get_recipe("k3p", energy_transport_weight=.25)
    torch.testing.assert_close(recipe.energy_transport_loss(x, y), .25 * terms.loss)
    monkeypatch.setattr(energy_module, "energy_transport_loss", lambda *a: pytest.fail("evaluated at weight 0"))
    assert get_recipe("k3p").energy_transport_loss(x, y).item() == 0
    with pytest.raises(ValueError, match="energy transport needs"):
        energy_transport_loss(x[:1], y)
    with pytest.raises(ValueError, match="energy transport needs"):
        energy_transport_loss(x, y.float())


def test_degenerate_target_sets_are_rejected_not_divided_by_eps():
    # A one-row or zero-spread target set has no scale; clamping it to eps
    # would turn the objective into a ~1e7 loss with a huge gradient.
    x = torch.randn(9, 2, generator=torch.Generator().manual_seed(5), requires_grad=True)
    for target in (torch.tensor([[3., 0.]]), torch.tensor([[3., 0.]]).repeat(16, 1)):
        with pytest.raises(ValueError, match="energy transport"):
            energy_transport_terms(x, target)
    with pytest.raises(ValueError, match="no spread"):
        energy_transport_terms(x, torch.full((16, 2), float("nan")))
    for fields in (dict(energy_transport_memory=1), dict(batch_size=1)):
        with pytest.raises(ValueError, match="at least two target rows"):
            get_recipe("k3p_energy", **fields)
    # The trainer reports the degenerate batch as an error, never a corrupted update.
    trainer = make_trainer(memory=0)
    point = torch.tensor([[3., 0.]]).repeat(BATCH, 1)
    with pytest.raises(ValueError, match="no spread"):
        trainer.step(ring(BATCH, torch.Generator().manual_seed(6)), generator_real=point)


def test_recipe_fields_validate_and_serialize():
    recipe = get_recipe("k3p", energy_transport_weight=1, energy_transport_memory=64, energy_transport_population=True)
    saved = recipe.to_dict()
    assert saved["energy_transport_weight"] == 1.0 and type(saved["energy_transport_weight"]) is float
    assert Recipe.from_dict(saved) == recipe.replace(energy_transport_weight=1.0)
    assert not set(ENERGY_FIELDS) & set(get_recipe("k3p").to_dict())
    for bad in (dict(energy_transport_weight=-1.), dict(energy_transport_weight=float("nan")),
                dict(energy_transport_weight=True), dict(energy_transport_weight=1., energy_transport_memory=-1),
                dict(energy_transport_weight=1., energy_transport_memory=2.), dict(energy_transport_weight=1., energy_transport_population=1),
                dict(energy_transport_memory=8), dict(energy_transport_population=True)):
        with pytest.raises(ValueError, match="energy_transport"):
            get_recipe("k3p", **bad)
    from experiments.forge.boundaries import RECIPE_FIELD_SPECS, TUNABLE_FIELDS, recipe_field_owner
    for name in ENERGY_FIELDS:
        assert recipe_field_owner(name) == "technique" and name not in TUNABLE_FIELDS
        assert RECIPE_FIELD_SPECS[name].default == asdict(get_recipe("k3p"))[name]
    # The disabled fields are registered search defaults (recipes.py, like the
    # rhbi additions), so every projection omits them and never drops active ones.
    # task-card-pinned recipe_compat.py stays byte-identical to develop.
    from particlegan.recipe_compat import SEARCH_RECIPE_DEFAULTS, without_default_additions
    assert {name: SEARCH_RECIPE_DEFAULTS[name] for name in ENERGY_FIELDS} == {
        name: asdict(get_recipe("k3p"))[name] for name in ENERGY_FIELDS}
    off, on = asdict(get_recipe("k3p")), asdict(recipe)
    assert not set(ENERGY_FIELDS) & set(without_default_additions(off))
    assert {name: on[name] for name in ENERGY_FIELDS} == {
        name: without_default_additions(on)[name] for name in ENERGY_FIELDS}
    # Forge planning lists its later additions; a candidate written before the
    # energy fields existed keeps its exact revision identity.
    from experiments.forge.planning import candidate_revision_for
    historical = {name: value for name, value in off.items() if name not in ENERGY_FIELDS}
    candidate = lambda resolved: {"resolved_recipe": resolved, "prior": {"kind": "particles"}}
    assert candidate_revision_for("0" * 64, candidate(off)) == candidate_revision_for("0" * 64, candidate(historical))
    assert candidate_revision_for("0" * 64, candidate(on)) != candidate_revision_for(
        "0" * 64, candidate({**on, "energy_transport_weight": 0., "energy_transport_memory": 0,
                             "energy_transport_population": False}))


def test_energy_transport_is_refused_outside_the_gantrainer_envelope():
    # Encoder, DDGAN and routed-row hosts own their loops; none consumes the
    # objective, so the recipe refuses instead of letting it drop silently.
    for envelope in (dict(model="ddgan"), dict(row_policy="routed_paired"),
                     dict(conditioning="conditional", num_classes=4)):
        with pytest.raises(ValueError, match="GANTrainer envelope"):
            get_recipe("k3p_energy").replace(**envelope)
    from benchmarks.toy_audit import api_model_ae  # encoder_mode="ae" host
    case = next(iter(api_model_ae.CASES))
    with pytest.raises(ValueError, match="GANTrainer envelope"):
        api_model_ae.task_recipe({"id": case}, recipe_name="k3p_energy")


@pytest.mark.parametrize("module, case_id", [
    ("api_ring16", "api-ring16-bigan-continuous"),
    ("api_conditionals", "api-misgan-block-recovery"),
    ("api_cgan_overlap", "api-cgan-overlapping-classes"),
    ("api_cyclegan", "api-cyclegan-unpaired-constellation"),
    ("api_cramergan", "api-cramer-energy-overlap"),
    ("api_infogan", "api-infogan-anisotropic-modes"),
    ("api_pacgan", "api-pacgan-unequal-mass"),
])
def test_caller_owned_scalar_gan_hosts_block_instead_of_dropping_energy(module, case_id):
    # These hosts own their training loops and do not bind
    # Recipe.energy_transport_loss; a k3p_energy request is BLOCKED.
    import importlib
    from benchmarks.toy_audit.api_contract import UnsupportedRecipeOverrides
    host = importlib.import_module(f"benchmarks.toy_audit.{module}")
    with pytest.raises(UnsupportedRecipeOverrides, match="energy_transport_weight"):
        host.build_case(case_id, device="cpu", seed=0, recipe_name="k3p_energy")


# (b) Weight 0 is fully off.

def test_weight_zero_is_fully_off(monkeypatch):
    monkeypatch.setattr(energy_module, "energy_transport_terms", lambda *a: pytest.fail("objective evaluated"))
    monkeypatch.setattr(SeenTargetMemory, "push", lambda *a: pytest.fail("memory written"))
    off = make_recipe(weight=0.)
    explicit = make_recipe(weight=0.).replace(energy_transport_weight=0., energy_transport_memory=0,
                                              energy_transport_population=False)
    without = Recipe.from_dict({k: v for k, v in asdict(off).items() if k not in ENERGY_FIELDS})
    assert off == explicit == without and off.to_dict() == without.to_dict()
    first, second = make_trainer(off), make_trainer(without)
    results = []
    run(first, 0, 5, results)
    run(second, 0, 5)
    state = first.state_dict()
    assert_same(state, second.state_dict())
    assert "energy_transport" not in state and "energy_generator" not in state["streams"]
    assert first.energy_generator is None and first.energy_transport_memory() is None
    assert not any(key.startswith("energy") for result in results for key in result)
    # The enabled objective draws only from its own stream: every other stream matches weight 0.
    monkeypatch.undo()
    enabled = run(make_trainer(make_recipe()), 0, 5).state_dict()
    assert_same({k: v for k, v in enabled["streams"].items() if k != "energy_generator"}, state["streams"])
    assert not torch.equal(enabled["models"]["G"]["2.weight"], state["models"]["G"]["2.weight"])


# (c) FIFO semantics.

def test_seen_target_memory_fifo_order_capacity_and_fill_boundary():
    memory = SeenTargetMemory(5)
    assert memory.contents() is None and len(memory) == 0
    rows = torch.arange(20, dtype=torch.float32)[:, None]
    memory.push(rows[0:2])
    assert memory.contents().flatten().tolist() == [0, 1] and (memory.count, memory.next) == (2, 2)
    memory.push(rows[2:4])
    memory.push(rows[4:6])  # crosses the fill boundary
    assert memory.contents().flatten().tolist() == [1, 2, 3, 4, 5] and (memory.count, memory.next) == (5, 1)
    memory.push(rows[6:8])
    assert memory.contents().flatten().tolist() == [3, 4, 5, 6, 7]
    assert sorted(memory.targets().flatten().tolist()) == [3, 4, 5, 6, 7]
    copy = SeenTargetMemory(5)
    copy.load_state_dict(memory.state_dict())
    memory.push(rows[8:15])  # at least the capacity: keep the last five
    copy.push(rows[8:15])
    assert memory.contents().flatten().tolist() == [10, 11, 12, 13, 14] == copy.contents().flatten().tolist()
    assert (memory.count, memory.next) == (5, 0)
    with pytest.raises(ValueError, match="changed"):
        memory.push(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="capacity"):
        SeenTargetMemory(4).check_state(memory.state_dict())
    with pytest.raises(ValueError):
        SeenTargetMemory(0)


def test_trainer_memory_holds_the_most_recent_consumed_generator_targets():
    trainer = make_trainer()
    results, seen = [], []
    for step in range(5):
        real, real_g = batches(step)
        seen.append(real_g)
        results.append(trainer.step(real, generator_real=real_g))
        expected = torch.cat(seen)[-MEMORY:]
        assert torch.equal(trainer.energy_transport_memory(), expected)
        assert results[-1]["energy_transport_stats"]["targets"] == min(BATCH * (step + 1), MEMORY)
        assert results[-1]["energy_transport_stats"]["generated"] == N
    batch_only = make_trainer(memory=0, population=False)
    result = batch_only.step(*batches(0)[:1], generator_real=batches(0)[1])
    assert batch_only.energy_transport_memory() is None
    assert result["energy_transport_stats"]["targets"] == BATCH == result["energy_transport_stats"]["generated"]


def test_trainer_evaluates_the_declared_objects_population_against_memory():
    """Semantic control: reported energy equals the formula on G(all rows) vs the pushed FIFO."""
    trainer = make_trainer(mog=False, output_noise=0.)
    seen = []
    for step in range(4):
        real, real_g = batches(step)
        seen.append(real_g)
        with torch.no_grad():
            population = trainer.G(trainer.prior.z).clone()
        result = trainer.step(real, generator_real=real_g)
        expected = energy_transport_loss(population, torch.cat(seen)[-MEMORY:])
        torch.testing.assert_close(result["energy_transport_stats"]["energy"], expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result["energy_transport"], expected * trainer.recipe.energy_transport_weight,
                                   rtol=1e-5, atol=1e-6)
    # A pure-atom prior without output noise needs no energy randomness.
    assert torch.equal(trainer.energy_generator.get_state(),
                       torch.Generator().manual_seed(ENERGY_STREAM_OFFSET).get_state())


# (d) Exact continuation in fresh processes across the memory fill boundary, population on.

def _worker(mode, path, split, total, source=None):
    torch.set_num_threads(1)
    torch.manual_seed(1234)  # this interpreter's default RNG seed is per-process; the caller fixes it
    trainer = make_trainer()
    if mode == "full":
        run(trainer, 0, total)
        torch.save(trainer.state_dict(), path)
    elif mode == "prefix":
        for stop in split:
            run(trainer, trainer.completed_steps, stop)
            torch.save(trainer.state_dict(), f"{path}.{stop}")
    elif mode == "resume":
        trainer.load_state_dict(torch.load(f"{source}.{split[0]}", map_location="cpu", weights_only=True))
        run(trainer, split[0], total)
        torch.save(trainer.state_dict(), path)


def _spawn(*arguments):
    code = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tests']; "
            "import test_energy_transport as t; t._worker(sys.argv[2], sys.argv[3], "
            "[int(v) for v in sys.argv[4].split(',')], int(sys.argv[5]), *sys.argv[6:])")
    subprocess.run([sys.executable, "-c", code, str(ROOT), *map(str, arguments)], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONHASHSEED": "0"}, timeout=600)


def test_exact_continuation_in_fresh_process_across_fill_boundary(tmp_path):
    total = 6
    _spawn("full", tmp_path / "full.pt", "0", total)
    _spawn("prefix", tmp_path / "ckpt", "2,3", total)  # 16 rows (before the boundary), 20 rows (wrapped)
    uninterrupted = torch.load(tmp_path / "full.pt", map_location="cpu", weights_only=True)
    for split in (2, 3):
        saved = torch.load(tmp_path / f"ckpt.{split}", map_location="cpu", weights_only=True)
        memory = saved["energy_transport"]["memory"]
        assert (memory["count"], memory["next"]) == ((16, 16) if split == 2 else (20, 4))
        _spawn("resume", tmp_path / f"resumed{split}.pt", f"{split}", total, tmp_path / "ckpt")
        resumed = torch.load(tmp_path / f"resumed{split}.pt", map_location="cpu", weights_only=True)
        assert_same(uninterrupted, resumed)
    assert uninterrupted["energy_transport"]["memory"]["count"] == MEMORY
    assert not torch.equal(uninterrupted["streams"]["energy_generator"],
                           torch.Generator().manual_seed(ENERGY_STREAM_OFFSET).get_state())


def test_dv12_feature_cell_population_continues_exactly_and_draws_its_own_stream():
    """Atlas perturbs latents (DV12 + feature cells); population draws must use the energy stream."""
    recipe = get_recipe("atlas", num_particles=32, z_dim=2, batch_size=BATCH, energy_transport_weight=1.,
                        energy_transport_memory=MEMORY, energy_transport_population=True)

    def build():
        prior = recipe.make_prior(generator=torch.Generator().manual_seed(0), init_std=.5)
        G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
        D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
        init.deterministic_orthogonal_(G, seed=0)
        init.deterministic_orthogonal_(D, seed=1)
        init.deterministic_orthogonal_(prior, seed=2)
        return GANTrainer(recipe, G, D, prior=prior, seed=0, max_steps=12)

    first, second = build(), build()
    run(first, 0, 2)
    saved = first.state_dict()
    run(first, 2, 5)
    second.load_state_dict(saved)
    run(second, 2, 5)
    assert_same(first.state_dict(), second.state_dict())
    assert not torch.equal(first.energy_generator.get_state(),
                           torch.Generator().manual_seed(ENERGY_STREAM_OFFSET).get_state())


# (e) Incompatible checkpoints are rejected before any mutation.

def test_incompatible_checkpoints_are_rejected():
    trainer = run(make_trainer(), 0, 3)
    resized_trainer, disabled = make_trainer(memory=24), make_trainer(weight=0.)
    off = run(make_trainer(weight=0.), 0, 1).state_dict()
    saved = trainer.state_dict()
    before = trainer.state_dict()

    def rejects(state, match, target=trainer):
        with pytest.raises(ValueError, match=match):
            target.load_state_dict(state)

    missing = dict(saved)
    del missing["energy_transport"]
    rejects(missing, "energy transport state is missing")
    no_stream = {**saved, "streams": {k: v for k, v in saved["streams"].items() if k != "energy_generator"}}
    rejects(no_stream, "RNG schema")
    resized = {**saved, "energy_transport": {**saved["energy_transport"], "memory_capacity": 24}}
    rejects(resized, "memory size")
    short = {**saved, "energy_transport": {**saved["energy_transport"], "memory": {
        **saved["energy_transport"]["memory"], "rows": saved["energy_transport"]["memory"]["rows"][:10]}}}
    rejects(short, "memory rows")
    flipped = {**saved, "energy_transport": {**saved["energy_transport"], "population": False}}
    rejects(flipped, "population")
    rejects(saved, "recipe does not match", resized_trainer)
    rejects(saved, "energy transport state", disabled)
    rejects(off, "energy transport state is missing")
    rejects({**off, "energy_transport": saved["energy_transport"]}, "energy transport state", disabled)
    assert_same(before, trainer.state_dict())
    trainer.load_state_dict(saved)  # the intact checkpoint still loads


# (f) Population: every row exactly once per update, other streams untouched.

def test_population_visits_every_row_once_and_leaves_training_streams_unchanged():
    def traced(population):
        trainer = make_trainer(population=population)
        calls, original = [], trainer.policy.generate

        def spy(latent, *, sigma=None, stream=None, averaged=False, model=None, rows=None):
            calls.append((trainer.completed_steps, stream is trainer.energy_generator and stream is not None,
                          None if rows is None else rows.clone()))
            return original(latent, sigma=sigma, stream=stream, averaged=averaged, model=model, rows=rows)

        trainer.policy.generate = spy
        run(trainer, 0, 4)
        return trainer, calls

    on, on_calls = traced(True)
    off, off_calls = traced(False)
    population_calls = [(step, rows) for step, energy, rows in on_calls if energy]
    assert [step for step, _ in population_calls] == [0, 1, 2, 3]
    for _, rows in population_calls:
        assert torch.equal(rows, torch.arange(N))
    ordinary_on = [(step, rows) for step, energy, rows in on_calls if not energy]
    ordinary_off = [(step, rows) for step, energy, rows in off_calls if not energy]
    assert len(ordinary_on) == len(ordinary_off)
    for (step_a, rows_a), (step_b, rows_b) in zip(ordinary_on, ordinary_off):
        assert step_a == step_b and torch.equal(rows_a, rows_b)
    on_streams, off_streams = on.state_dict()["streams"], off.state_dict()["streams"]
    assert set(on_streams) - set(off_streams) == {"energy_generator"}
    for name in off_streams:
        assert torch.equal(on_streams[name], off_streams[name]), name
    assert off.energy_generator is None
    # MoG kernel noise and output noise of the population come from the energy stream.
    assert not torch.equal(on_streams["energy_generator"],
                           torch.Generator().manual_seed(ENERGY_STREAM_OFFSET).get_state())


def test_population_rejects_a_manipulator_index_law():
    with pytest.raises(ValueError, match="row law"):
        make_trainer(mog=False, output_noise=0.,
                     manipulators=StratifiedSamplingManipulator(N, jitter_std=0.))


def test_explicit_energy_stream_is_bound_separate_and_only_with_population():
    stream = torch.Generator().manual_seed(77)
    recipe = make_recipe()

    def build(recipe, **streams):
        G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2))
        D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1))
        return GANTrainer(recipe, G, D, prior=recipe.make_prior(
            generator=torch.Generator().manual_seed(0), init_std=.5, sigma=.05), seed=0, **streams)

    assert build(recipe, energy_generator=stream).energy_generator is stream
    with pytest.raises(ValueError, match="only by energy_transport_population"):
        build(make_recipe(population=False), energy_generator=stream)
    with pytest.raises(ValueError, match="only by energy_transport_population"):
        build(make_recipe(weight=0.), energy_generator=stream)
    with pytest.raises(ValueError, match="separate from the other trainer streams"):
        build(recipe, latent_generator=stream, energy_generator=stream)
    with pytest.raises(ValueError, match="evaluation stream must be separate"):
        build(recipe, eval_generator=stream, energy_generator=stream)


# Forge binds the energy stream as a named stream, so its context checkpoint,
# RNG consistency check and exact continuation cover population runs.

def _forge_energy_context(population=True):
    from experiments.forge.api import FormulationContext
    overrides = dict(num_particles=12, z_dim=2, batch_size=4, total_steps=6)
    if population is not None:
        overrides.update(energy_transport_weight=1., energy_transport_memory=8,
                         energy_transport_population=population)
    context = FormulationContext(recipe_preset="k3p", recipe_overrides=overrides)
    G = context.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 2)),
                          component="generator")
    D = context.construct(lambda: nn.Sequential(nn.Linear(2, 6), nn.LeakyReLU(.2), nn.Linear(6, 1)),
                          component="discriminator")
    return context, context.build_trainer(G, D)


def _forge_steps(trainer, start, stop):
    for step in range(start, stop):
        generator = torch.Generator().manual_seed(2000 + step)
        trainer.step(ring(4, generator), generator_real=ring(4, generator))


def test_forge_population_checkpoint_round_trip_and_exact_continuation():
    from experiments.forge.api import RECIPE_STREAM_BINDINGS
    from experiments.forge.state import require_consistent_rng, require_same_formulation
    context, trainer = _forge_energy_context()
    energy_key = next(key for key, binding in context.streams.manifest()["bindings"].items()
                      if (binding["family"], binding["component"], binding["purpose"])
                      == RECIPE_STREAM_BINDINGS["energy_generator"])
    assert trainer.energy_generator is context.streams._streams[energy_key]
    _forge_steps(trainer, 0, 2)
    saved = context.state_dict()
    require_consistent_rng(saved)
    resumed_context, resumed = _forge_energy_context()
    resumed_context.load_state_dict(saved)
    require_same_formulation(saved, resumed_context.state_dict())
    assert_same(saved, resumed_context.state_dict())
    _forge_steps(trainer, 2, 4)
    _forge_steps(resumed, 2, 4)
    assert_same(context.state_dict(), resumed_context.state_dict())
    # A checkpoint without the energy stream is incomplete, not silently reseeded.
    damaged = {**saved, "trainer": {**saved["trainer"], "streams": {
        key: value for key, value in saved["trainer"]["streams"].items() if key != "energy_generator"}}}
    with pytest.raises(ValueError, match="incomplete"):
        require_consistent_rng(damaged)


def test_forge_binds_no_energy_stream_without_population():
    for population in (None, False):
        context, trainer = _forge_energy_context(population)
        assert trainer.energy_generator is None
        assert not any(binding["component"] == "energy"
                       for binding in context.streams.manifest()["bindings"].values())
        _forge_steps(trainer, 0, 1)
        saved = context.state_dict()
        from experiments.forge.state import require_consistent_rng
        require_consistent_rng(saved)
        fresh, _ = _forge_energy_context(population)
        fresh.load_state_dict(saved)


# (g) Damage control: the objective prefers the target law over collapsed or shifted sets.

def test_damage_control_collapsed_or_shifted_fakes_score_worse():
    data = torch.Generator().manual_seed(11)
    target = ring(4096, data)
    faithful = ring(256, data)
    one_mode = torch.tensor([3., 0.]) + .07 * torch.randn(256, 2, generator=data)
    point = torch.zeros(256, 2) + torch.tensor([3., 0.])
    shifted = ring(256, data, shift=1.)
    good = energy_transport_loss(faithful, target)
    for damaged in (one_mode, point, shifted, faithful * .5):
        assert energy_transport_loss(damaged, target) > good
    scalar = 2 + .5 * torch.randn(4096, 1, generator=data)
    sample = 2 + .5 * torch.randn(256, 1, generator=data)
    good = energy_transport_loss(sample, scalar)
    for damaged in (sample + .5, 2 + .25 * (sample - 2), torch.full_like(sample, 2.)):
        assert energy_transport_loss(damaged, scalar) > good
