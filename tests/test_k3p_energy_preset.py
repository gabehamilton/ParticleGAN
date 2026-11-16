"""The named ``k3p_energy`` preset is k3p plus exactly its declared delta.

It must resolve, on the two selection hosts, to the complete Recipe that the
native-baseline round-2 sweep executed (source ed3a393a5, seed 0) apart from
the ``name`` label. The digests below are sha256 of the receipts' resolved
``recipe`` objects, JSON-serialized with sorted keys. This is a software
identity check, not qualification: the preset FAILS ring8 numerically.
"""
import hashlib
import json

import pytest

from benchmarks.toy_audit import api_contract
from particlegan import get_recipe

DELTA = {
    "energy_transport_weight": 300.,
    "energy_transport_memory": 4096,
    "energy_transport_population": True,
    "output_noise_std": 0.,
    "lr_anneal_start": .2,
    "lr_floor": .01,
    "network_lr_floor": .001,
}

# Resolved recipe digests of S1-atoms-w300-m4096-pop/<case>/receipt.json (label "k3p").
SWEPT_RECIPE_SHA256 = {
    "api-ring8-acquire": "8530304583deceba6570b7de709330009fd9a41fa872f558d59f4096ce43c3af",
    "api-gaussian1d-acquisition": "71aeb833bb4e829612358ad0ba7ae1fa92e00ed895a50921dd021f98d34745cd",
}


def _json(recipe):
    return json.loads(json.dumps(recipe.to_dict()))


def test_k3p_energy_is_k3p_plus_exactly_the_declared_delta():
    named = get_recipe("k3p_energy")
    assert named.name == "k3p_energy"
    for field, value in DELTA.items():
        assert getattr(named, field) == value, field
    assert named == get_recipe("k3p", **DELTA).replace(name="k3p_energy")
    base, selected = _json(get_recipe("k3p")), _json(named)
    changed = {key for key in base.keys() | selected.keys() if base.get(key) != selected.get(key)}
    assert changed == set(DELTA) | {"name"}
    # No other mechanism enters through the preset.
    assert named.continuous_policy is None and named.particle_birth_death is False
    assert named.serve_average == 0. and named.kinetic_transport_weight == 0.


@pytest.mark.parametrize("case_id", sorted(SWEPT_RECIPE_SHA256))
def test_k3p_energy_resolves_on_selection_hosts_to_the_swept_recipe(case_id):
    case = api_contract.discover()[case_id]
    fixture = api_contract.build(case, recipe_name="k3p_energy", seed=0, max_steps=1)
    # The sweep built get_recipe("k3p", **task resources, **delta) as a complete Recipe.
    resources = dict(num_particles=case["particles"], z_dim=case["z_dim"],
                     batch_size=case["batch_size"], total_steps=case["default_steps"])
    if case.get("prior_options", {}).get("prior_kind") == "mog":
        resources.update(prior_kind="mog", standardize=case["prior_options"]["standardize"],
                         sigma_rel=case["prior_options"]["sigma_rel"])
    swept = get_recipe("k3p", **resources, **DELTA)
    assert fixture.recipe == swept.replace(name="k3p_energy")
    receipt_view = {**_json(fixture.recipe), "name": "k3p"}
    digest = hashlib.sha256(json.dumps(receipt_view, sort_keys=True).encode()).hexdigest()
    assert digest == SWEPT_RECIPE_SHA256[case_id]
