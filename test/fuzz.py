"""Randomized differential testing against the stdlib types.

Baselines:
    dict and set, driven through the same operation sequence and compared after every step, so
    a divergence names the exact operation that caused it.

Run:
    python -m pytest test/fuzz.py -v
    SMASHTABLE_TESTS_SEED=42 python -m pytest test/fuzz.py -v
"""

import pytest

import smashtable as st

from .base import (
    assert_same_state,
    key_types,
    make,
    map_class_names,
    random_map_ops,
    random_set_ops,
    replay,
    set_class_names,
    value_types,
)


class _Abort(Exception):
    """Aborts a batch, distinct from anything the library raises."""


@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_a_random_sequence_matches_dict(container, keygen, valuegen, rng):
    """A random walk over the mapping surface leaves the container agreeing with a dict."""
    keys = keygen(32)
    values = valuegen(16)
    replay(container, {}, random_map_ops(rng, keys, values, count=150))


@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_random_sequence_matches_set(container, keygen, rng):
    """A random walk over the set surface leaves the container agreeing with a set."""
    members = keygen(32)
    replay(container, set(), random_set_ops(rng, members, count=150))


@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
def test_random_transactions_match_the_model(container_class, key_type, keygen, valuegen, rng):
    """Each batch either commits and the model advances, or aborts and it must not."""
    container = make(container_class, key_type)
    keys = keygen(16)
    values = valuegen(8)
    model = {}

    for _ in range(12):
        aborting = rng.random() < 0.35
        shadow = dict(model)
        batch = [(rng.choice(keys), rng.choice(values)) for _ in range(rng.randint(1, 6))]
        try:
            with st.atomic(container) as (view,):
                for key, value in batch:
                    view[key] = value
                    shadow[key] = value
                if aborting:
                    raise _Abort
        except _Abort:
            pass
        else:
            model = shadow
        assert_same_state(container, model)


@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_random_scan_windows_match_the_sorted_model(container, keygen, rng):
    """Every random window agrees with the same slice of a sorted model."""
    keys = keygen(40)
    model = {}
    for key in keys:
        container[key] = key
        model[key] = key

    for _ in range(25):
        low = rng.choice(keys)
        high = rng.choice(keys)
        got = [key for key, _ in container.scan(low, high)]
        want = sorted(key for key in model if low <= key < high)
        assert got == want, f"scan({low}, {high}) gave {got}, model gave {want}"


@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_random_interleaved_iteration_terminates(container, keygen, rng):
    """A walk under random mutation always ends, and never yields a key twice."""
    keys = keygen(30)
    for key in keys:
        container[key] = key

    seen = []
    steps = 0
    for key in container:
        seen.append(key)
        steps += 1
        if steps > 1000:
            pytest.fail("iteration failed to terminate under mutation")
        roll = rng.random()
        if roll < 0.3:
            container[rng.randint(0, 1000)] = 0
        elif roll < 0.5:
            victim = rng.choice(keys)
            if victim in container:
                del container[victim]
    assert len(set(seen)) == len(seen), "a key was yielded twice"
