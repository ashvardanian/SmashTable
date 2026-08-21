"""The lazy-cursor contract: what iteration promises while the container is being changed.

Baselines:
    dict, contrasted rather than matched. A dict raises RuntimeError on mutation during
    iteration; these containers deliberately do not, and one test asserts the divergence so it
    stays a decision rather than an accident.

Run:
    python -m pytest test/iteration.py -v
"""

import gc

import pytest

from .base import (
    enumerable_class_names,
    is_map_class,
    key_types,
    populate,
    sorted_class_names,
)

# region Termination


@pytest.mark.parametrize("class_name", enumerable_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_iterating_nothing_stops_immediately(container):
    """An empty container yields nothing rather than hanging or raising."""
    assert list(container) == []
    assert list(iter(container)) == []


@pytest.mark.parametrize("class_name", enumerable_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_each_key_is_visited_exactly_once(container, keygen):
    """A walk sees every element and never repeats one."""
    keys = keygen(50)
    model = populate(container, keys, keys)
    walked = list(container)
    assert len(walked) == len(model)
    assert len(set(map(repr, walked))) == len(walked)


# endregion Termination

# region Mutation during iteration


@pytest.mark.parametrize("class_name", sorted_class_names)
def test_insertion_ahead_of_the_cursor_is_observed(container_class):
    """A key inserted beyond the cursor appears, because the position is re-derived each step."""
    container = container_class(key="int")
    populate(container, [0, 10], [0, 10])
    walked = []
    for key in container:
        walked.append(key)
        if key == 0:
            populate(container, [5], [5])
    assert walked == [0, 5, 10]


@pytest.mark.parametrize("class_name", sorted_class_names)
def test_insertion_behind_the_cursor_is_not_observed(container_class):
    """A key inserted below the cursor is missed, which is the cost of not snapshotting."""
    container = container_class(key="int")
    populate(container, [5, 10], [5, 10])
    walked = []
    for key in container:
        walked.append(key)
        if key == 5:
            populate(container, [1], [1])
    assert walked == [5, 10]


@pytest.mark.parametrize("class_name", sorted_class_names)
def test_deleting_the_current_key_does_not_strand_the_walk(container_class):
    """Erasing the key just yielded is harmless: the cursor is a value, not a node pointer."""
    container = container_class(key="int")
    keys = [1, 2, 3, 4]
    populate(container, keys, keys)
    erase = container.__delitem__ if is_map_class(container_class) else container.discard
    walked = []
    for key in container:
        walked.append(key)
        erase(key)
    assert walked == keys
    assert len(container) == 0


@pytest.mark.parametrize("class_name", sorted_class_names)
def test_deletion_ahead_of_the_cursor_is_skipped(container_class):
    """A key removed before the cursor reaches it is simply never yielded."""
    container = container_class(key="int")
    keys = [1, 2, 3]
    populate(container, keys, keys)
    erase = container.__delitem__ if is_map_class(container_class) else container.discard
    walked = []
    for key in container:
        walked.append(key)
        if key == 1:
            erase(2)
    assert walked == [1, 3]


# One sequence drives the two tests that contrast a container with a dict: walk a store holding
# `seeded_keys` and insert `key_above_the_seed`, which every seeded key sorts below, so an ordered
# walk can still reach it.
seeded_keys = list(range(10))
key_above_the_seed = 100

# A walk that has not ended by here is not going to, and the assertion after the loop never runs.
runaway_ceiling = 100


@pytest.mark.parametrize("class_name", enumerable_class_names)
def test_mutation_during_iteration_does_not_raise(container_class):
    """Unlike dict, changing the container mid-walk is legal here."""
    container = container_class(key="int")
    populate(container, seeded_keys, seeded_keys)
    walked = 0
    for index, _ in enumerate(container):
        walked += 1
        if index == 2:
            populate(container, [key_above_the_seed], [0])
        if walked > runaway_ceiling:
            pytest.fail(f"iteration failed to terminate, {walked} steps in")
    assert walked == len(seeded_keys) + 1, f"walked {walked}, wanted {len(seeded_keys) + 1}"


def test_a_dict_would_have_raised_on_the_same_sequence():
    """The contrast is deliberate: a dict raises where these containers keep going."""
    model = dict.fromkeys(seeded_keys, 0)
    with pytest.raises(RuntimeError):
        for index, _ in enumerate(model):
            if index == 2:
                model[key_above_the_seed] = 0


@pytest.mark.parametrize("class_name", enumerable_class_names)
def test_clear_during_iteration_terminates(container_class):
    """Emptying the container mid-walk ends the walk on the next step rather than looping."""
    container = container_class(key="int")
    keys = list(range(20))
    populate(container, keys, keys)
    walked = 0
    for _ in container:
        walked += 1
        container.clear()
        if walked > runaway_ceiling:
            pytest.fail(f"iteration failed to terminate after clear(), {walked} steps in")
    assert walked == 1, f"walked {walked} steps over a container emptied after the first"


# endregion Mutation during iteration

# region Lifetime and independence


@pytest.mark.parametrize("class_name", enumerable_class_names)
def test_an_iterator_keeps_its_container_alive(container_class):
    """Dropping the last named reference mid-walk must not free the store underneath."""
    container = container_class(key="int")
    keys = list(range(5))
    populate(container, keys, keys)
    walk = iter(container)
    first = next(walk)
    del container
    gc.collect()
    remaining = list(walk)
    assert first == 0
    assert remaining == keys[1:]


@pytest.mark.parametrize("class_name", enumerable_class_names)
def test_two_walks_do_not_interfere(container_class):
    """Each cursor carries its own position, so nested walks are independent."""
    container = container_class(key="int")
    keys = list(range(4))
    populate(container, keys, keys)
    pairs = [(outer, inner) for outer in container for inner in container]
    assert len(pairs) == len(keys) * len(keys)


@pytest.mark.parametrize("class_name", enumerable_class_names)
def test_an_exhausted_walk_stays_exhausted(container_class):
    """A cursor that ended does not restart when the container grows again."""
    container = container_class(key="int")
    populate(container, [1], [1])
    walk = iter(container)
    assert list(walk) == [1]
    populate(container, [2], [2])
    assert list(walk) == []


# endregion Lifetime and independence
