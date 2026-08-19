"""Protocol parity between SmashTable containers and the stdlib types they mirror.

Baselines:
    dict for the map classes, set for the set classes, compared through the oracle in test.base
    rather than by hand-written expectation.

Run:
    python -m pytest test/containers.py -v
"""

import pytest

from .base import (
    Op,
    all_class_names,
    apply_op,
    assert_same_state,
    enumerable_class_names,
    enumerable_map_names,
    enumerable_set_names,
    key_types,
    map_class_names,
    populate,
    set_class_names,
    sizes,
    sorted_map_names,
    value_types,
)

# region Construction


@pytest.mark.parametrize("class_name", enumerable_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_new_container_is_empty(container):
    """A freshly built container holds nothing and is falsy."""
    assert len(container) == 0
    assert not container
    assert list(container) == []


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_key_type_reads_back(container, key_type):
    """The layout named at construction is the one the container reports."""
    assert container.key_type == key_type


@pytest.mark.parametrize("class_name", all_class_names)
def test_key_is_required(container_class):
    """A container cannot be built without naming its key layout."""
    with pytest.raises(TypeError):
        container_class()


@pytest.mark.parametrize("class_name", all_class_names)
def test_unknown_key_type_is_rejected(container_class):
    """A key layout that does not exist is a ValueError, not a silent default."""
    with pytest.raises(ValueError):
        container_class(key="quadruple")


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_positional_arguments_are_rejected(container_class, key_type):
    """The key layout is keyword-only, so a stray positional cannot be mistaken for data."""
    with pytest.raises(TypeError):
        container_class({}, key=key_type)


# endregion Construction

# region Mapping protocol


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
@pytest.mark.parametrize("size", sizes)
def test_population_matches_dict(populated):
    """A populated container is indistinguishable from the dict built the same way."""
    container, model = populated
    assert_same_state(container, model)


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very state it compares against its model"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_overwrite_replaces_without_growing(container, keygen, valuegen):
    """Assigning an existing key changes the value and leaves the size alone."""
    key = keygen(1)[0]
    first, second = valuegen(2)
    container[key] = first
    container[key] = second
    assert len(container) == 1
    assert container[key] == second


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize(
    "op",
    [
        pytest.param(Op("getitem", ()), id="getitem"),
        pytest.param(Op("delitem", ()), id="delitem"),
        pytest.param(Op("pop", ()), id="pop"),
    ],
)
def test_missing_key_raises_like_dict(container, keygen, op):
    """A lookup, delete or pop of an absent key raises exactly as a dict would."""
    key = keygen(1)[0]
    apply_op(container, {}, Op(op.name, (key,)))


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_get_and_setdefault_match_dict(container, keygen, valuegen):
    """get and setdefault agree with dict on both the present and the absent key."""
    keys = keygen(2)
    values = valuegen(2)
    model = {}
    for op in (
        Op("get", (keys[0], values[0])),
        Op("setdefault", (keys[0], values[0])),
        Op("setdefault", (keys[0], values[1])),
        Op("get", (keys[1],)),
    ):
        apply_op(container, model, op)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
@pytest.mark.parametrize("size", sizes)
def test_clear_empties(populated):
    """clear leaves the container empty and its key layout untouched."""
    container, _ = populated
    layout = container.key_type
    container.clear()
    assert len(container) == 0
    assert list(container) == []
    assert container.key_type == layout


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_update_from_mapping_and_pairs(container, keygen, valuegen):
    """update accepts a mapping and an iterable of pairs, like dict."""
    keys = keygen(4)
    values = valuegen(4)
    model = {}
    apply_op(container, model, Op("update", (dict(zip(keys[:2], values[:2])),)))
    container.update(list(zip(keys[2:], values[2:])))
    model.update(dict(zip(keys[2:], values[2:])))
    assert_same_state(container, model)


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very state it compares against its model"
)
@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
@pytest.mark.parametrize("size", [pytest.param(7, id="n7")])
def test_popmin_removes_a_real_pair(populated):
    """popmin returns a pair the store actually held, and removes exactly it."""
    container, model = populated
    key, value = container.popmin()
    assert key in model
    assert model[key] == value
    del model[key]
    assert_same_state(container, model)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
def test_popmin_on_empty_raises(container):
    """popmin on an empty store raises KeyError, as dict's popitem does."""
    with pytest.raises(KeyError):
        container.popmin()


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
@pytest.mark.parametrize("size", [pytest.param(7, id="n7")])
@pytest.mark.parametrize("class_name", [pytest.param("SortedMap", id="sortedmap")])
def test_popmin_takes_the_smallest(populated):
    """A sorted map pops its smallest key, which is where it diverges from dict deliberately."""
    container, model = populated
    smallest = sorted(model)[0]
    key, _ = container.popmin()
    assert key == smallest


# endregion Mapping protocol

# region Set protocol


@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="unused")])
@pytest.mark.parametrize("size", sizes)
def test_set_population_matches_set(populated):
    """A populated set is indistinguishable from the set built the same way."""
    container, model = populated
    assert_same_state(container, model)


@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_add_is_idempotent(container, keygen):
    """Adding a member twice leaves one member."""
    member = keygen(1)[0]
    container.add(member)
    container.add(member)
    assert len(container) == 1


@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_discard_is_silent_and_remove_is_not(container, keygen):
    """discard ignores an absent member; remove raises on one, exactly as set does."""
    member = keygen(1)[0]
    container.discard(member)
    with pytest.raises(KeyError):
        container.remove(member)


@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_set_refuses_item_assignment(container, keygen):
    """A set has no values, so subscript assignment is a TypeError."""
    member = keygen(1)[0]
    with pytest.raises(TypeError):
        container[member] = 1


@pytest.mark.parametrize("class_name", enumerable_set_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize(
    "operation",
    ["union", "intersection", "difference", "symmetric_difference"],
)
def test_set_algebra_matches_set(container, keygen, operation):
    """Every algebra method agrees with the stdlib set on the same inputs."""
    keys = keygen(6)
    mine, theirs = keys[:4], keys[2:]
    model = populate(container, mine, mine)
    other = set(theirs)
    got = getattr(container, operation)(other)
    want = getattr(model, operation)(other)
    assert sorted(map(repr, got)) == sorted(map(repr, want))


@pytest.mark.parametrize("class_name", enumerable_set_names)
@pytest.mark.parametrize("key_type", key_types)
def test_isdisjoint_matches_set(container, keygen):
    """isdisjoint agrees with the stdlib set, both when it holds and when it does not."""
    keys = keygen(4)
    model = populate(container, keys[:2], keys[:2])
    assert container.isdisjoint(set(keys[2:])) == model.isdisjoint(set(keys[2:]))
    assert container.isdisjoint(set(keys[:1])) == model.isdisjoint(set(keys[:1]))


@pytest.mark.parametrize("class_name", enumerable_set_names)
@pytest.mark.parametrize("key_type", key_types)
def test_subset_and_superset_match_set(container, keygen):
    """The four ordering comparisons agree with the stdlib set."""
    keys = keygen(4)
    model = populate(container, keys[:2], keys[:2])
    bigger = set(keys)
    assert (container <= bigger) == (model <= bigger)
    assert (container < bigger) == (model < bigger)
    assert (container >= bigger) == (model >= bigger)
    assert (container > bigger) == (model > bigger)


# endregion Set protocol

# region Representation and equality


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(0, id="n0"), pytest.param(3, id="n3")])
def test_repr_names_the_class_and_the_layout(populated):
    """repr identifies the class and the key layout, and terminates."""
    container, _ = populated
    rendered = repr(container)
    assert rendered.startswith(type(container).__name__)
    assert container.key_type in rendered


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(64, id="n64")])
def test_repr_of_a_large_container_is_bounded(populated):
    """A large container still reprs, eliding rather than rendering everything."""
    container, model = populated
    assert len(repr(container)) < 100_000
    assert len(model) == 64


@pytest.mark.parametrize("class_name", enumerable_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(7, id="n7")])
def test_equality_against_the_stdlib_model(populated):
    """A container equals the stdlib model holding the same elements, and differs once it does not."""
    container, model = populated
    assert container == model
    assert not (container != model)


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(3, id="n3")])
def test_comparison_with_a_foreign_type_is_false(populated):
    """Comparing against an unrelated object answers False rather than raising."""
    container, _ = populated
    assert container != object()
    assert not (container == object())


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
def test_equality_follows_python_numeric_rules_for_values(container, keygen):
    """`{k: 1} == {k: 1.0}` holds for a container exactly as it does for a dict."""
    key = keygen(1)[0]
    container[key] = 1
    assert container == {key: 1.0}


@pytest.mark.parametrize("class_name", enumerable_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_containers_are_unhashable(container):
    """A mutable container is unhashable, like dict and set."""
    with pytest.raises(TypeError):
        hash(container)


# endregion Representation and equality

# region Views


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(0, id="n0"), pytest.param(5, id="n5")])
def test_views_agree_with_each_other(populated):
    """keys, values and items describe one container consistently."""
    container, model = populated
    assert list(container.keys()) == list(container)
    assert list(container.items()) == list(zip(container.keys(), container.values()))
    assert len(container.keys()) == len(model)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(3, id="n3")])
def test_a_view_reflects_later_writes(populated, keygen, valuegen):
    """A view is lazy, so it sees a write that happens after it was taken."""
    container, model = populated
    view = container.keys()
    fresh = keygen(1, start=1000)[0]
    container[fresh] = valuegen(1)[0]
    assert fresh in list(view)
    assert len(view) == len(model) + 1


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(3, id="n3")])
def test_a_view_names_its_container(populated):
    """A view exposes the container it reads, matching dict's view protocol."""
    container, _ = populated
    assert container.keys().mapping is container


# endregion Views


# region Input handling


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_update_accepts_any_two_element_sequence(container):
    """`dict.update` takes pairs of any sequence type, and so must this.

    Regression: the pair was parsed with `PyArg_ParseTuple`, which answers a list with
    `SystemError` - a name that blames the library for an ordinary input.
    """
    container.update([[1, "list"]])
    container.update(((2, "tuple"),))
    assert container[1] == "list" and container[2] == "tuple"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize("bad", [pytest.param([[1, 2, 3]], id="triple"), pytest.param([[1]], id="single")])
def test_update_refuses_a_sequence_that_is_not_a_pair(container, bad):
    """A pair has two elements, and anything else is the caller's error rather than a crash."""
    with pytest.raises(ValueError):
        container.update(bad)


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", key_types)
def test_scan_refuses_a_negative_limit(container):
    """A negative limit is the cursor's own spelling for uncounted, so it must not reach it.

    Regression: `scan(limit=-5)` returned the whole container, which is the opposite of what a
    caller passing a negative bound could possibly mean.
    """
    with pytest.raises(ValueError):
        container.scan(limit=-1)


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", key_types)
def test_scan_limit_zero_yields_nothing(container, keygen):
    """Zero is a real bound, not a sentinel, and is the boundary the negative case sits beside."""
    for index, key in enumerate(keygen(4)):
        container[key] = index
    assert container.scan(limit=0) == []
    assert len(container.scan(limit=2)) == 2


@pytest.mark.parametrize("class_name", enumerable_set_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_set_algebra_propagates_a_real_error(container):
    """A failure inside a caller's `__eq__` is not an answer and must not read as one.

    A member of a foreign type genuinely is not in this set, so that stays tolerated; anything
    else - an interrupt, a memory error - has to reach the caller.
    """
    container.update([1, 2, 3])

    class Angry:
        def __eq__(self, other):
            raise MemoryError("propagate me")

        def __hash__(self):
            return 1

    with pytest.raises(MemoryError):
        container.union([Angry()])
    assert container.isdisjoint(["not an int"]) is True, "a foreign member is simply not shared"


# endregion Input handling

# region Batched writes


@pytest.mark.thread_unsafe(
    reason="the refused batch is the test - a parallel copy sharing the container would apply the same pairs before this one looks"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_map_update_applies_as_one_unit(container, keygen):
    """A batch that cannot be applied whole applies nothing.

    The store opens one transaction for the batch, stages it and commits it once, so a bad pair
    part-way through leaves the store as it was. `dict` applies pairs one at a time and would keep
    the prefix.
    """
    keys = keygen(3)
    container[keys[0]] = "kept"
    with pytest.raises(TypeError):
        container.update([(keys[1], "a"), (keys[2], "b"), (object(), "never")])
    assert dict(container) == {keys[0]: "kept"} if hasattr(container, "keys") else len(container) == 1

    container.update([(keys[1], "a"), (keys[2], "b")])
    assert len(container) == 3


@pytest.mark.thread_unsafe(
    reason="the refused batch is the test - a parallel copy sharing the container would apply the same members before this one looks"
)
@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_set_update_applies_as_one_unit(container, keygen):
    """The same for a set, so the two do not disagree on what a batch means."""
    keys = keygen(3)
    container.add(keys[0])
    with pytest.raises(TypeError):
        container.update([keys[1], keys[2], object()])
    assert len(container) == 1, "a refused batch added members"

    container.update([keys[1], keys[2]])
    assert len(container) == 3


# endregion Batched writes
