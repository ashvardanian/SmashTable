"""Randomized differential testing against the stdlib types.

Baselines:
    dict and set, driven through the same operation sequence and compared after every step, so
    a divergence names the exact operation that caused it. A group is compared participant by
    participant, since all-or-nothing is a statement about the set of them rather than about one.

Matches C++ suite:
    scripts/test_fuzz.hpp, which drives the same shapes against `reference_store` as its oracle.

Run:
    python -m pytest test/fuzz.py -v
    SMASHTABLE_TESTS_SEED=42 python -m pytest test/fuzz.py -v
"""

import operator

import pytest

import smashtable as st

from .base import (
    adopt_shadows,
    assert_group_state,
    assert_same_state,
    enumerable_map_names,
    isolation_levels,
    key_types,
    make,
    make_participants,
    map_class_names,
    ordering_of,
    random_map_ops,
    random_set_ops,
    replay,
    set_class_names,
    shadows_of,
    sharing_modes,
    stage_group_writes,
    value_modes,
    value_types,
)


class _Abort(Exception):
    """Aborts a batch, distinct from anything the library raises."""


def _permitted(call) -> bool:
    """Runs a phase call, treating a refusal as an answer and anything else as a defect."""
    try:
        call()
    except (st.StateError, st.ConflictError):
        return False
    return True


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_a_random_sequence_matches_dict(container, keygen, valuegen, rng):
    """A random walk over the mapping surface leaves the container agreeing with a dict.

    Swept over both map classes: an unordered one cannot be listed, so the oracle compares it by
    length and by probing every key the model holds instead of by walking it.
    """
    keys = keygen(32)
    values = valuegen(16)
    replay(container, {}, random_map_ops(rng, keys, values, count=150, ordering=ordering_of(type(container))))


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", set_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_random_sequence_matches_set(container, keygen, rng):
    """A random walk over the set surface leaves the container agreeing with a set.

    Swept over both set classes: an unordered one cannot be listed, so the oracle compares it by
    length and by probing every member the model holds instead of by walking it.
    """
    members = keygen(32)
    replay(container, set(), random_set_ops(rng, members, count=150, ordering=ordering_of(type(container))))


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
            with st.transaction(container) as (view,):
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
@pytest.mark.parametrize("class_name", enumerable_map_names)
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


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very state it compares against its model"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", enumerable_map_names)
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


# region Mixed Axes


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int"), pytest.param("str", id="str")])
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("isolation", isolation_levels, indirect=True)
@pytest.mark.parametrize("sharing", sharing_modes, indirect=True)
def test_a_random_sequence_matches_dict_at_every_level(container, keygen, valuegen, rng):
    """The answers a container gives do not depend on the level it delivers them at.

    Isolation governs what a *transaction* may observe, never what a single-threaded sequence of
    direct calls returns, so the same walk must agree with the same dict at all four levels and
    under either sharing mode.
    """
    keys = keygen(24)
    values = valuegen(12)
    replay(container, {}, random_map_ops(rng, keys, values, count=120, ordering=ordering_of(type(container))))


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("value_mode", value_modes, indirect=True)
def test_a_random_map_sequence_matches_dict_in_either_mode(container, keygen, valuegen, rng):
    """The object-mode store path answers the same as the scalar one.

    Object mode holds the interpreter lock where scalar mode drops it, so it is a different code
    path through every store call rather than a different payload, and nothing else fuzzes it.
    Swept over the maps rather than the sets because only a map has values to hold by reference -
    `make` passes no value mode to a set, so both parameters would build the same store.
    """
    keys = keygen(24)
    values = valuegen(12)
    replay(container, {}, random_map_ops(rng, keys, values, count=120, ordering=ordering_of(type(container))))


# endregion Mixed Axes

# region Groups

_GROUP_SPECS = [
    ("SortedMap", "int", "scalar", None, None),
    ("SortedSet", "str", "scalar", "snapshot", None),
    ("SortedMap", "bytes", "object", "serializable", None),
    ("HashMap", "int", "scalar", "snapshot", "partitioned"),
    ("HashSet", "uint", "scalar", "strict_serializable", "partitioned"),
]


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize(
    "group_size",
    [pytest.param(2, id="g2"), pytest.param(3, id="g3"), pytest.param(4, id="g4"), pytest.param(5, id="g5")],
)
def test_a_random_group_publishes_all_of_itself_or_none(group_size, rng):
    """Every round leaves every participant advanced together, or none of them advanced.

    The participants deliberately differ - map beside set, enumerable beside not, scalar beside
    object, locked beside partitioned, four key layouts, four levels - because the group's own
    passes pick the strictest mode across participants and sort them into a canonical order, and a
    group of identical containers exercises neither.
    """
    participants = make_participants(rng, _GROUP_SPECS[:group_size])
    containers = [one.container for one in participants]

    for round_number in range(10):
        aborting = rng.random() < 0.35
        shadows = shadows_of(participants)
        try:
            with st.transaction(*containers) as views:
                stage_group_writes(views, participants, shadows, rng)
                if aborting:
                    raise _Abort
        except _Abort:
            pass
        else:
            participants = adopt_shadows(participants, shadows)
        try:
            assert_group_state(participants)
        except AssertionError as error:
            raise AssertionError(f"round {round_number} ({'aborted' if aborting else 'committed'})\n{error}") from None


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("isolation", isolation_levels, indirect=True)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_refused_group_commit_publishes_no_participant(isolation, keygen, rng):
    """A commit refused over a watched key leaves every participant exactly as it was.

    Regression: a group used to publish the participants ahead of the refusing one and then seal
    itself, so the containers disagreed permanently and `rollback` refused to unwind what was left.
    """
    left = make(st.SortedMap, "int", isolation=isolation)
    right = make(st.SortedMap, "int", isolation=isolation)
    keys = keygen(6)
    for key in keys:
        left[key] = 0
        right[key] = 0

    for _ in range(6):
        watched = rng.choice(keys)
        group = st.transaction(left, right)
        left_view, right_view = group.begin()
        left_view[rng.choice(keys)] = 1
        right_view[watched] = 2
        right_view.watch(watched)
        group.stage()

        right[watched] = 99  # published over the watch while the group sits staged
        before_left, before_right = dict(left), dict(right)
        with pytest.raises(st.ConflictError):
            group.commit()

        assert dict(left) == before_left, "a refused commit published its first participant"
        assert dict(right) == before_right, "a refused commit published its second participant"
        group.rollback()  # the group stayed staged, so it can still be unwound


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.repeat(4)
@pytest.mark.parametrize("isolation", isolation_levels, indirect=True)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_random_lifecycle_never_crashes_and_never_half_applies(isolation, keygen, rng):
    """Random legal and illegal phase orders are refused, never crashed, and never half-applied.

    Regression: a write accepted through a staged participant, followed by a rollback that reported
    failure but reopened the group anyway, moved a version twice and segfaulted the interpreter.
    """
    container = make(st.SortedMap, "int", isolation=isolation)
    keys = keygen(8)
    for key in keys:
        container[key] = 0
    model = dict(container)

    for _ in range(40):
        group = st.transaction(container)
        (view,) = group.begin()
        shadow = dict(model)
        published = False

        for _ in range(rng.randint(1, 6)):
            match rng.choice(("write", "read", "watch", "stage", "commit", "rollback", "reset")):
                case "write":
                    key, value = rng.choice(keys), rng.randrange(100)
                    if _permitted(lambda: operator.setitem(view, key, value)):
                        shadow[key] = value
                case "read":
                    _permitted(lambda: view[rng.choice(keys)])
                case "watch":
                    _permitted(lambda: view.watch(rng.choice(keys)))
                case "stage":
                    _permitted(group.stage)
                case "commit":
                    if _permitted(group.commit):
                        published = True
                # Either one leaves the transaction holding nothing, so the shadow starts over too.
                case "rollback":
                    if _permitted(group.rollback):
                        shadow = dict(model)
                case "reset":
                    if _permitted(group.reset):
                        shadow = dict(model)

        if published:
            model = shadow
        # Whatever the sequence did, the store holds one of the two whole states and never a mix.
        assert dict(container) == model, "a refused or abandoned lifecycle left the store half-applied"


# endregion Groups
