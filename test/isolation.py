"""What each isolation level and sharing strategy actually promises, and what it reports.

Baselines:
    Jepsen's consistency-model names, which the C++ layer publishes as `isolation_k` and this
    layer reports verbatim. The property is asserted against `effective_isolation`, which encodes
    the one place a container promises less than it was asked for.

Matches C++ suite:
    scripts/test_consistency.hpp, which checks the same repeated-read guarantee one layer down.

Run:
    python -m pytest test/isolation.py -v
"""

import pytest

import smashtable as st

from .base import (
    all_class_names,
    effective_isolation,
    isolation_levels,
    key_types,
    make,
    populate,
    sharing_modes,
)

# region Reporting


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("sharing", sharing_modes, indirect=True)
@pytest.mark.parametrize("isolation", isolation_levels, indirect=True)
@pytest.mark.parametrize("key_type", key_types)
def test_isolation_reports_the_effective_level(container, isolation, sharing):
    """The level a container delivers, which is not always the level it was asked for."""
    assert container.isolation == effective_isolation(isolation, sharing)


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_the_default_is_the_stronger_sharing(container_class, key_type):
    """Sharing defaults to one lock, which is the only way snapshot survives above a single key."""
    assert make(container_class, key_type).isolation == "monotonic_atomic_view"
    assert make(container_class, key_type, isolation="snapshot").isolation == "snapshot"


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_sharding_caps_a_stampless_level(container_class, key_type):
    """A monotonic reader holds no stamp, so sharding it can never promise more than Read Committed.

    Stated on its own because it is the permanent half of the cap: what a partitioned snapshot
    container reports may rise, but this cannot.
    """
    sharded = make(container_class, key_type, isolation="monotonic_atomic_view", sharing="partitioned")
    assert sharded.isolation == "read_committed"


# endregion Reporting

# region Refusals


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize("argument", ["isolation", "sharing"])
def test_an_unknown_choice_is_refused(container_class, key_type, argument):
    """A misspelled level is a ValueError at construction, not a silently weaker container."""
    with pytest.raises(ValueError):
        container_class(**{"key": key_type, argument: "serialisable"})


@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize("argument", ["isolation", "sharing"])
def test_a_non_string_choice_is_refused(container_class, key_type, argument):
    """Both axes name a promise, so neither accepts anything but the names."""
    with pytest.raises(TypeError):
        container_class(**{"key": key_type, argument: 3})


# endregion Refusals

# region Guarantees


@pytest.mark.parametrize("sharing", sharing_modes, indirect=True)
@pytest.mark.parametrize("isolation", isolation_levels, indirect=True)
@pytest.mark.parametrize("class_name", all_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_every_configuration_round_trips(container, keygen):
    """Whatever the level and the sharing, the container is still a container."""
    keys = keygen(8)
    populate(container, keys, range(len(keys)))
    assert len(container) == len(keys)
    for key in keys:
        assert key in container


@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_snapshot_repeats_its_reads(keygen):
    """A snapshot transaction answers every read at the instant it opened."""
    key = keygen(1)[0]
    container = make(st.SortedMap, "int", isolation="snapshot", sharing="locked")
    container[key] = "first"

    group = st.transaction(container)
    (view,) = group.begin()
    assert view[key] == "first"

    container[key] = "second"  # Committed by an outsider while the transaction is open
    assert view[key] == "first", "a snapshot read must not follow a later commit"


@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_sharded_snapshot_repeats_reads_across_partitions(keygen):
    """Sharding does not cost a snapshot container its level, and the guarantee spans partitions.

    Enough keys to reach every partition, so this fails if any one of them answers from a clock of
    its own rather than the shared one.
    """
    keys = keygen(64)
    container = make(st.SortedMap, "int", isolation="snapshot", sharing="partitioned")
    for key in keys:
        container[key] = 0

    group = st.transaction(container)
    (view,) = group.begin()
    assert [view[key] for key in keys] == [0] * len(keys)

    for key in keys:  # An outsider rewrites every partition while the transaction is open
        container[key] = 1
    assert [view[key] for key in keys] == [0] * len(keys), "a sharded snapshot must not tear"

    later = st.transaction(container)
    (fresh_view,) = later.begin()
    assert [fresh_view[key] for key in keys] == [1] * len(keys), "a later transaction sees the commits"


@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_monotonic_does_not_repeat_its_reads(keygen):
    """The weaker level is allowed to see the newer value, and the contrast is the whole point.

    Asserted rather than left implicit so that strengthening `monotonic_store` cannot silently make
    the two levels indistinguishable, which would leave the keyword buying nothing.
    """
    key = keygen(1)[0]
    container = make(st.SortedMap, "int", isolation="monotonic_atomic_view", sharing="locked")
    container[key] = "first"

    group = st.transaction(container)
    (view,) = group.begin()
    assert view[key] == "first"

    container[key] = "second"
    assert view[key] == "second"


# endregion Guarantees

# region Write Skew


def _doctors_on_call(isolation: str, keygen) -> tuple[bool, int]:
    """Runs the two-doctors schedule and reports whether the second commit was refused.

    Two transactions each read the *other* doctor, find them on call, and take themselves off.
    Neither writes what the other wrote, so a write-set check alone sees no conflict - only a
    reader that validates what it read can catch this.
    """
    alice, bob = keygen(2)
    container = make(st.SortedMap, "int", isolation=isolation, sharing="locked")
    container[alice] = 1
    container[bob] = 1

    hers = st.transaction(container)
    (her_view,) = hers.begin()
    his = st.transaction(container)
    (his_view,) = his.begin()

    her_view[bob]  # each checks that the other is covering
    his_view[alice]
    her_view[alice] = 0  # and so goes off call
    his_view[bob] = 0

    hers.stage()
    hers.commit()
    try:
        his.stage()
        his.commit()
        refused = False
    except st.ConflictError:
        refused = True
    return refused, container[alice] + container[bob]


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_snapshot_admits_write_skew(keygen):
    """Snapshot validates only what a transaction wrote, so disjoint writes never conflict.

    Asserted as an anomaly the level permits rather than left untested, because it is the whole
    reason the level above it exists. Berenson lists A5B as possible under Snapshot.
    """
    refused, on_call = _doctors_on_call("snapshot", keygen)
    assert not refused, "disjoint write sets give a snapshot commit nothing to conflict on"
    assert on_call == 0, "both doctors went off call, which is the anomaly"


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_serializable_refuses_write_skew(keygen):
    """Serializable validates what was read too, which is exactly what catches this.

    The acceptance criterion for the level: the same schedule that slips through a snapshot
    container is refused here, and the invariant the two transactions were each relying on holds.
    """
    refused, on_call = _doctors_on_call("serializable", keygen)
    assert refused, "a key this transaction read was written under it, so the commit must refuse"
    assert on_call == 1, "one doctor stays on call, which is the invariant write skew broke"


# endregion Write Skew

# region Refusal Causes


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_written_key_refuses_by_write_conflict(keygen):
    """Both transactions write one key, so the write set alone catches it."""
    container = make(st.SortedMap, "int", isolation="serializable", sharing="locked")
    (key,) = keygen(1)
    container[key] = 0

    hers = st.transaction(container)
    (her_view,) = hers.begin()
    his = st.transaction(container)
    (his_view,) = his.begin()
    her_view[key] = 10
    his_view[key] = 20

    hers.stage()
    hers.commit()
    with pytest.raises(st.WriteConflictError):
        his.stage()
        his.commit()


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_read_key_refuses_by_read_conflict(keygen):
    """One transaction reads a key the other then writes, and writes elsewhere itself.

    The write sets are disjoint, so only validating what was read can refuse this - which is the
    difference `serializable` buys over `snapshot`.
    """
    container = make(st.SortedMap, "int", isolation="serializable", sharing="locked")
    read_key, written_key = keygen(2)
    container[read_key] = 0
    container[written_key] = 0

    hers = st.transaction(container)
    (her_view,) = hers.begin()
    his = st.transaction(container)
    (his_view,) = his.begin()
    his_view[read_key]  # he reads it
    his_view[written_key] = 5  # and writes somewhere else entirely
    her_view[read_key] = 10  # she writes what he read

    hers.stage()
    hers.commit()
    with pytest.raises(st.ReadConflictError):
        his.stage()
        his.commit()


def test_every_refusal_cause_is_catchable_as_one():
    """The three causes subclass `ConflictError`, so a retry loop need not know which it got."""
    for cause in (st.WriteConflictError, st.ReadConflictError, st.PhantomConflictError):
        assert issubclass(cause, st.ConflictError)
        assert issubclass(cause, st.SmashTableError)
        assert issubclass(cause, RuntimeError)


def _phantom_at(level: str, keygen, *, lower_bound: str):
    """Runs the phantom schedule at `level`, returning what refused the scanner and what it saw.

    One transaction scans a window that comes back empty and another commits a key into it, with
    disjoint write sets, so only validating the window itself can refuse this. A `lower_bound` of
    "open" names no floor, leaving the store rather than the binding to decide what the window
    covers.
    """
    container = make(st.SortedMap, "int", isolation=level, sharing="locked")
    # `keygen` ascends, so the phantom lands inside the window scanned and `written_key` outside it.
    lower, inside, upper, written_key = keygen(4)

    hers = st.transaction(container)
    (her_view,) = hers.begin()
    seen = her_view.scan(lower if lower_bound == "named" else None, upper)

    his = st.transaction(container)
    (his_view,) = his.begin()
    his_view[inside] = "phantom"
    his.stage()
    his.commit()

    her_view[written_key] = "elsewhere"
    try:
        hers.stage()
        hers.commit()
        return None, seen
    except st.ConflictError as refusal:
        return refusal, seen


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("level", ["serializable", "strict_serializable"])
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_window_open_at_the_bottom_still_catches_a_phantom(keygen, level):
    """A scan bounded only from above records a window, so a key committed into it refuses."""
    refusal, seen = _phantom_at(level, keygen, lower_bound="open")
    assert seen == [], "the window was empty, so only the window itself was read"
    assert isinstance(refusal, st.PhantomConflictError)


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("level", ["serializable", "strict_serializable"])
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_scanned_window_refuses_by_phantom_conflict(keygen, level):
    """A key committed into a window this transaction read is what the third cause names."""
    refusal, seen = _phantom_at(level, keygen, lower_bound="named")
    assert seen == [], "the window was empty, so no key was read and only the window itself was"
    assert isinstance(refusal, st.PhantomConflictError)


@pytest.mark.thread_unsafe(
    reason="the schedule is the test - a parallel copy sharing the container would commit between the two transactions"
)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_snapshot_admits_a_phantom(keygen):
    """Snapshot validates only what a transaction wrote, so a window it read can gain a member.

    Asserted as an anomaly the level permits rather than left untested, for the same reason write
    skew is: it is what the level above it exists to refuse. Berenson's A3 under a snapshot read.
    """
    refusal, seen = _phantom_at("snapshot", keygen, lower_bound="named")
    assert seen == []
    assert refusal is None, "disjoint write sets give a snapshot commit nothing to conflict on"


# endregion Refusal Causes
