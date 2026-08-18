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
    for index, key in enumerate(keys):
        if hasattr(container, "add"):
            container.add(key)
        else:
            container[key] = index
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
