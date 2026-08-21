"""Iteration order and the scan window, for the classes that promise ordering.

Baselines:
    Python's own `sorted`, applied to the model built alongside each container.

Run:
    python -m pytest test/ordering.py -v
"""

import pytest

import smashtable as st

from .base import (
    hash_map_names,
    key_types,
    populate,
    sorted_class_names,
    sorted_map_names,
    sorted_set_names,
)

# region Iteration order


@pytest.mark.parametrize("class_name", sorted_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", [pytest.param("int", id="vint")])
@pytest.mark.parametrize("size", [pytest.param(32, id="n32")])
def test_iteration_matches_sorted(populated):
    """A sorted container walks its keys in exactly Python's own order."""
    container, model = populated
    assert list(container) == sorted(model)


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize(
    ("key_type", "keys"),
    [
        pytest.param("int", [5, -5, 0, -(2**62), 2**62], id="negatives-below-zero"),
        pytest.param("uint", [1, 2**63, 2**63 + 1, 2**64 - 1, 0], id="above-the-sign-bit"),
        pytest.param("str", ["b", "a", "ab", "", "\U0001f600", "Z", "z"], id="beyond-ascii"),
        pytest.param("bytes", [b"ab", b"a", b"", b"\xff", b"b"], id="bytes-prefixes"),
    ],
)
def test_awkward_keys_order_like_python(container, keys):
    """Each layout orders exactly as Python does, where a comparator of the wrong width would not."""
    for key in keys:
        container[key] = "x"
    assert list(container) == sorted(keys)


# endregion Iteration order

# region Scan


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_scan_returns_sorted_pairs(container, keygen):
    """An unbounded scan is every pair, in order."""
    keys = keygen(8)
    model = populate(container, keys, ["v"] * len(keys))
    assert container.scan() == sorted(model.items())


# The half-open window is one contract, exercised through the store and through a participant, so
# the cases live in one table rather than in two that drift apart. Every bound and every expected
# key is spelled in the `int` layout both tests pin.
scan_window_keys = list(range(8))
scan_windows = [
    pytest.param(2, 5, [2, 3, 4], id="half-open"),
    pytest.param(0, 0, [], id="empty-window"),
    pytest.param(5, 2, [], id="inverted"),
    pytest.param(None, 3, [0, 1, 2], id="only-stop"),
    pytest.param(6, None, [6, 7], id="only-start"),
    pytest.param(-100, 100, scan_window_keys, id="outside-both"),
    pytest.param(None, None, scan_window_keys, id="unbounded"),
]


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(("start", "stop", "wanted"), scan_windows)
def test_scan_window_is_half_open(container, start, stop, wanted):
    """start is inclusive and stop exclusive, and bounds need not be present keys."""
    populate(container, scan_window_keys, ["v"] * len(scan_window_keys))
    got = [key for key, _ in container.scan(start, stop)]
    assert got == wanted, f"scan({start}, {stop}) yielded {got}"


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_scan_of_an_empty_container(container):
    """Scanning nothing yields nothing rather than raising."""
    assert container.scan() == []
    assert container.scan(0, 10) == []


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_scan_rejects_a_foreign_bound(container):
    """A bound of the wrong layout is refused, like a key of the wrong layout."""
    with pytest.raises(TypeError):
        container.scan("nope", None)


# endregion Scan


# region Range erase

# The window a slice names, erased through the store and through a participant, so the cases live
# in one table rather than in two that drift apart. Bounds are indices into the eight generated
# keys, since what a key looks like is the layout's business.
slice_windows = [
    pytest.param(slice(2, 5), [0, 1, 5, 6, 7], id="both-bounds"),
    pytest.param(slice(None, 3), [3, 4, 5, 6, 7], id="open-lower"),
    pytest.param(slice(5, None), [0, 1, 2, 3, 4], id="open-upper"),
    pytest.param(slice(None, None), [], id="both-open"),
]

# Both rules a slice is held to, at the store and at a participant alike.
slice_refusals = [
    pytest.param(slice(0, 10, 2), ValueError, id="step"),
    pytest.param(slice("nope", "neither"), TypeError, id="foreign-bound"),
]


def key_bounds(keys, window: slice) -> slice:
    """The index window rewritten over actual keys, an open end left open."""
    return slice(
        None if window.start is None else keys[window.start],
        None if window.stop is None else keys[window.stop],
    )


@pytest.mark.thread_unsafe(
    reason="the window is the test - a parallel copy sharing the container would erase what this one asserts is still there"
)
@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize(("window", "kept"), slice_windows)
def test_deleting_a_slice_erases_the_window(container, keygen, window, kept):
    """`del m[a:b]` removes the half-open window, with either end left out."""
    keys = keygen(8)
    for index, key in enumerate(keys):
        container[key] = index
    del container[key_bounds(keys, window)]
    assert list(container) == [keys[index] for index in kept]


@pytest.mark.thread_unsafe(
    reason="the window is the test - a parallel copy sharing the container would erase what this one scanned"
)
@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_slice_window_matches_scan(container, keygen):
    """What a slice erases is exactly what the same window would have scanned."""
    keys = keygen(10)
    for index, key in enumerate(keys):
        container[key] = index
    doomed = [key for key, _ in container.scan(keys[3], keys[7])]
    del container[keys[3] : keys[7]]
    assert all(key not in container for key in doomed)
    assert len(container) == 10 - len(doomed)


@pytest.mark.parametrize("class_name", sorted_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(("bounds", "failure"), slice_refusals)
def test_a_slice_is_held_to_the_layout(container, bounds, failure):
    """A step names no window over keys, and a bound is a key, held to the layout a key is."""
    with pytest.raises(failure):
        del container[bounds]


@pytest.mark.thread_unsafe(
    reason="the window is the test - a parallel copy sharing the container would erase what this one asserts is still there"
)
@pytest.mark.parametrize("class_name", [pytest.param("SortedSet", id="sortedset")])
@pytest.mark.parametrize("key_type", key_types)
def test_a_set_erases_a_slice_too(container, keygen):
    """A set names a window the same way, its only use for a subscript."""
    keys = keygen(6)
    for key in keys:
        container.add(key)
    del container[keys[1] : keys[4]]
    assert list(container) == [keys[0], keys[4], keys[5]]


# endregion Range erase


# region Inside a transaction


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(("start", "stop", "wanted"), scan_windows)
def test_a_view_scans_the_same_window(container, start, stop, wanted):
    """A participant's window is the store's window, on the same half-open terms."""
    populate(container, scan_window_keys, ["v"] * len(scan_window_keys))
    with st.transaction(container) as (view,):
        got = [key for key, _ in view.scan(start, stop)]
    assert got == wanted, f"view.scan({start}, {stop}) yielded {got}"


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy sharing the container would commit the very write this asserts is invisible"
)
@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_view_scan_sees_its_own_writes(container, keygen):
    """A staged write is in the window before it is anywhere else, which point reads promise too."""
    keys = keygen(4)
    populate(container, keys, ["v"] * len(keys))
    with st.transaction(container) as (view,):
        view[2] = "staged"
        assert dict(view.scan(0, 4))[2] == "staged"
        assert container[2] == "v", "nothing is visible outside until the commit"


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_view_scan_stops_at_its_limit(container, keygen):
    """`limit` caps the list, and a negative one is refused rather than read as uncounted."""
    keys = keygen(8)
    populate(container, keys, ["v"] * len(keys))
    with st.transaction(container) as (view,):
        assert [key for key, _ in view.scan(limit=3)] == [0, 1, 2]
        assert [key for key, _ in view.scan(2, 7, limit=2)] == [2, 3]
        with pytest.raises(ValueError):
            view.scan(limit=-1)


@pytest.mark.parametrize("class_name", sorted_set_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_set_view_scans_members(container, keygen):
    """A set participant yields bare members, exactly as the set's own scan does."""
    keys = keygen(6)
    for key in keys:
        container.add(key)
    with st.transaction(container) as (view,):
        assert view.scan(keys[1], keys[4]) == keys[1:4]


@pytest.mark.thread_unsafe(
    reason="the window is the test - a parallel copy sharing the container would commit its own delete between the stage and the assertion"
)
@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize(("window", "kept"), slice_windows)
def test_a_view_deletes_a_slice(container, keygen, window, kept):
    """`del view[a:b]` stages a tombstone per member, and the store sees none of it until commit."""
    keys = keygen(8)
    for index, key in enumerate(keys):
        container[key] = index
    group = st.transaction(container)
    (view,) = group.begin()
    del view[key_bounds(keys, window)]
    assert [key for key, _ in view.scan()] == [keys[index] for index in kept]
    assert list(container) == keys, "nothing is erased outside until the commit"
    group.stage()
    group.commit()
    assert list(container) == [keys[index] for index in kept]


@pytest.mark.parametrize("class_name", sorted_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(("bounds", "failure"), slice_refusals)
def test_a_view_slice_is_held_to_the_layout(container, bounds, failure):
    """A staged window is held to the same rules the store's own slice is."""
    with st.transaction(container) as (view,):
        with pytest.raises(failure):
            del view[bounds]


@pytest.mark.parametrize("class_name", hash_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_an_unordered_view_has_no_window(container):
    """One view type speaks for every core, so an unordered participant refuses at the call."""
    with st.transaction(container) as (view,):
        with pytest.raises(TypeError):
            view.scan()
        with pytest.raises(TypeError):
            del view[0:10]


# endregion Inside a transaction
