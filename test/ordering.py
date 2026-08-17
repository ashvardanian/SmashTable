"""Iteration order and the scan window, for the classes that promise ordering.

Baselines:
    Python's own `sorted`, applied to the model built alongside each container.

Run:
    python -m pytest test/ordering.py -v
"""

import pytest

from .base import key_types, populate, sorted_class_names, sorted_map_names

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
def test_signed_keys_order_negatives_first(container_class):
    """A signed layout puts negatives below zero, which an unsigned comparison would not."""
    container = container_class(key="int")
    for key in (5, -5, 0, -(2**62), 2**62):
        container[key] = "x"
    assert list(container) == sorted([5, -5, 0, -(2**62), 2**62])


@pytest.mark.parametrize("class_name", sorted_map_names)
def test_unsigned_keys_order_above_the_sign_bit(container_class):
    """Keys past 2**63 must order above small ones; a signed comparator gets this backwards."""
    container = container_class(key="uint")
    keys = [1, 2**63, 2**63 + 1, 2**64 - 1, 0]
    for key in keys:
        container[key] = "x"
    assert list(container) == sorted(keys)


@pytest.mark.parametrize("class_name", sorted_map_names)
def test_text_keys_order_by_code_point(container_class):
    """Text ordering agrees with Python's, including beyond ASCII."""
    container = container_class(key="str")
    keys = ["b", "a", "ab", "", "\U0001f600", "Z", "z"]
    for key in keys:
        container[key] = "x"
    assert list(container) == sorted(keys)


@pytest.mark.parametrize("class_name", sorted_map_names)
def test_bytes_keys_order_lexicographically(container_class):
    """Byte ordering agrees with Python's, prefixes included."""
    container = container_class(key="bytes")
    keys = [b"ab", b"a", b"", b"\xff", b"b"]
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


@pytest.mark.parametrize("class_name", sorted_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(
    ("start", "stop", "wanted"),
    [
        pytest.param(2, 5, [2, 3, 4], id="half-open"),
        pytest.param(0, 0, [], id="empty-window"),
        pytest.param(5, 2, [], id="inverted"),
        pytest.param(None, 3, [0, 1, 2], id="only-stop"),
        pytest.param(6, None, [6, 7], id="only-start"),
        pytest.param(-100, 100, [0, 1, 2, 3, 4, 5, 6, 7], id="outside-both"),
        pytest.param(None, None, [0, 1, 2, 3, 4, 5, 6, 7], id="unbounded"),
    ],
)
def test_scan_window_is_half_open(container, keygen, start, stop, wanted):
    """start is inclusive and stop exclusive, and bounds need not be present keys."""
    keys = keygen(8)
    populate(container, keys, ["v"] * len(keys))
    got = container.scan(start, stop) if (start, stop) != (None, None) else container.scan()
    assert [key for key, _ in got] == wanted


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
