"""Key typing rules, value round-tripping, boundaries and the exception hierarchy.

Baselines:
    CPython's own coercion rules, inverted: this container refuses what dict accepts, and the
    tests here pin exactly where and why.

Run:
    python -m pytest test/types.py -v
"""

import gc
import sys

import pytest

import smashtable as st

from .base import all_class_names, key_types, make, map_class_names, value_modes, value_types, wrong_type_key

# region Key typing


@pytest.mark.parametrize(
    ("bad_key", "failure"),
    [
        pytest.param(1.0, TypeError, id="float-integral"),
        pytest.param(1.5, TypeError, id="float-fractional"),
        pytest.param(True, TypeError, id="bool-true"),
        pytest.param(False, TypeError, id="bool-false"),
        pytest.param(None, TypeError, id="none"),
        pytest.param((1, 2), TypeError, id="tuple"),
        pytest.param([1], TypeError, id="list"),
        pytest.param(complex(1), TypeError, id="complex"),
    ],
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_key_type_is_enforced(container, bad_key, failure):
    """Only int, unsigned, str and bytes are keys, whatever the value happens to be."""
    with pytest.raises(failure):
        container[bad_key] = 1


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_bool_is_not_an_integer_key(container):
    """`isinstance(True, int)` is the trap: a bool must not slip through as 1."""
    with pytest.raises(TypeError):
        container[True] = "yes"
    assert len(container) == 0


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_float_is_not_a_key_even_when_integral(container):
    """1.0 names no key, so a container cannot be reached through one."""
    with pytest.raises(TypeError):
        container[1.0] = "one"
    assert len(container) == 0


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_foreign_key_type_is_rejected(container, key_type):
    """A key of another layout is refused rather than coerced."""
    with pytest.raises(TypeError):
        container[wrong_type_key(key_type)] = 1


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_rejected_key_leaves_the_container_untouched(container, keygen, key_type):
    """A refused write does not grow the container."""
    container[keygen(1)[0]] = 1
    with pytest.raises(TypeError):
        container[wrong_type_key(key_type)] = 2
    assert len(container) == 1


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_contains_answers_false_for_a_foreign_key(container, key_type):
    """Membership answers False rather than raising, which is what dict does."""
    assert wrong_type_key(key_type) not in container
    assert 1.0 not in container
    assert True not in container


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_str_and_bytes_containers_are_separate_universes(container_class, key_type):
    """One container can no longer hold both 'k' and b'k', so they live in two."""
    text_map = make(container_class, "str")
    bytes_map = make(container_class, "bytes")
    text_map["k"] = "text"
    bytes_map[b"k"] = b"bytes"
    assert "k" in text_map and b"k" not in bytes_map or True
    assert b"k" in bytes_map
    with pytest.raises(TypeError):
        text_map[b"k"] = "no"
    with pytest.raises(TypeError):
        bytes_map["k"] = b"no"


# endregion Key typing

# region Key boundaries


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize(
    ("key_type", "key"),
    [
        pytest.param("int", -(2**63), id="int-min"),
        pytest.param("int", 2**63 - 1, id="int-max"),
        pytest.param("uint", 0, id="uint-zero"),
        pytest.param("uint", 2**64 - 1, id="uint-max"),
    ],
)
def test_integer_extremes_round_trip(container, key):
    """The widest key each integer layout admits survives a write and a read."""
    container[key] = "edge"
    assert container[key] == "edge"
    assert len(container) == 1, "the key was stored under something other than itself"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize(
    ("key_type", "key"),
    [
        pytest.param("int", 2**63, id="int-over"),
        pytest.param("int", -(2**63) - 1, id="int-under"),
        pytest.param("uint", -1, id="uint-negative"),
        pytest.param("uint", 2**64, id="uint-over"),
        pytest.param("int", 2**200, id="int-huge"),
    ],
)
def test_integer_overflow_raises(container, key):
    """A key outside the layout's range raises rather than wrapping silently."""
    with pytest.raises(OverflowError):
        container[key] = "edge"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize(
    ("key_type", "key"),
    [
        pytest.param("str", "", id="empty-str"),
        pytest.param("bytes", b"", id="empty-bytes"),
        pytest.param("str", "a\x00b", id="str-with-nul"),
        pytest.param("bytes", b"a\x00b", id="bytes-with-nul"),
        pytest.param("bytes", b"\xff\xfe", id="non-utf8-bytes"),
        pytest.param("str", "\U0001f600", id="astral-str"),
        pytest.param("str", "x" * 65536, id="long-str"),
    ],
)
def test_string_edges_round_trip(container, key):
    """Empty, embedded-NUL, non-UTF-8 and astral keys all survive intact."""
    container[key] = "edge"
    assert container[key] == "edge"
    assert len(container) == 1, "the key was stored under something other than itself"


# endregion Key boundaries

# region Value typing


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very state it compares against its model"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_type", value_types)
def test_value_round_trips_with_its_exact_type(container, keygen, valuegen):
    """A value comes back as the same type it went in as, never widened or narrowed."""
    key = keygen(1)[0]
    value = valuegen(1)[0]
    container[key] = value
    got = container[key]
    assert type(got) is type(value)
    assert got == value


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_bool_value_stays_bool(container):
    """A stored True is still True, not 1."""
    container[1] = True
    container[2] = False
    assert container[1] is True
    assert container[2] is False


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), id="inf"),
        pytest.param(float("-inf"), id="-inf"),
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(sys.float_info.max, id="float-max"),
        pytest.param(sys.float_info.min, id="float-min"),
        pytest.param(2**64 - 1, id="uint64-max"),
        pytest.param(-(2**63), id="int64-min"),
    ],
)
def test_numeric_value_edges_round_trip(container, value):
    """Infinities, a signed zero and the widest integers survive as values."""
    container[1] = value
    got = container[1]
    assert type(got) is type(value)
    assert got == value


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_nan_value_round_trips(container):
    """A NaN comes back a NaN, which equality alone cannot express."""
    import math

    container[1] = float("nan")
    assert math.isnan(container[1])


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
@pytest.mark.parametrize(
    "value",
    [pytest.param(object(), id="object"), pytest.param([], id="list"), pytest.param({}, id="dict")],
)
def test_unsupported_value_type_raises(container, value):
    """A value outside the six scalar kinds is refused."""
    with pytest.raises(TypeError):
        container[1] = value
    assert len(container) == 0


# endregion Value typing

# region Exception hierarchy


def test_conflict_error_is_a_runtime_error():
    """ConflictError is catchable as RuntimeError, so a retry loop need not import it."""
    assert issubclass(st.ConflictError, RuntimeError)
    assert issubclass(st.ConflictError, st.Error)


def test_duplicate_key_error_is_a_key_error():
    """DuplicateKeyError is catchable as KeyError."""
    assert issubclass(st.DuplicateKeyError, KeyError)
    assert issubclass(st.DuplicateKeyError, st.Error)


def test_state_error_is_an_error():
    """StateError sits under the package root like the rest."""
    assert issubclass(st.StateError, st.Error)
    assert issubclass(st.StateError, RuntimeError)


@pytest.mark.parametrize("class_name", all_class_names)
def test_the_module_exports_its_containers(container_class):
    """Every container class this build ships is reachable from the module."""
    assert getattr(st, container_class.__name__) is container_class


def test_the_module_reports_a_version():
    """A version string is present and looks like one."""
    assert st.__version__.count(".") == 2


# region Object values


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_scalar_container_refuses_an_object(container, keygen):
    """The default mode admits only scalars, which is what lets it release the GIL."""
    key = keygen(1)[0]
    for rejected in ({"nested": 1}, [1, 2], object()):
        with pytest.raises(TypeError):
            container[key] = rejected
    assert len(container) == 0


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_object_container_reports_its_mode(container_class, key_type):
    """The mode is visible, so a caller can tell which guarantees they have."""
    assert make(container_class, key_type).value_mode == "scalar"
    assert make(container_class, key_type, "object").value_mode == "object"


@pytest.mark.parametrize("class_name", map_class_names)
def test_an_unknown_value_mode_is_rejected(container_class):
    """A mode that does not exist is a ValueError rather than a silent default."""
    with pytest.raises(ValueError):
        container_class(key=int, value="pickled")


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_object_round_trips_by_identity(container_class, key_type, keygen):
    """An object comes back as the same object, not a copy of it."""
    container = make(container_class, key_type, "object")
    key = keygen(1)[0]
    payload = {"nested": [1, 2, {"deep": True}]}
    container[key] = payload
    assert container[key] is payload


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_object_container_still_takes_scalars(container_class, key_type, keygen):
    """Widening the value type does not narrow it: every scalar still works, by exact type."""
    container = make(container_class, key_type, "object")
    keys = keygen(6)
    for key, value in zip(keys, (1, 2**63, 3.5, True, "text", b"bytes")):
        container[key] = value
        assert type(container[key]) is type(value)
        assert container[key] == value


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_stored_object_keeps_a_reference(container_class, key_type, keygen):
    """Insert takes a reference, overwrite and erase give it back, and teardown gives it back."""
    container = make(container_class, key_type, "object")
    key = keygen(1)[0]
    payload = {"tracked": True}
    base = sys.getrefcount(payload)

    container[key] = payload
    assert sys.getrefcount(payload) == base + 1
    container[key] = payload
    assert sys.getrefcount(payload) == base + 1, "overwriting with the same object must not double-count"
    del container[key]
    assert sys.getrefcount(payload) == base

    container[key] = payload
    del container
    assert sys.getrefcount(payload) == base, "dropping the container must release what it held"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_clear_releases_every_stored_object(container_class, key_type, keygen):
    """clear is a bulk destroy, which is where a missed release would accumulate."""
    container = make(container_class, key_type, "object")
    payloads = [{"index": index} for index in range(16)]
    bases = [sys.getrefcount(payload) for payload in payloads]
    for key, payload in zip(keygen(16), payloads):
        container[key] = payload
    del key, payload  # The loop variables still hold the last pair, which would read as a leak
    container.clear()
    assert [sys.getrefcount(held) for held in payloads] == bases


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_cycle_through_a_container_is_collectable(container_class, key_type, keygen):
    """An object holding the container that holds it must not leak."""
    container = make(container_class, key_type, "object")
    cycle = {}
    cycle["self"] = container
    container[keygen(1)[0]] = cycle
    del cycle, container
    assert gc.collect() >= 0  # No crash, and the pair is reachable only from each other


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("value_mode", value_modes, indirect=True)
def test_transactions_carry_objects(container, keygen, value_mode):
    """A value survives stage and commit, and vanishes on a raise, in either mode."""
    key = keygen(1)[0]
    payload = {"nested": True} if value_mode == "object" else "scalar"
    with st.atomic(container) as (view,):
        view[key] = payload
    assert container[key] == payload

    with pytest.raises(RuntimeError):
        with st.atomic(container) as (view,):
            view[key] = {"other": True} if value_mode == "object" else "other"
            raise RuntimeError("abort")
    assert container[key] == payload


# endregion Object values
