"""Shared matrices, generators, oracle and gates for the SmashTable suite.

Baselines:
    dict for SortedMap and HashMap, set for SortedSet and HashSet, compared structurally
    rather than by repr, since only the sorted pair has a defined order.

Matches C++ suite:
    scripts/test_basic.hpp and scripts/test_consistency.hpp, which check the same invariants
    one layer down.

Run:
    python -m pytest test/ -v
    SMASHTABLE_TESTS_SEED=42 python -m pytest test/ -v
    python -m pytest test/ -k "sortedmap and str"
"""

import dataclasses
import math
import random
import sys
import sysconfig
from collections.abc import Iterable, Sequence

import pytest

import smashtable as st

# region Matrices

# Container classes named as strings and resolved by a fixture, so a class this build does not
# export yet skips loudly under its own id instead of vanishing from the matrix.
map_class_names = [
    pytest.param("SortedMap", id="sortedmap"),
    pytest.param("HashMap", id="hashmap"),  # phase 2
]
set_class_names = [
    pytest.param("SortedSet", id="sortedset"),
    pytest.param("HashSet", id="hashset"),  # phase 2
]
all_class_names = map_class_names + set_class_names
sorted_map_names = [map_class_names[0]]
sorted_set_names = [set_class_names[0]]
sorted_class_names = [map_class_names[0], set_class_names[0]]

# Classes whose contents can be walked, which is what the structural oracle needs: it compares
# against `dict` and `set`, and cannot run against a container it cannot enumerate.
#
# A separate axis from `sorted_*` even though the two name the same classes today. Ordering and
# enumerability are different properties - an unordered core that gains a `for_each` becomes
# enumerable without becoming ordered - and a test that needs to list a container should say so
# rather than borrowing a name that happens to coincide.
enumerable_map_names = [map_class_names[0]]
enumerable_set_names = [set_class_names[0]]
enumerable_class_names = enumerable_map_names + enumerable_set_names

# Every key type the containers accept. Floats and booleans are deliberately absent and are swept
# as rejections in test/types.py instead.
key_types = [pytest.param(name, id=name) for name in ("int", "uint", "str", "bytes")]

# Every value type: the four key layouts plus the two that may only ever be values.
value_types = [pytest.param(name, id="v" + name) for name in ("int", "uint", "float", "bool", "str", "bytes")]

# A container either admits only scalars, releasing the GIL around every store call, or admits any
# object and holds it. The axis exists because the second is the path where a missed acquisition
# corrupts rather than fails.
value_modes = [pytest.param("scalar", id="scalar"), pytest.param("object", id="object")]

sizes = [pytest.param(0, id="n0"), pytest.param(1, id="n1"), pytest.param(7, id="n7"), pytest.param(64, id="n64")]

group_sizes = [pytest.param(1, id="g1"), pytest.param(2, id="g2"), pytest.param(5, id="g5")]

transaction_styles = [pytest.param("context", id="with"), pytest.param("explicit", id="phases")]

# What a reader is promised, and how the store is shared between threads. Two independent axes: the
# level is asked for at construction, while the sharing decides how far up it actually survives.
isolation_levels = [pytest.param("monotonic", id="monotonic"), pytest.param("snapshot", id="snapshot")]
sharing_modes = [pytest.param("locked", id="locked"), pytest.param("partitioned", id="partitioned")]

# endregion Matrices

# region Gates


def free_threaded() -> bool:
    """Whether this interpreter is a free-threaded build with the GIL actually off."""
    if sys.version_info < (3, 13):
        return False
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    return not sys._is_gil_enabled()


def skip_unless_free_threaded() -> None:
    """Skips a test that only means something without the GIL."""
    if sys.version_info < (3, 13):
        pytest.skip("free-threading requires Python 3.13t or newer")
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded build")
    if sys._is_gil_enabled():
        pytest.skip("the GIL is enabled in this process")


def exported_container_names() -> list[str]:
    """Which container classes this build actually ships."""
    return [name for name in ("SortedMap", "SortedSet", "HashMap", "HashSet") if hasattr(st, name)]


def is_sorted_class(container_class: type) -> bool:
    """Whether iteration over this class promises key order."""
    return container_class.__name__.startswith("Sorted")


def is_map_class(container_class: type) -> bool:
    """Whether this class stores values as well as keys."""
    return container_class.__name__.endswith("Map")


# endregion Gates

# region Generators

_KEY_FACTORIES = {
    "int": lambda count, start, rng: [start + index for index in range(count)],
    "uint": lambda count, start, rng: [start + index for index in range(count)],
    "str": lambda count, start, rng: [f"key{start + index:08d}" for index in range(count)],
    "bytes": lambda count, start, rng: [b"key%08d" % (start + index) for index in range(count)],
}

_VALUE_FACTORIES = {
    "int": lambda count, rng: [rng.randint(-(2**40), 2**40) for _ in range(count)],
    "uint": lambda count, rng: [rng.randint(2**63, 2**64 - 1) for _ in range(count)],
    "float": lambda count, rng: [rng.uniform(-1e6, 1e6) for _ in range(count)],
    "bool": lambda count, rng: [rng.random() < 0.5 for _ in range(count)],
    "str": lambda count, rng: [f"value{rng.randint(0, 10**9)}" for _ in range(count)],
    "bytes": lambda count, rng: [b"value%d" % rng.randint(0, 10**9) for _ in range(count)],
    # Only storable by an object-mode container, and deliberately nested so the round trip has to
    # preserve the object rather than a flattening of it.
    "object": lambda count, rng: [{"nested": [rng.randint(0, 999), {"deep": True}]} for _ in range(count)],
}


def make_keys(key_type: str, count: int, rng: random.Random, *, start: int = 0) -> list:
    """`count` distinct keys of `key_type`, ascending so ordering assertions are meaningful."""
    return _KEY_FACTORIES[key_type](count, start, rng)


def make_values(value_type: str, count: int, rng: random.Random) -> list:
    """`count` values of `value_type`."""
    return _VALUE_FACTORIES[value_type](count, rng)


def wrong_type_key(key_type: str):
    """A key of a layout this container refuses, for the cross-type rejection tests."""
    return "not-an-int" if key_type in ("int", "uint") else 12345


# endregion Generators

# region Builders


def make(
    container_class: type,
    key_type: str,
    value_mode: str = "scalar",
    isolation: str | None = None,
    sharing: str | None = None,
):
    """The single place the suite spells container construction.

    An axis left as None is not passed at all, so the container's own default applies and the
    default is exercised by every test that does not sweep it.
    """
    arguments = {"key": key_type}
    if is_map_class(container_class) and value_mode != "scalar":
        arguments["value"] = value_mode
    if isolation is not None:
        arguments["isolation"] = isolation
    if sharing is not None:
        arguments["sharing"] = sharing
    return container_class(**arguments)


def effective_isolation(isolation: str, sharing: str) -> str:
    """What a container actually promises, which is not always what was asked for.

    A snapshot container carries its level across partitions, because visibility there is a stamp
    comparison and every partition draws from one clock. A monotonic one cannot: its reader holds no
    stamp to answer at, so a walk across partitions can catch a commit half-applied and only Read
    Committed survives above a single key.

    The single place the cap is spelled, so a core that changes what it can carry is one edit here
    rather than a sweep through the suite.
    """
    if isolation == "snapshot":
        return "snapshot"
    return "read_committed" if sharing == "partitioned" else "monotonic_atomic_view"


def populate(container, keys: Sequence, values: Sequence):
    """Fills a container and returns the stdlib model built the same way."""
    if is_map_class(type(container)):
        model = {}
        for key, value in zip(keys, values):
            container[key] = value
            model[key] = value
        return model
    model = set()
    for key in keys:
        container.add(key)
        model.add(key)
    return model


# endregion Builders

# region Oracle

_ERROR_CATEGORIES = (KeyError, TypeError, OverflowError, ValueError, StopIteration, RuntimeError)


def _category(error: BaseException) -> type:
    """The stdlib base a SmashTable error may be compared as.

    `DuplicateKeyError` is a `KeyError` and `ConflictError` is a `RuntimeError`, so the oracle
    compares the family rather than the exact class; the exact classes are pinned once, by name,
    in test/types.py.
    """
    for base in _ERROR_CATEGORIES:
        if isinstance(error, base):
            return base
    raise AssertionError(f"unclassifiable error {error!r}")


def outcome(call):
    """`('ok', value)` or `('raised', KeyError)` - never a traceback escaping into the diff."""
    try:
        return ("ok", call())
    except Exception as error:  # noqa: BLE001 - classification is the point
        return ("raised", _category(error))


def same_scalar(left, right) -> bool:
    """Equal and the same type - `True` must not pass for `1`, nor `1` for `1.0`.

    A round-tripped NaN is a correct round trip, so NaN compares equal to itself here, and the
    sign of a zero is compared because `-0.0 == 0.0` would otherwise hide a lost sign bit.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        if math.isnan(left) and math.isnan(right):
            return True
        if left == 0.0 and right == 0.0:
            return math.copysign(1.0, left) == math.copysign(1.0, right)
    return left == right


def same_result(left, right) -> bool:
    """Compares two returned values, descending into tuples so `popitem` pairs compare exactly."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(same_result(a, b) for a, b in zip(left, right))
    if left is None or right is None:
        return left is right
    return same_scalar(left, right)


@dataclasses.dataclass(frozen=True)
class Op:
    """One container call, replayable against a SmashTable container and a stdlib model."""

    name: str
    args: tuple = ()
    compare: str = "exact"  # "exact" | "member" | "ignore"

    def __str__(self) -> str:
        return f"{self.name}{self.args!r}"


def run_map(target, op: Op):
    """Dispatches one op onto anything with the mapping surface - container or dict."""
    key = op.args[0] if op.args else None
    return {
        "setitem": lambda: target.__setitem__(*op.args),
        "getitem": lambda: target[key],
        "delitem": lambda: target.__delitem__(key),
        "contains": lambda: key in target,
        "len": lambda: len(target),
        "get": lambda: target.get(*op.args),
        "pop": lambda: target.pop(*op.args),
        "popitem": lambda: target.popitem(),
        "setdefault": lambda: target.setdefault(*op.args),
        "update": lambda: target.update(op.args[0]),
        "clear": lambda: target.clear(),
    }[op.name]()


def run_set(target, op: Op):
    """Dispatches one op onto anything with the set surface - container or set."""
    member = op.args[0] if op.args else None
    return {
        "add": lambda: target.add(member),
        "discard": lambda: target.discard(member),
        "remove": lambda: target.remove(member),
        "contains": lambda: member in target,
        "len": lambda: len(target),
        "pop": lambda: target.pop(),
        "update": lambda: target.update(op.args[0]),
        "clear": lambda: target.clear(),
    }[op.name]()


def is_enumerable(container) -> bool:
    """Whether this container can list its contents, which decides how it is compared."""
    return hasattr(type(container), "__iter__")


def assert_same_state(container, model) -> None:
    """Everything the container can be asked, asked of the model too.

    Iteration is walked on every call rather than in a handful of dedicated tests, so the lazy
    cursor is exercised thousands of times across a run.

    A container that cannot enumerate itself is compared by length and by probing every key the
    model holds. That is an equivalence rather than a weaker check: matching lengths mean the
    container holds no key the model does not, so proving every model key present with the right
    value proves the two agree.
    """
    assert len(container) == len(model), f"len {len(container)} vs {len(model)}"

    if not is_enumerable(container):
        for key in model:
            assert key in container, f"model key {key!r} is absent from the container"
            if isinstance(model, dict):
                assert same_scalar(container[key], model[key]), f"{key!r}: {container[key]!r} vs {model[key]!r}"
        return

    walked = list(container)
    assert len(walked) == len(model), f"iteration visited {len(walked)} keys, model holds {len(model)}"
    assert len(set(map(repr, walked))) == len(walked), f"iteration repeated a key: {walked}"

    if isinstance(model, dict):
        pairs = list(container.items())
        assert [key for key, _ in pairs] == walked, "items() and __iter__ disagree"
        assert list(container.keys()) == walked, "keys() and __iter__ disagree"
        assert [value for _, value in pairs] == list(container.values()), "values() and items() disagree"
        for key, value in pairs:
            assert key in model, f"phantom key {key!r}"
            assert same_scalar(value, model[key]), f"{key!r}: {value!r} vs {model[key]!r}"
    assert sorted(map(repr, walked)) == sorted(map(repr, model)), "contents differ"

    if is_sorted_class(type(container)):
        assert walked == sorted(model), "a sorted container iterated out of order"

    for key in model:
        assert key in container, f"model key {key!r} missing from the container"


def apply_op(container, model, op: Op) -> None:
    """Runs `op` on both sides and asserts they agreed, on the value and on the state."""
    dispatch = run_map if isinstance(model, dict) else run_set
    got = outcome(lambda: dispatch(container, op))

    if op.compare == "member":
        # `popitem`/`pop` remove *some* element; only the sorted classes promise which one, so the
        # model is advanced to match the container's choice rather than run independently.
        assert got[0] == "ok" or len(model) == 0, f"{op}: {got}"
        if got[0] == "ok":
            taken = got[1][0] if isinstance(got[1], tuple) else got[1]
            assert taken in model, f"{op}: invented element {taken!r}"
            model.pop(taken) if isinstance(model, dict) else model.discard(taken)
        else:
            assert len(model) == 0, f"{op}: raised on a non-empty container"
    else:
        want = outcome(lambda: dispatch(model, op))
        assert got[0] == want[0], f"{op}: container {got} vs model {want}"
        if got[0] == "ok" and op.compare == "exact":
            assert same_result(got[1], want[1]), f"{op}: returned {got[1]!r}, model returned {want[1]!r}"

    assert_same_state(container, model)


def random_map_ops(rng: random.Random, keys: Sequence, values: Sequence, count: int) -> list[Op]:
    """A weighted walk over the mapping surface, biased toward mutation and toward key reuse."""
    ops: list[Op] = []
    for _ in range(count):
        key = rng.choice(keys)
        roll = rng.random()
        if roll < 0.30:
            ops.append(Op("setitem", (key, rng.choice(values))))
        elif roll < 0.42:
            ops.append(Op("delitem", (key,)))
        elif roll < 0.52:
            ops.append(Op("pop", (key, rng.choice(values))))
        elif roll < 0.60:
            ops.append(Op("setdefault", (key, rng.choice(values))))
        elif roll < 0.66:
            ops.append(Op("popitem", (), compare="member"))
        elif roll < 0.70:
            ops.append(Op("update", ({other: rng.choice(values) for other in rng.sample(list(keys), 3)},)))
        elif roll < 0.72:
            ops.append(Op("clear", ()))
        elif roll < 0.86:
            ops.append(Op("getitem", (key,)))
        else:
            ops.append(Op("get", (key, rng.choice(values))))
    return ops


def random_set_ops(rng: random.Random, members: Sequence, count: int) -> list[Op]:
    """The same walk over the set surface."""
    ops: list[Op] = []
    for _ in range(count):
        member = rng.choice(members)
        roll = rng.random()
        if roll < 0.40:
            ops.append(Op("add", (member,)))
        elif roll < 0.55:
            ops.append(Op("discard", (member,)))
        elif roll < 0.62:
            ops.append(Op("remove", (member,)))
        elif roll < 0.68:
            ops.append(Op("pop", (), compare="member"))
        elif roll < 0.72:
            ops.append(Op("update", (rng.sample(list(members), 3),)))
        elif roll < 0.74:
            ops.append(Op("clear", ()))
        else:
            ops.append(Op("contains", (member,)))
    return ops


def replay(container, model, ops: Iterable[Op]) -> None:
    """Applies every op to both sides, so a failure names the op that diverged."""
    for index, op in enumerate(ops):
        try:
            apply_op(container, model, op)
        except AssertionError as error:
            raise AssertionError(f"diverged at op {index}: {op}\n{error}") from None


# endregion Oracle
