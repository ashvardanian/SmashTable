"""Shared matrices, generators, oracle and gates for the SmashTable suite.

Baselines:
    dict for SortedMap and HashMap, set for SortedSet and HashSet, compared structurally
    rather than by repr, since only the sorted pair has a defined order.

Matches C++ suite:
    the fixture half of test/fixtures.hpp, which builds the same matrices for the layer
    below. This module declares no test of its own; the suites that import it hold those.

Run:
    python -m pytest test/ -v
    SMASHTABLE_SEED=random python -m pytest test/ -v
    SMASHTABLE_FILTER="SortedMap.*str" python -m pytest test/

File: test/base.py
Author: Ash Vardanian
Date: August 17, 2026
"""

import enum
import math
import os
import random
import re
import sys
import sysconfig
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import NoReturn

import pytest

import smashtable as st

# region Shared Settings
_UINT64_MASK = 2**64 - 1


def env_text(variable: str) -> str | None:
    """The stripped text of the environment `variable`, or None when unset or blank."""
    text = os.environ.get(variable, "").strip()
    return text or None


def exit_unparsed(variable: str, text: str, expected: str) -> NoReturn:
    """Stops the session on a setting that does not parse, naming it as the C and C++ suites do."""
    raise SystemExit(f'{variable}="{text}" does not parse, expected {expected}')


def env_seed(variable: str) -> int:
    """An unsigned 64-bit seed: 42 when unset, drawn afresh for `random`."""
    text = env_text(variable)
    if text is None:
        return 42
    if text == "random":
        return random.SystemRandom().getrandbits(64)
    if not text.isdigit() or int(text) > _UINT64_MASK:
        exit_unparsed(variable, text, "an unsigned integer or random")
    return int(text)


def env_real(variable: str, fallback: float) -> float:
    """A positive real, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0 or value == float("inf"):
        exit_unparsed(variable, text, "a positive number like 0.1 or 10")
    return value


def env_count(variable: str, fallback: int) -> int:
    """A whole number, `fallback` when unset."""
    text = env_text(variable)
    if text is None:
        return fallback
    if not text.isdigit():
        exit_unparsed(variable, text, "a whole number")
    return int(text)


def env_flag(variable: str, fallback: bool = False) -> bool:
    """A switch, `fallback` when unset: 0, false, off and no turn it off."""
    text = env_text(variable)
    if text is None:
        return fallback
    return text not in {"0", "false", "off", "no"}


@dataclass(frozen=True)
class Filter:
    """Which node ids a run selects: a regex search when the pattern compiles, else a substring, all when empty."""

    pattern: str
    regex: re.Pattern[str] | None

    def selects(self, nodeid: str) -> bool:
        """Whether `nodeid` passes the filter."""
        if not self.pattern:
            return True
        if self.regex is not None:
            return self.regex.search(nodeid) is not None
        return self.pattern in nodeid


def env_filter(variable: str) -> Filter:
    """The filter the environment `variable` spells, compiled once."""
    pattern = env_text(variable) or ""
    try:
        regex = re.compile(pattern) if pattern else None
    except re.error:
        regex = None
    return Filter(pattern, regex)


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def stream_key(seed: int, name: str, index: int = 0) -> int:
    """The `index`-th stream of the family `name` under `seed`, identical in C, C++, Rust and Mojo."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _UINT64_MASK
    return mix((mix(seed ^ hashed) + index) & _UINT64_MASK)


# endregion Shared Settings


@dataclass(frozen=True)
class Settings:
    """Every `SMASHTABLE_*` variable the Python suite reads, parsed once at import."""

    seed: int
    filter: Filter
    scale: float


SETTINGS = Settings(
    seed=env_seed("SMASHTABLE_SEED"),
    filter=env_filter("SMASHTABLE_FILTER"),
    scale=env_real("SMASHTABLE_SCALE", 1.0),
)

BACKEND_AXES: set[str] = set()
"""The axes the seed and the case cap leave out, so every backend draws the same inputs: none, as there
is no backend axis.
"""

# region Matrices

MAP_CLASSES = [pytest.param(st.SortedMap, id="st.SortedMap"), pytest.param(st.HashMap, id="st.HashMap")]
"""The map classes, swept directly as the `container_class` axis."""
SET_CLASSES = [pytest.param(st.SortedSet, id="st.SortedSet"), pytest.param(st.HashSet, id="st.HashSet")]
ALL_CLASSES = MAP_CLASSES + SET_CLASSES
CONTAINER_CLASS_NAMES = ("SortedMap", "SortedSet", "HashMap", "HashSet")
"""Every container the stub declares, by the name the stub checks read it under."""
SORTED_MAP_CLASSES = [MAP_CLASSES[0]]
SORTED_SET_CLASSES = [SET_CLASSES[0]]
SORTED_CLASSES = [MAP_CLASSES[0], SET_CLASSES[0]]
HASH_MAP_CLASSES = [MAP_CLASSES[1]]
HASH_SET_CLASSES = [SET_CLASSES[1]]

ENUMERABLE_MAP_CLASSES = [MAP_CLASSES[0]]
"""Classes whose contents can be walked, which is what the structural oracle needs: it compares
against `dict` and `set`, and cannot run against a container it cannot enumerate.

A separate axis from `SORTED_*` even though the two name the same classes today. Ordering and
enumerability are different properties - an unordered core that gains a `for_each` becomes
enumerable without becoming ordered - and a test that needs to list a container should say so
rather than borrowing a name that happens to coincide.
"""
ENUMERABLE_SET_CLASSES = [SET_CLASSES[0]]
ENUMERABLE_CLASSES = ENUMERABLE_MAP_CLASSES + ENUMERABLE_SET_CLASSES

KEY_TYPES = [pytest.param(name, id=name) for name in ("int", "uint", "str", "bytes")]
"""Every key type the containers accept. Floats and booleans are deliberately absent and are swept as
rejections in test/key_types.py instead.
"""

VALUE_TYPES = [pytest.param(name, id="v" + name) for name in ("int", "uint", "float", "bool", "str", "bytes")]
"""Every value type: the four key layouts plus the two that may only ever be values."""

VALUE_MODES = [pytest.param("scalar", id="scalar"), pytest.param("object", id="object")]
"""A container either admits only scalars, releasing the GIL around every store call, or admits any
object and holds it. The axis exists because the second is the path where a missed acquisition
corrupts rather than fails.
"""

COUNTS = [pytest.param(0, id="n0"), pytest.param(1, id="n1"), pytest.param(7, id="n7"), pytest.param(64, id="n64")]

GROUP_SIZES = [pytest.param(1, id="g1"), pytest.param(2, id="g2"), pytest.param(5, id="g5")]

TRANSACTION_STYLES = [pytest.param("context", id="with"), pytest.param("explicit", id="phases")]

ISOLATION_LEVELS = [
    pytest.param("monotonic_atomic_view", id="monotonic_atomic_view"),
    pytest.param("snapshot", id="snapshot"),
    pytest.param("serializable", id="serializable"),
    pytest.param("strict_serializable", id="strict_serializable"),
]
"""What a reader is promised, and how the store is shared between threads. Two independent axes: the
level is asked for at construction, while the sharing decides how far up it actually survives.
"""
STAMPED_ISOLATION_LEVELS = ("snapshot", "serializable", "strict_serializable")
"""The levels whose reader answers at a stamp, which is what lets its reads repeat."""
SHARING_MODES = [pytest.param("locked", id="locked"), pytest.param("partitioned", id="partitioned")]

MAX_CASES_PER_TEST = 192
"""The most cases one test function may expand to, today's largest, so a new axis cannot multiply a
matrix past it unnoticed.
"""

# endregion Matrices

# region Gates


FREE_THREADED = (
    sys.version_info >= (3, 13) and bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and not sys._is_gil_enabled()
)
"""Whether this interpreter is a free-threaded build with the GIL actually off."""

requires_free_threading = pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded build with the GIL off")
"""Marks a test that only means something on a free-threaded interpreter with the GIL off."""


def is_sorted_class(container_class: type) -> bool:
    """Whether iteration over this class promises key order."""
    return container_class.__name__.startswith("Sorted")


def ordering_of(container_class: type) -> str:
    """Named ordering of this class: "ordered" carries `popmin`, "unordered" does not."""
    return "ordered" if is_sorted_class(container_class) else "unordered"


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

    A stamp-based container - snapshot upwards - carries its level across partitions,
    because visibility there is a stamp comparison and every partition draws from one order. A
    monotonic one cannot: its reader holds no stamp to answer at, so a walk across partitions can
    catch a commit half-applied and only Read Committed survives above a single key.

    The single place the cap is spelled, so a core that changes what it can carry is one edit here
    rather than a sweep through the suite.
    """
    if isolation in STAMPED_ISOLATION_LEVELS:
        return isolation
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
    in test/key_types.py.
    """
    for base in _ERROR_CATEGORIES:
        if isinstance(error, base):
            return base
    raise AssertionError(f"unclassifiable error {error!r}")


class Verdict(enum.Enum):
    """Whether a call answered with a value or raised."""

    returned = enum.auto()
    raised = enum.auto()


@dataclass(frozen=True)
class Outcome:
    """What one call did: the value it returned, or the stdlib family of the error it raised."""

    verdict: Verdict
    value: object

    def __str__(self) -> str:
        return f"{self.verdict.name} {self.value!r}"


def outcome(call) -> Outcome:
    """What `call` did, with an error reduced to its family - never a traceback in the diff."""
    try:
        return Outcome(Verdict.returned, call())
    except Exception as error:  # noqa: BLE001 - classification is the point
        return Outcome(Verdict.raised, _category(error))


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
    """Compares two returned values, descending into tuples so `popmin` pairs compare exactly."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(same_result(one, other) for one, other in zip(left, right))
    if left is None or right is None:
        return left is right
    return same_scalar(left, right)


class Compare(enum.Enum):
    """How an op's returned value is held against the model's.

    `any_member` is what `popmin` needs: only an ordered class promises which element it takes, so
    the model is advanced to match the container's choice rather than run independently.
    """

    exact = enum.auto()
    any_member = enum.auto()


@dataclass(frozen=True)
class Op:
    """One container call, replayable against a SmashTable container and a stdlib model."""

    name: str
    args: tuple = ()
    compare: Compare = Compare.exact

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
        "popmin": lambda: target.popmin(),
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
        "popmin": lambda: target.popmin(),
        "update": lambda: target.update(op.args[0]),
        "clear": lambda: target.clear(),
    }[op.name]()


def is_enumerable(container) -> bool:
    """Whether this container can list its contents, which decides how it is compared."""
    return hasattr(type(container), "__iter__")


def assert_same_state(container, model) -> None:
    """Everything the container can be asked, asked of the model too.

    A container that cannot be walked is pinned by its length and by probing every model key,
    which is an equivalence rather than a weaker check: equal lengths leave the container no room
    to hold a key the model does not.
    """
    assert len(container) == len(model), f"len {len(container)} vs {len(model)}"
    for key in model:
        assert key in container, f"model key {key!r} is absent from the container"

    if not is_enumerable(container):
        if isinstance(model, dict):
            for key, value in model.items():
                assert same_scalar(container[key], value), f"{key!r}: {container[key]!r} vs {value!r}"
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


def apply_op(container, model, op: Op) -> None:
    """Runs `op` on both sides and asserts they agreed, on the value and on the state."""
    dispatch = run_map if isinstance(model, dict) else run_set
    got = outcome(lambda: dispatch(container, op))

    if op.compare is Compare.any_member:
        if got.verdict is Verdict.raised:
            assert len(model) == 0, f"{op}: raised on a non-empty container"
        else:
            taken = got.value[0] if isinstance(got.value, tuple) else got.value
            assert taken in model, f"{op}: invented element {taken!r}"
            model.pop(taken) if isinstance(model, dict) else model.discard(taken)
    else:
        want = outcome(lambda: dispatch(model, op))
        assert got.verdict is want.verdict, f"{op}: container {got} vs model {want}"
        if got.verdict is Verdict.returned:
            assert same_result(got.value, want.value), f"{op}: returned {got.value!r}, model returned {want.value!r}"
        else:
            assert got.value is want.value, f"{op}: raised {got.value.__name__}, model raised {want.value.__name__}"

    assert_same_state(container, model)


def random_map_ops(rng: random.Random, keys: Sequence, values: Sequence, count: int, *, ordering: str) -> list[Op]:
    """A weighted walk over the mapping surface, biased toward mutation and toward key reuse.

    `ordering` is what `ordering_of` reports: only an ordered store carries `popmin`, so an
    unordered one spends that share on a read instead of dropping the op and shortening the walk.
    """
    popmin_or_contains = (
        (lambda key: Op("popmin", (), compare=Compare.any_member))
        if ordering == "ordered"
        else (lambda key: Op("contains", (key,)))
    )
    weighted = (
        (30, lambda key: Op("setitem", (key, rng.choice(values)))),
        (12, lambda key: Op("delitem", (key,))),
        (10, lambda key: Op("pop", (key, rng.choice(values)))),
        (8, lambda key: Op("setdefault", (key, rng.choice(values)))),
        (6, popmin_or_contains),
        (4, lambda key: Op("update", ({other: rng.choice(values) for other in rng.sample(keys, 3)},))),
        (2, lambda key: Op("clear", ())),
        (14, lambda key: Op("getitem", (key,))),
        (14, lambda key: Op("get", (key, rng.choice(values)))),
    )
    weights, factories = zip(*weighted)
    return [rng.choices(factories, weights)[0](rng.choice(keys)) for _ in range(count)]


def random_set_ops(rng: random.Random, members: Sequence, count: int, *, ordering: str) -> list[Op]:
    """The same walk over the set surface, with the same `ordering` rule for `popmin`."""
    popmin_or_contains = (
        (lambda member: Op("popmin", (), compare=Compare.any_member))
        if ordering == "ordered"
        else (lambda member: Op("contains", (member,)))
    )
    weighted = (
        (40, lambda member: Op("add", (member,))),
        (15, lambda member: Op("discard", (member,))),
        (7, lambda member: Op("remove", (member,))),
        (6, popmin_or_contains),
        (4, lambda member: Op("update", (rng.sample(members, 3),))),
        (2, lambda member: Op("clear", ())),
        (26, lambda member: Op("contains", (member,))),
    )
    weights, factories = zip(*weighted)
    return [rng.choices(factories, weights)[0](rng.choice(members)) for _ in range(count)]


def replay(container, model, ops: Iterable[Op]) -> None:
    """Applies every op to both sides, so a failure names the op that diverged."""
    for index, op in enumerate(ops):
        try:
            apply_op(container, model, op)
        except AssertionError as error:
            raise AssertionError(f"diverged at op {index}: {op}\n{error}") from None


# endregion Oracle


# region Groups


@dataclass(frozen=True)
class Participant:
    """One container in a group, beside the stdlib model standing in for it."""

    container: object
    model: object
    keys: Sequence
    values: Sequence

    @property
    def is_map(self) -> bool:
        return isinstance(self.model, dict)


def make_participants(rng: random.Random, specs: Sequence[tuple], count_each: int = 12) -> list[Participant]:
    """Builds one participant per spec, each a `(container_class, key_type, value_mode, isolation, sharing)`.

    Mixing the specs is the point: a group spanning a sorted map and a hashed set, or a scalar
    container and an object one, exercises the group-wide passes that pick the strictest mode and
    the canonical order, which a group of identical containers cannot.
    """
    participants = []
    for container_class, key_type, value_mode, isolation, sharing in specs:
        container = make(container_class, key_type, value_mode, isolation, sharing)
        keys = make_keys(key_type, count_each, rng)
        values = [rng.randrange(1000) for _ in range(count_each)]
        model = {} if is_map_class(container_class) else set()
        participants.append(Participant(container, model, keys, values))
    return participants


def stage_group_writes(views, participants: Sequence[Participant], shadows, rng: random.Random) -> None:
    """Writes a random batch through every view, advancing the shadow model in step."""
    for view, participant, shadow in zip(views, participants, shadows):
        for _ in range(rng.randint(1, 4)):
            key = rng.choice(participant.keys)
            if participant.is_map:
                value = rng.choice(participant.values)
                view[key] = value
                shadow[key] = value
            else:
                view.add(key)
                shadow.add(key)


def assert_group_state(participants: Sequence[Participant]) -> None:
    """Every participant agrees with its own model, which is what all-or-nothing means here."""
    for index, participant in enumerate(participants):
        try:
            assert_same_state(participant.container, participant.model)
        except AssertionError as error:
            raise AssertionError(f"participant {index} diverged\n{error}") from None


def shadows_of(participants: Sequence[Participant]) -> list[dict | set]:
    """A copy of every model, to be adopted only if the group publishes."""
    return [dict(one.model) if one.is_map else set(one.model) for one in participants]


def adopt_shadows(participants: Sequence[Participant], shadows) -> list[Participant]:
    """Replaces each participant's model with the shadow the round advanced."""
    return [replace(one, model=shadow) for one, shadow in zip(participants, shadows)]


# endregion Groups
