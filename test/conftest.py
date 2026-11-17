"""Fixtures and the session banner. No test functions and no oracle live here.

File: test/conftest.py
Author: Ash Vardanian
Date: August 17, 2026
"""

import collections
import concurrent.futures
import enum
import random
import sys
import threading

import pytest
from base import (
    BACKEND_AXES,
    CONTAINER_CLASS_NAMES,
    FREE_THREADED,
    MAX_CASES_PER_TEST,
    SETTINGS,
    make,
    make_keys,
    make_values,
    populate,
    stream_key,
)

import smashtable as st


def pytest_report_header() -> list[str]:
    """What this run exercises, printed where pytest prints its own header."""
    return [
        f"SmashTable {st.__version__} from {st.__file__}",
        f"- Free-threaded: {str(FREE_THREADED).lower()}",
        f"- Containers: {', '.join(CONTAINER_CLASS_NAMES)}",
        f"- Seed: {SETTINGS.seed}, scale: {SETTINGS.scale:g}",
    ]


# region Shared Pytest
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Refuses a test whose cases outgrow MAX_CASES_PER_TEST."""
    variants: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for item in items:
        callspec = getattr(item, "callspec", None)
        parameters = dict(callspec.params) if callspec is not None else {}
        parameters.pop("__pytest_repeat_step_number", None)
        kept = {key: value for key, value in parameters.items() if key not in BACKEND_AXES}
        variants[(item.module.__name__, getattr(item, "originalname", item.name))].add(repr(kept))
    oversized = [
        f"{module}::{name} has {len(cases)}"
        for (module, name), cases in variants.items()
        if len(cases) > MAX_CASES_PER_TEST
    ]
    if oversized:
        raise pytest.UsageError(f"over MAX_CASES_PER_TEST={MAX_CASES_PER_TEST}: " + "; ".join(oversized))


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    """Names enum parameters as the code spells them, `Backend.LINUX_URING618`, so an id greps to its source."""
    return f"{type(val).__name__}.{val.name}" if isinstance(val, enum.Enum) else None


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """Share inputs across backends while keeping other parameters and repeat steps distinct."""
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    repeat_step = parameters.pop("__pytest_repeat_step_number", 0)
    name = request.node.originalname or request.node.name
    name += repr({key: value for key, value in parameters.items() if key not in BACKEND_AXES})
    return stream_key(SETTINGS.seed, name, repeat_step)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collects what the interpreter reports as unraisable, like a finalizer's exception, for the test to check."""
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# endregion Shared Pytest


@pytest.fixture
def key_type() -> str:
    """The key type of a test that does not sweep it: `int`, as its keys only need to exist."""
    return "int"


@pytest.fixture
def keygen(key_type: str, rng: random.Random):
    """A factory for distinct, ascending keys of the container's key type."""

    def generate(count: int, *, start: int = 0) -> list:
        return make_keys(key_type, count, rng, start=start)

    return generate


@pytest.fixture
def valuegen(value_type: str, rng: random.Random):
    """A factory for values of the swept value type."""

    def generate(count: int) -> list:
        return make_values(value_type, count, rng)

    return generate


@pytest.fixture
def value_mode() -> str:
    """The value mode of a test that does not sweep it: scalars."""
    return "scalar"


@pytest.fixture
def isolation() -> str | None:
    """The level of a test that does not sweep it: None, which leaves the container's default in place."""
    return None


@pytest.fixture
def sharing() -> str | None:
    """The sharing of a test that does not sweep it: None, which leaves the container's default in place."""
    return None


@pytest.fixture
def container(container_class: type, key_type: str, value_mode: str, isolation, sharing):
    """An empty container of the swept class, key type, value mode, isolation and sharing."""
    return make(container_class, key_type, value_mode, isolation, sharing)


@pytest.fixture
def populated(container, keygen, valuegen, count: int):
    """`(container, model)` already holding `count` elements."""
    keys = keygen(count)
    values = valuegen(count)
    model = populate(container, keys, values)
    return container, model


@pytest.fixture
def drive_threads():
    """Runs `worker(index)` on `count` threads, re-raising whatever fired inside one of them."""

    def drive(worker, count: int, timeout: float = 60.0) -> None:
        # Every worker is submitted before any is awaited, or the threads would run one at a time.
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=count)
        try:
            futures = [pool.submit(worker, index) for index in range(count)]
            raised = (future.exception(timeout=timeout) for future in futures)
            failures = [error for error in raised if error is not None]
        finally:
            # Never wait here: a worker that hung is what the timeout above is reporting.
            pool.shutdown(wait=False)
        # A broken barrier is how one worker's failure reaches the others, so it never outranks theirs.
        if failures:
            raise min(failures, key=lambda error: isinstance(error, threading.BrokenBarrierError))

    return drive


@pytest.fixture
def crossing_gate():
    """A barrier holding the writer until every reader is in its loop, so the two provably overlap.

    Expires below `drive_threads`, so a thread that never arrives breaks the barrier and names
    itself rather than expiring the driver's wait on an unrelated future.
    """

    def gate(parties: int, timeout: float = 30.0) -> threading.Barrier:
        return threading.Barrier(parties, timeout=timeout)

    return gate
