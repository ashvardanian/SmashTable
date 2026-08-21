"""Fixtures and the session banner. No test functions and no oracle live here."""

import concurrent.futures
import os
import platform
import random

import pytest

import smashtable as st

from .base import exported_container_names, free_threaded, make, make_keys, make_values, populate

_RUN_SEED = int(os.environ.get("SMASHTABLE_TESTS_SEED", int.from_bytes(os.urandom(4), "little")))


def pytest_report_header() -> list[str]:
    """What this run exercises, printed where pytest prints its own header."""
    return [
        f"python: {platform.python_version()} (free-threaded: {free_threaded()})",
        f"smashtable: {st.__version__} from {st.__file__}",
        f"containers: {', '.join(exported_container_names())}",
        f"seed: {_RUN_SEED}, pin with SMASHTABLE_TESTS_SEED",
    ]


@pytest.fixture
def seed(__pytest_repeat_step_number) -> int:
    """A per-test seed that moves with the repeat step, so repetitions differ but reproduce.

    The parameter carries no default on purpose: pytest builds a fixture's closure from the
    parameters that have none, so a defaulted one is never injected and every repeat replays the
    first step's draws. `pytest-repeat` hands `None` to a test that is not repeated, which is what
    the `or 0` is for.
    """
    return _RUN_SEED + (__pytest_repeat_step_number or 0)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """An RNG private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def container_class(class_name: str) -> type:
    """The exported class named by the `class_name` axis, or a skip while phase 2 is pending.

    A fixture rather than an axis: the axis is the name, which exists in every build, so the
    matrix and its ids stay stable and a missing class reports as a skip rather than a gap.
    """
    resolved = getattr(st, class_name, None)
    if resolved is None:
        pytest.skip(f"{class_name} is not exported by this build")
    return resolved


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
def value_mode(request) -> str:
    """The container's value mode, defaulting to scalars when a test does not sweep it."""
    return getattr(request, "param", "scalar")


@pytest.fixture
def isolation(request) -> str | None:
    """The level asked for, or None to leave the container's default in place."""
    return getattr(request, "param", None)


@pytest.fixture
def sharing(request) -> str | None:
    """How the store is shared, or None to leave the container's default in place."""
    return getattr(request, "param", None)


@pytest.fixture
def container(container_class: type, key_type: str, value_mode: str, isolation, sharing):
    """An empty container of the swept class, key type, value mode, isolation and sharing."""
    return make(container_class, key_type, value_mode, isolation, sharing)


@pytest.fixture
def populated(container, keygen, valuegen, size: int):
    """`(container, model)` already holding `size` elements."""
    keys = keygen(size)
    values = valuegen(size)
    model = populate(container, keys, values)
    return container, model


@pytest.fixture
def drive_threads():
    """Runs `worker(index)` on `count` threads, re-raising whatever fired inside one of them."""

    def drive(worker, count: int, timeout: float = 60.0) -> None:
        # Every worker is submitted before any is awaited, or the threads would run one at a time.
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            for future in [pool.submit(worker, index) for index in range(count)]:
                future.result(timeout=timeout)

    return drive
