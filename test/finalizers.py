"""What happens when a stored object's finalizer touches the container that is freeing it.

Baselines:
    `dict`, which permits it: `d[k] = other` may run the displaced value's `__del__`, and that
    finalizer may read `d`. A container that deadlocks there is not a drop-in.

The deadlock cases run as a subprocess, because the failure mode is a hang and a hang inside the
suite stops the run rather than reporting it. They are ordinary functions rather than source
strings, so a mistake in them is a syntax error at collection rather than a runtime surprise, and
they all run in one subprocess rather than one each.

Run:
    python -m pytest test/finalizers.py -v
    python -m test.finalizers            # the same cases, directly, for debugging a hang
"""

import subprocess
import sys

import pytest

import smashtable as st

from .base import sharing_modes

# region The cases

# Each stores an object whose `__del__` reads the container, then releases it through a different
# write path. The store destroys what it displaces while holding its own non-recursive lock, so the
# release has to happen after that call has returned rather than inside it.


def _watcher(container, seen):
    class Watcher:
        def __del__(self):
            seen.append(len(container))

    return Watcher


def overwrite(m, boom):
    m[1] = boom()
    m[1] = None


def delete_item(m, boom):
    m[1] = boom()
    del m[1]


def clear(m, boom):
    m[1] = boom()
    m.clear()


def delete_slice(m, boom):
    m[1] = boom()
    del m[0:9]


def handle_write(m, boom):
    m[1] = boom()
    with st.atomic(m) as (view,):
        view[1] = None


def handle_delete(m, boom):
    m[1] = boom()
    with st.atomic(m) as (view,):
        del view[1]


def teardown(m, boom):
    m[1] = boom()
    replacement = st.SortedMap(key=int, value="object")
    replacement[1] = 1


RELEASE_PATHS = (overwrite, delete_item, clear, delete_slice, handle_write, handle_delete, teardown)


def exercise_every_path() -> None:
    """Every release path against both sharing strategies. Hangs if a finalizer is run under a lock."""
    for sharing in ("locked", "partitioned"):
        for path in RELEASE_PATHS:
            container = st.SortedMap(key=int, value="object", sharing=sharing)
            seen: list[int] = []
            path(container, _watcher(container, seen))


# endregion The cases

# region Tests


@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="it spawns an interpreter and asserts on its exit")
def test_no_release_path_deadlocks():
    """A `__del__` reaching into its own container must not block against the write that freed it.

    Regression: the store destroyed a displaced value inside its own callback, with a non-recursive
    mutex held, so a finalizer touching that container waited on a lock its own write was holding.
    """
    try:
        finished = subprocess.run(
            [sys.executable, "-m", "test.finalizers"], capture_output=True, text=True, timeout=60, cwd="."
        )
    except subprocess.TimeoutExpired:
        pytest.fail("deadlocked: a finalizer reaching into its own container hung on a store lock")
    assert finished.returncode == 0, f"exited {finished.returncode}: {finished.stderr[-500:]}"


@pytest.mark.parametrize("sharing", sharing_modes, indirect=True)
def test_a_finalizer_sees_the_write_completed(sharing):
    """The finalizer runs after the operation, so it observes the store the write left behind.

    In process, because nothing here can hang: it asserts the ordering, not the absence of a
    deadlock, and if the deadlock returns the test above is the one that reports it.
    """
    container = st.SortedMap(key=int, value="object", sharing=sharing)
    seen: list[int] = []
    watcher = _watcher(container, seen)

    container[1] = watcher()
    container[1] = None
    assert seen == [1], "the overwrite must be complete before its finalizer runs"

    container[2] = watcher()
    del container[2]
    assert seen == [1, 1], "the erase must be complete before its finalizer runs"


# endregion Tests


if __name__ == "__main__":
    exercise_every_path()
