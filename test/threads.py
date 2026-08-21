"""Concurrency, with the GIL and without it.

Baselines:
    An arithmetic invariant rather than a stdlib type: N threads times M increments under watch
    must land at exactly N*M, which no interleaving may perturb.

Run:
    python -m pytest test/threads.py -v
    python -m pytest test/threads.py -v -m "not slow"
"""

import contextlib
import threading

import pytest

import smashtable as st

from .base import (
    enumerable_map_names,
    make,
    map_class_names,
    skip_unless_free_threaded,
)

# region With the GIL


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_concurrent_writers_to_disjoint_keys(container, drive_threads):
    """Threads writing keys nobody else touches all land."""
    per_thread = 50
    threads = 4

    def worker(index: int) -> None:
        for offset in range(per_thread):
            container[index * per_thread + offset] = index

    drive_threads(worker, threads)
    assert len(container) == threads * per_thread


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_concurrent_readers_never_raise(container, drive_threads):
    """Threads reading one container between them never raise and never miss a seeded key."""
    for key in range(200):
        container[key] = key

    def worker(_: int) -> None:
        for _ in range(50):
            assert len(container) >= 200
            assert container[7] == 7

    drive_threads(worker, 4)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_iteration_concurrent_with_mutation_never_crashes(container, drive_threads):
    """A walk running alongside writers terminates and never yields one key twice."""
    seeded, per_writer, writers, walks = 100, 100, 2, 20
    for key in range(seeded):
        container[key] = key

    def worker(index: int) -> None:
        if index == 0:
            for _ in range(walks):
                walked = list(container)
                assert len(set(walked)) == len(walked), f"{len(walked)} yields, {len(set(walked))} distinct"
        else:
            # Writer blocks start above the seeded keys and never reach each other's.
            for offset in range(per_writer):
                container[seeded + (index - 1) * per_writer + offset] = offset

    drive_threads(worker, 1 + writers)
    wanted = seeded + writers * per_writer
    assert len(container) == wanted, f"len {len(container)} vs {wanted}"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_swapped_argument_order_does_not_deadlock(container_class, key_type, drive_threads):
    """Two groups naming the same containers in opposite orders both finish.

    Participants stage in a process-wide canonical order rather than argument order, which is
    what makes this safe; without it the two groups would each hold what the other wants.
    """
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    threads, rounds = 4, 25

    def worker(index: int) -> None:
        pair = (first, second) if index % 2 == 0 else (second, first)
        # The stride is the bound, so no two workers can name one key.
        for round_number in range(rounds):
            with st.transaction(*pair) as (left, right):
                left[index * rounds + round_number] = index
                right[index * rounds + round_number] = index

    drive_threads(worker, threads)
    assert len(first) == threads * rounds, f"len {len(first)} vs {threads * rounds}"
    assert len(second) == threads * rounds, f"len {len(second)} vs {threads * rounds}"


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_group_is_never_half_visible(container_class, key_type, drive_threads):
    """A reader sampling both containers never catches a group part-way in."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    rounds, sampled = 200, 20
    stop = threading.Event()

    def worker(index: int) -> None:
        # Index 0 writes the groups, aborting every third; every other thread samples until it stops.
        if index != 0:
            while not stop.is_set():
                # Nothing here ever removes a key, so a key seen in one container must already be
                # in the other: the group published both or neither.
                for key in list(first)[:sampled]:
                    assert key in second, f"key {key} landed in first and not in second"
                    assert first[key] == second[key] == key, f"{key}: {first[key]!r} and {second[key]!r}"
                for key in list(second)[:sampled]:
                    assert key in first, f"key {key} landed in second and not in first"
                    assert first[key] == second[key] == key, f"{key}: {first[key]!r} and {second[key]!r}"
            return
        try:
            for round_number in range(rounds):
                with contextlib.suppress(RuntimeError), st.transaction(first, second) as (left, right):
                    left[round_number] = round_number
                    right[round_number] = round_number
                    if round_number % 3 == 0:
                        raise RuntimeError("abort")
        finally:
            stop.set()

    drive_threads(worker, 2)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int"), pytest.param("str", id="str")])
def test_one_iterator_shared_by_many_threads(container, keygen, drive_threads):
    """Threads pulling from one iterator between them see every key exactly once.

    A step releases the GIL, so without a lock over the cursor two threads read one position, both
    step from it and both assign it back - a duplicate for an integer layout, and a torn string for
    a text one.
    """
    keys = keygen(500)
    for key in keys:
        container[key] = 1
    walk = iter(container)
    seen = []
    sink = threading.Lock()

    def worker(_: int) -> None:
        # Pulled first and merged after, so the lock never serializes the walk it is testing.
        mine = []
        for key in walk:
            mine.append(key)
        with sink:
            seen.extend(mine)

    drive_threads(worker, 8)
    assert len(seen) == len(keys), "a key was yielded twice or lost"
    assert set(seen) == set(keys)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_one_transaction_staged_by_many_threads(container, keygen, drive_threads):
    """Exactly one thread stages a shared group; the rest are told it is no longer open.

    The staging pass releases the GIL, so without a lock over the group every thread would pass the
    state test and stage the same participants again.
    """
    threads = 8
    keys = keygen(8)
    group = st.transaction(container)
    (view,) = group.begin()
    for key in keys:
        view[key] = 1
    staged = []
    refused = []
    sink = threading.Lock()

    def worker(index: int) -> None:
        try:
            group.stage()
            outcome = staged
        except st.StateError:
            outcome = refused
        with sink:
            outcome.append(index)

    drive_threads(worker, threads)
    assert len(staged) == 1, f"{len(staged)} threads staged the same group"
    assert len(refused) == threads - 1
    group.commit()
    assert all(container[key] == 1 for key in keys)


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.slow
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_transactional_counter_converges(container, drive_threads):
    """N threads times M increments under watch land at exactly N*M.

    The strongest claim the suite makes: no increment may be lost, and the retry loop must
    terminate for every one of them.
    """
    threads, increments = 4, 25
    container[0] = 0

    def worker(_: int) -> None:
        landed = 0
        while landed < increments:
            with contextlib.suppress(st.ConflictError):
                group = st.transaction(container)
                (view,) = group.begin()
                view.watch(0)
                view[0] = view[0] + 1
                group.stage()
                group.commit()
                landed += 1

    drive_threads(worker, threads, timeout=300.0)
    assert container[0] == threads * increments


# endregion With the GIL

# region Free-threaded


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_parallel_disjoint_writers(container, drive_threads):
    """Without the GIL, disjoint writers still all land."""
    skip_unless_free_threaded()
    per_thread, threads = 200, 8

    def worker(index: int) -> None:
        for offset in range(per_thread):
            container[index * per_thread + offset] = index

    drive_threads(worker, threads)
    assert len(container) == threads * per_thread


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_parallel_readers_see_no_torn_values(container, drive_threads):
    """Without the GIL, a read is always exactly one of the values written, never a mix."""
    skip_unless_free_threaded()
    written = {"alpha", "beta", "gamma"}
    container[0] = "alpha"

    def worker(index: int) -> None:
        if index == 0:
            for _ in range(500):
                for value in written:
                    container[0] = value
        else:
            for _ in range(500):
                assert container[0] in written

    drive_threads(worker, 4)


# endregion Free-threaded
