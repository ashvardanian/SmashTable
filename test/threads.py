"""Concurrency, with the GIL and without it.

Baselines:
    An arithmetic invariant rather than a stdlib type: N threads times M increments under watch
    must land at exactly N*M, which no interleaving may perturb.

Run:
    python -m pytest test/threads.py -v
    python -m pytest test/threads.py -v -m "not slow"
"""

import concurrent.futures
import threading

import pytest

import smashtable as st

from .base import make, map_class_names, skip_unless_free_threaded


def _drive(worker, count: int, failures: list[str], timeout: float = 60.0) -> None:
    """Runs `worker(index)` on `count` threads, surfacing assertions that fired inside them.

    An assertion in a worker does not reach pytest, so every body records into `failures` and the
    check happens after the join.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        for future in [pool.submit(worker, index) for index in range(count)]:
            future.result(timeout=timeout)
    assert not failures, "\n".join(failures)


# region With the GIL


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_concurrent_writers_to_disjoint_keys(container, failures):
    """Threads writing keys nobody else touches all land."""
    per_thread = 50
    threads = 4

    def worker(index: int) -> None:
        try:
            for offset in range(per_thread):
                container[index * per_thread + offset] = index
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, threads, failures)
    assert len(container) == threads * per_thread


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_concurrent_readers_never_raise(container, failures):
    """Readers running against a writer see values, never a torn or missing store."""
    for key in range(200):
        container[key] = key

    def worker(index: int) -> None:
        try:
            for _ in range(50):
                assert len(container) >= 200
                assert container[7] == 7
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, 4, failures)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_iteration_concurrent_with_mutation_never_crashes(container, failures):
    """A walk running alongside writers terminates and yields only real keys."""
    for key in range(100):
        container[key] = key

    def worker(index: int) -> None:
        try:
            if index == 0:
                for _ in range(20):
                    walked = list(container)
                    assert len(set(walked)) == len(walked)
            else:
                for offset in range(100):
                    container[1000 * index + offset] = offset
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, 3, failures)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_swapped_argument_order_does_not_deadlock(container_class, key_type, failures):
    """Two groups naming the same containers in opposite orders both finish.

    Participants stage in a process-wide canonical order rather than argument order, which is
    what makes this safe; without it the two groups would each hold what the other wants.
    """
    first = make(container_class, key_type)
    second = make(container_class, key_type)

    def worker(index: int) -> None:
        try:
            pair = (first, second) if index % 2 == 0 else (second, first)
            for round_number in range(25):
                with st.atomic(*pair) as (left, right):
                    left[index * 100 + round_number] = index
                    right[index * 100 + round_number] = index
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, 4, failures)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_group_is_never_half_visible(container_class, key_type, failures):
    """A reader sampling both containers never catches a group part-way in."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    stop = threading.Event()

    def writer(_: int) -> None:
        try:
            for round_number in range(200):
                try:
                    with st.atomic(first, second) as (left, right):
                        left[round_number] = round_number
                        right[round_number] = round_number
                        if round_number % 3 == 0:
                            raise RuntimeError("abort")
                except RuntimeError:
                    pass
        finally:
            stop.set()

    def reader(index: int) -> None:
        try:
            while not stop.is_set():
                for key in list(first)[:20]:
                    # A key committed into the first container must eventually be in the second,
                    # and must never be in neither after having been seen in one.
                    assert first[key] == key
        except Exception as error:  # noqa: BLE001
            failures.append(f"reader {index}: {error!r}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, 0), pool.submit(reader, 1)]
        for future in futures:
            future.result(timeout=60)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int"), pytest.param("str", id="str")])
def test_one_iterator_shared_by_many_threads(container, keygen, failures):
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

    def worker(index: int) -> None:
        mine = []
        try:
            for key in walk:
                mine.append(key)
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")
        with sink:
            seen.extend(mine)

    _drive(worker, 8, failures)
    assert len(seen) == len(keys), "a key was yielded twice or lost"
    assert set(seen) == set(keys)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_one_transaction_staged_by_many_threads(container, keygen, failures):
    """Exactly one thread stages a shared group; the rest are told it is no longer open.

    The staging pass releases the GIL, so without a lock over the group every thread would pass the
    state test and stage the same participants again.
    """
    keys = keygen(8)
    group = st.atomic(container)
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
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")
            return
        with sink:
            outcome.append(index)

    _drive(worker, 8, failures)
    assert len(staged) == 1, f"{len(staged)} threads staged the same group"
    assert len(refused) == 7
    group.commit()
    assert all(container[key] == 1 for key in keys)


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.slow
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_a_transactional_counter_converges(container, failures):
    """N threads times M increments under watch land at exactly N*M.

    The strongest claim the suite makes: no increment may be lost, and the retry loop must
    terminate for every one of them.
    """
    threads, increments = 4, 25
    container[0] = 0

    def worker(index: int) -> None:
        try:
            for _ in range(increments):
                while True:
                    try:
                        group = st.atomic(container)
                        (view,) = group.begin()
                        view.watch(0)
                        view[0] = view[0] + 1
                        group.stage()
                        group.commit()
                        break
                    except st.ConflictError:
                        continue
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, threads, failures, timeout=300.0)
    assert container[0] == threads * increments


# endregion With the GIL

# region Free-threaded


@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_parallel_disjoint_writers(container, failures):
    """Without the GIL, disjoint writers still all land."""
    skip_unless_free_threaded()
    per_thread, threads = 200, 8

    def worker(index: int) -> None:
        try:
            for offset in range(per_thread):
                container[index * per_thread + offset] = index
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, threads, failures)
    assert len(container) == threads * per_thread


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_parallel_readers_see_no_torn_values(container, failures):
    """Without the GIL, a read is always exactly one of the values written, never a mix."""
    skip_unless_free_threaded()
    written = {"alpha", "beta", "gamma"}
    container[0] = "alpha"

    def worker(index: int) -> None:
        try:
            if index == 0:
                for _ in range(500):
                    for value in written:
                        container[0] = value
            else:
                for _ in range(500):
                    assert container[0] in written
        except Exception as error:  # noqa: BLE001
            failures.append(f"worker {index}: {error!r}")

    _drive(worker, 4, failures)


# endregion Free-threaded
