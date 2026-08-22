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
    seeded = 200
    for key in range(seeded):
        container[key] = key

    def worker(_: int) -> None:
        for _ in range(50):
            assert len(container) == seeded, f"len {len(container)} vs {seeded}"
            assert container[7] == 7

    drive_threads(worker, 4)


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_iteration_concurrent_with_mutation_never_crashes(container_class, key_type, drive_threads, crossing_gate):
    """A walk running alongside writers terminates and never yields one key twice.

    The container is private so a parallel copy of the body cannot write into it, the walker's span
    contains every writer's, and whether any one walk met a write is the scheduler's to decide, so
    only distinctness is asserted of those.
    """
    container = make(container_class, key_type)
    seeded, per_writer, writers, walks = 100, 100, 2, 20
    for key in range(seeded):
        container[key] = key
    stop = threading.Event()
    gate = crossing_gate(1 + writers)
    finished = []
    tally = {"walks": 0}

    def walk_once() -> list:
        walked = list(container)
        assert len(set(walked)) == len(walked), f"{len(walked)} yields, {len(set(walked))} distinct"
        return walked

    def walk_across_writes() -> None:
        before = walk_once()
        gate.wait()
        assert len(before) == seeded, f"{len(before)} keys walked before the writers were released"
        while tally["walks"] < walks or not stop.is_set():
            tally["walks"] += 1
            walk_once()

    def write_one_block(index: int) -> None:
        gate.wait()
        try:
            # Writer blocks start above the seeded keys and never reach each other's.
            for offset in range(per_writer):
                container[seeded + (index - 1) * per_writer + offset] = offset
        finally:
            finished.append(index)
            if len(finished) == writers:
                stop.set()

    def worker(index: int) -> None:
        if index == 0:
            walk_across_writes()
        else:
            write_one_block(index)

    drive_threads(worker, 1 + writers)
    wanted = seeded + writers * per_writer
    # Every writer has finished, so this walk spans all of their keys.
    after = walk_once()
    assert len(after) == wanted, f"{len(after)} keys walked after the writers"
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
def test_only_the_group_in_flight_is_ever_half_visible(container_class, key_type, drive_threads, crossing_gate):
    """A settled group is in both containers, an aborted one in neither, and the skew never reverses.

    Publication follows creation order rather than argument order, so the group names the two
    reversed and the container made second may never be found ahead of the one made first. The
    sampler meets the writer at each round and stays in it until the round publishes, so it reads
    across every commit rather than wherever the scheduler happened to drop it.
    """
    earlier = make(container_class, key_type)
    later = make(container_class, key_type)
    rounds, batch = 200, 100
    gate = crossing_gate(2)
    # The round the writer is about to publish, and the last one it finished.
    entered = [0]
    finished = [-1]
    tally = {"later_ahead": 0, "samples": 0}

    def sample() -> None:
        tally["samples"] += 1
        round_number = entered[0]
        # `later` is read first, so finding it there and not in `earlier` really is a lead.
        leading = round_number * batch
        tally["later_ahead"] += leading in later and leading not in earlier
        if round_number == 0:
            return
        settled = (round_number - 1) * batch
        if (round_number - 1) % 3:
            assert settled in earlier and settled in later, f"round {round_number - 1} in one container only"
        else:
            assert settled not in earlier and settled not in later, f"aborted round {round_number - 1} landed"

    def publish_one_round(round_number: int) -> None:
        base = round_number * batch
        with contextlib.suppress(RuntimeError), st.transaction(later, earlier) as (into_later, into_earlier):
            # A batch rather than one key, so publishing the first container is not instant.
            for offset in range(batch):
                into_later[base + offset] = base + offset
                into_earlier[base + offset] = base + offset
            # Every third round aborts, so the sampler must find it in neither container.
            if round_number % 3 == 0:
                raise RuntimeError("abort")

    def publish_every_round() -> None:
        for round_number in range(rounds):
            entered[0] = round_number
            gate.wait()
            try:
                publish_one_round(round_number)
            finally:
                # A round the sampler is still spinning on must be marked however it ended.
                finished[0] = round_number

    def sample_every_round() -> None:
        for round_number in range(rounds):
            gate.wait()
            sample()
            while finished[0] < round_number:
                sample()

    def worker(index: int) -> None:
        try:
            if index == 0:
                publish_every_round()
            else:
                sample_every_round()
        except BaseException:
            gate.abort()
            raise

    drive_threads(worker, 2)
    assert not tally["later_ahead"], f"the container staged second led {tally['later_ahead']} of {tally['samples']}"
    committed = {
        key
        for round_number in range(rounds)
        if round_number % 3
        for key in range(round_number * batch, round_number * batch + batch)
    }
    assert set(earlier) == committed, f"{len(earlier)} keys in the container made first, {len(committed)} committed"
    assert set(later) == committed, f"{len(later)} keys in the container made second, {len(committed)} committed"


@pytest.mark.parametrize("class_name", enumerable_map_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int"), pytest.param("str", id="str")])
def test_one_iterator_shared_by_many_threads(container, keygen, drive_threads, crossing_gate):
    """Threads pulling from one iterator between them see every key exactly once.

    A step releases the GIL so the cursor needs a lock of its own, and every thread takes one step
    before any of them drains, so that lock is provably contended rather than left to a scheduler
    that may hand the whole walk to whoever starts first.
    """
    threads = 8
    keys = keygen(30_000)
    for key in keys:
        container[key] = 1
    walk = iter(container)
    gate = crossing_gate(threads)
    seen = []
    pulled = []
    sink = threading.Lock()

    def pull_one_share() -> list:
        gate.wait()
        # There are far more keys than threads, so every one of these steps answers.
        first = next(walk, None)
        mine = [] if first is None else [first]
        gate.wait()
        mine.extend(walk)
        return mine

    def worker(_: int) -> None:
        try:
            mine = pull_one_share()
        except BaseException:
            gate.abort()
            raise
        # Merged after the pull, so the sink never serializes the walk it is testing.
        with sink:
            seen.extend(mine)
            pulled.append(len(mine))

    drive_threads(worker, threads)
    assert sum(1 for count in pulled if count) == threads, "the cursor reported exhaustion with keys left"
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
