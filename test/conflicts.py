"""Optimistic concurrency: watch, ConflictError, and the retry loop that makes them useful.

Baselines:
    None. This is the library's own guarantee, so the invariants are stated directly rather
    than diffed against a stdlib type that cannot express them.

Run:
    python -m pytest test/conflicts.py -v
"""

import threading

import pytest

import smashtable as st

from .base import key_types, make, map_class_names


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_external_write_to_a_watched_key_conflicts(container, keygen):
    """A watch turns a lost update into a refusal at stage time."""
    key = keygen(1)[0]
    container[key] = 100
    group = st.atomic(container)
    (view,) = group.begin()
    view.watch(key)
    view[key] = 50
    container[key] = 999
    with pytest.raises(st.ConflictError):
        group.stage()


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_conflicted_stage_applies_nothing(container, keygen):
    """A refused stage leaves the store exactly as the other writer left it."""
    key = keygen(1)[0]
    container[key] = 100
    group = st.atomic(container)
    (view,) = group.begin()
    view.watch(key)
    view[key] = 50
    container[key] = 999
    with pytest.raises(st.ConflictError):
        group.stage()
    assert container[key] == 999


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_undisturbed_watch_commits(container, keygen):
    """A watch that nobody trips is invisible."""
    key = keygen(1)[0]
    container[key] = 100
    with st.atomic(container) as (view,):
        view.watch(key)
        view[key] = 50
    assert container[key] == 50


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_unwatched_key_does_not_conflict(container, keygen):
    """Only watched keys arm the check, so an unrelated write is not a conflict."""
    keys = keygen(2)
    container[keys[0]] = 1
    group = st.atomic(container)
    (view,) = group.begin()
    view.watch(keys[0])
    view[keys[0]] = 2
    container[keys[1]] = 99
    group.stage()
    group.commit()
    assert container[keys[0]] == 2


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_watching_an_absent_key_conflicts_when_it_appears(container, keygen):
    """A watch covers absence too, so an insert by someone else is a conflict."""
    key = keygen(1)[0]
    group = st.atomic(container)
    (view,) = group.begin()
    view.watch(key)
    view[key] = "mine"
    container[key] = "theirs"
    with pytest.raises(st.ConflictError):
        group.stage()


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_watching_the_same_key_twice_is_idempotent(container, keygen):
    """Watching twice is not an error and does not double-arm anything."""
    key = keygen(1)[0]
    container[key] = 1
    with st.atomic(container) as (view,):
        view.watch(key)
        view.watch(key)
        view[key] = 2
    assert container[key] == 2


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_watch_rejects_a_foreign_key_type(container):
    """A watch on a key this container cannot hold is refused like any other key use."""
    with st.atomic(container) as (view,):
        with pytest.raises(TypeError):
            view.watch(1.5)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_conflict_in_one_container_aborts_the_whole_group(container_class, key_type, keygen):
    """A group is all-or-nothing, so one participant's conflict discards the others' work."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    key = keygen(1)[0]
    first[key] = 1
    group = st.atomic(first, second)
    left, right = group.begin()
    left.watch(key)
    left[key] = 2
    right[key] = "companion"
    first[key] = 999
    with pytest.raises(st.ConflictError):
        group.stage()
    assert key not in second


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_the_retry_loop_converges(container, keygen):
    """The documented retry pattern terminates and lands the intended value."""
    key = keygen(1)[0]
    container[key] = 100
    interference = [True]
    attempts = 0
    while True:
        attempts += 1
        try:
            group = st.atomic(container)
            (view,) = group.begin()
            view.watch(key)
            view[key] = container[key] - 10
            if interference[0]:
                container[key] = 500
                interference[0] = False
            group.stage()
            group.commit()
            break
        except st.ConflictError:
            continue
        if attempts > 10:
            pytest.fail("retry loop failed to converge")
    assert container[key] == 490
    assert attempts == 2


@pytest.mark.parametrize("sharing", ["locked", "partitioned"])
@pytest.mark.parametrize("isolation", ["monotonic", "snapshot"])
def test_a_refused_stage_applied_nothing(isolation, sharing):
    """`ConflictError` must mean nothing landed, which is what makes a retry safe.

    The contract the README leads with, and the one every retry loop rests on. Forced rather than
    raced, so it holds every level to the promise without depending on an interleaving.
    """
    keys = list(range(64))
    container = make(st.SortedMap, "int", isolation=isolation, sharing=sharing)
    for key in keys:
        container[key] = -1

    group = st.atomic(container)
    (view,) = group.begin()
    view.watch(keys[0])
    container[keys[0]] = 999  # An outsider commits under the watch

    for key in keys:
        view[key] = 12345
    with pytest.raises(st.ConflictError):
        group.stage()

    assert 12345 not in {container[key] for key in keys}, "a refused transaction published its writes"


@pytest.mark.slow
@pytest.mark.thread_unsafe(reason="it runs its own threads and asserts on their interleaving")
@pytest.mark.parametrize("sharing", ["locked", "partitioned"])
def test_a_refused_commit_applied_nothing(sharing):
    """The same contract for a refusal that arrives at `commit` rather than at `stage`.

    Snapshot only: it is the level that validates writes as well as watches, so it is the only one
    where a transaction can pass `stage` and still be refused. The precondition at the end is
    load-bearing - a run in which nothing was refused has not exercised the path, and must not be
    read as evidence that the path is sound.

    Regression: a sharded snapshot commit once published its partitions and then reported a
    conflict, so the retry applied the transaction a second time.
    """
    keys = list(range(64))
    container = make(st.SortedMap, "int", isolation="snapshot", sharing=sharing)
    for key in keys:
        container[key] = -1

    ghosts: list[int] = []
    refusals = [0]
    guard = threading.Lock()

    def writer(index: int) -> None:
        attempt = 0
        for _ in range(30):
            while True:
                attempt += 1
                marker = index * 1_000_000 + attempt
                try:
                    with st.atomic(container) as (view,):
                        for key in keys:
                            view[key] = marker
                    break
                except st.ConflictError:
                    with guard:
                        refusals[0] += 1
                        if any(container[key] == marker for key in keys):
                            ghosts.append(marker)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "a writer never finished"

    assert not ghosts, f"{len(ghosts)} refused transactions published their writes: {ghosts[:5]}"
    assert refusals[0] > 0, "nothing was refused, so this run proved nothing"
