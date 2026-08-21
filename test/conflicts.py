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
    group = st.transaction(container)
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
    group = st.transaction(container)
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
    with st.transaction(container) as (view,):
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
    group = st.transaction(container)
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
    group = st.transaction(container)
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
    with st.transaction(container) as (view,):
        view.watch(key)
        view.watch(key)
        view[key] = 2
    assert container[key] == 2


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_watch_rejects_a_foreign_key_type(container):
    """A watch on a key this container cannot hold is refused like any other key use."""
    with st.transaction(container) as (view,):
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
    group = st.transaction(first, second)
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
    attempts = 0
    while attempts < 10:
        attempts += 1
        try:
            group = st.transaction(container)
            (view,) = group.begin()
            view.watch(key)
            view[key] = container[key] - 10
            if attempts == 1:
                container[key] = 500  # An outsider trips the watch on the first attempt only
            group.stage()
            group.commit()
            break
        except st.ConflictError:
            continue
    assert container[key] == 490
    assert attempts == 2


@pytest.mark.parametrize("sharing", ["locked", "partitioned"])
@pytest.mark.parametrize("isolation", ["monotonic_atomic_view", "snapshot"])
def test_a_refused_stage_applied_nothing(isolation, sharing):
    """`ConflictError` must mean nothing landed, which is what makes a retry safe.

    The contract the README leads with, and the one every retry loop rests on. Forced rather than
    raced, so it holds every level to the promise without depending on an interleaving.
    """
    keys = list(range(64))
    container = make(st.SortedMap, "int", isolation=isolation, sharing=sharing)
    for key in keys:
        container[key] = -1

    group = st.transaction(container)
    (view,) = group.begin()
    view.watch(keys[0])
    container[keys[0]] = 999  # An outsider commits under the watch

    for key in keys:
        view[key] = 12345
    with pytest.raises(st.ConflictError):
        group.stage()

    assert all(container[key] != 12345 for key in keys), "a refused transaction published its writes"


@pytest.mark.parametrize("key_type", key_types)
def test_a_refused_stage_keeps_the_pending_writes(key_type, keygen):
    """A refused stage unwinds what it staged, not what the caller wrote.

    Two stores, with the conflict on the one that stages second, so the first has already staged
    when the refusal arrives and has to be unwound. Rolling it back returns its staged writes to the
    transaction, where they are still pending; resetting would discard the caller's writes with
    them, and a later commit would silently drop work the caller believed it had done.

    The watch stays recorded and stays stale, so re-staging refuses again - that is correct, and it
    is why the documented retry loop opens a fresh transaction. What is asserted here is the writes.

    Regression: the binding reset every participant where `transaction_group::stage` rolls back only
    the prefix that took.
    """
    keys = keygen(3)
    # Participants stage in creation order, so the conflict has to be in the store made second.
    staged_first = make(st.SortedMap, key_type)
    refuses = make(st.SortedMap, key_type)
    refuses[keys[0]] = "original"

    group = st.transaction(staged_first, refuses)
    early, late = group.begin()
    early[keys[1]] = "the caller's own write"
    late.watch(keys[0])
    late[keys[2]] = "and this one"

    refuses[keys[0]] = "an outsider got there first"
    with pytest.raises(st.ConflictError):
        group.stage()

    assert early[keys[1]] == "the caller's own write", "the unwound participant lost a pending write"
    assert late[keys[2]] == "and this one"
    assert keys[1] not in staged_first, "a refused stage must publish nothing"


@pytest.mark.thread_unsafe(reason="it runs its own threads and asserts on their interleaving")
@pytest.mark.slow
@pytest.mark.parametrize("sharing", ["locked", "partitioned"])
def test_a_refused_commit_applied_nothing(sharing, drive_threads):
    """No refused writer publishes anything, under writers that all rewrite every shared key.

    Snapshot only: it is the weakest level that validates the write set and not just the watches,
    so an unwatched rewrite is refused here and accepted below. How many refusals a run draws is the
    scheduler's to decide - pinned to one core the writers serialize and a legitimate run refuses
    nobody - so the count is not asserted here. That a refusal happens at all, and publishes
    nothing when it does, is pinned by `test_a_commit_refused_after_staging_applied_nothing`
    against a fixed schedule.

    Regression: a sharded snapshot commit once published its partitions and then reported a
    conflict, so the retry applied the transaction a second time.
    """
    keys = list(range(64))
    writers, rounds = 3, 30
    container = make(st.SortedMap, "int", isolation="snapshot", sharing=sharing)
    for key in keys:
        container[key] = -1

    ghosts: list[int] = []
    guard = threading.Lock()

    def writer(index: int) -> None:
        attempt, landed = 0, 0
        while landed < rounds:
            attempt += 1
            marker = index * 1_000_000 + attempt
            try:
                with st.transaction(container) as (view,):
                    for key in keys:
                        view[key] = marker
                landed += 1
            except st.ConflictError:
                with guard:
                    if any(container[key] == marker for key in keys):
                        ghosts.append(marker)

    drive_threads(writer, writers, timeout=120)

    assert not ghosts, f"{len(ghosts)} refused transactions published their writes: {ghosts[:5]}"


@pytest.mark.parametrize("sharing", ["locked", "partitioned"])
def test_a_commit_refused_after_staging_applied_nothing(sharing):
    """A write landing between `stage` and `commit` refuses the commit and leaves it unpublished.

    The window the two-phase commit opens is what makes this a schedule rather than a race: the
    stage succeeds because nothing has moved yet, and the outside write lands before the commit
    reads the store again. Both halves of the threaded sibling's contract - that a refusal comes,
    and that it publishes nothing - hold here without a thread.

    The outsider takes the *last* key on purpose. Keys map to partitions in order, so a regression
    that published each partition before validating the next would refuse on partition 0 having
    published nothing, and a conflict on the first key could not tell the two apart.
    """
    keys = list(range(8))
    container = make(st.SortedMap, "int", isolation="snapshot", sharing=sharing)
    for key in keys:
        container[key] = -1

    group = st.transaction(container)
    (view,) = group.begin()
    for key in keys:
        view[key] = 50
    group.stage()

    container[keys[-1]] = 999
    with pytest.raises(st.WriteConflictError):
        group.commit()

    assert container[keys[-1]] == 999, "the outside write is the one that stands"
    assert all(container[key] == -1 for key in keys[:-1]), "a refused commit published part of itself"
