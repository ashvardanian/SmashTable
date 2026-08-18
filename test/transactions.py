"""Single-container and cross-container atomicity, and the explicit two-phase protocol.

Baselines:
    A shadow dict advanced only when a block commits, so an aborted block must leave the model
    and the container agreeing that nothing happened.

Run:
    python -m pytest test/transactions.py -v
"""

import pytest

import smashtable as st

from .base import group_sizes, key_types, make, map_class_names, sorted_class_names, transaction_styles


class _Abort(Exception):
    """Raised inside a block to abort it, distinct from anything the library raises."""


# region Single container


@pytest.mark.iterations(1)
@pytest.mark.thread_unsafe(
    reason="not idempotent - it asserts an absolute state of its container, so re-running the body against one fixture, whether by --iterations or by --parallel-threads, falsifies it"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_changes_are_invisible_until_the_block_ends(container, keygen):
    """A write inside a transaction is not visible outside it until commit."""
    key = keygen(1)[0]
    group = st.transaction(container)
    (view,) = group.begin()
    view[key] = "staged"
    assert key not in container
    group.stage()
    assert key not in container
    group.commit()
    assert container[key] == "staged"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_view_reads_its_own_writes(container, keygen):
    """Inside the block, a write is immediately readable through the same view."""
    key = keygen(1)[0]
    with st.transaction(container) as (view,):
        view[key] = "mine"
        assert view[key] == "mine"
        assert key in view


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_raise_discards_everything(container, keygen):
    """A block that raises applies nothing, and the exception propagates unchanged."""
    keys = keygen(20)
    with pytest.raises(_Abort):
        with st.transaction(container) as (view,):
            for key in keys:
                view[key] = "doomed"
            raise _Abort
    assert len(container) == 0


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_batch_lands_whole(container, keygen):
    """Every write of a committed block is visible together."""
    keys = keygen(20)
    with st.transaction(container) as (view,):
        view.update({key: "batch" for key in keys})
    assert len(container) == len(keys)
    assert all(container[key] == "batch" for key in keys)


@pytest.mark.thread_unsafe(
    reason="its premise is a single writer - a parallel copy of the test sharing the container would disturb the very watch or count it asserts on"
)
@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_erase_reports_presence(container, keygen):
    """A view's erase says whether the key was there, so a caller need not look first."""
    keys = keygen(2)
    container[keys[0]] = "present"
    with st.transaction(container) as (view,):
        assert view.discard(keys[0]) is True
        assert view.discard(keys[1]) is False


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_transaction_sees_prior_state(container, keygen):
    """A block reads what was committed before it opened."""
    key = keygen(1)[0]
    container[key] = "before"
    with st.transaction(container) as (view,):
        assert view[key] == "before"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_an_empty_transaction_commits(container):
    """A block that writes nothing still completes cleanly."""
    with st.transaction(container) as (_,):
        pass
    assert len(container) == 0


# endregion Single container

# region Explicit phases


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_stage_then_rollback_applies_nothing(container, keygen):
    """Rolling a staged transaction back pulls the changes out again."""
    key = keygen(1)[0]
    group = st.transaction(container)
    (view,) = group.begin()
    view[key] = "staged"
    group.stage()
    group.rollback()
    assert key not in container


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_commit_without_stage_is_a_state_error(container, keygen):
    """The two phases are ordered, and skipping the first is refused."""
    group = st.transaction(container)
    (view,) = group.begin()
    view[keygen(1)[0]] = "x"
    with pytest.raises(st.StateError):
        group.commit()


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_rollback_without_stage_is_a_state_error(container):
    """Rolling back something never staged is refused rather than silently ignored."""
    group = st.transaction(container)
    group.begin()
    with pytest.raises(st.StateError):
        group.rollback()


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_reset_discards_pending_changes(container, keygen):
    """reset throws away the block's work and leaves the store untouched."""
    key = keygen(1)[0]
    group = st.transaction(container)
    (view,) = group.begin()
    view[key] = "x"
    group.reset()
    assert key not in container


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_finished_group_refuses_more_work(container, keygen):
    """After commit the group is done, and its views say so rather than writing anywhere."""
    key = keygen(1)[0]
    group = st.transaction(container)
    (view,) = group.begin()
    view[key] = "x"
    group.stage()
    group.commit()
    with pytest.raises(st.StateError):
        view[key] = "again"


@pytest.mark.parametrize("key_type", key_types)
def test_a_finished_transaction_cannot_be_reset(key_type, keygen):
    """A committed transaction is over, and `reset` must not hand its views a second turn.

    Regression: reset wrote the open state unconditionally, so a finished group could be reopened
    and committed again, applying a fresh set of writes through a transaction that had ended.
    """
    container = make(st.SortedMap, key_type)
    keys = keygen(2)

    group = st.transaction(container)
    (view,) = group.begin()
    view[keys[0]] = "first"
    group.stage()
    group.commit()
    assert dict(container) == {keys[0]: "first"}

    with pytest.raises(st.StateError):
        group.reset()
    with pytest.raises(st.StateError):
        group.begin()
    assert dict(container) == {keys[0]: "first"}, "a finished transaction wrote again"


@pytest.mark.parametrize("key_type", key_types)
def test_reset_is_allowed_before_the_transaction_finishes(key_type, keygen):
    """The state gate must not break what reset is for: retrying after a conflict."""
    container = make(st.SortedMap, key_type)
    key = keygen(1)[0]

    group = st.transaction(container)
    (view,) = group.begin()
    view[key] = "pending"
    group.reset()  # open, not staged
    assert key not in container

    (view,) = group.begin()
    view[key] = "staged"
    group.stage()
    group.reset()  # staged, not yet committed
    assert key not in container


# endregion Explicit phases

# region Cross container


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("style", transaction_styles)
def test_two_indexes_flip_together(container_class, key_type, style, keygen):
    """Both containers show the batch or neither does, through either spelling."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    key = keygen(1)[0]
    if style == "context":
        with st.transaction(first, second) as (left, right):
            left[key] = "a"
            right[key] = "b"
    else:
        group = st.transaction(first, second)
        left, right = group.begin()
        left[key] = "a"
        right[key] = "b"
        group.stage()
        group.commit()
    assert first[key] == "a"
    assert second[key] == "b"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_raise_leaves_neither_container_touched(container_class, key_type, keygen):
    """An aborted group is invisible in every participant, not just the first."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    key = keygen(1)[0]
    with pytest.raises(_Abort):
        with st.transaction(first, second) as (left, right):
            left[key] = "a"
            right[key] = "b"
            raise _Abort
    assert len(first) == 0 and len(second) == 0


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("group_size", group_sizes)
def test_a_group_may_span_many_containers(container_class, key_type, group_size, keygen):
    """A group is not limited to two, and every participant lands together."""
    containers = [make(container_class, key_type) for _ in range(group_size)]
    key = keygen(1)[0]
    with st.transaction(*containers) as views:
        assert len(views) == group_size
        for view in views:
            view[key] = "shared"
    assert all(one[key] == "shared" for one in containers)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_views_arrive_in_argument_order(container_class, key_type, keygen):
    """Whatever order participants stage in, the views come back as the caller wrote them."""
    first = make(container_class, key_type)
    second = make(container_class, key_type)
    key = keygen(1)[0]
    with st.transaction(second, first) as (left, right):
        left[key] = "into-second"
        right[key] = "into-first"
    assert second[key] == "into-second"
    assert first[key] == "into-first"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_the_same_container_twice_is_rejected(container):
    """Two participants over one store would each believe they owned its staged state."""
    with pytest.raises(ValueError):
        st.transaction(container, container)


def test_atomic_needs_a_container():
    """A group with no participants is meaningless."""
    with pytest.raises(TypeError):
        st.transaction()


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_atomic_rejects_a_foreign_object(container):
    """Only containers may participate."""
    with pytest.raises(TypeError):
        st.transaction(container, {})


@pytest.mark.parametrize("class_name", sorted_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_group_may_mix_maps_and_sets(container_class, key_type, keygen):
    """A map and a set commit together, which the single-class binding could not express."""
    from .base import is_map_class

    mapping = make(st.SortedMap, key_type)
    members = make(st.SortedSet, key_type)
    key = keygen(1)[0]
    assert is_map_class(st.SortedMap)
    with st.transaction(mapping, members) as (map_view, set_view):
        map_view[key] = "value"
        set_view.add(key)
    assert mapping[key] == "value"
    assert key in members


@pytest.mark.parametrize("key_type", key_types)
def test_a_set_participant_refuses_a_value(key_type, keygen):
    """Assigning a value to a set participant is a TypeError pointing at add()."""
    members = make(st.SortedSet, key_type)
    key = keygen(1)[0]
    with pytest.raises(TypeError):
        with st.transaction(members) as (view,):
            view[key] = "value"


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_a_map_participant_refuses_add(container, keygen):
    """Calling add on a map participant is a TypeError pointing at assignment."""
    key = keygen(1)[0]
    with pytest.raises(TypeError):
        with st.transaction(container) as (view,):
            view.add(key)


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", [pytest.param("int", id="int")])
def test_containers_may_use_different_key_types(container_class, key_type, keygen):
    """A group does not require its participants to agree on a key layout."""
    by_number = make(container_class, "int")
    by_name = make(container_class, "str")
    with st.transaction(by_number, by_name) as (numbers, names):
        numbers[42] = "carol"
        names["carol"] = 42
    assert by_number[42] == "carol"
    assert by_name["carol"] == 42


@pytest.mark.parametrize("class_name", map_class_names)
@pytest.mark.parametrize("key_type", key_types)
def test_single_container_transaction_shorthand(container, keygen):
    """A container opens a one-participant group over itself."""
    key = keygen(1)[0]
    with container.transaction() as (view,):
        view[key] = "solo"
    assert container[key] == "solo"


@pytest.mark.parametrize("key_type", key_types)
@pytest.mark.parametrize("object_first", [pytest.param(False, id="scalar-first"), pytest.param(True, id="object-first")])
def test_a_group_may_mix_value_modes(key_type, keygen, object_first):
    """Each participant writes under its own value mode, not under the group's first.

    The two modes differ in whether the interpreter lock is held around a stored value, so a
    participant reading a neighbour's mode picks the wrong policy - the path where a missed
    acquisition corrupts rather than fails. Swept in both orders because reading participant zero
    happens to be right for whichever container is there.
    """
    scalars = make(st.SortedMap, key_type)
    objects = make(st.SortedMap, key_type, value_mode="object")
    key = keygen(1)[0]
    payload = {"nested": [1, 2]}

    first, second = (objects, scalars) if object_first else (scalars, objects)
    with st.transaction(first, second) as (first_view, second_view):
        object_view, scalar_view = (first_view, second_view) if object_first else (second_view, first_view)
        object_view[key] = payload
        scalar_view[key] = 42

    assert objects[key] is payload, "an object-mode value must round-trip by identity"
    assert scalars[key] == 42


# endregion Cross container
