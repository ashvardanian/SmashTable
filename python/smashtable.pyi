"""Type stubs for the SmashTable extension module.

The module is a single C extension with no Python wrapper, so this file is the whole
description a type checker or an editor ever sees. Requires Python 3.12, so the modern
`X | Y` unions and builtin generics are used throughout rather than `typing.Union`.

Each closed set of names the constructors accept is a `Literal`, so a misspelling is a
type error rather than the `ValueError` the parser raises at runtime.

File: python/smashtable.pyi
Author: Ash Vardanian
Date: August 22, 2026
"""

from collections.abc import Iterable, Iterator, Mapping
from types import TracebackType
from typing import Any, ClassVar, Literal

__version__: str

# region Names the Constructors Accept

type _KeyTypeName = Literal["int", "uint", "str", "bytes"]
"""The key layout a store is built around, which decides how a key is compared and hashed."""

type _ValueModeName = Literal["scalar", "object"]
"""Whether a map holds scalars it can store inline, or arbitrary Python objects."""

type _IsolationName = Literal["monotonic_atomic_view", "snapshot", "serializable", "strict_serializable"]
"""What a reader is promised, named as Jepsen names it and ordered by strength. The top two refuse
the same schedules; the stricter one additionally waits for its own publication, so a transaction
opening after a commit returned cannot be ordered before it.
"""

type _SharingName = Literal["locked", "partitioned"]
"""How a store is shared between threads: one lock over the whole store, or a shard per bucket."""

type _DeliveredIsolationName = _IsolationName | Literal["read_committed"]
"""What a store reports it actually delivers, which is wider than what may be asked for: sharding a
store in no order falls back to a rung the constructor does not accept.
"""

type _KeySpec = type[int | str | bytes] | _KeyTypeName
"""A key may be named by its layout or by the Python type standing for it, so `key=int` and
`key="int"` build the same store. There is no Python type for an unsigned key.
"""

type _ValueSpec = _ValueModeName | type[object]
"""A value mode may likewise be named by `object` itself rather than by the string."""

type _Key = int | str | bytes
"""Whatever the key layout admits. A store built for one layout refuses the others."""

# endregion Names the Constructors Accept

# region Errors

class SmashTableError(Exception):
    """Base of every refusal this module raises, so one `except` catches the family.

    Also raised on its own for a failure no caller can act on, such as a staged version
    missing when a transaction unwound. Retrying one of those cannot help.
    """

class ConflictError(SmashTableError, RuntimeError):
    """Base of the three ways an optimistic validation turns a transaction away.

    Never raised on its own - catch it to retry whichever of the three arrived.
    """

class WriteConflictError(ConflictError):
    """An earlier commit already wrote a key this transaction staged."""

class ReadConflictError(ConflictError):
    """A key this transaction read was committed over before it could commit."""

class PhantomConflictError(ConflictError):
    """A key was committed into a window this transaction had read."""

class DuplicateKeyError(SmashTableError, KeyError):
    """A key was inserted where the caller asked for it to be absent."""

class StateError(SmashTableError, RuntimeError):
    """A transaction was asked for something its current state does not allow."""

# endregion Errors

# region Lazy Views

# Named privately because the module never exports them: they are reached through `keys`, `values`
# and `items`, and `from smashtable import ItemsView` raises `ImportError`.

class _KeysView:
    """A lazy walk over the keys of a sorted map, in key order."""

    @property
    def mapping(self) -> SortedMap:
        """The container this view reads."""

    def __iter__(self) -> Iterator[_Key]: ...
    def __len__(self) -> int: ...

class _ValuesView:
    """A lazy walk over the values of a sorted map, in key order."""

    @property
    def mapping(self) -> SortedMap:
        """The container this view reads."""

    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...

class _ItemsView:
    """A lazy walk over the pairs of a sorted map, in key order."""

    @property
    def mapping(self) -> SortedMap:
        """The container this view reads."""

    def __iter__(self) -> Iterator[tuple[_Key, Any]]: ...
    def __len__(self) -> int: ...

# endregion Lazy Views

# region Ordered Containers

class SortedMap:
    """An ordered mapping whose writes can be grouped into transactions.

    Keys are homogeneous and their type is fixed at construction, which is what lets
    every comparison skip type dispatch. Floats and booleans may be values but never
    keys.

    Args:
      key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.
      value (str): 'scalar' to store copies, 'object' to hold arbitrary objects.
      isolation (str): One of the four levels the isolation property names.
      sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.

    Raises:
      TypeError: If key is missing, or names float or bool.
      ValueError: If key, value, isolation or sharing names an unknown choice.
    """

    def __init__(
        self,
        *,
        key: _KeySpec,
        value: _ValueSpec = "scalar",
        isolation: _IsolationName = "monotonic_atomic_view",
        sharing: _SharingName = "locked",
    ) -> None: ...
    @property
    def key_type(self) -> _KeyTypeName:
        """The key layout this store was built around: 'int', 'uint', 'str' or 'bytes'."""
    @property
    def value_mode(self) -> _ValueModeName:
        """'scalar' when values must be scalars, 'object' when any object is stored."""
    @property
    def isolation(self) -> _DeliveredIsolationName:
        """What this store actually promises a reader, as Jepsen names it.

        Four levels may be asked for, in ascending strength: 'monotonic_atomic_view',
        'snapshot', 'serializable' and 'strict_serializable'. The last two both refuse a
        schedule no serial order explains; the stricter one additionally waits for every
        earlier commit to become visible, so a transaction opening after a commit returns
        cannot be ordered before it.

        The effective level rather than the one asked for. A stamped store keeps its level
        across partitions, since visibility there is a comparison against a stamp rather
        than a lock somebody holds. A monotonic one sharded across partitions reports
        'read_committed', because a reader holding no stamp can catch a commit
        half-applied.
        """
    @property
    def sharing(self) -> _SharingName:
        """How this store is shared between threads: 'locked' or 'partitioned'.

        What was asked for, unlike isolation, which reports what is delivered.
        """
    def __len__(self) -> int: ...
    def __contains__(self, key: _Key, /) -> bool: ...
    def __getitem__(self, key: _Key, /) -> Any: ...
    def __setitem__(self, key: _Key, value: Any, /) -> None: ...
    def __delitem__(self, key: _Key | slice, /) -> None: ...
    def __iter__(self) -> Iterator[_Key]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    __hash__: ClassVar[None]  # type: ignore[assignment]
    """Mutable and compared by content, so it is deliberately unhashable."""
    def get(self, key: _Key, default: Any = None, /) -> Any:
        """Value for a key, or default when the key is absent.

        Raises:
          TypeError: If key is not of this store's key type.
        """
    def pop(self, key: _Key, default: Any = ..., /) -> Any:
        """Remove a key and return its value.

        Raises:
          KeyError: If the key is absent and no default was given.
        """
    def popmin(self) -> tuple[_Key, Any]:
        """Remove and return the smallest (key, value) pair.

        Named for what it does rather than borrowing dict's popitem, which removes
        the most recently inserted pair. The store steps forward only, so the
        largest key would cost a full walk while the smallest costs one lookup.

        Raises:
          KeyError: If the store is empty.
        """
    def setdefault(self, key: _Key, default: Any = None, /) -> Any:
        """Value for a key, inserting default first when the key is absent.

        The insertion is one strict operation rather than a lookup followed by a
        write, so two threads racing on the same key cannot both believe they won.

        Raises:
          TypeError: If key is not of this store's key type, or default is not a
            value this store can hold.
        """
    def update(self, other: Mapping[_Key, Any] | Iterable[tuple[_Key, Any]], /) -> None:
        """Apply every pair of a mapping or an iterable of pairs.

        The batch lands as one unit: the store stages it and commits it once, so a
        failure part-way leaves nothing applied. This is stronger than dict, which
        applies pairs one at a time.

        Raises:
          TypeError: If other is neither a mapping nor an iterable of pairs, or a
            key or value in it is not one this store can hold.
          ValueError: If an element of other is not a pair.
        """
    def clear(self) -> None:
        """Remove every entry. The key type stays as it was."""
    def scan(
        self, start: _Key | None = None, stop: _Key | None = None, *, limit: int | None = None
    ) -> list[tuple[_Key, Any]]:
        """List of (key, value) pairs in key order, over the half-open window [start, stop).

        Bounds need not be present keys. Read in one span rather than lazily, so the list
        holds a single committed state unless isolation is 'read_committed'; iterate the
        container itself to walk it lazily.

        Raises:
          TypeError: If a bound is not of this store's key type.
          ValueError: If limit is negative.
        """
    def keys(self) -> _KeysView:
        """A lazy view over the keys, in order."""
    def values(self) -> _ValuesView:
        """A lazy view over the values, in key order."""
    def items(self) -> _ItemsView:
        """A lazy view over the (key, value) pairs, in key order."""
    def transaction(self) -> Transaction:
        """Open a transaction over this store alone."""

class SortedSet:
    """An ordered set whose writes can be grouped into transactions.

    Members are homogeneous and their type is fixed at construction. Floats and
    booleans are not valid members, for the same reason they are not valid keys.

    Args:
      key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.
      isolation (str): One of the four levels the isolation property names.
      sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.
    """

    def __init__(
        self,
        *,
        key: _KeySpec,
        isolation: _IsolationName = "monotonic_atomic_view",
        sharing: _SharingName = "locked",
    ) -> None: ...
    @property
    def key_type(self) -> _KeyTypeName:
        """The key layout this store was built around: 'int', 'uint', 'str' or 'bytes'."""
    @property
    def isolation(self) -> _DeliveredIsolationName:
        """What this store actually promises a reader, as Jepsen names it.

        Four levels may be asked for, in ascending strength: 'monotonic_atomic_view',
        'snapshot', 'serializable' and 'strict_serializable'. The last two both refuse a
        schedule no serial order explains; the stricter one additionally waits for every
        earlier commit to become visible, so a transaction opening after a commit returns
        cannot be ordered before it.

        The effective level rather than the one asked for. A stamped store keeps its level
        across partitions, since visibility there is a comparison against a stamp rather
        than a lock somebody holds. A monotonic one sharded across partitions reports
        'read_committed', because a reader holding no stamp can catch a commit
        half-applied.
        """
    @property
    def sharing(self) -> _SharingName:
        """How this store is shared between threads: 'locked' or 'partitioned'.

        What was asked for, unlike isolation, which reports what is delivered.
        """
    def __len__(self) -> int: ...
    def __contains__(self, member: _Key, /) -> bool: ...
    def __iter__(self) -> Iterator[_Key]: ...
    def __delitem__(self, window: slice, /) -> None:
        """A set is subscripted only by a slice, which erases the window it names."""
    # `__setitem__` exists at runtime only to raise, so it is deliberately absent here: a checker
    # rejecting `members[key] = value` outright is the answer, not a signature that always fails.
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    # Ordering is subset containment, so only another set answers; an iterable does not.
    def __lt__(self, other: SortedSet | set[_Key] | frozenset[_Key], /) -> bool: ...
    def __le__(self, other: SortedSet | set[_Key] | frozenset[_Key], /) -> bool: ...
    def __gt__(self, other: SortedSet | set[_Key] | frozenset[_Key], /) -> bool: ...
    def __ge__(self, other: SortedSet | set[_Key] | frozenset[_Key], /) -> bool: ...
    __hash__: ClassVar[None]  # type: ignore[assignment]
    """Mutable and compared by content, so it is deliberately unhashable."""
    def add(self, member: _Key, /) -> None:
        """Insert a member. Adding one that is already present does nothing.

        Raises:
          TypeError: If member is not of this set's key type.
        """
    def discard(self, member: _Key, /) -> None:
        """Remove a member if present, and say nothing if not."""
    def remove(self, member: _Key, /) -> None:
        """Remove a member.

        Raises:
          KeyError: If the member is absent.
        """
    def popmin(self) -> _Key:
        """Remove and return the smallest member.

        Unlike set, which pops an arbitrary member, this pops the
        smallest, which the ordered store reaches in one lookup.

        Raises:
          KeyError: If the set is empty.
        """
    def update(self, other: Iterable[_Key], /) -> None:
        """Add every member of an iterable.

        The batch lands as one unit: the store stages it and commits it once, so a
        failure part-way leaves nothing applied. This is stronger than set, which
        adds members one at a time.

        Raises:
          TypeError: If other is not iterable, or a member of it is not of this
            set's key type.
        """
    def clear(self) -> None:
        """Remove every entry. The key type stays as it was."""
    def scan(self, start: _Key | None = None, stop: _Key | None = None, *, limit: int | None = None) -> list[_Key]:
        """List of members in order, over the half-open window [start, stop).

        Bounds need not be present members. Read in one span rather than lazily, so the
        list holds a single committed state unless isolation is 'read_committed'; iterate
        the set to walk it lazily.

        Raises:
          TypeError: If a bound is not of this set's key type.
          ValueError: If limit is negative.
        """
    def union(self, other: Iterable[_Key], /) -> SortedSet:
        """Members of either side, as a new set."""
    def intersection(self, other: Iterable[_Key], /) -> SortedSet:
        """Members of both sides."""
    def difference(self, other: Iterable[_Key], /) -> SortedSet:
        """Members of this side only."""
    def symmetric_difference(self, other: Iterable[_Key], /) -> SortedSet:
        """Members of exactly one side."""
    def isdisjoint(self, other: Iterable[_Key], /) -> bool:
        """Whether the two sides share no member."""
    def transaction(self) -> Transaction:
        """Open a transaction over this store alone."""

# endregion Ordered Containers

# region Unordered Containers

class HashMap:
    """An unordered mapping whose writes can be grouped into transactions.

    Point access only. The open-addressed core supplies no ordering, so this class has
    no iteration, no keys/values/items, no scan and no range erase - reaching for one
    is an AttributeError rather than an empty result. Use SortedMap when order or
    enumeration matters.

    Args:
      key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.
      value (str): 'scalar' to store copies, 'object' to hold arbitrary objects.
      isolation (str): One of the four levels the isolation property names.
      sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.
    """

    def __init__(
        self,
        *,
        key: _KeySpec,
        value: _ValueSpec = "scalar",
        isolation: _IsolationName = "monotonic_atomic_view",
        sharing: _SharingName = "locked",
    ) -> None: ...
    @property
    def key_type(self) -> _KeyTypeName:
        """The key layout this store was built around: 'int', 'uint', 'str' or 'bytes'."""
    @property
    def value_mode(self) -> _ValueModeName:
        """'scalar' when values must be scalars, 'object' when any object is stored."""
    @property
    def isolation(self) -> _DeliveredIsolationName:
        """What this store actually promises a reader, as Jepsen names it.

        Four levels may be asked for, in ascending strength: 'monotonic_atomic_view',
        'snapshot', 'serializable' and 'strict_serializable'. The last two both refuse a
        schedule no serial order explains; the stricter one additionally waits for every
        earlier commit to become visible, so a transaction opening after a commit returns
        cannot be ordered before it.

        The effective level rather than the one asked for. A stamped store keeps its level
        across partitions, since visibility there is a comparison against a stamp rather
        than a lock somebody holds. A monotonic one sharded across partitions reports
        'read_committed', because a reader holding no stamp can catch a commit
        half-applied.
        """
    @property
    def sharing(self) -> _SharingName:
        """How this store is shared between threads: 'locked' or 'partitioned'.

        What was asked for, unlike isolation, which reports what is delivered.
        """
    def __len__(self) -> int: ...
    def __contains__(self, key: _Key, /) -> bool: ...
    def __getitem__(self, key: _Key, /) -> Any: ...
    def __setitem__(self, key: _Key, value: Any, /) -> None: ...
    def __delitem__(self, key: _Key, /) -> None: ...
    def get(self, key: _Key, default: Any = None, /) -> Any:
        """Value for a key, or default when the key is absent.

        Raises:
          TypeError: If key is not of this store's key type.
        """
    def pop(self, key: _Key, default: Any = ..., /) -> Any:
        """Remove a key and return its value.

        Raises:
          KeyError: If the key is absent and no default was given.
        """
    def setdefault(self, key: _Key, default: Any = None, /) -> Any:
        """Value for a key, inserting default first when the key is absent.

        The insertion is one strict operation rather than a lookup followed by a
        write, so two threads racing on the same key cannot both believe they won.

        Raises:
          TypeError: If key is not of this store's key type, or default is not a
            value this store can hold.
        """
    def update(self, other: Mapping[_Key, Any] | Iterable[tuple[_Key, Any]], /) -> None:
        """Apply every pair of a mapping or an iterable of pairs.

        The batch lands as one unit: the store stages it and commits it once, so a
        failure part-way leaves nothing applied. This is stronger than dict, which
        applies pairs one at a time.

        Raises:
          TypeError: If other is neither a mapping nor an iterable of pairs, or a
            key or value in it is not one this store can hold.
          ValueError: If an element of other is not a pair.
        """
    def clear(self) -> None:
        """Remove every entry. The key type stays as it was."""
    def transaction(self) -> Transaction:
        """Open a transaction over this store alone."""

class HashSet:
    """An unordered set whose writes can be grouped into transactions.

    Membership and mutation only. The core supplies no ordering, so this class has no
    iteration, no scan, no pop and none of the set algebra, each of which would have to
    walk at least one side. Use SortedSet when any of those matter.
    """

    def __init__(
        self,
        *,
        key: _KeySpec,
        isolation: _IsolationName = "monotonic_atomic_view",
        sharing: _SharingName = "locked",
    ) -> None: ...
    @property
    def key_type(self) -> _KeyTypeName:
        """The key layout this store was built around: 'int', 'uint', 'str' or 'bytes'."""
    @property
    def isolation(self) -> _DeliveredIsolationName:
        """What this store actually promises a reader, as Jepsen names it.

        Four levels may be asked for, in ascending strength: 'monotonic_atomic_view',
        'snapshot', 'serializable' and 'strict_serializable'. The last two both refuse a
        schedule no serial order explains; the stricter one additionally waits for every
        earlier commit to become visible, so a transaction opening after a commit returns
        cannot be ordered before it.

        The effective level rather than the one asked for. A stamped store keeps its level
        across partitions, since visibility there is a comparison against a stamp rather
        than a lock somebody holds. A monotonic one sharded across partitions reports
        'read_committed', because a reader holding no stamp can catch a commit
        half-applied.
        """
    @property
    def sharing(self) -> _SharingName:
        """How this store is shared between threads: 'locked' or 'partitioned'.

        What was asked for, unlike isolation, which reports what is delivered.
        """
    def __len__(self) -> int: ...
    def __contains__(self, member: _Key, /) -> bool: ...
    def add(self, member: _Key, /) -> None:
        """Insert a member. Adding one that is already present does nothing.

        Raises:
          TypeError: If member is not of this set's key type.
        """
    def discard(self, member: _Key, /) -> None:
        """Remove a member if present, and say nothing if not."""
    def remove(self, member: _Key, /) -> None:
        """Remove a member.

        Raises:
          KeyError: If the member is absent.
        """
    def update(self, other: Iterable[_Key], /) -> None:
        """Add every member of an iterable.

        The batch lands as one unit: the store stages it and commits it once, so a
        failure part-way leaves nothing applied. This is stronger than set, which
        adds members one at a time.

        Raises:
          TypeError: If other is not iterable, or a member of it is not of this
            set's key type.
        """
    def clear(self) -> None:
        """Remove every entry. The key type stays as it was."""
    def transaction(self) -> Transaction:
        """Open a transaction over this store alone."""

# endregion Unordered Containers

# region Transactions

class Participant:
    """One container's slice of a transaction.

    Reads see this transaction's own writes and the state as of when the group
    opened, never another transaction's staged-but-uncommitted writes. Writes stay
    invisible to every other reader until the transaction commits, and vanish if it does
    not. A view stops working once its transaction finishes.
    """

    def __contains__(self, key: _Key, /) -> bool: ...
    def __getitem__(self, key: _Key, /) -> Any: ...
    def __setitem__(self, key: _Key, value: Any, /) -> None: ...
    def __delitem__(self, key: _Key | slice, /) -> None: ...
    def get(self, key: _Key, default: Any = None, /) -> Any:
        """Value for a key inside this transaction, or default when the key is absent.

        Reads this transaction's own uncommitted writes, and otherwise the state as of
        when the transaction opened. It does not see writes another transaction has staged but
        not committed, so a read is never dirty.

        Repeating a read is not guaranteed to return the same value: a committed write
        from elsewhere becomes visible immediately. Call watch(key) to make the
        transaction refuse to commit if that happens.

        Raises:
          TypeError: If key is not of this store's key type, or this is a set.
          StateError: If the transaction has already finished.
        """
    def add(self, member: _Key, /) -> None:
        """Insert one member into a set participant, invisibly until the transaction commits.

        Idempotent, and subject to the same visibility rule as upsert.

        Raises:
          TypeError: If this participant is a map, which needs a value.
          StateError: If the transaction has already finished.
        """
    def discard(self, key: _Key, /) -> bool:
        """Remove a key if it is there, reporting whether it was.

        `del participant[key]` removes it too; this is the form that answers.

        Returns:
          bool: True when the key was present, so a caller need not look first.

        The removal is invisible outside the transaction until commit, and is undone by
        rollback or by the block raising.

        Raises:
          TypeError: If key is not of this store's key type.
          StateError: If the transaction has already finished.
        """
    def update(self, other: Mapping[_Key, Any] | Iterable[Any], /) -> None:
        """Apply every pair of a mapping to this participant.

        Atomic, unlike SortedMap.update: every pair lands with the rest of the transaction or
        none of them does. A failure part-way leaves the transaction abandonable by
        rollback with nothing applied.

        Raises:
          TypeError: If other is not a mapping, or a key is of the wrong type.
          ValueError: If an element of other is not a pair.
          StateError: If the transaction has already finished.
        """
    def watch(self, key: _Key, /) -> None:
        """Refuse to commit if another writer touches this key first.

        This is what turns read-modify-write into a lost-update-free operation. Without
        it a transaction is atomic but not serializable: two transactions can each read the
        same value, each write back, and one update is silently lost.

        Covers absence as well as presence, so watching a key that does not exist still
        conflicts if someone inserts it. The conflict surfaces at stage(), not at the
        write, and nothing is applied when it does.

        Raises:
          ConflictError: At stage() time, never here.
          TypeError: If key is not of this store's key type.
          StateError: If the transaction has already finished.
        """
    def scan(self, start: _Key | None = None, stop: _Key | None = None, *, limit: int | None = None) -> list[Any]:
        """List of (key, value) pairs in key order over [start, stop), or of members for a set.

        Sees this transaction's own staged writes, and otherwise the state as of when it
        opened. Materialized in one span rather than lazily, because a walk records the
        window it read and nothing else may run inside that record.

        Under 'serializable' or stricter, a key another transaction commits into that
        window makes this transaction's commit raise PhantomConflictError, rather than
        let a repeat of the walk answer differently.

        Raises:
          TypeError: If a bound is not of this store's key type, or this participant is unordered.
          ValueError: If limit is negative.
          StateError: If the transaction has already finished.
        """

class Transaction:
    """A transaction of containers staged together and then published in turn.

    Either every store is published or none is: a body that raises resets each
    participant, and a stage that one refuses unwinds the rest. Publication itself
    walks the containers in turn rather than moving them at one instant, so a thread
    reading two of them while a commit runs may find one a step ahead - take a
    transaction of your own if the pair must agree. No transaction ever reads
    another's uncommitted writes. What more is promised depends on the store: its
    isolation property reports the level, and below 'snapshot' a value already
    read may change underneath, so a read-modify-write needs watch() to be safe.

    Durability is memory-only: this is not a database and nothing survives the
    process.

    Used as a context manager, the block is exactly begin(), the body, stage(),
    commit(); an exception discards everything instead.
    """

    def begin(self) -> tuple[Participant, ...]:
        """Tuple of per-container views, in the order the stores were given.

        Participants stage in a process-wide canonical order rather than argument order,
        which is what lets two groups naming the same containers in opposite orders run
        concurrently without deadlocking. The views you receive ignore that and follow
        your arguments.

        Raises:
          StateError: If the transaction has already finished.
        """
    def stage(self) -> None:
        """Validate every watch and reserve the writes. Nothing becomes visible.

        This is the first of the two phases. After it succeeds, commit() cannot fail for
        a reason this transaction could have avoided. It can still refuse over a watched
        key another transaction published while this one sat staged.

        A partial stage is never observable: if one participant refuses, every other is
        unwound before this returns.

        Raises:
          ConflictError: If a watched key changed since the transaction opened.
          StateError: If the transaction is not open - already staged, or finished.
        """
    def commit(self) -> None:
        """Publish every staged change, one container at a time.

        Not a snapshot across containers. Each is applied in turn, so a thread reading
        two of them while this runs may find one a step ahead of the other; a reader that
        needs the pair to agree should take its own transaction. Readers are never
        blocked while this runs.

        A refusal before anything is published leaves the group staged, so it can still
        be rolled back. A refusal after one container has published cannot be pulled
        back, so the group stops calling itself staged and only reset() clears what the
        rest still hold.

        Raises:
          StateError: If the transaction was not staged first.
          ConflictError: If a watched key was published over while the group sat staged.
        """
    def rollback(self) -> None:
        """Pull staged changes back out of the stores, leaving them as they were.

        Valid only after stage() and before commit(). Since nothing staged was ever
        visible, no reader can have observed what this undoes.

        Raises:
          StateError: If the transaction was not staged.
          SmashTableError: If a staged version was gone before the rollback reached
            it, which is a broken invariant rather than a race a retry could win.
        """
    def reset(self) -> None:
        """Discard everything pending and start the transaction over.

        Unlike rollback, valid at any point before the handle closes - open, staged or
        already committed. This is what a retry loop calls after catching ConflictError.
        """
    def __enter__(self) -> tuple[Participant, ...]: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

def transaction(*stores: SortedMap | SortedSet | HashMap | HashSet) -> Transaction:
    """Open one transaction spanning every given store.

    Used as a context manager, the block either makes every change visible or none
    of them. Stores may mix maps and sets, ordered and unordered, and may use
    different key types and isolation levels.

    Args:
      *stores: One or more SortedMap, SortedSet, HashMap or HashSet, each at
        most once. A store's own transaction() is this with one argument.

    Returns:
      Transaction: Whose participants arrive in the order the stores were given.

    Raises:
      TypeError: If no store was given, or one is not a store.
      ValueError: If the same store is given twice.
    """

# endregion Transactions
