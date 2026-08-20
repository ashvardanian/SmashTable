# SmashTable

SmashTable makes __one update span several containers__.
Either every change lands or none does: a group stages into every container or into none, and a commit spanning two or more of them asks each for permission before the first one writes.
A partitioned container is the one that cannot be asked, so a group holding one publishes its participants in turn.
It's implemented in __C++ 20__ as header-only templates, and exposed to __Python 3__ through the raw CPython API.

Three `std::map`s cannot do this, and neither can three `dict`s.
It needs no threads to be useful — the same guarantee that keeps two indexes consistent under contention is what keeps them consistent when an exception unwinds a loop halfway through.

![SmashTable Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/SmashTable.jpg?raw=true)

## What's Inside

```bash
  basic_*            → cores, single-threaded     vector · AVL · weight-balanced · hash
  atomic_*           → pinned, GPU-capable        hash
       ↑ ::adopt() moves one allocation between the two, no rehash, no copy

  *_store            → transactions over a core   monotonic · snapshot · reference
       ↓ wrap any of those for thread safety
  locked_store       → one lock per transaction
  partitioned_store  → one lock per partition, 16 by default

  transaction_group  → one commit spanning several stores
```

Every store publishes what it promises as a compile-time `isolation_k`, and the suite checks that constant against behaviour rather than against itself — a container that quietly changes level fails.
Each rung implies the ones beneath it, so one column names the highest a store reaches rather than five ticking the same fact five times.
Every row below the cores wraps something: a transactional store wraps a plain container, and either thread-safety wrapper wraps a store.
So a usable type reads `locked_store<snapshot_store<basic_avl_tree<…>>>` for one lock over the whole store, or `partitioned_store<…>` for sixteen.
Nesting the two wrappers is redundant rather than clever: every call would take an inner lock inside a partition lock that already excludes.

|                             | Isolation               |    Writers    | Transactions |      Ordered      |
| --------------------------- | ----------------------- | :-----------: | :----------: | :---------------: |
| `basic_vector`              | —                       |  one thread   |      —       |         —         |
| `basic_avl_tree`            | —                       |  one thread   |      —       |         ✔         |
| `basic_wb_tree`             | —                       |  one thread   |      —       | `rank` · `select` |
| `basic_hash_table`          | —                       |  one thread   |      —       |         —         |
| `atomic_hash_table`         | — ⁴                     |  per slot ⁴   |      —       |         —         |
| `monotonic_store`           | Monotonic Atomic View ⁶ |  one thread   |      ✔       |     inherits      |
| `snapshot_store`            | Snapshot ⁷              |  one thread   |      ✔       |    inherits ²     |
| `serializable_store`        | Serializable ⁸          |  one thread   |      ✔       |    inherits ²     |
| `strict_serializable_store` | Strict Serializable ⁹   |  one thread   |      ✔       |    inherits ²     |
| `reference_store` ⁵         | Monotonic Atomic View ⁶ |  one thread   |      ✔       |         ✔         |
| `locked_store`              | inherits                |   one call    |      ✔       |     inherits      |
| `partitioned_store`         | inherits ¹              | per partition |      ✔       |    inherits ³     |

> ¹ A stamp-based store keeps its level: one clock, and the watermark moves only once the last partition has published, so a reader sees a whole commit or none.
> A clock-less store has no stamp to hold, so only Read Committed survives; atomicity costs in proportion to the partitions touched.
> ² `select` and `rank` are exact at the newest commit; an older snapshot gets a merged walk, since one count per node cannot answer an unbounded parameter.
> ³ A bound, a range and an ordinal hold every partition's lock for the whole walk, so each is decided at one moment rather than by probes that can disagree.
> An unbounded enumeration takes one partition at a time and answers unordered, while the cursor behind iteration walks the merged order and holds no lock between its steps.
> ⁴ Per-slot spin locks: atomic over one slot and nothing wider, so `size()` is a relaxed read, a stalled thread blocks its slot, and there are no transactions.
> ⁵ A `std::set`-backed oracle: it installs alongside every other header, and the suites hold the tree containers against it.
> ⁶ Permits a [lost update](https://jepsen.io/consistency/phenomena/p4) and a repeated read that moves.
> ⁷ Refuses both; permits [write skew](https://jepsen.io/consistency/phenomena/a5b), since a read you did not `watch` is not validated.
> ⁸ Refuses write skew and phantoms too: every key and window read is re-checked at commit.
> ⁹ Same refusals as ⁸; [strict](https://jepsen.io/consistency/models/strict-serializable) also waits for its own publication, so a transaction opening after a commit cannot precede it — which changes only sharded commits and writes outside a transaction.

The ladder runs from [Read Committed](https://jepsen.io/consistency/models/read-committed) through [Monotonic Atomic View](https://jepsen.io/consistency/models/monotonic-atomic-view) and [Snapshot Isolation](https://jepsen.io/consistency/models/snapshot-isolation) to the two serializable rungs, and every level from Monotonic Atomic View upward is delivered by [multi-version concurrency control](https://en.wikipedia.org/wiki/Multiversion_concurrency_control): a key keeps one version per commit that touched it, a reader is answered from the newest version its own snapshot can name, and versions below the oldest live reader are reclaimed.

## Python Quick Start

```bash
pip install smashtable
```

The wheel ships type stubs, so an editor completes the container methods and a checker rejects a misspelled isolation level or key layout before the call ever runs.

Two indexes over the same entities have to move together, or a lookup by name finds an identifier that no longer resolves:

```python
import smashtable as st

by_id = st.SortedMap(key=int)     # entity id → record
by_name = st.SortedMap(key=str)   # name      → entity id

with st.transaction(by_id, by_name) as (ids, names):
    ids[42] = "carol"
    names["carol"] = 42
# both indexes became visible together
```

Each container is a mapping in its own right, so the usual vocabulary works:

```python
list(by_id)                  # [42]           — keys, in order
dict(by_name)                # {'carol': 42}
by_id.items()                # a lazy view, not a copy
by_id.scan(0, 100)           # the half-open window [0, 100)
```

If the block raises, nothing was applied:

```python
try:
    with st.transaction(by_id, by_name) as (ids, names):
        ids[43] = "dave"
        names["dave"] = 43
        raise ValueError("failed validation")
except ValueError:
    pass

assert 43 not in by_id and "dave" not in by_name
```

Without this, the alternative is an O(n) defensive copy that every alias to the container can still read through mid-update:

```python
backup = dict(cache)                      # copies the whole thing
try:
    for key, value in updates.items():
        cache[key] = transform(value)     # raises on item 40 of 100
except Exception:
    cache.clear(); cache.update(backup)   # O(n) again, and readers already saw the torn state
```

### Optimistic Concurrency

`watch` makes a transaction fail rather than overwrite a key someone else touched first.
`ConflictError` is the retry signal, and the only exception a correct program is expected to catch routinely:

```python
while True:
    try:
        with st.transaction(accounts) as (view,):
            view.watch("alice")
            view["alice"] = view["alice"] - 100
        break
    except st.ConflictError:
        continue    # nothing was applied, so retrying is safe
```

### The Two Phases

`with` is exactly `begin()`, the body, `stage()`, `commit()`.
Both phases are also available directly:

```python
group = st.transaction(by_id, by_name)
ids, names = group.begin()
ids[42] = "carol"; names["carol"] = 42
group.stage()      # validates watches and reserves; still invisible, and may raise ConflictError
group.commit()     # flips visibility
```

Separating the fallible phase from the applying phase is what lets independent transactions compose into one all-or-nothing unit.
`stage` is where a conflict or an allocation failure normally surfaces.
Between the two the participants take no more writes: `ids[43] = 'dave'` after `stage()` raises `StateError`, since staging already reserved and validated what the transaction would change.
A read still answers.
`commit` validates again and can still refuse, because a watched key may be committed over while the group sits staged.
A group of two or more containers, none of them partitioned, is asked in full before any of them writes, so a refusal there publishes nothing and leaves the group staged and retryable.
A lone participant commits in one call, having nothing to tear against, and so does a group holding a `sharing='partitioned'` container, where a refusal part-way leaves the participants before it published.
A refusal by the first participant asked has published nothing and leaves the group staged, which is the state `rollback()` accepts.
A refusal by any later one returns the group to open — the position decides that, not whether anything was published — and `reset()` rather than `rollback()` clears what the rest still hold.
Inside a `with` block the exit discards that remainder itself, so `reset()` is the remedy on the manual `stage()` and `commit()` path.
Even where every participant is asked first, one of them may be committed over between the two passes: what the split buys is that the second pass cannot refuse, not that nothing moves beneath it.

### What It Guarantees, and What It Does Not

- __A group stages in full or not at all.__
  A body that raises resets every participant.
  A stage that fails on one participant unwinds them all.
- __A commit publishes in full or not at all, wherever every participant can be asked first.__
  That is a group of two or more containers, none of them partitioned.
  Elsewhere the participants publish in turn, and only a refusal by the first leaves the group staged — any later one returns it to open, where `reset()` and not `rollback()` clears the remainder.
- __Staged writes are invisible to everyone else__, including a transaction opened after the stage.
- __A transaction reads its own writes, until it stages.__
  `view[k]` sees what `view[k] = v` put there; after `stage()` the write is in the store carrying no stamp, so it is invisible to everyone including its own transaction until `commit()`.
- __Commit is not a snapshot.__
  It applies each container in turn, so another thread reading two containers while a commit runs may find one of them a step ahead.
  A reader that needs the pair to agree should take its own transaction or read after the writer's block returns.
- __Scans are not snapshots either.__ `scan()` walks in key order and never yields a key twice or raises mid-walk, but a key inserted behind the cursor is missed.
  That is the container's own `scan`; the same call on a transaction participant records the window it read, so at `serializable` and above a key committed into that window refuses the commit rather than being silently missed.

### Ordered and Unordered

`SortedMap` and `SortedSet` keep their keys in order, which is what iteration, `scan` and slice erase rest on.
`HashMap` and `HashSet` are the same stores over an open-addressed core, and offer point access only:

```python
cache = st.HashMap(key=str)
cache['a'] = 1
cache['a'], len(cache), 'a' in cache      # (1, 1, True)

cache.scan()                              # AttributeError — no ordering to scan
list(cache)                               # TypeError — not iterable
```

The absent methods are absent from the type rather than refused at the call, so reaching for one fails the way a typo fails.
Everything else is shared — the same key layouts, the same isolation and sharing choices, the same transactions — so one `st.transaction()` may span ordered and unordered stores together.

### Choosing an Isolation Level

A container names what it promises a reader, and reports back what it actually delivers:

```python
cache = st.SortedMap(key=int)                                    # monotonic atomic view, the default
ledger = st.SortedMap(key=int, isolation='snapshot')
books = st.SortedMap(key=int, isolation='serializable')
audit = st.SortedMap(key=int, isolation='strict_serializable')

cache.isolation    # 'monotonic_atomic_view'
ledger.isolation   # 'snapshot'
books.isolation    # 'serializable'
audit.isolation    # 'strict_serializable'
```

Four names, refused by `ValueError` if misspelled — and the shipped type stubs turn a misspelling into a checker error before it ever runs.

Under `snapshot` every read a transaction makes is answered at the instant the transaction opened, so a repeated read returns what it first saw:

```python
with st.transaction(ledger) as (view,):
    first = view[account]
    # ... another thread commits a new value for `account` here ...
    assert view[account] == first     # holds under snapshot, not under monotonic
```

That is what `monotonic` does not give you, and the reason to pay for the extra versions.
The cost is memory: a key keeps every version a live reader can still name, and the tail is freed once the last transaction closes.

Snapshot validates what a transaction __wrote__, plus any key it explicitly `watch`ed, and nothing more, so two transactions that each read what the other writes and then write elsewhere both land — [write skew](https://jepsen.io/consistency/phenomena/a5b), which Snapshot Isolation permits by construction because no two writes ever collide.
`serializable` validates the read set as well, which is also what makes a [phantom](https://jepsen.io/consistency/phenomena/a3) detectable: a window a transaction scanned is remembered as a window, so a key committed into it afterwards is a conflict even though no key either transaction touched overlaps.

```python
with st.transaction(books) as (view,):
    empty = view.scan(20, 80)          # reads a window, and records that it did
    # ... another thread commits key 50 here ...
    ...                                # PhantomConflictError on leaving the block
```

`PhantomConflictError` derives from `ConflictError`, so a caller that retries on conflict needs no new branch.
`strict_serializable` refuses exactly what `serializable` refuses; it adds only that a transaction opening after a commit returned cannot be ordered before it.

`sharing` decides how many writers can proceed at once:

```python
st.SortedMap(key=int, sharing='locked')        # one lock, the default
st.SortedMap(key=int, sharing='partitioned')   # sixteen partitions, concurrent writers
```

`isolation` reports what a container delivers, not what it was asked for, and sharding is the one place those differ:

```python
sharded_snapshot = st.SortedMap(key=int, isolation='snapshot', sharing='partitioned')
sharded_monotonic = st.SortedMap(key=int, isolation='monotonic_atomic_view', sharing='partitioned')

sharded_snapshot.isolation    # 'snapshot'        — asked for snapshot, got snapshot
sharded_monotonic.isolation   # 'read_committed'  — asked for monotonic, got less
```

A monotonic container has no stamp for a reader to hold, so a walk crossing partitions can catch a commit half-applied, and only [Read Committed](https://jepsen.io/consistency/models/read-committed) survives above a single key.
A snapshot container does have one — visibility there is a comparison against a stamp rather than a lock somebody holds — so its commits become visible everywhere at once and its level survives however many partitions it is spread across.

Keeping the guarantee is not free, though.
Making a commit atomic across partitions costs in proportion to how many of them a transaction touches: one writing a single key is unaffected, while one writing across all sixteen serializes against every other transaction that does the same.
Sharding pays off for point-heavy work spread over many keys, and least for transactions that touch everything.

`locked` is the default because it is the setting under which the default level means what it says.

### Stored Types

Values are `int`, `float`, `bool`, `str` and `bytes`, deep-copied into memory the library owns, so a stored value outlives the object it came from.
`str` and `bytes` stay distinct, and `bool` round-trips as `bool` rather than as `1`.

A container may instead hold arbitrary objects, which it stores by reference rather than by copy:

```python
documents = st.SortedMap(key=str, value='object')
documents['doc'] = {'nested': [1, 2]}   # the very same object comes back out
```

The mode is named rather than inferred, because it is what the scalar guarantee rests on.
In the default scalar mode nothing stored can touch a reference count, so the interpreter lock is released around every store; in object mode it is held, and the container is correspondingly slower.

__Keys are typed and homogeneous.__
A container names its key layout at construction and refuses every other type:

```python
st.SortedMap(key=int)      # signed 64-bit
st.SortedMap(key='uint')   # unsigned 64-bit, the only way to ask, since Python has no unsigned type
st.SortedMap(key=str)      # UTF-8, ordered bytewise
st.SortedMap(key=bytes)    # opaque, never decoded
```

That is what lets every comparison skip type dispatch: the ordering function is chosen once, at construction, rather than re-derived from a tag on each of the millions of comparisons a tree makes.

`float` and `bool` are values but never keys.
A float key would make ordering depend on a total order over NaN, and `bool` is a subclass of `int` in Python, so accepting it would silently alias two key spaces:

```python
m = st.SortedMap(key=int)
m[1] = 'one'      # fine
m[1.0] = 'one'    # TypeError: float is not a valid key
m[True] = 'one'   # TypeError: bool is not a valid key
```

This is the one place the library is deliberately stricter than `dict`, which treats `1`, `1.0` and `True` as one key.

## C++ Quick Start

```cmake
include(FetchContent)
FetchContent_Declare(
    smashtable
    GIT_REPOSITORY https://github.com/ashvardanian/SmashTable
    GIT_TAG main
)
FetchContent_MakeAvailable(smashtable)
target_link_libraries(your_target PRIVATE smashtable::smashtable)
```

For a system-wide install, standard CMake package discovery works:

```bash
sudo cmake --install build_release
```

```cmake
find_package(smashtable REQUIRED)
target_link_libraries(your_target PRIVATE smashtable::smashtable)
```

The library throws nowhere, and will not let you make it.
Every API returns `status_t`, reads included, and a read delivers its answer through `noexcept` callbacks rather than through the return.
A read is fallible from `serializable_k` up, where it has to write down what it read before the commit can validate it.
A key whose copy or comparison can throw fails to compile rather than being quietly accepted.
All containers take custom allocators and can be pre-allocated.

The same shape as the Python example, with no threads in sight:

```cpp
#include <smashtable/basic_avl_tree.hpp>
#include <smashtable/monotonic_store.hpp>

namespace st = ashvardanian::smashtable;

using id_t = std::uint64_t;
using by_id_t = st::monotonic_avl_map<id_t, record_t>;      // id   → record
using by_name_t = st::monotonic_avl_map<name_t, id_t>;      // name → id

auto by_id = *by_id_t::make();
auto by_name = *by_name_t::make();

auto group = *st::make_transaction_group(by_id, by_name);
auto &ids = group.participant<0>();
auto &names = group.participant<1>();

_ = ids.upsert({id, record});
_ = names.upsert({record.name, id});
_ = ids.watch(id);                  // fail rather than clobber a concurrent write

_ = group.stage();                  // both reserve, or neither does
_ = group.commit();                 // both become visible
```

Staging the two by hand would leave the first one staged when the second refuses, which is the bug a group exists to remove.
Participants are visited in order of the store's address rather than of the argument list, so two groups naming the same stores in opposite orders cannot deadlock on each other.
A refused `stage()` rolls the staged prefix back rather than resetting it, so the writes survive and the group can be retried.

Basic containers — `basic_avl_tree`, `basic_wb_tree` — carry STL-style bidirectional iterators.
Transactional containers do not, because keeping an iterator valid across concurrent updates costs more than it returns:

- `find()`, `lower_bound()` and `upper_bound()` return `status_t` and take two `noexcept` callbacks, for the found and missing cases.
- `equal_range()`, `sample_one()` and `sample_reservoir()` return `status_t` and take one callback, invoked per element.
- `find_copy()`, `lower_bound_copy()` and `upper_bound_copy()` return an `expected<value_t>` for callers that cannot use a callback.

## Why The C++ Library

The standard library has loose ends around failure.
`std::set::insert(first, last)` has no defined behaviour on partial failure, and in practice an allocation failure leaves some elements inserted and the rest not:

```cpp
struct budget_t { std::size_t used = 0, limit = 0; };

template <typename value_type_> // ? rebind, converting constructor and `deallocate` elided
struct failing_allocator {
    using value_type = value_type_;
    budget_t *budget {};

    value_type_ *allocate(std::size_t count) {
        if (!budget || budget->used + count > budget->limit) throw std::bad_alloc {};
        budget->used += count;
        return static_cast<value_type_ *>(::operator new(count * sizeof(value_type_)));
    }
};

int main() {
    budget_t budget {0, 3};
    std::set<int, std::less<>, failing_allocator<int>> values {std::less<> {}, failing_allocator<int> {&budget}};

    try { for (int value : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) values.insert(value); }
    catch (std::bad_alloc const &) { std::println("std::bad_alloc after {} allocations", budget.used); }

    std::println("set contains {} elements:", values.size());
    for (int value : values) std::print("{} ", value);
    std::print("\n");
}
```

Under Clang++ 21 and G++ 15 that prints:

```
std::bad_alloc after 3 allocations
set contains 3 elements:
0 1 2
```

Where this matters is secondary-index consistency inside a storage engine — an `id → slot` map, a `slot → vector` map and a deleted set that have to move together, and where "the crash left index B disagreeing with index A" is a corruption bug someone has already debugged.

## Stores

Terminology first, since both words are overloaded:

- "Concurrency" does not imply threads.
  Several transactions can be open on one thread.
- "Consistency" does not imply [strict serializability](https://jepsen.io/consistency/models/strict-serializable).
  [Weaker levels exist](https://jepsen.io/consistency/models), and each store here names the one it targets as a compile-time `isolation_k` rather than leaving you to infer it.

Headers group as:

```bash
  basic_*               → Cores. Own their memory, no transactions, one thread.
    ├─ basic_vector<T, Alloc>
    ├─ basic_avl_tree<T, Comparator, Alloc>
    ├─ basic_wb_tree<T, Comparator, Alloc>          # also `rank` and `select`
    └─ basic_hash_table<T, Hash, Equals, Alloc>     # grows, iterates, rehashes

  atomic_*              → Pinned core. Fixed capacity, atomic per slot, callback reads.
    └─ atomic_hash_table<T, Hash, Equals, Alloc>    # the one type that runs on a GPU
       ↑ ::adopt(std::move(growable).release())     ↓ ::adopt(std::move(pinned).release())

  *_store               → 2-phase commit, watch and CAS.
    ├─ monotonic_store<Core>                        # one visible version per key
    ├─ snapshot_store<Core>                         # every version a live reader can still name
    └─ reference_store<T, Comparator, Alloc>        # the same contract over `std::set`, kept as the oracle
       ↓ either of the first two wraps a core; either of these wraps a store
    ├─ locked_store<Store, Mutex>                   # one lock spans a whole commit
    └─ partitioned_store<Store, Hash, Mutex, N>     # one lock per partition, one shared commit clock

  transaction_group<Stores...>  → One 2-phase commit spanning several stores
```

Transactions and thread-safety are two independent axes, not one ladder.
A wrapper takes its mutex per __call__, not across a transaction: `stage` and `commit` each take it and drop it in between.
What carries the inner store's level across that gap is the reservation staging made, since no other transaction can claim those keys until this one publishes or unwinds.
A clock-less store gives its reader no stamp to hold, so a monotonic reader spanning partitions can catch a commit half-applied and only [Read Committed](https://jepsen.io/consistency/models/read-committed) survives — but a snapshot reader answers at its own stamp, and every partition draws from one clock, so it keeps Snapshot across all sixteen.
Every container publishes what it promises as `isolation_k`, so the level is checkable rather than folklore.
`atomic_hash_table` has no transactions at all — it offers per-__operation__ atomicity, which is a different product, and is why it is not a `*_store`.

`status_t` reports `out_of_memory_heap_k == ENOMEM`, `invalid_argument_k == EINVAL`, `key_not_found_k == ENOENT` and others, and is `[[nodiscard]]` on every mutating API.

### Making `std::set` Transactional

> `smashtable/reference_store.hpp`

The baseline reference design, and the yardstick the tree containers are held against — it runs the same test suites they do.
Updates group into transactions that stage and roll back before committing, which is what lets inter-dependent updates span several stores.

Reads are consistent at the [Monotonic Atomic View](https://jepsen.io/consistency/models/monotonic-atomic-view) level, so a transaction never sees another's partial update.
That is weaker than [Snapshot Isolation](https://jepsen.io/consistency/models/snapshot-isolation): there is no guarantee that every read within one transaction sees the same snapshot.
It keeps a version per key all the same, dated by a generation rather than by a shared commit stamp, which is why a watch is re-resolved at commit instead of compared against one.
`snapshot_store` is the sibling that does make that guarantee, and the section below states what it costs.

### Adelson-Velsky and Landis Trees

> `smashtable/basic_avl_tree.hpp`

Rarely called by its full name, the AVL tree is the simplest clean self-balancing binary search tree, from 1962.
`basic_avl_tree<value, comparator, allocator>` is an ordered store in the shape of `std::set` or `std::map`.
Standard libraries usually pick Red-Black trees; AVL balances more rigidly, trading slightly slower updates for faster lookups.

Its API differs from the STL where exception-free reporting demands it:

- `insert()` returns `std::pair<iterator, bool>`, as the STL does.
- `erase(iterator)` returns `erase_result_t { iterator next; status_t status; }`.
- Range `insert(first, last)` returns `status_t`.
- Hint-based `insert(hint, value)` and `emplace_hint()` are deleted, since AVL trees gain nothing from a hint.

### Weight-Balanced Trees

> `smashtable/basic_wb_tree.hpp`

`basic_wb_tree` balances on subtree sizes rather than heights, using Δ=3 and Γ=2 — the only proven integer solution, per Hirai and Yamamoto.
Storing sizes buys __order statistics__: `select(k)` finds the k-th smallest and `rank(x)` finds a key's position, both in O(log n).
That is what pagination, percentiles and quantiles need, and it is the one thing the AVL tree cannot offer.

Expect depth around 1.88 log₂(n) against AVL's 1.44, in exchange for O(1) amortized rotations per update.

### Transactional Stores

> `smashtable/monotonic_store.hpp`

`monotonic_store<Core>` adds two-phase commit, watches and CAS on top of any key-addressable core.
`monotonic_avl_set`, `monotonic_avl_map`, `monotonic_wb_set` and `monotonic_wb_map` are the aliases you'll name directly.
`monotonic_hash_set` and `monotonic_hash_map` back the same store with the open-addressed table, which supplies no ordering, so bounds, ranges and order statistics are gated out of those instantiations at compile time.

Which version a reader sees is decided by one number, stamped when a transaction commits.
A version that has not committed carries no stamp, so it is invisible to readers and cannot make anyone else's validation fail — a transaction that stages and then rolls back costs its peers nothing.
Two writers on one key are ordered by their stamps, and the later one wins.
That is a [lost update](https://jepsen.io/consistency/phenomena/p4), which this level permits: Monotonic Atomic View constrains what a reader observes, not what two writers do to each other, and a level that is available under partition provably cannot prevent it.
A writer that needs the first commit to win must `watch()` the key, or use a snapshot container, where the write set is validated whether you ask or not.

`find` never records what it read, here.
A read set is memory, and a read that allocates is a read that can fail, so on this engine the recording variant is a different name:

```cpp
_ = transaction.find(key, on_found, on_missing);             // answers, and records nothing
_ = transaction.find_and_watch(key, on_found, on_missing);   // records, and can report out of memory
```

Both return `status_t`, because a read that cannot write down what it read has to say so where it happened rather than poisoning the transaction silently until commit.
`snapshot_store` at `serializable_k` and above needs no second name: every read records, and every read reports.

An erase leaves a tombstone that readers skip, and `vacuum()` is what returns that space:

```cpp
expected<std::size_t> const reclaimed = store.vacuum();          // everything no reader can name
expected<std::size_t> const window = store.vacuum(lower, upper); // ordered cores, one slice at a time
```

### Snapshot Isolation

> `smashtable/snapshot_store.hpp`

`snapshot_store<Core>` is the sibling that fixes a transaction's reads to one instant, and it is where [multi-version concurrency control](https://en.wikipedia.org/wiki/Multiversion_concurrency_control) is at its most literal here: a transaction draws a snapshot when it opens, every read is answered from the newest version at or below that snapshot, and a writer never overwrites what a reader can still name.
It rebinds the same cores, exports `snapshot_avl_set`, `_avl_map`, `_wb_set`, `_wb_map`, `_hash_set` and `_hash_map`, and publishes `isolation_k == snapshot_k`.
A repeated read returns what it first saw, a repeated range admits no phantoms, and neither holds in its sibling.

The same template answers at three levels, selected by a second parameter rather than by a second implementation, and each has aliases of its own:

```cpp
snapshot_store<Core>                        // isolation_k == snapshot_k
serializable_store<Core>                    // reads validated too
strict_serializable_store<Core>             // and a commit waits for its own publication

serializable_avl_map<Key, Value>            // and `_avl_set`, `_wb_*`, `_hash_*`
strict_serializable_avl_map<Key, Value>     // the same six for the top rung
```

Versions are kept as ordinary entries keyed by `(key, generation)` rather than chained off one entry, on ordered and unordered cores alike.
The hash core still hashes the bare key, so every version of a key shares one probe run and a lookup walks that run instead of stopping at the first match.

Reclamation is the other half of multi-version concurrency control, and the whole of it here: old versions are freed against a low-water mark that sits at the oldest open snapshot, so it can never pass a snapshot somebody still holds, and it advances as soon as the oldest reader leaves rather than waiting for the last.
Pruning happens on every commit, and `vacuum()` reaches the keys nobody writes again.
There is no background thread and no epoch registry.
`clear()` refuses while a reader is open rather than dropping versions out from under it.

Range writes publish under one stamp, so a reader on an older snapshot sees all of a range erase or none of it.
`update_range` builds new versions rather than rewriting the visible one, which is what open readers make unsound.
`select` and `rank` descend on the weight-balanced tree's augmented count, exact at the newest commit — a reader at an older snapshot gets the merged walk instead, because one number per node cannot answer for an unbounded parameter.

What it costs, on an AVL core over `mapping<key, int>`:

|                               | `monotonic_store` | `snapshot_store`    |
| ----------------------------- | ----------------- | ------------------- |
| Resident per key, one version | 80 B              | 72 B                |
| Each retained older version   | 48 B chain node   | 72 B, a full entry  |
| Read of one key               | one chain head    | the key's whole run |

At rest it is the cheaper of the two, because a key carries no chain-head pointer and no per-entry allocator.
It becomes the more expensive one exactly when a long-lived reader pins history — and on the hash core a pinned version occupies a real slot, so it lengthens the probe runs of its neighbours as well.
With no transaction open the low-water mark sits at the newest commit and the whole version tail is freed on the next write, so a store nobody is reading costs what its sibling costs.

### Hash Tables

> `smashtable/basic_hash_table.hpp`

`hash_set<key>` and `hash_map<key, value>` are open-addressing containers with linear probing.
They differ from `google::dense_hash_map` and `tsl::robin_map` by not reserving sentinel key values for tombstones, by unpacking keys and values into disjoint arrays, and by carrying two bits of metadata per slot rather than a byte.

Keys and values live in separate regions, so a lookup that only compares keys never pulls values into cache — which is most of them, since a miss touches no value at all.
Thirty-two slots share one 64-bit header holding two parallel bitmasks, encoding free, deleted, populated and locked in two bits each.
Sixty-four bits is the widest atomic every target supports, which is what lets a whole bucket header move in one instruction.

Growing repoints the key, value and header regions that every live slot reference has cached, so a table that can grow can never be read concurrently.
The type system enforces that rather than a sentence: two types, neither of which includes the other's header.
`basic_hash_table` grows, iterates and rehashes; `atomic_hash_table` is pinned and atomic over the slot each operation touches.
What passes between them is the allocation itself — a `hash_storage` — so the hand-off is a move of a value rather than one table reaching into the other:

```cpp
_ = growable.reserve(1u << 20);                                          // pin the capacity first
auto pinned = atomic_hash_map<key_t, value_t>::adopt(std::move(growable).release());

_ = pinned.find(key, [](auto const &slot) noexcept { use(slot.value()); });
if (failed(pinned.emplace(key, value))) report_full();                   // a pinned table can fill up

auto compacted = hash_map<key_t, value_t>::adopt(std::move(pinned).release());   // iterators return
```

Every operation on the pinned table is atomic over the slot it touches: it takes that slot by setting both its bits with a `fetch_or` and releases it with a `fetch_xor` of the difference to the desired state, so neither path needs a compare-and-swap loop.
It is not lock-free, though — `fetch_or` spins until it wins the slot, so a thread descheduled while holding one blocks every other prober that walks onto it.
No global lock and no reallocation is what the pinning buys; a stalled thread stalling its neighbours is what it does not.
Reads take a callback rather than returning a reference or an iterator, since both would dangle the moment another thread erased the slot.
Tombstones only accumulate while pinned, because compaction needs a rehash — `deleted_count()` is what says it is time to hand the storage back.

Readers take the slot lock like writers do.
A reader that merely loaded the header would be unsound, and measurably so: the two bits per slot cannot distinguish "untouched" from "locked, rewritten and unlocked", because a completed write cycle restores exactly the bits the reader first saw.
Adding a per-bucket version counter would fix it and would cost half the slots per bucket, which is not worth breaking the one-warp-per-bucket geometry for.

Order-dependent operations are absent by construction: there is no `range`, `erase_range`, `lower_bound` or `select`.

The small-string specialization mentioned in the header is a design note, not shipped behaviour.

## Testing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DSMASHTABLE_WERROR=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`SMASHTABLE_FILTER` selects a subset by substring, matched against `suite.name`, and fails the binary when it matches nothing:

```bash
SMASHTABLE_FILTER=transactional_consistency ./build/smashtable_test_avl_tree
```

One binary per container family runs the same suites — the `std::set` store, both trees, both thread-safety wrappers, the bare hash table and the transactional stores over it — so a behavioural difference between them shows up as a failure rather than a surprise.

The suite asserts costs, not only answers.
`select` over the augmented count is pinned to an exact node count rather than to a bound: an independent descent measures 13 nodes at 4096 entries and 19 at 262144, one per level of a tree that ascending keys leave perfectly balanced.
The counting comparator tallies `rank` alone, because `select_augmented` navigates on subtree sizes and asks the comparator nothing.
An element type that tallies its own construction and destruction turns a leak into arithmetic — a key leaked inside a correctly freed node is invisible to a sanitizer and not to the tally.
A key whose hash keeps only its group forces the long probe runs a well-spread hash never produces, which is where tombstone reuse and severed runs actually show.

`isolation_k` is checked against behaviour rather than against itself.
The shared suite branches on the level a container declares and asserts both outcomes, so a container that quietly starts snapshotting without raising its level fails just as one that claims a level it does not deliver.

For the Python side:

```bash
pip install -e . --group test && pytest test/
```
