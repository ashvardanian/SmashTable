# SmashTable

SmashTable makes __one update span several containers__.
Either every change lands or none does, and the containers never disagree about which happened.
It's implemented in __C++ 20__ as header-only templates, and exposed to __Python 3__ through the raw CPython API.

Three `std::map`s cannot do this, and neither can three `dict`s.
It needs no threads to be useful — the same guarantee that keeps two indexes consistent under contention is what keeps them consistent when an exception unwinds a loop halfway through.

![SmashTable Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/SmashTable.jpg?raw=true)

## Python Quick Start

```bash
pip install smashtable
```

Two indexes over the same entities have to move together, or a lookup by name finds an identifier that no longer resolves:

```python
import smashtable as st

by_id = st.SortedMap(key=int)     # entity id → record
by_name = st.SortedMap(key=str)   # name      → entity id

with st.atomic(by_id, by_name) as (ids, names):
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
    with st.atomic(by_id, by_name) as (ids, names):
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
        with st.atomic(accounts) as (view,):
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
group = st.atomic(by_id, by_name)
ids, names = group.begin()
ids[42] = "carol"; names["carol"] = 42
group.stage()      # validates watches and reserves; still invisible, and may raise ConflictError
group.commit()     # flips visibility
```

Separating the fallible phase from the applying phase is what lets independent transactions compose into one all-or-nothing unit.
`stage` is where a conflict or an allocation failure surfaces; by `commit` there is nothing left to fail on.

### What It Guarantees, and What It Does Not

- __A group applies in full or not at all.__ A body that raises resets every participant. A stage that fails on one participant unwinds them all.
- __Staged writes are invisible to everyone else__, including a transaction opened after the stage.
- __A transaction reads its own writes__ — `view[k]` sees what `view[k] = v` put there.
- __Commit is not a snapshot.__ It applies each container in turn, so another thread reading two containers while a commit runs may find one of them a step ahead. A reader that needs the pair to agree should take its own transaction or read after the writer's block returns.
- __Scans are not snapshots either.__ `scan()` walks in key order and never yields a key twice or raises mid-walk, but a key inserted behind the cursor is missed.

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
    GIT_REPOSITORY https://github.com/ashvardanian/smashtable
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

The library throws nowhere.
Mutating APIs return `status_t`; read-only operations cannot fail and deliver results through `noexcept` callbacks.
All containers take custom allocators and can be pre-allocated.

The same shape as the Python example, with no threads in sight:

```cpp
#include <smashtable/basic_avl_tree.hpp>
#include <smashtable/transactional_binary_tree.hpp>

namespace st = ashvardanian::smashtable;

using id_t = std::uint64_t;
using by_id_t = st::transactional_avl_map<id_t, record_t>;      // id   → record
using by_name_t = st::transactional_avl_map<name_t, id_t>;      // name → id

auto by_id = *by_id_t::make();
auto by_name = *by_name_t::make();

auto ids = *by_id.transaction();
auto names = *by_name.transaction();

_ = ids.upsert({id, record});
_ = names.upsert({record.name, id});
_ = ids.watch(id);                  // fail rather than clobber a concurrent write

_ = ids.stage();                    // both reserve, nothing visible yet
_ = names.stage();
_ = ids.commit();                   // both become visible
_ = names.commit();
```

Basic containers — `basic_avl_tree`, `basic_wb_tree` — carry STL-style bidirectional iterators.
Transactional containers do not, because keeping an iterator valid across concurrent updates costs more than it returns:

- `find()`, `lower_bound()` and `upper_bound()` return `void` and take two `noexcept` callbacks, for the found and missing cases.
- `equal_range()` and `sample_range()` return `void` and take one callback, invoked per element.
- `find_copy()`, `lower_bound_copy()` and `upper_bound_copy()` return an `expected<value_t>` for callers that cannot use a callback.

## Why The C++ Library

The standard library has loose ends around failure.
`std::set::insert(first, last)` has no defined behaviour on partial failure, and in practice an allocation failure leaves some elements inserted and the rest not:

```cpp
#include <cstddef>  // `std::size_t`
#include <iostream> // `std::cout`
#include <new>      // `std::bad_alloc`
#include <set>      // `std::set`
#include <vector>   // `std::vector`

struct failing_allocator_state_t {
    std::size_t count = 0, limit = 0;
};

template <typename value_type_>
struct failing_allocator {
    using value_type = value_type_;
    template <typename other_type_>
    struct rebind { using other = failing_allocator<other_type_>; };

    failing_allocator_state_t *state_ {};
    explicit failing_allocator(failing_allocator_state_t *state) noexcept : state_ {state} {}

    failing_allocator() noexcept = default;
    template <typename other_type_>
    failing_allocator(failing_allocator<other_type_> const &other) noexcept : state_ {other.state_} {}
    value_type_ *allocate(std::size_t count) {
        if (!state_ || state_->count + count > state_->limit) throw std::bad_alloc {};
        state_->count += count;
        return static_cast<value_type_ *>(::operator new(count * sizeof(value_type_)));
    }
    void deallocate(value_type_ *pointer, std::size_t) noexcept { ::operator delete(pointer); }
};

int main() {
    failing_allocator_state_t state {0, 3};
    std::set<int, std::less<>, failing_allocator<int>> values {std::less<> {}, failing_allocator<int> {&state}};
    std::vector<int> inputs {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    try { values.insert(inputs.begin(), inputs.end()); }
    catch (std::bad_alloc const &) { std::cout << "std::bad_alloc after " << state.count << " allocations\n"; }

    std::cout << "set contains " << values.size() << " elements:\n";
    for (int value : values) std::cout << value << ' ';
    std::cout << '\n';
}
```

Under Clang++ 21 and G++ 15 that prints:

```
std::bad_alloc after 3 allocations
set contains 3 elements:
0 1 2
```

Where this matters is secondary-index consistency inside a storage engine — an `id → slot` map, a `slot → vector` map and a deleted set that have to move together, and where "the crash left index B disagreeing with index A" is a corruption bug someone has already debugged.

## Collections

Terminology first, since both words are overloaded:

- "Concurrency" does not imply threads. Several transactions can be open on one thread.
- "Consistency" does not imply strict serializability. [Weaker levels exist](https://jepsen.io/consistency/models), and this library targets one of them.

Headers group as:

```bash
  basic_*               → Core structures. Own memory, no transactions, one thread.
    ├─ basic_vector<T, Alloc>
    ├─ basic_avl_tree<T, Comparator, Alloc>
    ├─ basic_wb_tree<T, Comparator, Alloc>        # also `rank` and `select`
    └─ basic_hash_table<T, Hash, Equals, Alloc>   # grows, iterates, rehashes

  concurrent_*          → Pinned cores. Fixed capacity, lock-free, callback reads.
    └─ concurrent_hash_table<T, Hash, Equals, Alloc>
       ↑ ::adopt(std::move(growable).release())   ↓ ::adopt(std::move(pinned).release())

  transactional_*       → 2-phase commit, watch and CAS
    ├─ transactional_binary_tree<Tree>            # generic over both trees
    └─ transactional_std_store<T, Comparator, Alloc>

  *_collection          → Thread-safety wrappers
    ├─ locked_collection<Collection, Mutex>
    └─ partitioned_collection<Collection, Hash, Mutex, PartsCount>
```

Both wrappers serialize whole __transactions__, so the unit of exclusion is a two-phase commit rather than a single operation.
`locked_collection` holds one lock across the whole commit, so whatever its inner store promises survives intact.
`partitioned_collection` takes and releases one partition lock at a time, so a reader spanning partitions can catch a commit half-applied — above a single partition only [Read Committed](https://jepsen.io/consistency/models/read-committed) survives.
Every partition walk acquires in ascending index order, which is what keeps two of them from waiting on each other.
`concurrent_hash_table` has no transactions at all — it offers per-__operation__ atomicity, which is a different product, and is why it is not a `*_collection`.

`status_t` reports `out_of_memory_heap_k == ENOMEM`, `invalid_argument_k == EINVAL`, `key_not_found_k == ENOENT` and others, and is `[[nodiscard]]` on every mutating API.

### Making `std::set` Transactional

> `smashtable/transactional_std_store.hpp`

The baseline reference design, and the yardstick the tree containers are held against — it runs the same test suites they do.
Updates group into transactions that stage and roll back before committing, which is what lets inter-dependent updates span several collections.

Reads are consistent at the [Monotonic Atomic View](https://jepsen.io/consistency/models/monotonic-atomic-view) level, so a transaction never sees another's partial update.
That is weaker than strict serializability — there is no guarantee that every read within one transaction sees the same snapshot — and much cheaper than MVCC in both time and memory.

### Adelson-Velsky and Landis Trees

> `smashtable/basic_avl_tree.hpp`

Rarely called by its full name, the AVL tree is the simplest clean self-balancing binary search tree, from 1962.
`basic_avl_tree<value, comparator, allocator>` is an ordered collection in the shape of `std::set` or `std::map`.
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

### Transactional Trees

> `smashtable/transactional_store.hpp`

`transactional_binary_tree<Tree>` adds two-phase commit, watches and CAS on top of either tree.
`transactional_avl_set`, `transactional_avl_map`, `transactional_wb_set` and `transactional_wb_map` are the aliases you'll name directly.

### Hash Tables

> `smashtable/basic_hash_table.hpp`

`hash_set<key>` and `hash_map<key, value>` are open-addressing containers with linear probing.
They differ from `google::dense_hash_map` and `tsl::robin_map` by not reserving sentinel key values for tombstones, by unpacking keys and values into disjoint arrays, and by carrying two bits of metadata per slot rather than a byte.

Keys and values live in separate regions, so a lookup that only compares keys never pulls values into cache — which is most of them, since a miss touches no value at all.
Thirty-two slots share one 64-bit header holding two parallel bitmasks, encoding free, deleted, populated and locked in two bits each.
Sixty-four bits is the widest atomic every target supports, which is what lets a whole bucket header move in one instruction.

Growing repoints the key, value and header regions that every live slot reference has cached, so a table that can grow can never be read concurrently.
That used to be a sentence here that nothing enforced.
It is now two types, neither of which includes the other's header.
`basic_hash_table` grows, iterates and rehashes; `concurrent_hash_table` is pinned and lock-free.
What passes between them is the allocation itself — a `hash_storage` — so the hand-off is a move of a value rather than one table reaching into the other:

```cpp
_ = growable.reserve(1u << 20);                                          // pin the capacity first
auto pinned = concurrent_hash_map<key_t, value_t>::adopt(std::move(growable).release());

pinned.find(key, [](auto const &slot) noexcept { use(slot.value()); });
if (!pinned.emplace(key, value)) report_full();                          // a pinned table can fill up

auto compacted = hash_map<key_t, value_t>::adopt(std::move(pinned).release());   // iterators return
```

Every operation on the pinned table is lock-free: it takes a slot by setting both its bits with a `fetch_or` and releases it with a `fetch_xor` of the difference to the desired state, so neither path needs a compare-and-swap loop.
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

`SMASHTABLE_FILTER` selects a subset by substring, matched against `suite.name`:

```bash
SMASHTABLE_FILTER=transactional_consistency ./build/smashtable_test_avl_tree
```

Four binaries run the same suites over every container family — the `std::set` store, both trees, and both thread-safety wrappers — so a behavioural difference between them shows up as a failure rather than a surprise.

For the Python side:

```bash
pip install -e . && pytest test.py
```
