# SmashTable

SmashTable is a library of data-structures, with Atomic, Consistent, and Isolated transactions, similar to databases, but at the level of individual in-memory containers, without any Durability promises.
It's implemented in __C++ 20__ as header-only templates, and also exposed to __Python 3__ via raw CPython API.
At the lower C++ level it enables Systems Engineers to build safer concurrent software, avoiding exuberant costs of Multi-Version Concurrency Control (MVCC) in favor of cheaper mechanisms.
At the higher Python level, it simplifies multi-core programming for data-intensive scripts, by providing shared collections that work across GIL-free sub-interpreters.

![SmashTable Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/SmashTable.jpg?raw=true)

## Python Quick Start

To install SmashTable for Python, simply run:

```bash
pip install smashtable
```

Then, you can use it as follows to create a parallel Python application.
Let's say you've also installed `stringzilla` and want to find all of the unique words in a large text file using multiple CPU cores:

```python
import stringzilla as sz
import smashtable as st
import concurrent.futures

doc = sz.File("enwik9.txt")

```

## C++ Quick Start

To use SmashTable in C++, simply borrow the desired header files or install via CMake:

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

For system-wide installation, SmashTable supports standard CMake package discovery:

```bash
sudo cmake --install build_release
```

Then in your CMakeLists.txt:

```cmake
find_package(smashtable REQUIRED)
target_link_libraries(your_target PRIVATE smashtable::smashtable)
```

The library is designed to avoid exceptions entirely, no `throw` anywhere.
Mutation APIs (`upsert`, `insert`, `erase`, `reserve`, `clear`, etc.) return `status_t` to indicate success or failure.
All containers are compatible with custom memory allocators, and can be pre-allocated to reduce runtime memory allocations.
Basic containers (`basic_avl_tree`, `basic_hash_table`) provide full STL-style iterator support with bidirectional traversal.
Transactional containers don't support iterators due to complexity of maintaining validity in presence of concurrent updates:

- Query APIs like `find()`, `lower_bound()`, and `upper_bound()` return `void` and use 2 noexcept callbacks for found/missing cases.
- Range operations like `equal_range()` and `sample_range()` return `void` and use a single noexcept callback invoked for each element.
- Callbacks must be `noexcept` as they're invoked directly without exception wrapping.

Most APIs are similar to STL containers:

```cpp
#include <smashtable/transactional_std_store.hpp>

namespace st = ashvardanian::smashtable;

int main() {
    using pair_t = association<std::string_view, int>;  // cheaper than `std::pair`
    using map_t = st::transactional_std_store<pair_t>;  // builds on top of `std::map`
    auto map = *map_t::make();                          // instead of constructors to return optionals
    _ = map.reserve(100);                               // optionally reserve space
    
    // STL-style operations outside of transactions
    _ = map.clear(); _ = map.merge(another_map);
    _ = map.insert({"alice", 2}); _ = map.insert_or_assign({"bob", 2}); _ = map.erase("alice");
    _ = map.insert({"carol", 3});

    // Some operations will look different from STL
    _ = map.upsert({"dave", 4});                        // "upsert" = insert or update
    _ = map.insert_if_missing({"carol", 5});            // similar to `try_emplace` - skips if exists
    map.find("alice",                                   // "find" won't return iterators!
        [](pair_t const &existing) noexcept { ... },    // scoped processing of a found element
        []() noexcept { ... });                         // handle the missing case
    map.lower_bound(..., [](auto) noexcept { }, []() noexcept { });
    map.upper_bound(..., [](auto) noexcept { }, []() noexcept { });
    map.equal_range(..., [](auto const &) noexcept { });     // one key, invokes callback for matches

    // Transactions are the crucial part
    auto t1 = *map.transaction();                       // can't copy or move transactions
    _ = t1.reserve(5);                                  // optionally reserve space for 5 updates
    _ = t1.insert("al", 1); _ = t1.upsert("bob", 2);    // update some key-value pairs
    _ = t1.watch("carol");                              // transaction will fail if "carol" is later modified

    // Even within a single thread, many transactions can be active simultaneously
    auto t2 = *map.transaction();
    _ = t2.insert("carol", 5); _ = t2.erase("dave");

    // Committing the transactions can happen in any order
    _ = t2.commit();                                    // will succeed
    _ = t1.commit();                                    // must fail, as "carol" was modified by `t2`

    return 0;
}
```

## Why Do You Need The Python Library?

Python is famous for its Global Interpreter Lock (GIL) and the frequency of Twitter flame wars about it.
More recently, sub-interpreters have been added to CPython to enable multi-core parallelism without the GIL.
However, sub-interpreters can't share Python objects directly, as each interpreter has its own memory space and object management.
The only [recommended sharing structures](https://docs.python.org/3/library/concurrent.interpreters.html#communication-between-interpreters) are the `memoryview` and `concurrent.interpreters.Queue`, which are limited in functionality and performance.
One way to address this could be to:

- allow read-only access to parent interpreter objects from sub-interpreters.
- allow returning sub-interpreter created objects to the parent interpreter on completion.

This, however, generally requires serializing and deserializing objects, which is expensive and error-prone... especially if you `pickle`!
To address this gap, SmashTable provides shared associative and set containers of trivially copyable types, where keys and values of basic types (integers, floats, strings) can be shared directly between sub-interpreters without serialization.
Create it once in the parent interpreter, and use it from multiple sub-interpreters concurrently!

## Why Do You Need The C++ Library?

Like any library, the C++ standard library has many loose ends when it comes to consistency.
For example, the `std::set` container provides an API to insert a range of elements - `insert(first, last)`.
But if an allocation failure happens halfway through the insertion, some elements will have already been inserted, while others won't.

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

Compiled with Clang++ 21 and G++ 15, this program produces:

```
std::bad_alloc after 3 allocations
set contains 3 elements:
0 1 2 
```

The standard doesn't define which exceptions can be thrown by `insert(first, last)`, but in practice, it's usually `std::bad_alloc` from memory allocation failures.


## Collections

SmashTable implements several collections with different consistency and concurrency models.
Same tree structures and hash tables can be used for both "sets" and associative "maps", storing key-comparable key-value pairs.
But before enumerating them, let's constrain our terminology:

- "Concurrency" doesn't imply thread-safety or multi-threaded access... it can be simultaneous operations on the same thread.
- "Consistency" doesn't imply "strict serializability" or "linearizability"... [weaker consistency levels exist too](https://jepsen.io/consistency/models).

All of the header files are grouped as follows:

- `smashtable/basic_*.hpp` - "use at your own risk" building blocks
- `smashtable/transactional_*.hpp` - bringing 2-phase commit semantics
- `smashtable/*_collection.hpp` - composable wrappers to reduce contention

In more detail:

```bash
  basic_*               → Core data structures w/out transactions
    ├─ basic_vector<T, Alloc>
    ├─ basic_avl_tree<T, Comparator, Alloc>
    └─ basic_hash_table<T, Hash, KeyEqual, Alloc>

  transactional_*       → Add 2-phase commit + watch and CAS semantics
    ├─ transactional_avl_tree<T, Comparator, Alloc>
    └─ transactional_std_store<T, Comparator, Alloc>

  *_collection          → Thread-safety wrappers
    ├─ locked_collection<Collection, Mutex>
    └─ partitioned_collection<Collection, Hash, Mutex, PartsCount>
```

All collections support custom memory allocators, and avoid exceptions entirely.
Mutation APIs return `status_t` to indicate success or failure.
The `status_t` that can have non-`success_k` values include `out_of_memory_heap_k == ENOMEM`, `invalid_argument_k == EINVAL` and others marked `[[nodiscard]]`.
Read-only operations never fail, and use `noexcept` callbacks to return results instead of throwing exceptions.

### Making `std::set` Transactional

> Refers to `smashtable/transactional_std_store.hpp`.

The `std::set` was used to create a baseline reference design for the SmashTable functionality.
Beyond the underlying `std::set` and similar `std::map` containers, it adds "transactions".
Updates to the collection can be grouped together, and either all of them succeed, or none of them do, providing "Atomicity".
Those transactions can be "staged" and "rolled back" before being "committed", enabling inter-dependent updates across many such collections.
Read consistency is also provided, at the "Monotonic Atomic View" isolation level, so that transactions won't see partial updates from other concurrent transactions.
It's not as strong as "Strict Serializability", as we can't guarantee, that all of the reads happening within a transaction see the same snapshot of the collection.
On the bright side, it's much faster (in terms of runtime) and cheaper (in terms of memory consumption) than MVCC-based approaches.

### Adelson-Velsky and Landis Trees

#### Basic AVL Trees

> Refers to `smashtable/basic_avl_tree.hpp`.

Rarely referred to by the full name, the AVL tree is one of the simplest and cleanest self-balancing binary search tree structures, proposed in 1962.
The basic AVL tree template - `basic_avl_tree<entry, comparator, allocator>` provides a baseline ordered collection, similar to `std::set` or `std::map`.
The standard implementations typically use Red-Black trees, which are slightly more complex, but provide similar performance.
The AVL tree is more rigidly balanced, providing faster lookups at the cost of slightly slower insertions and deletions.

The `basic_avl_tree` provides STL-compatible iterators for traversal, but some APIs differ to accommodate exception-free error reporting:

- `insert()` returns `std::pair<iterator, bool>` (STL-compatible)
- `erase(iterator)` returns `erase_result_t { iterator next; status_t status; }` instead of just iterator
- Range `insert(first, last)` returns `status_t` instead of `void`
- Hint-based `insert(hint, value)` and `emplace_hint()` are explicitly deleted (AVL trees don't benefit from hints)

#### Transactional AVL Trees

> Refers to `smashtable/transactional_avl_tree.hpp`.

The higher-level `transactional_avl_tree<entry, comparator, allocator>` template builds on top of the basic AVL tree, adding transactional semantics similar to those described for `transactional_sset`.
Like the `transactional_std_store`, it provides Atomicity and Monotonic Atomic View isolation for grouped updates.

### Hash Tables

#### Somewhat Thread-Safe Hash Tables

> Refers to `smashtable/basic_hash_table.hpp`.

The `basic_hash_table<entry, hash, key_equal, allocator>` template implements a lock-free hash table using open addressing with constant probing - step size of 1.
There is no shortage of hash-table designs, like the `google::dense_hash_map` or `tsl::robin_map`, but there are several noticeable improvement areas.
Especially if you can optimize for around a certain hash function in StringZilla or parallel access patterns in ForkUnion.

Unlike Google's hash-tables:

- SmashTable doesn't reserve special key values to mark tombstones or empty slots, which simplifies usage with arbitrary key types.
- SmashTable unpacks key-value pairs into disjoint arrays to achieve higher density, especially for abnormally sized entries, like the 5-byte wide `uint40_t` in USearch.
- SmashTable provides Lock-Free mechanisms for concurrent reads and writes, avoiding Compare-And-Swap (CAS) loops and Mutexes for most operations, using only 2 bits of metadata per slot.
- SmashTable provides additional APIs for faster DBMS-style operations, like joins, merges, and random sampling.

#### Optimizing for String Keys

StringZilla's hash function is one of the fastest and highest quality non-cryptographic hash functions available.
On AVX-512 capable CPUs it has a fast path for short strings under 16 bytes, which is a common case for hash table keys in many applications.
In the spirit of co-design, SmashTable's hash table provides a specialization optimized for StringZilla's strings, using 2 separate tables under the hood:

1. A table for small keys (up to 15 bytes + length encoded in the last byte), storing them inline within the hash table slots in 16 bytes of space.
2. A table for large keys (16 bytes and above), storing a pointer to a null-terminated string and the 64-bit hash, also taking just 16 bytes together.

Thanks to that, probing within the hash-table can immediately check multiple strings for equality.
On AVX-512 machines, for small strings, we load and compare 4x 16-byte buffers at once.
For longer strings we can "gather" the pointers to the candidate strings and compare them in parallel, as long as we don't saturate the Line Fill Buffer (LFB).

- Intel's Sandy Bridge through Ice Lake can only handle 10 concurrent L1D cache misses.
- AMD's Zen 3/4 can handle 16-20 L1D cache misses, thus performing better with pointer-heavy workloads.

#### Transactional Hash Tables

> Refers to `smashtable/transactional_hash_table.hpp`.

The common design for parallel hash-tables is to take multiple independent serial hash-tables, wrap each with a mutex, and shard the keys across them.
With 4x or 16x the number of shards compared to CPU cores, this approach can work well for many workloads, especially if you interleave the buckets between NUMA nodes to flatten memory access latencies.
To further reduce contention, one will increase the number of locks.

