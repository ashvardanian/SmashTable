/**
 *  @file test/shared.cpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Tests for the primitives every container shares - the @c expected result type, the
 *      default allocator, the reader-writer lock, and the multi-store transaction group.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cerrno>  // `EIO`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `SIZE_MAX`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic`
#include <functional>  // `std::equal_to`, `std::hash`, `std::less`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator`, `std::unique_ptr`
#include <new>         // `std::nothrow_t`, `std::align_val_t`
#include <span>        // `std::span`
#include <string_view> // `std::string_view`
#include <thread>      // `std::thread`
#include <utility>     // `std::pair`
#include <vector>      // `std::vector`

#include <forkunion.hpp>

#include <smashtable/atomic_hash_table.hpp>
#include <smashtable/basic_hash_table.hpp>
#include <smashtable/reference_store.hpp>
#include <smashtable/shared.hpp>
#include <smashtable/hash_layout.hpp>
#include <smashtable/monotonic_store.hpp>
#include <smashtable/partitioned_store.hpp>
#include <smashtable/snapshot_store.hpp>

#include "harness.hpp"

namespace fu = ashvardanian::forkunion;

namespace ashvardanian::smashtable::test {

#pragma region Instrumented Value

/** A value that tallies every construction and destruction, so a leak or a double free shows up as
 *  an imbalance rather than as a sanitizer report alone. */
struct counted_t {
    static inline std::size_t constructions = 0;
    static inline std::size_t destructions = 0;

    int payload {0};

    counted_t() noexcept { ++constructions; }
    explicit counted_t(int value) noexcept : payload(value) { ++constructions; }
    counted_t(counted_t const &other) noexcept : payload(other.payload) { ++constructions; }
    counted_t(counted_t &&other) noexcept : payload(other.payload) { ++constructions; }
    counted_t &operator=(counted_t const &other) noexcept {
        payload = other.payload;
        return *this;
    }
    counted_t &operator=(counted_t &&other) noexcept {
        payload = other.payload;
        return *this;
    }
    ~counted_t() noexcept { ++destructions; }

    static std::size_t alive() noexcept { return constructions - destructions; }
};

/** A value with no copy at all, which is the shape a transaction has. */
struct move_only_t {
    int payload {0};

    move_only_t() noexcept = default;
    explicit move_only_t(int value) noexcept : payload(value) {}
    move_only_t(move_only_t &&) noexcept = default;
    move_only_t &operator=(move_only_t &&) noexcept = default;
    move_only_t(move_only_t const &) = delete;
    move_only_t &operator=(move_only_t const &) = delete;
};

#pragma endregion Instrumented Value

#pragma region Expected Tests

/** Every way of constructing an @c expected, checked for a discriminant that matches storage. */
static void test_expected_construction_matrix() {
    std::size_t const alive_before = counted_t::alive();

    // A default-constructed result reports failure and holds nothing.
    {
        expected<counted_t> const result;
        expect_true(!result);
        expect_true(!result.has_value());
        expect_equal(result.status(), unknown_k);
    }
    expect_equal(counted_t::alive(), alive_before);

    // A reason with no value behind it constructs nothing.
    {
        std::size_t const constructions_before = counted_t::constructions;
        expected<counted_t> const result {out_of_memory_heap_k};
        expect_true(!result);
        expect_equal(result.status(), out_of_memory_heap_k);
        expect_equal(counted_t::constructions, constructions_before);
    }
    expect_equal(counted_t::alive(), alive_before);

    // A success handed no value cannot claim to have one, or the destructor would run over
    // storage that was never constructed.
    {
        expected<counted_t> const result {success_k};
        expect_true(!result);
        expect_true(!result.has_value());
        expect_equal(result.status(), unknown_k);
    }
    expect_equal(counted_t::alive(), alive_before);

    // A value moved into a success is there, and dies exactly once.
    {
        expected<counted_t> const result {counted_t {7}, success_k};
        if (!expect_equal(result.status(), success_k)) return;
        expect_equal(result->payload, 7);
        expect_equal(counted_t::alive(), alive_before + 1);
    }
    expect_equal(counted_t::alive(), alive_before);

    // A value handed to a failure is left to its owner, so nothing outlives the statement.
    {
        expected<counted_t> const result {counted_t {9}, out_of_memory_heap_k};
        expect_true(!result);
        expect_equal(result.status(), out_of_memory_heap_k);
        expect_equal(counted_t::alive(), alive_before);
    }
    expect_equal(counted_t::alive(), alive_before);

    // The same two cases through the copying constructor.
    {
        counted_t const source {11};
        expected<counted_t> const kept {source, success_k};
        if (!expect_equal(kept.status(), success_k)) return;
        expect_equal(kept->payload, 11);
        expect_equal(counted_t::alive(), alive_before + 2);

        expected<counted_t> const dropped {source, key_not_found_k};
        expect_true(!dropped);
        expect_equal(counted_t::alive(), alive_before + 2);
    }
    expect_equal(counted_t::alive(), alive_before);

    // The default status is success, so a bare value is a value.
    {
        expected<counted_t> const result {counted_t {13}};
        if (!expect_equal(result.status(), success_k)) return;
        expect_equal(result->payload, 13);
    }
    expect_equal(counted_t::alive(), alive_before);
}

/** Moves between all four combinations of full and empty, checked for balance. */
static void test_expected_moves_stay_balanced() {
    std::size_t const alive_before = counted_t::alive();

    // A full source, into a fresh object.
    {
        expected<counted_t> source {counted_t {1}, success_k};
        expected<counted_t> const target {std::move(source)};
        if (!expect_equal(target.status(), success_k)) return;
        expect_equal(target->payload, 1);
        expect_equal(counted_t::alive(), alive_before + 2); // ? The moved-from value is still there
    }
    expect_equal(counted_t::alive(), alive_before);

    // An empty source, into a fresh object.
    {
        expected<counted_t> source {invalid_argument_k};
        expected<counted_t> const target {std::move(source)};
        expect_true(!target);
        expect_equal(target.status(), invalid_argument_k);
        expect_equal(counted_t::alive(), alive_before);
    }
    expect_equal(counted_t::alive(), alive_before);

    // Full over full, full over empty, empty over full, empty over empty.
    {
        expected<counted_t> target {counted_t {2}, success_k};
        expected<counted_t> full_source {counted_t {3}, success_k};
        target = std::move(full_source);
        if (!expect_equal(target.status(), success_k)) return;
        expect_equal(target->payload, 3);

        expected<counted_t> empty_source {consistency_k};
        target = std::move(empty_source);
        expect_true(!target);
        expect_equal(target.status(), consistency_k);
        expect_equal(counted_t::alive(), alive_before + 1); // ? Only `full_source` still holds one

        expected<counted_t> another_full {counted_t {4}, success_k};
        target = std::move(another_full);
        if (!expect_equal(target.status(), success_k)) return;
        expect_equal(target->payload, 4);

        expected<counted_t> another_empty {key_already_exists_k};
        target = std::move(another_empty);
        expect_true(!target);

        // Self-assignment must not destroy anything, reached through an alias so the compiler
        // cannot mistake the deliberate case for the accidental one.
        expected<counted_t> &alias = target;
        target = std::move(alias);
        expect_true(!target);
    }
    expect_equal(counted_t::alive(), alive_before);
}

/** The tuple protocol behind `auto [value, status]`, on both paths. */
static void test_expected_decomposes() {
    std::size_t const alive_before = counted_t::alive();
    {
        auto [value, status] = expected<counted_t> {counted_t {5}, success_k};
        expect_equal(status, success_k);
        expect_equal(value.payload, 5);
    }
    {
        auto [value, status] = expected<counted_t> {out_of_memory_heap_k};
        expect_equal(status, out_of_memory_heap_k);
        expect_equal(value.payload, 0); // ? A failure decomposes into a default, never into garbage
    }
    expect_equal(counted_t::alive(), alive_before);
}

/** A move-only payload, which is the case @c std::optional could not cover. */
static void test_expected_carries_move_only() {
    expected<move_only_t> held {move_only_t {21}, success_k};
    if (!expect_equal(held.status(), success_k)) return;
    expect_equal((*held).payload, 21);

    expected<move_only_t> const taken {std::move(held)};
    if (!expect_equal(taken.status(), success_k)) return;
    expect_equal(taken->payload, 21);

    expected<move_only_t> const refused {operation_would_block_k};
    expect_true(!refused);
    expect_equal(refused.status(), operation_would_block_k);
}

#pragma endregion Expected Tests

#pragma region Allocator Tests

/** A byte count that would wrap must be refused rather than served a tiny block. */
static void test_allocator_refuses_overflowing_counts() {
    default_allocator<std::uint64_t> wide_allocator;

    // Exactly the largest representable byte count, and the first count past it.
    std::size_t const largest = SIZE_MAX / sizeof(std::uint64_t);
    expect_equal(wide_allocator.allocate(largest + 1), nullptr);
    expect_equal(wide_allocator.allocate(SIZE_MAX), nullptr);
    expect_equal(wide_allocator.allocate(SIZE_MAX / 2), nullptr);

    // A count that fits still allocates and frees.
    std::uint64_t *const block = wide_allocator.allocate(16);
    expect_true(block != nullptr);
    block[0] = 1, block[15] = 2;
    expect_equal(block[0] + block[15], 3u);
    wide_allocator.deallocate(block, 16);

    // A single-byte element has no multiplication to overflow, and must keep working.
    default_allocator<char> narrow_allocator;
    char *const bytes = narrow_allocator.allocate(32);
    expect_true(bytes != nullptr);
    narrow_allocator.deallocate(bytes, 32);
}

#pragma endregion Allocator Tests

#pragma region Shared Mutex Tests

/** The mutex with its word owned through ForkUnion's reference for this build and its waits handed
 *  to ForkUnion's waiter, which no sanitizer sees into. */
using forkunion_mutex_t = spin_shared_mutex<fu::preferred_yield_t, fu::preferred_atomic_ref>;
using smashtable::spin_shared_mutex_t;

static_assert(an_extended_atomic_ref<fu::preferred_atomic_ref<std::uint64_t>, std::uint64_t>,
              "a reference short of the concept would pass as a ForkUnion case over the fallback loops");
static_assert(an_extended_atomic_ref<fu::preferred_atomic_ref<std::uint32_t>, std::uint32_t>,
              "the mutex word is narrower than a stamp, and must reach the same verbs");

/** Readers are admitted up to half the tally, and a reader refused at a full one takes its unit
 *  back, so the word reads idle once everyone has left. */
template <typename mutex_type_>
static void test_shared_mutex_keeps_room_above_its_readers() {
    mutex_type_ mutex;
    constexpr std::uint32_t readers_limit_k = 0xFFFFu >> 1;
    std::uint32_t admitted = 0;
    while (admitted != readers_limit_k && mutex.try_lock_shared()) ++admitted;
    expect_equal(admitted, readers_limit_k);
    expect_true(!mutex.try_lock_shared(), "the upper half of the tally is room for refused units");

    std::atomic<bool> entered {false};
    std::thread late([&]() noexcept {
        mutex.lock_shared();
        entered.store(true);
        mutex.unlock_shared();
    });
    for (; admitted != 0; --admitted) mutex.unlock_shared();
    late.join();
    expect_true(entered.load(), "the reader turned away at a full tally gets in once it drains");
    expect_true(mutex.try_lock(), "every refused unit was taken back");
    mutex.unlock();
}

/** Exclusion holds: a writer never overlaps a reader or another writer. */
template <typename mutex_type_>
static void test_shared_mutex_excludes() {
    mutex_type_ mutex;
    std::atomic<int> readers_inside {0};
    std::atomic<int> writers_inside {0};
    std::atomic<bool> violated {false};
    std::size_t guarded_left = 0, guarded_right = 0;

    constexpr std::size_t readers_count_k = 4;
    constexpr std::size_t writes_per_writer_k = 2000;
    constexpr std::size_t writers_count_k = 2;

    std::vector<std::thread> threads;
    for (std::size_t writer = 0; writer != writers_count_k; ++writer)
        threads.emplace_back([&]() noexcept {
            for (std::size_t iteration = 0; iteration != writes_per_writer_k; ++iteration) {
                unique_lock<mutex_type_> guard {mutex};
                if (writers_inside.fetch_add(1) != 0 || readers_inside.load() != 0) violated.store(true);
                ++guarded_left;
                ++guarded_right;
                if (writers_inside.fetch_sub(1) != 1) violated.store(true);
            }
        });

    std::atomic<bool> reading {true};
    for (std::size_t reader = 0; reader != readers_count_k; ++reader)
        threads.emplace_back([&]() noexcept {
            while (reading.load(std::memory_order_relaxed)) {
                shared_lock<mutex_type_> guard {mutex};
                readers_inside.fetch_add(1);
                if (writers_inside.load() != 0) violated.store(true);
                if (guarded_left != guarded_right) violated.store(true);
                readers_inside.fetch_sub(1);
            }
        });

    for (std::size_t writer = 0; writer != writers_count_k; ++writer) threads[writer].join();
    reading.store(false);
    for (std::size_t reader = writers_count_k; reader != threads.size(); ++reader) threads[reader].join();

    expect_true(!violated.load());
    expect_equal(guarded_left, writers_count_k * writes_per_writer_k);
    expect_equal(guarded_right, guarded_left);
}

/** Several writers under a steady read load all finish, which is what the waiting tally buys. One
 *  writer taking the lock must not erase the intent of the writers still queued, or a reader stream
 *  slips back in and parks them forever. */
template <typename mutex_type_>
static void test_shared_mutex_admits_every_writer() {
    // The readers re-enter fast enough that their count effectively never reaches zero, so a writer
    // only gets in by turning them away; the budget is fifty times the deepest run seen.
    constexpr std::size_t writers_count_k = 3;
    constexpr std::size_t readers_count_k = 10;
    constexpr std::size_t acquisitions_per_writer_k = 30;
    constexpr std::size_t acquisitions_per_reader_k = 250000;
    constexpr int reader_hold_steps_k = 20000;
    constexpr int writer_hold_steps_k = 2000;

    mutex_type_ mutex;
    std::atomic<bool> reading {true};
    std::atomic<std::size_t> readers_announced {0};
    std::atomic<std::size_t> readers_spent {0};

    std::vector<std::thread> readers;
    for (std::size_t reader = 0; reader != readers_count_k; ++reader)
        readers.emplace_back([&]() noexcept {
            bool announced = false;
            volatile unsigned hold = 0;
            std::size_t round = 0;
            for (; round < acquisitions_per_reader_k && reading.load(std::memory_order_relaxed); ++round) {
                shared_lock<mutex_type_> guard {mutex};
                if (!announced) {
                    announced = true;
                    readers_announced.fetch_add(1, std::memory_order_relaxed);
                }
                // Volatile, so the optimizer cannot drop the hold.
                for (int step = 0; step != reader_hold_steps_k; ++step) hold = hold + 1;
            }
            if (round == acquisitions_per_reader_k) readers_spent.fetch_add(1, std::memory_order_relaxed);
        });

    // Every reader has held the lock once before the first writer exists, so each writer arrives at
    // a lock the readers are already streaming through.
    expect_true(spin_until([&] { return readers_announced.load(std::memory_order_relaxed) >= readers_count_k; }));

    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer != writers_count_k; ++writer)
        writers.emplace_back([&]() noexcept {
            volatile unsigned hold = 0;
            for (std::size_t iteration = 0; iteration != acquisitions_per_writer_k; ++iteration) {
                unique_lock<mutex_type_> guard {mutex};
                for (int step = 0; step != writer_hold_steps_k; ++step) hold = hold + 1;
            }
        });

    for (auto &writer : writers) writer.join();
    reading.store(false, std::memory_order_relaxed);
    for (auto &reader : readers) reader.join();

    // A parked writer outlives the readers, so starvation reads as a reader that ran out of budget.
    expect_equal(readers_spent.load(), std::size_t {0}, "a reader spent its budget while a writer was parked");
}

#pragma endregion Shared Mutex Tests

#pragma region Waiting Policy Tests

/**
 *  @brief A waiting policy handing the core back to the scheduler, tallying what it was asked for.
 *
 *  Names nothing but the one pause every spin path needs, which is the shape a hardware hint takes:
 *  the parking a mutex also wants comes from @c standard_waiting_policy wrapped around this. Takes
 *  the watched word only by address, as a monitored wait arming its line must, so a spin path
 *  handing over a reference fails to compile here. The tally is static because a policy is
 *  default-constructed inside whatever spins on it, so there is nowhere to hand a counter in, and
 *  one test runs at a time.
 */
struct yielding_waiting_policy_t {

    /** How many single pauses the spin paths have asked for. */
    static inline std::atomic<std::size_t> pauses {0};

    /** Gives the core up for a scheduling quantum rather than burning it on a retry. */
    template <typename value_type_, typename attempts_type_>
    void operator()(value_type_ const *, value_type_, attempts_type_) const noexcept {
        pauses.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }
};

/** The mutex parks, so a pause-only policy reaches it through the standard parking wrapper. */
using yielding_mutex_t = spin_shared_mutex<standard_waiting_policy<yielding_waiting_policy_t>>;

/** The slot lock only ever spins, so a pause-only policy goes into the pinned table directly. */
using yielding_table_t =
    atomic_hash_set<std::uint64_t, default_hash_t, equal_to_t, default_allocator_t, yielding_waiting_policy_t>;

static_assert(a_parking_waiting_policy<standard_waiting_policy_t, std::uint32_t *, std::uint32_t>,
              "the default must satisfy the concept the mutex constrains its parameter on");
static_assert(a_waiting_policy<yielding_waiting_policy_t, std::uint64_t const *, std::uint64_t>,
              "the slot lock watches a word owned through atomic_ref rather than an atomic object");
static_assert(a_waiting_policy<yielding_waiting_policy_t, std::uint32_t *, std::uint32_t>,
              "the mutex hands a waiter the address of its word, as it does the slot lock's");
static_assert(sizeof(yielding_mutex_t) == sizeof(spin_shared_mutex_t), "a stateless policy costs the mutex no storage");
static_assert(sizeof(yielding_table_t) == sizeof(atomic_hash_set<std::uint64_t>),
              "a stateless policy costs the pinned table no storage");

/** The tallies every @c counting_extended_ref shares, whatever word it wraps, because the pinned
 *  table posts its population through a narrower offset type than its keys. */
struct extended_ref_tallies {

    /** How many bounded adds the mutex asked for, which is once per acquire attempt. */
    static inline std::atomic<std::size_t> bounded_adds {0};

    /** How many writes it posted without reading anything back, once per writer release. */
    static inline std::atomic<std::size_t> posted_writes {0};

    /** How many xors it posted, which is once per slot the pinned table lets go. */
    static inline std::atomic<std::size_t> posted_flips {0};

    /** How many single bits it set or cleared and read back, once per claim attempt. */
    static inline std::atomic<std::size_t> bit_claims {0};
};

/**
 *  @brief A reference offering the extended shapes, counting which one the mutex reached for.
 *
 *  Every operation lands on @c std::atomic_ref underneath, so the lock still behaves; what the
 *  tallies prove is which spelling the mutex chose, not that a bounded add is faster here.
 */
template <typename value_type_>
struct counting_extended_ref : extended_ref_tallies {

    /** The word this reference owns for the span of one operation. */
    value_type_ *word;

    explicit counting_extended_ref(value_type_ &owned) noexcept : word(&owned) {}

    [[nodiscard]] value_type_ load(std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).load(order);
    }
    [[nodiscard]] value_type_ fetch_add(value_type_ addend, std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).fetch_add(addend, order);
    }
    [[nodiscard]] value_type_ fetch_sub(value_type_ subtrahend, std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).fetch_sub(subtrahend, order);
    }
    [[nodiscard]] value_type_ fetch_or(value_type_ bits, std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).fetch_or(bits, order);
    }
    bool compare_exchange_weak(value_type_ &expected, value_type_ wanted, std::memory_order success,
                               std::memory_order failure) const noexcept {
        return std::atomic_ref<value_type_>(*word).compare_exchange_weak(expected, wanted, success, failure);
    }
    bool compare_exchange_strong(value_type_ &expected, value_type_ wanted, std::memory_order success,
                                 std::memory_order failure) const noexcept {
        return std::atomic_ref<value_type_>(*word).compare_exchange_strong(expected, wanted, success, failure);
    }

    void store_add(value_type_ addend, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_>(*word).fetch_add(addend, order);
    }
    void store_sub(value_type_ subtrahend, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_>(*word).fetch_sub(subtrahend, order);
    }
    void set_bits(value_type_ bits, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_>(*word).fetch_or(bits, order);
    }
    void clear_bits(value_type_ bits, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_>(*word).fetch_and(static_cast<value_type_>(~bits), order);
    }
    [[nodiscard]] value_type_ fetch_and(value_type_ mask, std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).fetch_and(mask, order);
    }
    [[nodiscard]] value_type_ fetch_xor(value_type_ bits, std::memory_order order) const noexcept {
        return std::atomic_ref<value_type_>(*word).fetch_xor(bits, order);
    }
    [[nodiscard]] value_type_ fetch_min(value_type_ ceiling, std::memory_order order) const noexcept {
        std::atomic_ref<value_type_> reference(*word);
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed > ceiling &&
               !reference.compare_exchange_weak(observed, ceiling, order, std::memory_order_acquire)) {}
        return observed;
    }
    [[nodiscard]] value_type_ fetch_max(value_type_ floor, std::memory_order order) const noexcept {
        std::atomic_ref<value_type_> reference(*word);
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed < floor &&
               !reference.compare_exchange_weak(observed, floor, order, std::memory_order_acquire)) {}
        return observed;
    }
    void store_min(value_type_ ceiling, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        [[maybe_unused]] value_type_ const before = fetch_min(ceiling, order);
    }
    void store_max(value_type_ floor, std::memory_order order) const noexcept {
        posted_writes.fetch_add(1, std::memory_order_relaxed);
        [[maybe_unused]] value_type_ const before = fetch_max(floor, order);
    }
    void flip_bits(value_type_ bits, std::memory_order order) const noexcept {
        posted_flips.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_>(*word).fetch_xor(bits, order);
    }
    [[nodiscard]] bool fetch_test_and_set_bit(unsigned bit, std::memory_order order) const noexcept {
        bit_claims.fetch_add(1, std::memory_order_relaxed);
        value_type_ const mask = static_cast<value_type_>(value_type_ {1} << bit);
        return (std::atomic_ref<value_type_>(*word).fetch_or(mask, order) & mask) != 0;
    }
    [[nodiscard]] bool fetch_test_and_reset_bit(unsigned bit, std::memory_order order) const noexcept {
        bit_claims.fetch_add(1, std::memory_order_relaxed);
        value_type_ const mask = static_cast<value_type_>(value_type_ {1} << bit);
        return (std::atomic_ref<value_type_>(*word).fetch_and(static_cast<value_type_>(~mask), order) & mask) != 0;
    }

    [[nodiscard]] value_type_ fetch_add_if_at_most(value_type_ addend, value_type_ limit,
                                                   std::memory_order order) const noexcept {
        bounded_adds.fetch_add(1, std::memory_order_relaxed);
        std::atomic_ref<value_type_> reference(*word);
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed <= limit && static_cast<value_type_>(limit - observed) >= addend &&
               !reference.compare_exchange_weak(observed, static_cast<value_type_>(observed + addend), order,
                                                std::memory_order_acquire)) {}
        return observed;
    }
    [[nodiscard]] value_type_ fetch_sub_if_at_least(value_type_ subtrahend, value_type_ floor,
                                                    std::memory_order order) const noexcept {
        std::atomic_ref<value_type_> reference(*word);
        value_type_ observed = reference.load(std::memory_order_acquire);
        while (observed >= floor && static_cast<value_type_>(observed - floor) >= subtrahend &&
               !reference.compare_exchange_weak(observed, static_cast<value_type_>(observed - subtrahend), order,
                                                std::memory_order_acquire)) {}
        return observed;
    }
};

/** The same mutex, owning its word through a reference that spells the extended shapes. */
using extended_mutex_t = spin_shared_mutex<standard_waiting_policy_t, counting_extended_ref>;

/** The pinned table over the same reference, so its slot lock and counters are seen going through
 *  it. */
using extended_table_t = atomic_hash_set<std::uint64_t, default_hash_t, equal_to_t, default_allocator_t,
                                         bare_waiting_policy_t, counting_extended_ref>;

static_assert(an_extended_atomic_ref<counting_extended_ref<std::uint32_t>, std::uint32_t>,
              "the fixture has to satisfy the concept, or the test proves nothing about dispatch");
static_assert(!an_extended_atomic_ref<atomic_ref<std::uint32_t>, std::uint32_t>,
              "and the default must not, or there is no fallback to fall back to");
static_assert(sizeof(extended_mutex_t) == sizeof(spin_shared_mutex_t),
              "the reference is chosen at compile time and costs the mutex no storage");
static_assert(sizeof(extended_table_t) == sizeof(atomic_hash_set<std::uint64_t>),
              "and costs the pinned table none either");

/** A reference spelling the extended shapes is the one the mutex drives its bounded acquires
 *  through, which are the ones that may be refused. */
static void test_extended_atomics_take_the_bounded_path() {

    counting_extended_ref<std::uint32_t>::bounded_adds.store(0);
    counting_extended_ref<std::uint32_t>::posted_writes.store(0);

    extended_mutex_t mutex;
    mutex.lock();
    expect_equal(counting_extended_ref<std::uint32_t>::bounded_adds.load(), std::size_t {0},
                 "a writer takes an idle word with one exchange, which no bound decides");
    expect_equal(mutex.try_lock_shared(), false, "a reader is refused while a writer holds");
    expect_equal(counting_extended_ref<std::uint32_t>::bounded_adds.load(), std::size_t {1},
                 "a reader that may be refused asks with the bounded add, which writes nothing when it loses");
    mutex.unlock();
    expect_equal(counting_extended_ref<std::uint32_t>::posted_writes.load(), std::size_t {1},
                 "and drops the held bit by posting it, since nothing reads what the word held");

    mutex.lock_shared();
    mutex.lock_shared();
    expect_equal(counting_extended_ref<std::uint32_t>::bounded_adds.load(), std::size_t {1},
                 "an arriving reader's first attempt is one blind add");
    expect_equal(mutex.try_lock(), false, "a writer is refused while readers hold");
    mutex.unlock_shared();
    mutex.unlock_shared();

    expect_equal(mutex.try_lock(), true, "and admitted once the last of them leaves");
    mutex.unlock();
}

/** The pinned table lets every slot go with one posted xor, and moves counters by posted adds. */
static void test_extended_atomics_reach_the_pinned_table() {

    counting_extended_ref<std::uint64_t>::posted_flips.store(0);
    counting_extended_ref<std::uint64_t>::posted_writes.store(0);

    auto [table, status] = hash_set<std::uint64_t>::make(std::size_t {64});
    if (!expect_equal(status, success_k, "the table to be pinned must build")) return;
    extended_table_t pinned = extended_table_t::adopt(std::move(table).release());

    expect_equal(pinned.emplace(std::uint64_t {42}), success_k);
    expect_equal(counting_extended_ref<std::uint64_t>::posted_flips.load(), std::size_t {1},
                 "an emplace takes one free slot and lets it go by posting the xor");
    expect_equal(counting_extended_ref<std::uint64_t>::posted_writes.load(), std::size_t {1},
                 "and counts the population with one posted add");

    expect_equal(pinned.contains(std::uint64_t {42}), true);
    expect_equal(counting_extended_ref<std::uint64_t>::posted_flips.load(), std::size_t {2},
                 "a probe that finds its key locks that slot alone");

    expect_equal(pinned.erase(std::uint64_t {42}), success_k);
    expect_equal(counting_extended_ref<std::uint64_t>::posted_flips.load(), std::size_t {3});
    expect_equal(counting_extended_ref<std::uint64_t>::posted_writes.load(), std::size_t {3},
                 "an erase moves both counters by posted writes");
}

/** Eight threads claiming bits of four words until none is left each win distinct bits, all of them
 *  between them, through the reference's own claim. */
static void test_extended_atomics_claim_each_bit_once() {
    constexpr std::size_t threads_count_k = 8;
    constexpr std::size_t words_count_k = 4;
    constexpr std::size_t bits_count_k = words_count_k * 64;
    counting_extended_ref<std::uint64_t>::bit_claims.store(0);

    std::array<std::uint64_t, words_count_k> words {};
    std::array<std::atomic<std::size_t>, bits_count_k> winners {};
    std::vector<std::thread> threads;
    for (std::size_t thread = 0; thread != threads_count_k; ++thread)
        threads.emplace_back([&]() noexcept {
            for (;;) {
                std::size_t const claimed =
                    atomic_claim_first_clear_bit<counting_extended_ref>(std::span<std::uint64_t>(words));
                if (claimed == bits_count_k) return;
                winners[claimed].fetch_add(1, std::memory_order_relaxed);
            }
        });
    for (std::thread &thread : threads) thread.join();

    for (std::size_t bit = 0; bit != bits_count_k; ++bit) expect_equal(winners[bit].load(), std::size_t {1});
    for (std::uint64_t const word : words) expect_equal(word, (std::numeric_limits<std::uint64_t>::max)());
    expect_true(counting_extended_ref<std::uint64_t>::bit_claims.load() >= bits_count_k,
                "every claim goes through the reference's own bit verb");

    expect_equal(atomic_fetch_test_and_reset_bit<counting_extended_ref>(words[1], 5u, memory_order_release_k), true);
    expect_equal(atomic_fetch_test_and_reset_bit<counting_extended_ref>(words[1], 5u, memory_order_release_k), false);
    expect_equal(atomic_claim_first_clear_bit<counting_extended_ref>(std::span<std::uint64_t>(words)),
                 std::size_t {64 + 5}, "the one bit given back is the one claimed next");
    expect_equal(atomic_fetch_test_and_set_bit(words[2], 7u, memory_order_acquire_k), true,
                 "the standard reference answers the same through its fallback");
}

/** The mutex asks the policy it was handed once per attempt, parking once the spin is spent. */
static void test_waiting_policy_substitutes_in_the_mutex() {
    yielding_waiting_policy_t::pauses.store(0);

    yielding_mutex_t mutex;
    unique_lock<yielding_mutex_t> held {mutex};

    // The contender cannot win the lock until the hold is released, so it spends its whole spin -
    // and the hold is given up only once it has, so the tally is read after the spinning stops.
    std::thread contender {[&]() noexcept { unique_lock<yielding_mutex_t> guard {mutex}; }};
    expect_true(spin_until([] { return yielding_waiting_policy_t::pauses.load() >= 64; }));
    held.release();
    contender.join();

    expect_true(yielding_waiting_policy_t::pauses.load() >= std::size_t {64}, "one pause per spent attempt");
}

/** The slot lock asks the same policy, reached through the table, while a prober waits for a held
 *  slot. */
static void test_waiting_policy_substitutes_in_the_slot_lock() {
    yielding_waiting_policy_t::pauses.store(0);

    auto [table, status] = hash_set<std::uint64_t>::make(std::size_t {64});
    if (!expect_equal(status, success_k, "the table to be pinned must build")) return;
    yielding_table_t pinned = yielding_table_t::adopt(std::move(table).release());
    expect_equal(pinned.emplace(std::uint64_t {42}), success_k);

    // A found callback runs under the slot's lock, which is the one way to hold a slot from outside.
    std::atomic<bool> holding {false};
    std::atomic<bool> release {false};
    std::thread holder {[&]() noexcept {
        [[maybe_unused]] status_t const found = pinned.find(std::uint64_t {42}, [&](auto const &) noexcept {
            holding.store(true);
            expect_true(spin_until([&] { return release.load(); }));
        });
    }};
    expect_true(spin_until([&] { return holding.load(); }));

    std::thread prober {
        [&]() noexcept { [[maybe_unused]] status_t const seen = pinned.contains(std::uint64_t {42}).status(); }};

    // Every attempt the prober spends on the held slot goes to the policy, so one tally is enough to
    // know the lock is spinning through it rather than through a hint of its own.
    expect_true(spin_until([] { return yielding_waiting_policy_t::pauses.load() != 0; }));
    release.store(true);
    holder.join();
    prober.join();

    expect_equal(pinned.contains(std::uint64_t {42}), true);
}

#pragma endregion Waiting Policy Tests

#pragma region Transaction Group Tests

/** The lifecycle step a logged entry names. */
enum class phase_t : std::uint8_t {
    open_k,
    stage_k,
    validate_k,
    publish_k,
    release_k,
    commit_k,
    rollback_k,
    reset_k,
};

/** One lifecycle call a recording store logged, and where in the group that store sits. */
struct recorded_call_t {
    phase_t phase {phase_t::open_k};
    std::size_t position {0};
};

/** Whether a store's transaction offers the two-step commit a group prefers over one call. */
enum class commit_shape_t : bool {
    one_call_k,
    split_k,
};

/** What one store hands back from each step, so a group can be steered into any refusal. Every
 *  member is that step's own status, and @c success_k lets it through. */
struct refusals_t {
    status_t opening {success_k};
    status_t stage {success_k};
    status_t validate_for_commit {success_k};
    status_t commit {success_k};
    status_t rollback {success_k};
    status_t reset {success_k};
};

/** A store that records the order its steps are visited in, and refuses where it is told to. */
template <commit_shape_t shape_ = commit_shape_t::one_call_k>
struct recording_store {
    using is_transactional = std::true_type;
    using identifier_t = int;
    static constexpr isolation_t isolation_k = isolation_t::monotonic_atomic_view_k;

    std::vector<recorded_call_t> &log;
    std::size_t position {0};
    refusals_t refusals {};

    struct transaction_t {
        // Pointers, not references: the transaction has to stay move-assignable.
        std::vector<recorded_call_t> *log {nullptr};
        std::size_t position {0};
        refusals_t const *refusals {nullptr};

        transaction_t() noexcept = default;
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;
        transaction_t(std::vector<recorded_call_t> *log, std::size_t position, refusals_t const *refusals) noexcept
            : log(log), position(position), refusals(refusals) {}

        [[nodiscard]] status_t reserve(std::size_t) noexcept { return success_k; }
        [[nodiscard]] status_t watch(identifier_t) noexcept { return success_k; }
        [[nodiscard]] status_t stage() noexcept { return record(phase_t::stage_k, refusals->stage); }
        [[nodiscard]] status_t commit() noexcept { return record(phase_t::commit_k, refusals->commit); }
        [[nodiscard]] status_t rollback() noexcept { return record(phase_t::rollback_k, refusals->rollback); }
        [[nodiscard]] status_t reset() noexcept { return record(phase_t::reset_k, refusals->reset); }

        [[nodiscard]] status_t validate_for_commit() noexcept
            requires(shape_ == commit_shape_t::split_k)
        {
            return record(phase_t::validate_k, refusals->validate_for_commit);
        }

        void publish_under() noexcept
            requires(shape_ == commit_shape_t::split_k)
        {
            [[maybe_unused]] status_t const published = record(phase_t::publish_k, success_k);
        }

        void release_validation() noexcept
            requires(shape_ == commit_shape_t::split_k)
        {
            [[maybe_unused]] status_t const released = record(phase_t::release_k, success_k);
        }

      private:
        status_t record(phase_t phase, status_t status) noexcept {
            log->push_back(recorded_call_t {phase, position});
            return status;
        }
    };

    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        if (failed(refusals.opening)) return refusals.opening;
        log.push_back(recorded_call_t {phase_t::open_k, position});
        return expected<transaction_t> {transaction_t {&log, position, &refusals}, success_k};
    }
};

/** A store in a shared commit order whose every reset lands an empty commit there, as another
 *  thread committing between the resets of two participants would. */
struct store_committing_on_reset_t {
    using is_transactional = std::true_type;
    using identifier_t = int;
    using order_t = commit_order_t;
    static constexpr isolation_t isolation_k = isolation_t::snapshot_k;

    order_t &commit_order;

    struct transaction_t {
        // A pointer, not a reference: the transaction has to stay move-assignable.
        order_t *commit_order {nullptr};
        generation_t read_at {0};

        transaction_t() noexcept = default;
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;
        transaction_t(order_t *commit_order, generation_t read_at) noexcept
            : commit_order(commit_order), read_at(read_at) {}

        [[nodiscard]] status_t watch(identifier_t) noexcept { return success_k; }
        [[nodiscard]] status_t stage() noexcept { return success_k; }
        [[nodiscard]] status_t commit() noexcept { return success_k; }
        [[nodiscard]] status_t rollback() noexcept { return success_k; }
        [[nodiscard]] status_t reset() noexcept { return success_k; }
        [[nodiscard]] status_t validate_for_commit() noexcept { return success_k; }
        void publish_under(commit_stamp_t) noexcept {}
        void release_validation() noexcept {}
        void adopt_snapshot(generation_t stamp) noexcept { read_at = stamp; }
        void reopen_at(opened_at_t opened) noexcept { read_at = opened.snapshot; }
        void prune_committed() noexcept {}
        [[nodiscard]] generation_t snapshot() const noexcept { return read_at; }

        [[nodiscard]] status_t reset_at(generation_t stamp) noexcept {
            read_at = stamp;
            order_t::commit_in_flight_t landing;
            commit_order->begin_commit(landing);
            commit_order->end_commit(landing);
            return success_k;
        }
    };

    [[nodiscard]] order_t &order() noexcept { return commit_order; }

    [[nodiscard]] expected<transaction_t> transaction_at(opened_at_t opened) noexcept {
        return transaction_t {&commit_order, opened.snapshot};
    }

    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        return transaction_at(opened_at_t {.snapshot = commit_order.published_stamp()});
    }
};

/** The stores @p log holds for @p phase by their position in the group, in visit order. */
static std::vector<std::size_t> stores_visited(std::vector<recorded_call_t> const &log, phase_t phase) {
    std::vector<std::size_t> visited;
    for (recorded_call_t const &call : log)
        if (call.phase == phase) visited.push_back(call.position);
    return visited;
}

/** Every phase walks the participants in one address order, rollback included. */
static void test_transaction_group_walks_one_order() {
    std::vector<recorded_call_t> log;
    recording_store<> first {log, 0}, second {log, 1}, third {log, 2};

    auto [group, status] = make_transaction_group(first, second, third);
    if (!expect_equal(status, success_k)) return;

    expect_equal(group.stage(), success_k);
    std::vector<std::size_t> const ordered = stores_visited(log, phase_t::stage_k);
    expect_equal(ordered.size(), 3u);
    expect_equal(group.rollback(), success_k);

    // Staging a second time takes that same order, and rollback follows it too.
    log.clear();
    expect_equal(group.stage(), success_k);
    expect_true(stores_visited(log, phase_t::stage_k) == ordered);
    expect_equal(group.rollback(), success_k);
    expect_true(stores_visited(log, phase_t::rollback_k) == ordered);

    // Commit and reset take it as well.
    log.clear();
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_true(stores_visited(log, phase_t::commit_k) == ordered);

    log.clear();
    expect_equal(group.reset(), success_k);
    expect_true(stores_visited(log, phase_t::reset_k) == ordered);

    // The phases refuse to run out of turn.
    expect_equal(group.rollback(), operation_not_permitted_k);
    expect_equal(group.commit(), operation_not_permitted_k);
}

/** A refused rollback stops where it was refused and leaves the group staged. */
static void test_transaction_group_rollback_stops_at_refusal() {
    std::vector<recorded_call_t> log;
    // Array members ascend in address, so the group visits these in the order they are named.
    std::array<recording_store<>, 3> stores {{{log, 0}, {log, 1}, {log, 2}}};
    stores[1].refusals.rollback = operation_would_block_k;

    auto [group, status] = make_transaction_group(stores[0], stores[1], stores[2]);
    if (!expect_equal(status, success_k)) return;

    expect_equal(group.stage(), success_k);
    log.clear();
    expect_equal(group.rollback(), operation_would_block_k);

    // The participant past the refusal was never asked, so it is still staged.
    std::vector<std::size_t> const reached {0, 1};
    expect_true(stores_visited(log, phase_t::rollback_k) == reached);

    // And the group still says so, which is what keeps the next `stage` from staging twice over.
    expect_equal(group.staging(), staging_t::staged_k);
    expect_equal(group.stage(), operation_not_permitted_k);

    // Lifting the refusal lets the rollback reach every participant and reopen the group.
    stores[1].refusals.rollback = success_k;
    log.clear();
    expect_equal(group.rollback(), success_k);
    std::vector<std::size_t> const every {0, 1, 2};
    expect_true(stores_visited(log, phase_t::rollback_k) == every);
    expect_equal(group.staging(), staging_t::pending_k);
}

/** One store named twice is refused before either participant is opened. */
static void test_transaction_group_refuses_a_duplicate_store() {
    std::vector<recorded_call_t> log;
    recording_store<> only {log, 0}, other {log, 1};

    expect_equal(make_transaction_group(only, only).status(), invalid_argument_k);
    expect_true(log.empty());

    // A repeat is caught wherever it sits, not only next to itself.
    expect_equal(make_transaction_group(only, other, only).status(), invalid_argument_k);
    expect_true(log.empty());

    // Two distinct stores are what the refusal is not about.
    expect_equal(make_transaction_group(only, other).status(), success_k);
    std::vector<std::size_t> const opened {0, 1};
    expect_true(stores_visited(log, phase_t::open_k) == opened);
}

/** A store refusing to open reports its own status, and the caller hears the first of them. */
static void test_transaction_group_reports_what_refused_to_open() {
    std::vector<recorded_call_t> log;
    recording_store<> first {log, 0}, second {log, 1};

    second.refusals.opening = operation_not_permitted_k;
    expect_equal(make_transaction_group(first, second).status(), operation_not_permitted_k);

    // Two refusals fold to the one the caller named first, in argument order.
    first.refusals.opening = capacity_exhausted_k;
    expect_equal(make_transaction_group(first, second).status(), capacity_exhausted_k);
}

/** A rollback failing inside a refused stage replaces the refusal and resets every participant. */
static void test_transaction_group_failed_unwind_resets() {
    using store_t = recording_store<>;
    std::vector<recorded_call_t> log;
    std::array<store_t, 2> stores {{{log, 0}, {log, 1}}};
    stores[0].refusals.rollback = consistency_k;
    stores[1].refusals.stage = write_conflict_k;

    auto [group, status] = make_transaction_group(stores[0], stores[1]);
    if (!expect_equal(status, success_k)) return;
    expect_equal(group.stage(), consistency_k);
    expect_equal(group.staging(), staging_t::pending_k);
    std::vector<std::size_t> const every {0, 1};
    expect_true(stores_visited(log, phase_t::reset_k) == every);
}

/** A commit that published nothing stays staged, and one that tore drops to pending. */
static void test_transaction_group_torn_commit_stops_claiming_staged() {
    using store_t = recording_store<>;
    static_assert(!transaction_group<store_t, store_t, store_t>::asks_before_writing_k,
                  "a one-call commit is what leaves a group able to tear");
    static_assert(!transaction_group<store_t, store_t, store_t>::shares_one_order_k,
                  "a store in no order has no stamp for a group to share");

    std::vector<recorded_call_t> log;
    // Array members ascend in address, so the group visits these in the order they are named.
    std::array<store_t, 3> stores {{{log, 0}, {log, 1}, {log, 2}}};

    // The first participant refusing publishes nothing, so the group is staged and still retryable.
    stores[0].refusals.commit = write_conflict_k;
    auto [intact, intact_status] = make_transaction_group(stores[0], stores[1], stores[2]);
    if (!expect_equal(intact_status, success_k)) return;
    expect_equal(intact.stage(), success_k);
    expect_equal(intact.commit(), write_conflict_k);
    expect_equal(intact.staging(), staging_t::staged_k);
    expect_equal(intact.rollback(), success_k);

    // A refusal after one participant published cannot be pulled back, so the group stops saying staged.
    stores[0].refusals.commit = success_k;
    stores[1].refusals.commit = write_conflict_k;
    auto [torn, torn_status] = make_transaction_group(stores[0], stores[1], stores[2]);
    if (!expect_equal(torn_status, success_k)) return;
    expect_equal(torn.stage(), success_k);
    expect_equal(torn.commit(), write_conflict_k);
    expect_equal(torn.staging(), staging_t::pending_k);

    // Neither phase that needs a staged group will run, and `reset` is what clears the rest.
    expect_equal(torn.commit(), operation_not_permitted_k);
    expect_equal(torn.rollback(), operation_not_permitted_k);
    log.clear();
    expect_equal(torn.reset(), success_k);
    std::vector<std::size_t> const every {0, 1, 2};
    expect_true(stores_visited(log, phase_t::reset_k) == every);
}

/** Where every participant splits its commit, a refusal publishes nothing and leaves no hold
 *  behind. */
static void test_transaction_group_split_commit_publishes_nothing_on_refusal() {
    using store_t = recording_store<commit_shape_t::split_k>;
    static_assert(transaction_group<store_t, store_t>::asks_before_writing_k,
                  "two split participants are asked before either writes");

    std::vector<recorded_call_t> log;
    std::array<store_t, 2> stores {{{log, 0}, {log, 1}}};
    stores[1].refusals.validate_for_commit = write_conflict_k;

    auto [group, status] = make_transaction_group(stores[0], stores[1]);
    if (!expect_equal(status, success_k)) return;

    expect_equal(group.stage(), success_k);
    log.clear();
    expect_equal(group.commit(), write_conflict_k);
    expect_true(stores_visited(log, phase_t::publish_k).empty());

    // Only the store asked before the refusal has a validation to give back.
    std::vector<std::size_t> const released {0};
    expect_true(stores_visited(log, phase_t::release_k) == released);

    // Nothing was written, so the group is staged and the caller may still unwind or retry it.
    expect_equal(group.staging(), staging_t::staged_k);
    stores[1].refusals.validate_for_commit = success_k;
    log.clear();
    expect_equal(group.commit(), success_k);
    std::vector<std::size_t> const every {0, 1};
    expect_true(stores_visited(log, phase_t::publish_k) == every);
    expect_equal(group.staging(), staging_t::pending_k);
}

/** A reset reads every participant at one snapshot, however many commits land between their
 *  resets. */
static void test_transaction_group_resets_every_participant_at_one_snapshot() {
    using values_t = partitioned_store<snapshot_avl_map<std::uint64_t, int>>;
    static_assert(
        transaction_group<store_committing_on_reset_t, store_committing_on_reset_t, values_t>::shares_one_order_k,
        "stores in one order are reset at a snapshot the group draws");

    commit_order_t order;
    store_committing_on_reset_t first {order}, second {order};
    values_t values {order};

    auto [group, status] = make_transaction_group(first, second, values);
    if (!expect_equal(status, success_k)) return;
    generation_t const drawn = order.published_stamp();
    expect_equal(group.reset(), success_k);
    expect_equal(order.published_stamp(), drawn + 2, "each committing participant landed a commit mid-reset");

    // Every reset but the first follows a landed commit, whatever order the group visits them in.
    auto [first_read, second_read, values_read] = group.participants();
    expect_equal(first_read.snapshot(), drawn);
    expect_equal(second_read.snapshot(), drawn);
    expect_equal(values_read.snapshot(), drawn, "a commit landing mid-reset must not split the group");
}

/** A map and a set, two kinds of store a database keeps any number of, each sharded and stamped. */
using grouped_map_t = partitioned_store<snapshot_avl_map<std::uint64_t, int>>;
using grouped_set_t = partitioned_store<snapshot_avl_set<std::uint64_t>>;
using grouped_member_t = grouped_map_t::value_t;
using grouped_order_t = grouped_map_t::order_t;
using dynamic_group_t = dynamic_transaction_group<grouped_map_t, grouped_set_t>;

/** Whether @p store holds @p key at @p stamp, read without a claim of its own. */
template <typename store_type_>
static bool holds_at(store_type_ const &store, generation_t stamp, std::uint64_t key) {
    bool found = false;
    expect_equal(store.reader_at(stamp).find(key, [&](auto const &) noexcept { found = true; }), success_k);
    return found;
}

/** A map and a set joined at run time publish under one stamp, which a reader sees whole or not at
 *  all. */
static void test_dynamic_transaction_group_commits_kinds_under_one_stamp() {
    grouped_order_t order;
    grouped_map_t values {order};
    grouped_set_t edges {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    generation_t const before = group.snapshot();
    auto const [values_joined, values_status] = group.join(values);
    if (!expect_equal(values_status, success_k)) return;
    auto const [edges_joined, edges_status] = group.join(edges);
    if (!expect_equal(edges_status, success_k)) return;
    expect_equal(values_joined->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(edges_joined->upsert(std::uint64_t {7}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);

    expect_equal(order.published_stamp(), before + 1, "one stamp for every kind the group spans");
    expect_equal(group.commit_stamp(), before + 1);
    expect_true(!holds_at(values, before, 1) && !holds_at(edges, before, 7));
    expect_true(holds_at(values, group.commit_stamp(), 1) && holds_at(edges, group.commit_stamp(), 7));
}

/** A conflict in one participant refuses the commit before any participant publishes. */
static void test_dynamic_transaction_group_conflict_publishes_nothing() {
    grouped_order_t order;
    grouped_map_t values {order};
    grouped_set_t edges {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    auto const [watching, watching_status] = group.join(values);
    if (!expect_equal(watching_status, success_k)) return;
    auto const [writing, writing_status] = group.join(edges);
    if (!expect_equal(writing_status, success_k)) return;
    expect_equal(watching->watch(5), success_k);
    expect_equal(watching->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(writing->upsert(std::uint64_t {7}), success_k);
    expect_equal(group.stage(), success_k);

    // The watched key moves between the stage and the commit, from outside the group.
    {
        auto [interloper, interloper_status] = values.transaction();
        if (!expect_equal(interloper_status, success_k)) return;
        expect_equal(interloper.upsert(grouped_member_t {5u, 500}), success_k);
        expect_equal(interloper.stage(), success_k);
        expect_equal(interloper.commit(), success_k);
    }

    generation_t const published = order.published_stamp();
    expect_equal(group.commit(), read_conflict_k);
    expect_equal(order.published_stamp(), published, "a refused group draws no stamp");
    expect_true(!holds_at(values, published, 1) && !holds_at(edges, published, 7));
    expect_equal(group.staging(), staging_t::staged_k);
    expect_equal(group.participants_count(), std::size_t {2});
    expect_equal(group.rollback(), success_k);
}

/** More stores of one kind than a type would list join one group, and their transactions stay put
 *  while the rest join. */
static void test_dynamic_transaction_group_commits_many_stores_of_one_kind() {
    constexpr std::size_t stores_count = 24;
    grouped_order_t order;
    std::vector<grouped_map_t> stores;
    stores.reserve(stores_count);
    for (std::size_t index = 0; index != stores_count; ++index) stores.emplace_back(order);

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    std::vector<grouped_map_t::transaction_t *> joined;
    for (grouped_map_t &store : stores) {
        auto const [participant, participant_status] = group.join(store);
        if (!expect_equal(participant_status, success_k)) return;
        joined.push_back(participant);
    }
    expect_equal(group.participants_count(), stores_count);

    // Written only once every store joined, through the pointers the earliest joins handed out.
    for (std::size_t index = 0; index != stores_count; ++index) {
        expect_true(group.participant(stores[index]) == joined[index]);
        expect_equal(joined[index]->upsert(grouped_member_t {index, static_cast<int>(index)}), success_k);
    }

    generation_t const before = order.published_stamp();
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_equal(order.published_stamp(), before + 1);
    for (std::size_t index = 0; index != stores_count; ++index)
        expect_true(holds_at(stores[index], group.commit_stamp(), index) && !holds_at(stores[index], before, index));
}

/** A store joins once, and a store it cannot stamp alongside the rest never joins. */
static void test_dynamic_transaction_group_joins_each_store_once() {
    grouped_order_t order;
    grouped_map_t values {order};
    grouped_set_t edges {order};
    grouped_map_t apart;

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    expect_true(group.participant(values) == nullptr);
    auto const [first, first_status] = group.join(values);
    if (!expect_equal(first_status, success_k)) return;
    auto const [again, again_status] = group.join(values);
    if (!expect_equal(again_status, success_k)) return;
    if (!expect_true(first == again)) return;
    expect_true(group.participant(values) == first);
    expect_equal(group.participants_count(), std::size_t {1});

    expect_equal(group.join(apart).status(), invalid_argument_k, "a store in its own order has another stamp");
    expect_equal(group.stage(), success_k);
    expect_equal(group.join(edges).status(), operation_not_permitted_k, "a staged group takes no one new");
    expect_equal(group.participants_count(), std::size_t {1});
}

/** A commit and a reset each leave the group empty, and the next join reads a newer snapshot. */
static void test_dynamic_transaction_group_empties_on_commit_and_reset() {
    grouped_order_t order;
    grouped_map_t values {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    expect_equal((*group.join(values))->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_equal(group.participants_count(), std::size_t {0});
    expect_true(group.participant(values) == nullptr);
    expect_true(group.snapshot() >= group.commit_stamp());

    auto const [reader, reader_status] = group.join(values);
    if (!expect_equal(reader_status, success_k)) return;
    expect_equal(reader->snapshot(), group.snapshot());
    expect_equal(reader->find_copy(std::uint64_t {1}).status(), success_k);

    {
        auto [interloper, interloper_status] = values.transaction();
        if (!expect_equal(interloper_status, success_k)) return;
        expect_equal(interloper.upsert(grouped_member_t {2u, 200}), success_k);
        expect_equal(interloper.stage(), success_k);
        expect_equal(interloper.commit(), success_k);
    }
    expect_equal(reader->find_copy(std::uint64_t {2}).status(), key_not_found_k);

    expect_equal(group.reset(), success_k);
    expect_equal(group.participants_count(), std::size_t {0});
    expect_equal((*group.join(values))->find_copy(std::uint64_t {2}).status(), success_k);

    // Emptied once more, the group has nothing to publish and draws no stamp for it.
    expect_equal(group.reset(), success_k);
    generation_t const published = order.published_stamp();
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_equal(order.published_stamp(), published);
    expect_equal(group.commit_stamp(), generation_t {0});
}

/** A long-lived group rewriting the same keys of the same stores asks the allocator for nothing
 *  once its participants and the stores' nodes are warm. */
static void test_dynamic_group_allocates_nothing_once_warm() {
    constexpr std::size_t warm_rounds_k = 2, counted_rounds_k = 64;
    grouped_order_t order;
    grouped_map_t values {order};
    grouped_set_t edges {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    std::size_t warming_allocations = 0, allocations = 0, distinct_stamps = 0;
    generation_t last_stamp = 0;
    for (std::size_t round = 0; round != warm_rounds_k + counted_rounds_k; ++round) {
        std::size_t const before = test_run.nothrow_allocations.load();
        bool passed = succeeded(group.reset());
        auto const [values_joined, values_status] = group.join(values);
        auto const [edges_joined, edges_status] = group.join(edges);
        passed = passed && succeeded(values_status) && succeeded(edges_status) &&
                 succeeded(values_joined->upsert(grouped_member_t {1u, static_cast<int>(round)})) &&
                 succeeded(edges_joined->upsert(std::uint64_t {7})) && succeeded(group.stage()) &&
                 succeeded(group.commit());
        group.clear();
        if (!expect_true(passed)) return;
        if (round < warm_rounds_k) {
            warming_allocations += test_run.nothrow_allocations.load() - before;
            continue;
        }
        allocations += test_run.nothrow_allocations.load() - before;
        distinct_stamps += group.commit_stamp() > last_stamp;
        last_stamp = group.commit_stamp();
    }
    expect_true(warming_allocations != 0, "the counter sees what the first joins and writes allocate");
    expect_equal(allocations, std::size_t {0}, "a warm group reuses its participants and the stores their nodes");
    expect_equal(distinct_stamps, counted_rounds_k);
    expect_equal(group.capacity(), std::size_t {2});
    expect_true(holds_at(values, last_stamp, 1) && holds_at(edges, last_stamp, 7));
}

/** A commit an interloper refuses is retried after a reset, and the retry allocates nothing. */
static void test_dynamic_group_allocates_nothing_across_retries() {
    constexpr std::size_t warm_rounds_k = 2, counted_rounds_k = 8;
    grouped_order_t order;
    grouped_map_t values {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    std::size_t allocations = 0;
    for (std::size_t round = 0; round != warm_rounds_k + counted_rounds_k; ++round) {
        std::size_t before = test_run.nothrow_allocations.load();
        bool passed = succeeded(group.reset());
        auto const [first_attempt, first_attempt_status] = group.join(values);
        passed = passed && succeeded(first_attempt_status) && succeeded(first_attempt->watch(5)) &&
                 succeeded(first_attempt->upsert(grouped_member_t {1u, static_cast<int>(round)})) &&
                 succeeded(group.stage());
        std::size_t spent = test_run.nothrow_allocations.load() - before;

        // The watched key moves between the stage and the commit, from outside the group.
        {
            auto [interloper, interloper_status] = values.transaction();
            if (!expect_equal(interloper_status, success_k)) return;
            expect_equal(interloper.upsert(grouped_member_t {5u, static_cast<int>(round)}), success_k);
            expect_equal(interloper.stage(), success_k);
            expect_equal(interloper.commit(), success_k);
        }

        before = test_run.nothrow_allocations.load();
        passed = passed && group.commit() == read_conflict_k && succeeded(group.reset());
        auto const [second_attempt, second_attempt_status] = group.join(values);
        passed = passed && succeeded(second_attempt_status) && second_attempt == first_attempt &&
                 succeeded(second_attempt->watch(5)) &&
                 succeeded(second_attempt->upsert(grouped_member_t {1u, static_cast<int>(round)})) &&
                 succeeded(group.stage()) && succeeded(group.commit());
        group.clear();
        spent += test_run.nothrow_allocations.load() - before;
        if (!expect_true(passed)) return;
        if (round >= warm_rounds_k) allocations += spent;

        auto const [committed, committed_status] = values.find_copy(std::uint64_t {1});
        expect_equal(committed_status, success_k);
        expect_equal(committed.mapped, static_cast<int>(round));
    }
    expect_equal(allocations, std::size_t {0}, "a retry reopens the participant the refused attempt closed");
    expect_equal(group.capacity(), std::size_t {1});
}

/** A spare participant opens on whichever store of its kind joins next, and every store ends up
 *  holding exactly the keys written through it. */
static void test_dynamic_group_rebinds_spares_to_other_stores() {
    grouped_order_t order;
    grouped_map_t first {order}, second {order}, third {order};
    grouped_set_t edges {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    expect_equal(group.capacity(), std::size_t {0});

    expect_equal((*group.join(first))->upsert(grouped_member_t {1u, 10}), success_k);
    expect_equal((*group.join(second))->upsert(grouped_member_t {2u, 20}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_equal(group.participants_count(), std::size_t {0});
    expect_equal(group.capacity(), std::size_t {2});

    expect_equal((*group.join(third))->upsert(grouped_member_t {3u, 30}), success_k);
    expect_equal((*group.join(first))->upsert(grouped_member_t {4u, 40}), success_k);
    expect_equal(group.capacity(), std::size_t {2}, "both joins reopened a spare");
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);

    expect_equal((*group.join(edges))->upsert(std::uint64_t {9}), success_k);
    expect_equal(group.participants_count(), std::size_t {1});
    expect_equal(group.capacity(), std::size_t {3}, "a spare of another kind is no use to a set");
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);

    generation_t const stamp = group.commit_stamp();
    expect_true(holds_at(first, stamp, 1) && !holds_at(first, stamp, 2) && !holds_at(first, stamp, 3) &&
                holds_at(first, stamp, 4));
    expect_true(!holds_at(second, stamp, 1) && holds_at(second, stamp, 2) && !holds_at(second, stamp, 3) &&
                !holds_at(second, stamp, 4));
    expect_true(!holds_at(third, stamp, 1) && !holds_at(third, stamp, 2) && holds_at(third, stamp, 3) &&
                !holds_at(third, stamp, 4));
    expect_true(holds_at(edges, stamp, 9));

    auto const [kept, kept_status] = group.join(second);
    if (!expect_equal(kept_status, success_k)) return;
    group.shrink_to_fit();
    expect_equal(group.capacity(), std::size_t {1});
    expect_true(group.participant(second) == kept);
    expect_equal(kept->find_copy(std::uint64_t {2}).status(), success_k);
}

/** A cleared group's spares name no store, so the stores they served may be freed and others
 *  built where they stood. */
static void test_spares_outlive_their_store() {
    grouped_order_t order;
    auto [group, status] = dynamic_group_t::make(order);
    if (!expect_equal(status, success_k)) return;

    std::unique_ptr<grouped_map_t> values = std::make_unique<grouped_map_t>(order);
    expect_equal((*group.join(*values))->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    group.clear();
    values.reset();

    std::unique_ptr<grouped_set_t> edges = std::make_unique<grouped_set_t>(order);
    expect_equal(group.reset(), success_k);
    expect_equal((*group.join(*edges))->upsert(std::uint64_t {7}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_true(holds_at(*edges, group.commit_stamp(), 7));
    group.clear();
    edges.reset();

    values = std::make_unique<grouped_map_t>(order);
    expect_equal(group.reset(), success_k);
    expect_equal((*group.join(*values))->upsert(grouped_member_t {2u, 200}), success_k);
    expect_equal(group.capacity(), std::size_t {2}, "the map's spare reopened on the new map");
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    expect_true(holds_at(*values, group.commit_stamp(), 2) && !holds_at(*values, group.commit_stamp(), 1));
    group.clear();
    values.reset();
}

/** A cleared group holds no claim, so the low-water mark passes it, and nothing joins it until a
 *  reset draws a snapshot again. */
static void test_cleared_group_pins_nothing() {
    grouped_order_t order;
    grouped_map_t values {order};

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    expect_equal((*group.join(values))->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    generation_t const pinned = group.snapshot();

    auto const commit_apart = [&](std::uint64_t key) {
        auto [apart, apart_status] = values.transaction();
        if (!expect_equal(apart_status, success_k)) return;
        expect_equal(apart.upsert(grouped_member_t {key, 0}), success_k);
        expect_equal(apart.stage(), success_k);
        expect_equal(apart.commit(), success_k);
    };
    commit_apart(2);
    expect_equal(order.low_water_mark(), pinned, "the group's claim still pins its snapshot");

    group.clear();
    expect_equal(group.participants_count(), std::size_t {0});
    commit_apart(3);
    expect_true(order.low_water_mark() > pinned, "a cleared group pins nothing");

    expect_equal(group.join(values).status(), operation_not_permitted_k, "a cleared group has no snapshot to join at");
    expect_equal(group.reset(), success_k);
    auto const [rejoined, rejoined_status] = group.join(values);
    if (!expect_equal(rejoined_status, success_k)) return;
    expect_equal(rejoined->find_copy(std::uint64_t {3}).status(), success_k);
}

#pragma endregion Transaction Group Tests

#pragma region Commit Order Tests

/** A resumed order answers its next reader at the resumed stamp and draws the stamp above it, and
 *  the mark rises past it once that reader leaves. */
template <typename order_type_>
static void check_resume_continues_stamps() {
    generation_t const durable = 1000;
    order_type_ order;
    expect_equal(order.resume(durable), success_k);
    expect_equal(order.published_stamp(), durable);
    expect_equal(order.drawn_stamp(), durable);
    expect_equal(order.low_water_mark(), durable, "a run built right after a restart packs its stamps from here");

    typename order_type_::snapshot_claim_t claim;
    expect_equal(order.take_snapshot(claim), durable);
    expect_true(order.next_generation() > durable);
    typename order_type_::commit_in_flight_t landing;
    order.begin_commit(landing);
    expect_equal(static_cast<generation_t>(landing.stamp()), durable + 1);
    order.end_commit(landing);
    expect_equal(order.published_stamp(), durable + 1);
    expect_equal(order.low_water_mark(), durable, "the open claim still pins its snapshot");

    claim.retire();
    expect_equal(order.low_water_mark(), durable + 1);
}

static void test_commit_order_resume_continues_stamps() {
    check_resume_continues_stamps<commit_order_t>();
    check_resume_continues_stamps<solitary_commit_order_t>();
}

/** Stores built into an order before it resumes commit above the resumed stamp, and a reader at
 *  that stamp sees none of it. */
static void test_commit_order_resume_reaches_its_stores() {
    generation_t const durable = 1 << 20;
    grouped_order_t order;
    grouped_map_t values {order};
    grouped_set_t edges {order};
    expect_equal(order.resume(durable), success_k, "building a store into an order draws nothing from it");

    auto [group, group_status] = dynamic_group_t::make(order);
    if (!expect_equal(group_status, success_k)) return;
    expect_equal(group.snapshot(), durable);
    auto const [values_joined, values_status] = group.join(values);
    if (!expect_equal(values_status, success_k)) return;
    auto const [edges_joined, edges_status] = group.join(edges);
    if (!expect_equal(edges_status, success_k)) return;
    expect_equal(values_joined->upsert(grouped_member_t {1u, 100}), success_k);
    expect_equal(edges_joined->upsert(std::uint64_t {7}), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);

    expect_equal(group.commit_stamp(), durable + 1);
    expect_true(!holds_at(values, durable, 1) && !holds_at(edges, durable, 7));
    expect_true(holds_at(values, durable + 1, 1) && holds_at(edges, durable + 1, 7));
}

/** An argument leaving no stamp to draw is refused, and so is an order something has already drawn
 *  from or reads at, with nothing changed either way. */
static void test_commit_order_resume_refusals() {
    {
        commit_order_t order;
        expect_equal(order.resume(-1), invalid_argument_k);
        expect_equal(order.resume(latest_snapshot_k), invalid_argument_k, "the next stamp would read uncommitted");
        expect_equal(order.published_stamp(), generation_t {0});
        expect_equal(order.resume(latest_snapshot_k - 1), success_k, "one stamp is still left to draw");
        expect_equal(order.resume(7), operation_not_permitted_k, "an order resumes once");
        expect_equal(order.published_stamp(), latest_snapshot_k - 1);
    }
    {
        commit_order_t order;
        expect_equal(order.next_generation(), generation_t {1});
        expect_equal(order.resume(7), operation_not_permitted_k);
        expect_equal(order.next_generation(), generation_t {2});
    }
    {
        solitary_commit_order_t order;
        solitary_commit_order_t::commit_in_flight_t landing;
        order.begin_commit(landing);
        order.end_commit(landing);
        expect_equal(order.resume(7), operation_not_permitted_k);
        expect_equal(order.published_stamp(), generation_t {1});
    }
    {
        commit_order_t order;
        commit_order_t::snapshot_claim_t claim;
        expect_equal(order.take_snapshot(claim), generation_t {0});
        expect_equal(order.resume(7), operation_not_permitted_k, "an open claim reads at the stamp it would move");
        claim.retire();
        expect_equal(order.resume(7), success_k, "a retired claim drew nothing");
        expect_equal(order.published_stamp(), generation_t {7});
    }
}

/** The order with every word owned through ForkUnion's reference. */
using forkunion_order_t =
    basic_commit_order<16, 4, order_sharing_t::shared_k, fu::preferred_yield_t, fu::preferred_atomic_ref>;

using smashtable::commit_order_t;
using smashtable::solitary_commit_order_t;

/** Two commits in flight, so four committers wait for room on most landings. */
using narrow_order_t = basic_commit_order<2>;

/** More commits in flight than half the word, so the window is what bounds a landing. */
using wide_order_t = basic_commit_order<64>;
using forkunion_narrow_order_t =
    basic_commit_order<2, 4, order_sharing_t::shared_k, fu::preferred_yield_t, fu::preferred_atomic_ref>;

/** The ring the parity word replaced, kept as its oracle: slot @c stamp % ring_ holds a stamp once
 *  that commit landed, and the watermark walks one slot at a time. */
template <std::size_t ring_>
struct ring_walk_order {
    std::atomic<generation_t> commits {0};
    std::atomic<generation_t> published {0};
    std::array<std::atomic<generation_t>, ring_> landed {};

    void resume(generation_t at) noexcept {
        commits.store(at);
        published.store(at);
    }

    generation_t begin_commit() noexcept { return commits.fetch_add(1) + 1; }

    void end_commit(generation_t stamp) noexcept {
        while (stamp - published.load() > static_cast<generation_t>(ring_)) std::this_thread::yield();
        landed[static_cast<std::size_t>(stamp) % ring_].store(stamp);
        generation_t seen = published.load();
        while (landed[static_cast<std::size_t>(seen + 1) % ring_].load() == seen + 1) {
            generation_t before = published.load();
            while (before < seen + 1 && !published.compare_exchange_weak(before, seen + 1)) {}
            seen = before < seen + 1 ? seen + 1 : before;
        }
    }
};

/** Lands @p landings stamps above the watermark of @p order on it and on the ring, four threads
 *  taking turns, each landing a random unlanded stamp inside the window, and expects the same
 *  watermark after every one. */
template <typename order_type_>
static void check_landings_match_ring_walk(order_type_ &order, std::size_t landings) {
    using flight_t = typename order_type_::commit_in_flight_t;
    constexpr std::size_t threads_k = 4;
    generation_t const resumed = order.published_stamp();
    ring_walk_order<order_type_::ring_k> ring;
    ring.resume(resumed);

    split_mix_generator_t generator {case_stream_key(static_cast<std::uint64_t>(resumed))};
    std::vector<std::size_t> landing_order;
    std::vector<std::uint8_t> is_landed(landings, 0);
    for (std::size_t lowest = 0; lowest != landings;) {
        std::size_t const left = landings - lowest;
        std::size_t const picked =
            lowest + generator.below(left < order_type_::window_k ? left : order_type_::window_k);
        if (is_landed[picked]) continue;
        is_landed[picked] = 1;
        landing_order.push_back(picked);
        while (lowest != landings && is_landed[lowest]) ++lowest;
    }

    std::vector<flight_t> flights(landings);
    std::vector<generation_t> word_watermarks(landings), ring_watermarks(landings);
    std::atomic<std::size_t> turn {0};
    std::size_t drawn = 0;
    std::vector<std::thread> threads;
    for (std::size_t thread = 0; thread != threads_k; ++thread)
        threads.emplace_back([&, thread]() noexcept {
            for (std::size_t step = thread; step < landings; step += threads_k) {
                while (turn.load() != step) std::this_thread::yield();
                std::size_t const picked = landing_order[step];
                for (; drawn <= picked; ++drawn) {
                    order.begin_commit(flights[drawn]);
                    ring.begin_commit();
                }
                order.end_commit(flights[picked]);
                ring.end_commit(static_cast<generation_t>(flights[picked].stamp()));
                word_watermarks[step] = order.published_stamp();
                ring_watermarks[step] = ring.published.load();
                turn.store(step + 1);
            }
        });
    for (std::thread &thread : threads) thread.join();

    expect_true(word_watermarks == ring_watermarks, "the word raises the watermark where the ring walked it");
    expect_equal(order.published_stamp(), resumed + static_cast<generation_t>(landings));
    expect_equal(order.drawn_stamp(), order.published_stamp());
    expect_equal(order.low_water_mark(), order.published_stamp());
}

/** One bit flipped per landing publishes what the ring walk did: from a fresh order, from one
 *  resumed just below a lap of the word and one far above it, and from one adopted mid-lap. */
template <typename order_type_>
static void test_commit_order_parity_matches_ring_walk() {
    constexpr std::size_t landings_k = 1000; // an odd count of laps and a part of one, so an adopted word is not zero
    order_type_ fresh;
    check_landings_match_ring_walk(fresh, landings_k);

    for (generation_t const durable : {generation_t {61}, generation_t {(1 << 20) + 127}}) {
        order_type_ resumed;
        expect_equal(resumed.resume(durable), success_k);
        check_landings_match_ring_walk(resumed, landings_k);
    }

    order_type_ adopted;
    adopted.adopt(fresh);
    check_landings_match_ring_walk(adopted, landings_k);
}

/** Committers racing through a window narrower than their count land every stamp: the watermark
 *  covers each one's stamp once it asks, and ends on the newest drawn, where the ring's ends. */
template <typename order_type_>
static void test_commit_order_parity_lands_racing_commits() {
    constexpr std::size_t committers_k = 4;
    std::size_t const commits = scaled_iterations(5000);
    order_type_ order;
    ring_walk_order<order_type_::ring_k> ring;
    std::atomic<bool> uncovered {false};

    std::vector<std::thread> threads;
    for (std::size_t committer = 0; committer != committers_k; ++committer)
        threads.emplace_back([&, committer]() noexcept {
            split_mix_generator_t generator {case_stream_key(committer)};
            for (std::size_t commit = 0; commit != commits; ++commit) {
                typename order_type_::commit_in_flight_t landing;
                order.begin_commit(landing);
                generation_t const ring_stamp = ring.begin_commit();
                if (generator.below(4) == 0) std::this_thread::yield();
                order.end_commit(landing);
                ring.end_commit(ring_stamp);
                order.await_published(landing.stamp());
                if (order.published_stamp() < static_cast<generation_t>(landing.stamp())) uncovered.store(true);
            }
        });
    for (std::thread &thread : threads) thread.join();

    generation_t const drawn = static_cast<generation_t>(committers_k * commits);
    expect_true(!uncovered.load(), "a committer reads its own stamp once the watermark was awaited");
    expect_equal(order.published_stamp(), drawn);
    expect_equal(ring.published.load(), drawn);
    expect_equal(order.drawn_stamp(), drawn);
    expect_equal(order.low_water_mark(), drawn);
}

#pragma endregion Commit Order Tests

#pragma region Ordering Tests

using versioning_t = versioning_for<std::size_t, less_t>;
using versioned_t = versioning_t::versioned_t;
using dated_identifier_t = versioning_t::dated_identifier_t;

/** A generation breaks ties only for operands that still name a key. */
static void test_versioned_comparator_orders_by_key_then_generation() {
    versioning_t::versioned_comparator_t comparator;

    versioned_t older {std::size_t {5}}, newer {std::size_t {5}}, other {std::size_t {6}};
    older.generation = 1, newer.generation = 2, other.generation = 0;

    expect_true(comparator.less(older, newer));
    expect_true(!comparator.less(newer, older));
    expect_true(comparator.less(newer, other));
    expect_true(!comparator.less(other, newer));
    expect_true(comparator.same(older, older));
    expect_true(!comparator.same(older, newer));

    // A dated identifier orders the same way, against an entry and against itself.
    dated_identifier_t const dated {std::size_t {5}, generation_t {2}};
    expect_true(comparator.less(older, dated));
    expect_true(!comparator.less(dated, older));
    expect_true(comparator.same(dated, newer));

    // A plain key carries no generation, so it orders by key alone.
    expect_true(comparator.less(std::size_t {4}, older));
    expect_true(!comparator.less(older, std::size_t {5}));

    // A watch dates an entry without naming one, so it is never an ordering operand.
    static_assert(a_generation_carrier<watch_t>, "a watch does carry a generation");
    static_assert(!a_version_ordered_operand<watch_t>, "but it names no key, so it cannot be ordered");
    static_assert(a_version_ordered_operand<versioned_t>, "an entry carries both");
    static_assert(a_version_ordered_operand<dated_identifier_t>, "and so does a dated identifier");
}

#pragma endregion Ordering Tests

#pragma region Optimistic Concurrency Tests

/** A stamp is visible to every snapshot at or past it, and an uncommitted one to none. */
static void test_commit_stamp_visibility_matrix() {
    constexpr generation_t stamps_k[] = {0, 1, 2, 7, 1000};
    constexpr generation_t snapshots_k[] = {0, 1, 2, 7, 1000, 1001};

    for (generation_t const stamp : stamps_k)
        for (generation_t const snapshot : snapshots_k)
            expect_equal(visible_at(static_cast<commit_stamp_t>(stamp), snapshot), stamp <= snapshot);

    // The sentinel sits past every snapshot a generation counter will ever hand out, which is the
    // whole reason it is the maximum rather than zero.
    for (generation_t const snapshot : snapshots_k) expect_true(!visible_at(commit_stamp_t::uncommitted_k, snapshot));
    expect_true(!visible_at(commit_stamp_t::uncommitted_k, (std::numeric_limits<generation_t>::max)() - 1));

    // Zero as a sentinel would have read as "visible since the beginning of time".
    expect_true(visible_at(static_cast<commit_stamp_t>(generation_t {0}), 0));
}

/**
 *  @brief The ladder weakest rung first, which is the order @c at_least is asked to agree with.
 *
 *  Written out rather than derived, because a walk generated from @c isolation_t could only confirm
 *  the enum agrees with itself. The count is pinned to the strongest rung so a new one cannot join
 *  the enum and skip the walk, which is how @c strict_serializable_k once went untested.
 */
constexpr std::array<isolation_t, 5> isolation_ladder_k {
    isolation_t::read_committed_k, isolation_t::monotonic_atomic_view_k, isolation_t::snapshot_k,
    isolation_t::serializable_k,   isolation_t::strict_serializable_k,
};
static_assert(isolation_ladder_k.size() == static_cast<std::size_t>(isolation_t::strict_serializable_k) + 1,
              "a rung joined or left `isolation_t` without joining the walk over it");

/** Every ordered pair of isolation levels, so the enum's order is load-bearing. */
static void test_isolation_levels_compare_by_strength() {
    for (std::size_t stronger = 0; stronger != isolation_ladder_k.size(); ++stronger)
        for (std::size_t weaker = 0; weaker != isolation_ladder_k.size(); ++weaker)
            expect_equal(at_least(isolation_ladder_k[stronger], isolation_ladder_k[weaker]), stronger >= weaker);

    static_assert(at_least(isolation_t::serializable_k, isolation_t::read_committed_k));
    static_assert(!at_least(isolation_t::read_committed_k, isolation_t::snapshot_k));
    static_assert(at_least(isolation_t::snapshot_k, isolation_t::snapshot_k));
    static_assert(at_least(isolation_t::strict_serializable_k, isolation_t::serializable_k));
    static_assert(!at_least(isolation_t::serializable_k, isolation_t::strict_serializable_k));
}

/** An entry standing in for what a store resolves a watched identifier to. */
struct resolved_entry_t {
    generation_t generation {0};
    presence_t presence {presence_t::present_k};

    bool operator==(watch_t const &watch) const noexcept {
        return generation == watch.generation && presence == watch.presence;
    }
};

/** Watches that still match pass, and the first that drifted reports a conflict. */
static void test_validate_watches_catches_drift() {
    using watched_t = watched_identifier<int>;
    std::vector<watched_t> watches;
    watches.push_back(watched_t {1, watch_t {5, presence_t::present_k}});
    watches.push_back(watched_t {2, watch_t {8, presence_t::present_k}});
    watches.push_back(watched_t {3, missing_watch()});

    // The store the watches are validated against: keys 1 and 2 are there, key 3 is not.
    auto resolve = [](std::vector<std::pair<int, resolved_entry_t>> const &entries) noexcept {
        return [&entries](int id, auto &&on_found, auto &&on_missing) noexcept {
            for (auto const &entry : entries)
                if (entry.first == id) return on_found(entry.second);
            on_missing();
        };
    };

    // Nothing moved, so every watch matches itself.
    std::vector<std::pair<int, resolved_entry_t>> unchanged {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
    };
    expect_equal(validate_watches(watches, resolve(unchanged)), success_k);

    // A watched key gained a newer version.
    std::vector<std::pair<int, resolved_entry_t>> bumped {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {9, presence_t::present_k}},
    };
    expect_equal(validate_watches(watches, resolve(bumped)), read_conflict_k);

    // A watched key was erased under us, which is a different shape from never having been there.
    std::vector<std::pair<int, resolved_entry_t>> erased {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::erased_k}},
    };
    expect_equal(validate_watches(watches, resolve(erased)), read_conflict_k);

    // A key watched as absent that has since appeared.
    std::vector<std::pair<int, resolved_entry_t>> appeared {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
        {3, resolved_entry_t {11, presence_t::present_k}},
    };
    expect_equal(validate_watches(watches, resolve(appeared)), read_conflict_k);

    // A committed tombstone resolves to missing, and the absent watch records that same shape, so
    // the two match rather than reporting a phantom conflict.
    std::vector<std::pair<int, resolved_entry_t>> tombstoned {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
        {3, resolved_entry_t {absent_generation_k, presence_t::erased_k}},
    };
    expect_equal(validate_watches(watches, resolve(tombstoned)), success_k);

    // A key that vanished entirely, resolved through the missing callback.
    std::vector<std::pair<int, resolved_entry_t>> vanished {{2, resolved_entry_t {8, presence_t::present_k}}};
    expect_equal(validate_watches(watches, resolve(vanished)), read_conflict_k);

    // No watches at all is trivially consistent.
    std::vector<watched_t> const none;
    expect_equal(validate_watches(none, resolve(unchanged)), success_k);
}

/** A bare key matches every version of it, while a dated identifier matches exactly one. */
static void test_per_version_equals_separates_versions() {
    per_version_equals<std::equal_to<std::size_t>> const equals;
    per_key_hasher<std::hash<std::size_t>> const hasher;

    versioned_t older {std::size_t {5}}, newer {std::size_t {5}}, other {std::size_t {6}};
    older.generation = 1, newer.generation = 2, other.generation = 1;

    // Two versions of one key are distinct entries.
    expect_true(!equals(older, newer));
    expect_true(equals(older, older));
    expect_true(!equals(older, other));

    // A bare key names the key alone, so it matches every version of it - the visibility walk.
    expect_true(equals(std::size_t {5}, older));
    expect_true(equals(std::size_t {5}, newer));
    expect_true(!equals(std::size_t {5}, other));

    // A dated identifier names one version - the commit.
    dated_identifier_t const dated {std::size_t {5}, generation_t {2}};
    expect_true(!equals(dated, older));
    expect_true(equals(dated, newer));

    // The hasher keeps peeling to the bare key, so every version shares one probe run.
    expect_equal(hasher(older), hasher(newer));
    expect_equal(hasher(older), hasher(std::size_t {5}));
    expect_equal(hasher(dated), hasher(std::size_t {5}));

    // Untouched by the widening: `per_key_equals` still collapses the versions of a key.
    per_key_equals<std::equal_to<std::size_t>> const collapsing;
    expect_true(collapsing(older, newer));
}

/** Both shipped stores are optimistically concurrent, and the seams they are read through exist. */
using avl_store_t = monotonic_avl_set<std::size_t, std::less<std::size_t>, std::allocator<std::size_t>>;
using std_store_t = reference_store<std::size_t, std::less<std::size_t>, std::allocator<std::size_t>>;

static_assert(an_optimistically_concurrent_store<avl_store_t>, "the tree-backed store validates and stages");
static_assert(an_optimistically_concurrent_store<std_store_t>, "and so does the `std::set`-backed one");
static_assert(!an_optimistically_concurrent_store<counted_t>, "a plain value is not a store");
static_assert(at_least(avl_store_t::isolation_k, isolation_t::read_committed_k), "the floor the concept states");

/** A transaction whose commit refuses with a planned status a set number of times, counting every
 *  call. */
struct conflicting_transaction_t {

    /** How many more commits refuse. */
    std::size_t refusals_left = 0;

    /** What a refusing commit answers. */
    status_t refusal = status_t::write_conflict_k;

    /** How many more resets refuse, which is the fault a retry cannot unwind. */
    std::size_t reset_refusals_left = 0;

    /** How many times @c stage ran. */
    std::size_t stages = 0;

    /** How many times @c commit ran. */
    std::size_t commits = 0;

    /** How many times @c reset ran. */
    std::size_t resets = 0;

    [[nodiscard]] status_t stage() noexcept {
        ++stages;
        return success_k;
    }
    [[nodiscard]] status_t commit() noexcept {
        ++commits;
        if (refusals_left == 0) return success_k;
        --refusals_left;
        return refusal;
    }
    [[nodiscard]] status_t reset() noexcept {
        ++resets;
        if (reset_refusals_left == 0) return success_k;
        --reset_refusals_left;
        return status_t::operation_not_permitted_k;
    }
};

/** Tests that a bounded retry restages after every conflict, gives up once its attempts run out,
 *  and returns any other failure at once. */
static void test_commit_with_retries_is_bounded() {
    std::size_t restaged = 0;
    auto const stage_changes = [&](conflicting_transaction_t &) noexcept {
        ++restaged;
        return status_t::success_k;
    };

    // Two conflicts and then a commit: three attempts, each restaging, with a reset between them.
    conflicting_transaction_t recovering;
    recovering.refusals_left = 2;
    expect_equal(commit_with_retries(recovering, 3, stage_changes), success_k);
    expect_equal(restaged, 3u, "every attempt recomputes what it stages");
    expect_equal(recovering.resets, 2u, "and a refused attempt is reset before the next one");

    // Every attempt conflicting is a retry that gave up, which is what the vocabulary names for it.
    conflicting_transaction_t hopeless;
    hopeless.refusals_left = 10;
    hopeless.refusal = status_t::phantom_conflict_k;
    restaged = 0;
    expect_equal(commit_with_retries(hopeless, 4, stage_changes), status_t::operation_would_block_k);
    expect_equal(hopeless.commits, 4u, "the bound is the number of attempts, not one more");
    expect_equal(hopeless.resets, 4u, "and the last refused attempt is reset too, leaving nothing staged");
    expect_equal(restaged, 4u, "every one of those attempts recomputed what it staged");

    // A failure that is not a conflict is not retried, and still leaves the transaction reset.
    conflicting_transaction_t refusing;
    std::size_t attempted = 0;
    status_t const answered = commit_with_retries(refusing, 5, [&](conflicting_transaction_t &) noexcept {
        ++attempted;
        return status_t::out_of_memory_heap_k;
    });
    expect_equal(answered, status_t::out_of_memory_heap_k);
    expect_equal(attempted, 1u, "retrying cannot mend a failure no other transaction caused");
    expect_equal(refusing.stages, 0u, "a transaction whose changes would not stage never reaches its own stage");
    expect_equal(refusing.resets, 1u);

    // A conflict the unwind cannot clear is answered with the unwind's own failure, not the conflict.
    conflicting_transaction_t unwinding;
    unwinding.refusals_left = 1;
    unwinding.reset_refusals_left = 1;
    expect_equal(commit_with_retries(unwinding, 3, stage_changes), status_t::operation_not_permitted_k);
    expect_equal(unwinding.commits, 1u, "a transaction that would not unwind is not retried over");

    // A failure of its own outranks the unwind's, because it is the one a caller can act on.
    conflicting_transaction_t faulted;
    faulted.reset_refusals_left = 1;
    expect_equal(commit_with_retries(
                     faulted, 3, [&](conflicting_transaction_t &) noexcept { return status_t::invalid_argument_k; }),
                 status_t::invalid_argument_k);
    expect_equal(faulted.resets, 1u, "and the attempt is still unwound before the failure goes back");

    // No attempts make no calls at all.
    conflicting_transaction_t untouched;
    expect_equal(commit_with_retries(untouched, 0, stage_changes), status_t::operation_would_block_k);
    expect_equal(untouched.stages + untouched.commits + untouched.resets, 0u);
}

/** Every status the vocabulary names, so a new one has to be classified here rather than fall
 *  through. */
constexpr status_t every_status_k[] = {
    status_t::success_k,
    status_t::unknown_k,
    status_t::consistency_k,
    status_t::write_conflict_k,
    status_t::read_conflict_k,
    status_t::phantom_conflict_k,
    status_t::out_of_memory_heap_k,
    status_t::invalid_argument_k,
    status_t::operation_not_permitted_k,
    status_t::operation_would_block_k,
    status_t::capacity_exhausted_k,
    status_t::input_output_k,
    status_t::key_already_exists_k,
    status_t::key_not_found_k,
};

/** Tests that the conflict statuses are exactly the three a retry can overcome, over the whole
 *  vocabulary, and that the I/O status names and numbers itself. */
static void test_status_vocabulary_names_conflicts_and_input_output() {
    for (status_t const status : every_status_k) {
        bool const retryable = status == status_t::write_conflict_k || status == status_t::read_conflict_k ||
                               status == status_t::phantom_conflict_k;
        expect_equal(conflicted(status), retryable, "{}", name_of(status));
        expect_true(std::string_view {name_of(status)} != std::string_view {"unrecognized"});
    }
    expect_equal(static_cast<int>(status_t::input_output_k), EIO, "the status follows errno where one fits");
    expect_equal(std::string_view {name_of(status_t::input_output_k)}, std::string_view {"input_output_k"});
}

#pragma endregion Optimistic Concurrency Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

#pragma region Counted Allocations

void *operator new(std::size_t bytes, std::nothrow_t const &) noexcept {
    test_run.nothrow_allocations.fetch_add(1, std::memory_order_relaxed);
    try {
        return ::operator new(bytes); // allocator primitive
    }
    catch (...) {
        return nullptr;
    }
}

void *operator new(std::size_t bytes, std::align_val_t alignment, std::nothrow_t const &) noexcept {
    test_run.nothrow_allocations.fetch_add(1, std::memory_order_relaxed);
    try {
        return ::operator new(bytes, alignment); // allocator primitive
    }
    catch (...) {
        return nullptr;
    }
}

void operator delete(void *pointer, std::nothrow_t const &) noexcept {
    ::operator delete(pointer); // allocator primitive
}

void operator delete(void *pointer, std::align_val_t alignment, std::nothrow_t const &) noexcept {
    ::operator delete(pointer, alignment); // allocator primitive
}

#pragma endregion Counted Allocations

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_expected_construction_matrix", test_expected_construction_matrix);
    run_test_case("test_expected_moves_stay_balanced", test_expected_moves_stay_balanced);
    run_test_case("test_expected_decomposes", test_expected_decomposes);
    run_test_case("test_expected_carries_move_only", test_expected_carries_move_only);

    run_test_case("test_allocator_refuses_overflowing_counts", test_allocator_refuses_overflowing_counts);

    run_test_case("test_shared_mutex_excludes<spin_shared_mutex_t>", test_shared_mutex_excludes<spin_shared_mutex_t>);
    run_test_case("test_shared_mutex_excludes<forkunion_mutex_t>", test_shared_mutex_excludes<forkunion_mutex_t>);
    run_test_case("test_shared_mutex_keeps_room_above_its_readers<spin_shared_mutex_t>",
                  test_shared_mutex_keeps_room_above_its_readers<spin_shared_mutex_t>);
    run_test_case("test_shared_mutex_keeps_room_above_its_readers<forkunion_mutex_t>",
                  test_shared_mutex_keeps_room_above_its_readers<forkunion_mutex_t>);
    run_test_case("test_shared_mutex_admits_every_writer<spin_shared_mutex_t>",
                  test_shared_mutex_admits_every_writer<spin_shared_mutex_t>);
    run_test_case("test_shared_mutex_admits_every_writer<forkunion_mutex_t>",
                  test_shared_mutex_admits_every_writer<forkunion_mutex_t>);

    run_test_case("test_extended_atomics_take_the_bounded_path", test_extended_atomics_take_the_bounded_path);
    run_test_case("test_extended_atomics_reach_the_pinned_table", test_extended_atomics_reach_the_pinned_table);
    run_test_case("test_extended_atomics_claim_each_bit_once", test_extended_atomics_claim_each_bit_once);
    run_test_case("test_waiting_policy_substitutes_in_the_mutex", test_waiting_policy_substitutes_in_the_mutex);
    run_test_case("test_waiting_policy_substitutes_in_the_slot_lock", test_waiting_policy_substitutes_in_the_slot_lock);

    run_test_case("test_transaction_group_walks_one_order", test_transaction_group_walks_one_order);
    run_test_case("test_transaction_group_rollback_stops_at_refusal", test_transaction_group_rollback_stops_at_refusal);
    run_test_case("test_transaction_group_refuses_a_duplicate_store", test_transaction_group_refuses_a_duplicate_store);
    run_test_case("test_transaction_group_reports_what_refused_to_open",
                  test_transaction_group_reports_what_refused_to_open);
    run_test_case("test_transaction_group_failed_unwind_resets", test_transaction_group_failed_unwind_resets);
    run_test_case("test_transaction_group_torn_commit_stops_claiming_staged",
                  test_transaction_group_torn_commit_stops_claiming_staged);
    run_test_case("test_transaction_group_split_commit_publishes_nothing_on_refusal",
                  test_transaction_group_split_commit_publishes_nothing_on_refusal);
    run_test_case("test_transaction_group_resets_every_participant_at_one_snapshot",
                  test_transaction_group_resets_every_participant_at_one_snapshot);
    run_test_case("test_dynamic_transaction_group_commits_kinds_under_one_stamp",
                  test_dynamic_transaction_group_commits_kinds_under_one_stamp);
    run_test_case("test_dynamic_transaction_group_conflict_publishes_nothing",
                  test_dynamic_transaction_group_conflict_publishes_nothing);
    run_test_case("test_dynamic_transaction_group_commits_many_stores_of_one_kind",
                  test_dynamic_transaction_group_commits_many_stores_of_one_kind);
    run_test_case("test_dynamic_transaction_group_joins_each_store_once",
                  test_dynamic_transaction_group_joins_each_store_once);
    run_test_case("test_dynamic_transaction_group_empties_on_commit_and_reset",
                  test_dynamic_transaction_group_empties_on_commit_and_reset);
    run_test_case("test_dynamic_group_allocates_nothing_once_warm", test_dynamic_group_allocates_nothing_once_warm);
    run_test_case("test_dynamic_group_allocates_nothing_across_retries",
                  test_dynamic_group_allocates_nothing_across_retries);
    run_test_case("test_dynamic_group_rebinds_spares_to_other_stores",
                  test_dynamic_group_rebinds_spares_to_other_stores);
    run_test_case("test_spares_outlive_their_store", test_spares_outlive_their_store);
    run_test_case("test_cleared_group_pins_nothing", test_cleared_group_pins_nothing);

    run_test_case("test_commit_order_resume_continues_stamps", test_commit_order_resume_continues_stamps);
    run_test_case("test_commit_order_resume_reaches_its_stores", test_commit_order_resume_reaches_its_stores);
    run_test_case("test_commit_order_resume_refusals", test_commit_order_resume_refusals);
    run_test_case("test_commit_order_parity_matches_ring_walk<commit_order_t>",
                  test_commit_order_parity_matches_ring_walk<commit_order_t>);
    run_test_case("test_commit_order_parity_matches_ring_walk<wide_order_t>",
                  test_commit_order_parity_matches_ring_walk<wide_order_t>);
    run_test_case("test_commit_order_parity_matches_ring_walk<forkunion_order_t>",
                  test_commit_order_parity_matches_ring_walk<forkunion_order_t>);
    run_test_case("test_commit_order_parity_matches_ring_walk<solitary_commit_order_t>",
                  test_commit_order_parity_matches_ring_walk<solitary_commit_order_t>);
    run_test_case("test_commit_order_parity_lands_racing_commits<narrow_order_t>",
                  test_commit_order_parity_lands_racing_commits<narrow_order_t>);
    run_test_case("test_commit_order_parity_lands_racing_commits<forkunion_narrow_order_t>",
                  test_commit_order_parity_lands_racing_commits<forkunion_narrow_order_t>);

    run_test_case("test_versioned_comparator_orders_by_key_then_generation",
                  test_versioned_comparator_orders_by_key_then_generation);

    run_test_case("test_commit_stamp_visibility_matrix", test_commit_stamp_visibility_matrix);
    run_test_case("test_isolation_levels_compare_by_strength", test_isolation_levels_compare_by_strength);
    run_test_case("test_validate_watches_catches_drift", test_validate_watches_catches_drift);
    run_test_case("test_per_version_equals_separates_versions", test_per_version_equals_separates_versions);
    run_test_case("test_commit_with_retries_is_bounded", test_commit_with_retries_is_bounded);
    run_test_case("test_status_vocabulary_names_conflicts_and_input_output",
                  test_status_vocabulary_names_conflicts_and_input_output);

    return finish_test_run();
}
