/**
 *  @file verification/atomic_hash_table/client.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief GenMC client for @c atomic_hash_table against its real header: two emplacers and an
 *      eraser over one bucket, and each mutant the same table over a reference that weakens the one
 *      order it changes.
 *
 *  The emplacers carry the keys of @c roomy.pml, each hashing to its own slot, or one key twice
 *  as in @c same_key.pml; the eraser looks for the first emplacer's key. All slots share one
 *  header word, so every lock and unlock lands on it. Keys are written and read as plain accesses
 *  under the slot lock, so a lock or unlock that publishes too little is a race GenMC reports.
 *  The bucket holds 32 slots, the smallest the header builds, of which the probes reach three.
 *
 *  - The populated count never goes below zero on the way.
 *  - Every slot ends unlocked, and the counters end equal to the occupied slots.
 *  - The key is never populated in two slots at once.
 *
 *  @verify pass sc,rc11,imm roomy
 *  @verify fail rc11,imm roomy_lock_acquire_false: with a relaxed @c fetch_or, the eraser reads the
 *      key of a populated slot it is not ordered after
 *  @verify fail rc11,imm roomy_unlock_release_false: with a relaxed @c fetch_xor, the eraser that
 *      takes the slot next reads the key of a populated slot it is not ordered after
 *  @verify pass sc,rc11,imm same_key
 *  @verify fail rc11,imm same_key_lock_acquire_false: with a relaxed @c fetch_or, the second
 *      emplacer reads the populated slot's key before it was written
 */
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <atomic> // `std::atomic_ref` - what a mutant's reference weakens

#include <genmc.hpp>

#include "smashtable/atomic_hash_table.hpp" // `atomic_hash_set` - the table under test

namespace st = ashvardanian::smashtable;

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool lock_acquire = true;
    bool unlock_release = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t lock_acquire_false_k {.lock_acquire = false};
constexpr knobs_t unlock_release_false_k {.unlock_release = false};

constexpr std::uint64_t first_key_k = 10;
constexpr std::uint64_t second_key_k = 11;

/** Sends the first key to the first slot and the second to the next. */
struct slot_per_key_hash_t {
    std::uint32_t operator()(std::uint64_t key) const noexcept { return static_cast<std::uint32_t>(key - first_key_k); }
};

/** Owns nothing: the storage is laid over a static arena, and is never grown or freed. */
struct arena_allocator_t {
    std::byte *allocate(std::size_t) noexcept { return nullptr; }
    void deallocate(std::byte *, std::size_t) noexcept {}
};

/** @c std::atomic_ref with the slot lock's @c fetch_or and the unlock's @c fetch_xor relaxed as a
 *  mutant asks; the table uses neither verb anywhere else. */
template <knobs_t const &knobs_>
struct weakened {
    template <typename scalar_type_>
    struct ref : std::atomic_ref<scalar_type_> {
        using std::atomic_ref<scalar_type_>::atomic_ref;
        scalar_type_ fetch_or(scalar_type_ bits, std::memory_order order) const noexcept {
            return std::atomic_ref<scalar_type_>::fetch_or(bits,
                                                           knobs_.lock_acquire ? order : std::memory_order_relaxed);
        }
        scalar_type_ fetch_xor(scalar_type_ bits, std::memory_order order) const noexcept {
            return std::atomic_ref<scalar_type_>::fetch_xor(bits,
                                                            knobs_.unlock_release ? order : std::memory_order_relaxed);
        }
    };
};

template <template <typename> class atomic_reference_>
using set_over = st::atomic_hash_set<std::uint64_t, slot_per_key_hash_t, st::equal_to_t, arena_allocator_t,
                                     st::bare_waiting_policy_t, atomic_reference_>;

/** The header's table for the pass entries, and one over a weakened reference for each mutant. */
template <knobs_t const &knobs_>
using table_t =
    std::conditional_t<&knobs_ == &header_k, set_over<st::atomic_ref>, set_over<weakened<knobs_>::template ref>>;

using storage_t = set_over<st::atomic_ref>::storage_type;

/** The bucket's memory, zeroed as a static: GenMC refuses the variable-length @c memset in
 *  @c hash_storage::make. */
alignas(64) std::byte arena[storage_t::memory_usage(st::hash_slots_count_t::from_slots(1))];

/** One bucket of storage over the arena, as @c hash_storage::make would lay it out. */
storage_t arena_storage() noexcept {
    storage_t storage;
    storage.memory = arena;
    storage.slots_count = st::hash_bucket_capacity_k;
    storage.retarget_regions();
    return storage;
}

template <knobs_t const &knobs_, std::uint64_t key_>
void *emplace(void *table) noexcept {
    [[maybe_unused]] auto const emplaced = static_cast<table_t<knobs_> *>(table)->emplace(key_);
    return nullptr;
}

template <knobs_t const &knobs_>
void *erase(void *table) noexcept {
    [[maybe_unused]] auto const erased = static_cast<table_t<knobs_> *>(table)->erase(first_key_k);
    return nullptr;
}

/** Checks one slot the probes reach, answering whether it holds the first emplacer's key. */
bool audit_slot(storage_t const &storage, st::hash_bucket_head_t const &head, std::uint32_t slot,
                std::uint32_t &populated, std::uint32_t &deleted) noexcept {
    st::hash_slot_state_t const state = st::hash_slot_state_of(head, std::uint32_t {1} << slot);
    verify(state != st::hash_slot_state_t::locked_k);
    populated += state == st::hash_slot_state_t::populated_k;
    deleted += state == st::hash_slot_state_t::deleted_k;
    return state == st::hash_slot_state_t::populated_k && storage.keys[slot] == first_key_k;
}

template <knobs_t const &knobs_, std::uint64_t second_key_>
int emplace_and_erase() noexcept {
    table_t<knobs_> table = table_t<knobs_>::adopt(arena_storage());
    thread_t const first_emplacer = spawn(emplace<knobs_, first_key_k>, &table);
    thread_t const second_emplacer = spawn(emplace<knobs_, second_key_>, &table);
    thread_t const eraser = spawn(erase<knobs_>, &table);
    verify(table.size() <= 2);
    join(eraser);
    join(second_emplacer);
    join(first_emplacer);

    storage_t const storage = std::move(table).release();
    // Copied whole: GenMC refuses a lane-sized read of a word the slot locks wrote as 64 bits.
    st::hash_bucket_head_t head;
    head.u64 = storage.headers[0].u64;
    std::uint32_t populated = 0, deleted = 0;
    int const holding_first_key = audit_slot(storage, head, 0, populated, deleted) +
                                  audit_slot(storage, head, 1, populated, deleted) +
                                  audit_slot(storage, head, 2, populated, deleted);
    verify(storage.populated_count == populated && storage.deleted_count == deleted);
    verify(holding_first_key <= 1);
    return 0;
}

extern "C" int roomy() { return emplace_and_erase<header_k, second_key_k>(); }
extern "C" int roomy_lock_acquire_false() { return emplace_and_erase<lock_acquire_false_k, second_key_k>(); }
extern "C" int roomy_unlock_release_false() { return emplace_and_erase<unlock_release_false_k, second_key_k>(); }
extern "C" int same_key() { return emplace_and_erase<header_k, first_key_k>(); }
extern "C" int same_key_lock_acquire_false() { return emplace_and_erase<lock_acquire_false_k, first_key_k>(); }
