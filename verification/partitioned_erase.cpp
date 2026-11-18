/**
 *  @file verification/partitioned_erase.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief GenMC client for a window write spanning every partition of @c partitioned_store under
 *      one stamp, spelled over @c std::atomic with an explicit order at every site.
 *
 *  An eraser holds both partitions for the whole span, draws one stamp through @c begin_commit,
 *  writes both tombstones under it and lands the stamp once through @c end_commit. A reader takes a
 *  claim and reads each partition under that partition's mutex; a watcher holds nothing and claims
 *  nothing, reading the watermark with acquire and the tombstones relaxed. No census, as in the
 *  model. The same scenario is <tt>partitioned_erase.pml</tt>.
 *
 *  @verify pass sc,rc11,imm partitioned_erase
 *  @verify fail sc,rc11,imm partitioned_erase_one_stamp_false: with a stamp per partition, a reader
 *      drawing between two publications names the first partition's stamp and not the second's, and
 *      both observers see the window half erased
 */
#include <atomic> // `std::atomic` - the mutexes, the tombstone stamps, the watermark and the word of landings
#include <bit>    // `std::countr_one`

#include <genmc.hpp>

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool one_stamp = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t one_stamp_false_k {.one_stamp = false};

/** A mutex as one word, taken by an acquiring exchange from zero, released by a releasing store. */
struct spin_mutex_t {
    std::atomic<int> word {0};

    void lock() noexcept {
        int expected = 0;
        while (!word.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed))
            expected = 0;
    }
    void unlock() noexcept { word.store(0, std::memory_order_release); }
};

/** One reader's claim on a snapshot, which with no census is the stamp alone. */
struct snapshot_claim_t {

    /** The stamp every read under this claim is answered at. */
    int snapshot {0};
};

constexpr int partitions_k = 2;             // the partitions one window write spans
constexpr int parity_bits_k = 4;            // the narrowest word whose window holds both stamps a mutant draws
constexpr int window_k = parity_bits_k / 2; // how far above the watermark a commit may land
constexpr unsigned parity_mask_k = (1u << parity_bits_k) - 1; // the bits of the word of landings

spin_mutex_t partition_mutexes[partitions_k];
std::atomic<int> tombstones[partitions_k] = {0, 0}; // zero while the key is present
std::atomic<int> commits {0};
std::atomic<int> published_stamp {0};
std::atomic<unsigned> landed_parity {0}; // bit `stamp % parity_bits_k` flips once that commit has landed

/** Raises @p word to @p floor and answers what it held, which is @c atomic_max_fetch. */
int raise_to(std::atomic<int> &word, int floor) noexcept {
    int observed = word.load(std::memory_order_acquire);
    while (observed < floor)
        if (word.compare_exchange_strong(observed, floor, std::memory_order_acq_rel, std::memory_order_acquire)) break;
    return observed;
}

/** Draws a stamp in one unconditional relaxed add, so a draw never waits and never refuses. Models
 *  @c basic_commit_order::begin_commit. */
int draw_stamp() noexcept { return commits.fetch_add(1, std::memory_order_relaxed) + 1; }

/** The word once every stamp through @p through landed and none above it. Models
 *  @c basic_commit_order::parity_through_. */
constexpr unsigned parity_through(int through) noexcept {
    unsigned const odd_laps = (through / parity_bits_k) & 1 ? parity_mask_k : 0u;
    unsigned const this_lap = parity_mask_k >> (parity_bits_k - 1 - through % parity_bits_k);
    return odd_laps ^ (this_lap & ~1u);
}

/** How many consecutive stamps above @p seen the word @p landed counts as landed. Models
 *  @c basic_commit_order::landed_run_. */
constexpr int landed_run(unsigned landed, int seen) noexcept {
    unsigned const above = landed ^ parity_through(seen);
    int const turns = (seen + 1) % parity_bits_k;
    return std::countr_one(((above >> turns) | (above << (parity_bits_k - turns))) & parity_mask_k);
}

/** Waits for room in the window, flips the stamp's bit and raises the watermark over the landed
 *  run. Models @c basic_commit_order::end_commit. */
void land_stamp(int stamp) noexcept {
    // A stamp a lap below shares this bit, so a flip outside the window reads as that stamp.
    while (stamp - published_stamp.load(std::memory_order_acquire) > window_k) {}
    unsigned const bit = 1u << (stamp % parity_bits_k);
    unsigned const landed = landed_parity.fetch_xor(bit, std::memory_order_acq_rel) ^ bit;
    // Loaded after the flip: a watermark older than the word only shortens the run.
    int const seen = published_stamp.load(std::memory_order_acquire);
    // The watermark passed this stamp since the word was read, so its bits are a lap stale.
    if (seen >= stamp) return;
    int const run = landed_run(landed, seen);
    if (run == 0) return;
    // Nobody parks here, so the wake that follows a raise has no step.
    raise_to(published_stamp, seen + run);
}

/** Reads the newest watermark, which is the whole claim where nothing prunes. Models
 *  @c basic_commit_order::take_snapshot. */
void take_snapshot(snapshot_claim_t &claim) noexcept {
    claim.snapshot = published_stamp.fetch_add(0, std::memory_order_acq_rel);
}

/** Holds both partitions ascending, writes the tombstones and lands the stamp once. Models
 *  @c partitioned_store::publish_every_part_. */
template <knobs_t const &knobs_>
void *erase_window(void *) noexcept {
    partition_mutexes[0].lock();
    partition_mutexes[1].lock();
    if constexpr (knobs_.one_stamp) {
        int const drawn = draw_stamp();
        for (int partition = 0; partition != partitions_k; ++partition)
            tombstones[partition].store(drawn, std::memory_order_relaxed);
        land_stamp(drawn);
    }
    else {
        for (int partition = 0; partition != partitions_k; ++partition) {
            int const drawn = draw_stamp();
            tombstones[partition].store(drawn, std::memory_order_relaxed);
            land_stamp(drawn);
        }
    }
    partition_mutexes[1].unlock();
    partition_mutexes[0].unlock();
    return nullptr;
}

/** The publication alone: the watermark acquired, the tombstones read with no partition held. */
void *watch_window(void *) noexcept {
    int const watermark = published_stamp.load(std::memory_order_acquire);
    bool covered[partitions_k] = {false, false};
    for (int partition = 0; partition != partitions_k; ++partition) {
        int const observed = tombstones[partition].load(std::memory_order_relaxed);
        covered[partition] = observed != 0 && observed <= watermark;
    }
    verify(covered[0] == covered[1]);
    return nullptr;
}

/** The eraser and the watcher on threads of their own, and the pinned reader on this one. */
template <knobs_t const &knobs_>
int erase_and_read() noexcept {
    thread_t const eraser = spawn(erase_window<knobs_>);
    thread_t const watcher = spawn(watch_window);

    // A pinned reader: the claim taken outside the partitions, each partition read under its own lock.
    snapshot_claim_t claim;
    take_snapshot(claim);
    bool erased[partitions_k] = {false, false};
    for (int partition = 0; partition != partitions_k; ++partition) {
        partition_mutexes[partition].lock();
        int const observed = tombstones[partition].load(std::memory_order_relaxed);
        erased[partition] = observed != 0 && observed <= claim.snapshot;
        partition_mutexes[partition].unlock();
    }
    verify(erased[0] == erased[1]);

    join(watcher);
    join(eraser);
    return 0;
}

extern "C" int partitioned_erase() { return erase_and_read<header_k>(); }
extern "C" int partitioned_erase_one_stamp_false() { return erase_and_read<one_stamp_false_k>(); }
