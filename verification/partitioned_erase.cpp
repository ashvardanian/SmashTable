/**
 *  @file verification/partitioned_erase.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief GenMC client for a window write spanning every partition of @c partitioned_store under
 *      one stamp, spelled over @c std::atomic with an explicit order at every site.
 *
 *  An eraser holds both partitions for the whole span, draws one stamp through @c begin_commit,
 *  writes both tombstones under it and lands the stamp once through @c end_commit, which is what
 *  @c publish_every_part_ does for @c erase_range and its neighbours. A reader takes a claim and
 *  reads each partition under that partition's mutex; a watcher holds nothing and claims nothing,
 *  reading the watermark with acquire and the tombstones relaxed, which is what shows the one stamp
 *  rather than the mutual exclusion makes the window whole. @c -Dwithout_one_stamp draws and lands
 *  a stamp per partition, and both of them catch the window half erased.
 *
 *  No census, as in @c partitioned_erase.pml: nothing here prunes, so a claim is the stamp
 *  @c take_snapshot hands back and the buckets, the floors and the low-water mark are left out.
 */
#include <atomic> // `std::atomic` - the mutexes, the tombstone stamps, the watermark and the ring

#include "genmc.hpp"

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

constexpr int partitions_k = 2; // the partitions one window write spans
constexpr int ring_k = 2;       // commits that may sit past the watermark, which is `ring_k`

spin_mutex_t partition_mutexes[partitions_k];
std::atomic<int> tombstones[partitions_k] = {0, 0}; // zero while the key is present
std::atomic<int> commits {0};
std::atomic<int> published_stamp {0};
std::atomic<int> landed[ring_k] = {0, 0};

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

/** Waits for the slot, releases the mark into it and walks the watermark in stamp order. Models
 *  @c basic_commit_order::end_commit. */
void land_stamp(int stamp) noexcept {
    // The slot still holds `stamp - ring_k`, and overwriting a mark the watermark has not consumed loses it.
    while (stamp - published_stamp.load(std::memory_order_acquire) > ring_k) {}
    landed[stamp % ring_k].store(stamp, std::memory_order_release);
    // Reading the newest, not merely loading: a plain load could miss a neighbour's mark and leave ours behind.
    int seen = published_stamp.fetch_add(0, std::memory_order_acq_rel);
    while (landed[(seen + 1) % ring_k].load(std::memory_order_acquire) == seen + 1) {
        int const before = raise_to(published_stamp, seen + 1);
        seen = before < seen + 1 ? seen + 1 : before;
    }
}

/** Reads the newest watermark, which is the whole claim where nothing prunes. Models
 *  @c basic_commit_order::take_snapshot. */
void take_snapshot(snapshot_claim_t &claim) noexcept {
    claim.snapshot = published_stamp.fetch_add(0, std::memory_order_acq_rel);
}

/** Holds both partitions ascending, writes the tombstones and lands the stamp once. Models
 *  @c partitioned_store::publish_every_part_. */
void *erase_window(void *) noexcept {
    partition_mutexes[0].lock();
    partition_mutexes[1].lock();
#ifdef without_one_stamp
    for (int partition = 0; partition != partitions_k; ++partition) {
        int const drawn = draw_stamp();
        tombstones[partition].store(drawn, std::memory_order_relaxed);
        land_stamp(drawn);
    }
#else
    int const drawn = draw_stamp();
    for (int partition = 0; partition != partitions_k; ++partition)
        tombstones[partition].store(drawn, std::memory_order_relaxed);
    land_stamp(drawn);
#endif
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

int main() {
    thread_t const eraser = spawn(erase_window);
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
