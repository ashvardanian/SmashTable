/**
 *  @file verification/snapshot_reader.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief GenMC client for @c snapshot_store::reader_t against a commit that prunes behind it,
 *      spelled over @c std::atomic with an explicit order at every site.
 *
 *  A commit draws its stamp through @c begin_commit outside the partition's mutex, joins its
 *  version to the key's run under it, lands the stamp through @c end_commit, and prunes the run
 *  under the mutex down to the mark that landing computed. Readers are counted per bucket, never
 *  listed one by one: @c take_snapshot joins the head bucket and then reads the watermark,
 *  @c share_snapshot counts a transaction in a live claim's bucket, @c retire_snapshot gives a
 *  member back, and @c republish_mark walks the watermark and the occupied floors, which is what
 *  @c basic_commit_order does in @c shared.hpp. A claim is a bucket index and a stamp.
 *
 *  @c -Dscenario=adoption has a transaction adopt the reader's stamp and read after the reader has
 *  closed; the default reads twice through the reader itself. @c -Dwithout_held_claim gives the
 *  claim back before the reads, @c -Dwithout_join_first reads the watermark before joining the
 *  bucket, and @c -Dwithout_shared_claim adopts the stamp with no member of its own: each lets a
 *  prune free a version still being read. The same scenarios are @c snapshot_reader.pml.
 *
 *  One commit, as in the model. With two, @c open_next_bucket_ has two openers that can each store
 *  one bucket's floor with a plain relaxed store and no order between them - benign, since either
 *  mark was computed before any member of that bucket read its own stamp - but GenMC's in-place
 *  revisiting refuses unordered writes outright, so a second committer makes the client unrunnable
 *  rather than wrong. @c -Dwithout_watermark_first is the weakening that needs the second commit,
 *  and @c snapshot_reader.pml carries it.
 *
 *  Left out, as in the model: the cached @c low_water_mark_ word, since a commit prunes at the mark
 *  it just computed, which is the freshest such a read can be; and @c generation_, which dates
 *  transactions rather than their visibility.
 */
#include <atomic> // `std::atomic` - the mutex, the stamp counter, the watermark, the ring and the census

#include "genmc.hpp"

/** The knob's values are integers, so a typo fails the range check below. */
#define reading 1
#define adoption 2
#ifndef scenario
#define scenario reading
#endif
#if scenario < reading || scenario > adoption
#error "scenario is reading or adoption"
#endif

/** A mutex in one word: an acquiring exchange takes it from zero, a releasing store returns it. */
struct spin_mutex_t {
    std::atomic<int> word {0};

    void lock() noexcept {
        int expected = 0;
        while (!word.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed))
            expected = 0;
    }
    void unlock() noexcept { word.store(0, std::memory_order_release); }
};

/** A bucket's word: open snapshots in the low half, the head that opened it in the high half. */
using bucket_word_t = unsigned long long;

/** One reader's claim on a snapshot, which is a bucket index and a stamp rather than a pointer. */
struct snapshot_claim_t {

    /** Which bucket counts this reader, meaningless while the claim is not held. */
    int bucket {0};

    /** The stamp every read under this claim is answered at. */
    int snapshot {0};
};

constexpr int versions_k = 2;  // the commit's version, beside the one every snapshot starts on
constexpr int none_k = -1;     // what a resolve covering nothing answers
constexpr int ring_k = 2;      // commits that may sit past the watermark, which is `ring_k`
constexpr int buckets_k = 2;   // floors the census keeps, which is `buckets_k`
constexpr int no_floor_k = -1; // what a scan over unoccupied buckets answers
constexpr bucket_word_t open_snapshots_mask_k = 0xFFFFFFFFull; // where a bucket's count ends and its head value begins

spin_mutex_t partition_mutex;
std::atomic<int> commits {0};
std::atomic<int> published_stamp {0};
std::atomic<int> landed[ring_k] = {0, 0};
std::atomic<bucket_word_t> head {0};
std::atomic<bucket_word_t> snapshots[buckets_k] = {0, 0};
std::atomic<int> floors[buckets_k] = {0, 0};

/*  The key's version run, ordered by the partition's mutex alone. */
int version_stamp[versions_k] = {0, none_k};
bool version_freed[versions_k] = {false, false};
int versions_written = 1;

/** Raises @p word to @p floor and answers what it held, never lowering it: @c atomic_max_fetch. */
template <typename word_type_>
word_type_ raise_to(std::atomic<word_type_> &word, word_type_ floor) noexcept {
    word_type_ observed = word.load(std::memory_order_acquire);
    while (observed < floor)
        if (word.compare_exchange_strong(observed, floor, std::memory_order_acq_rel, std::memory_order_acquire)) break;
    return observed;
}

/** Models @c basic_commit_order::record_snapshot_: counts one snapshot into the head bucket,
 *  refused only when a bucket reopened ahead of this thread carries a head above the one read. */
int record_snapshot() noexcept {
    for (;;) {
        bucket_word_t const opened = head.load(std::memory_order_acquire);
        int const bucket = static_cast<int>(opened % buckets_k);
        bucket_word_t const ceiling = ((opened & open_snapshots_mask_k) << 32) | open_snapshots_mask_k;
        bucket_word_t before = snapshots[bucket].load(std::memory_order_acquire);
        while (before + 1 <= ceiling)
            if (snapshots[bucket].compare_exchange_strong(before, before + 1, std::memory_order_acq_rel,
                                                          std::memory_order_acquire))
                break;
        if (before < ceiling) return bucket;
    }
}

/** Models @c basic_commit_order::open_next_bucket_: opens the bucket after @p opened, shuts it to
 *  arrivals, then stores its floor at @p watermark, so nobody is counted in as the floor moves. */
void open_next_bucket(bucket_word_t opened, int watermark) noexcept {
    bucket_word_t const next = opened + 1;
    int const bucket = static_cast<int>(next % buckets_k);
    bucket_word_t observed = snapshots[bucket].load(std::memory_order_acquire);
    if ((observed & open_snapshots_mask_k) != 0) return;
    // Another opener got here from the same head value: its floor store and this one would race.
    if ((observed >> 32) == (next & open_snapshots_mask_k)) return;
    if (!snapshots[bucket].compare_exchange_strong(observed, (next & open_snapshots_mask_k) << 32,
                                                   std::memory_order_acq_rel, std::memory_order_relaxed))
        return;
    floors[bucket].store(watermark, std::memory_order_relaxed);
    raise_to(head, next);
}

/** @p least lowered to bucket @p bucket's floor, where that bucket carries a member. */
int folded_floor(int bucket, int least) noexcept {
    if ((snapshots[bucket].load(std::memory_order_acquire) & open_snapshots_mask_k) == 0) return least;
    int const floor = floors[bucket].load(std::memory_order_relaxed);
    return least == no_floor_k || floor < least ? floor : least;
}

/** The least floor over the occupied buckets, or @c no_floor_k where nobody is counted anywhere.
 *  Spelled without a loop, since GenMC bounds every loop to the runner's unroll count. */
int scan_floors() noexcept { return folded_floor(1, folded_floor(0, no_floor_k)); }

/**
 *  @brief republish_mark_: the watermark read newest first, then the buckets, so a reader joining
 *      after that read draws at or above it and one joining before it is counted by the scan.
 *  @return The least of the watermark and every occupied bucket's floor.
 */
int republish_mark() noexcept {
#ifdef without_watermark_first
    int const least = scan_floors();
    bucket_word_t const opened = head.load(std::memory_order_acquire);
    int mark = published_stamp.fetch_add(0, std::memory_order_acq_rel);
#else
    int mark = published_stamp.fetch_add(0, std::memory_order_acq_rel);
    bucket_word_t const opened = head.load(std::memory_order_acquire);
    int const least = scan_floors();
#endif
    if (least != no_floor_k && least < mark) mark = least;
    // Sealed here rather than on a timer: a head bucket whose floor has fallen behind the mark is one
    // every later reader over-retains on.
    if (floors[opened % buckets_k].load(std::memory_order_relaxed) < mark) open_next_bucket(opened, mark);
    return mark;
}

/**
 *  @brief take_snapshot: the head bucket joined, then the watermark read newest, and nothing
 *      re-read - a mark computed after the join counts the member, and one computed before it
 *      stands at or below the stamp returned.
 *  @return The snapshot @p claim now reads at.
 */
int take_snapshot(snapshot_claim_t &claim) noexcept {
#ifdef without_join_first
    claim.snapshot = published_stamp.fetch_add(0, std::memory_order_acq_rel);
    claim.bucket = record_snapshot();
#else
    claim.bucket = record_snapshot();
    claim.snapshot = published_stamp.fetch_add(0, std::memory_order_acq_rel);
#endif
    return claim.snapshot;
}

/**
 *  @brief share_snapshot: one more member of the bucket @p held is counted in, posted with release
 *      and never read back, since a live claim's bucket carries a member and so cannot be stale.
 *  @return The snapshot @p claim now reads at.
 */
int share_snapshot(snapshot_claim_t const &held, snapshot_claim_t &claim) noexcept {
    snapshots[held.bucket].fetch_add(1, std::memory_order_release);
    claim.bucket = held.bucket;
    claim.snapshot = held.snapshot;
    return claim.snapshot;
}

/** Models @c basic_commit_order::retire_snapshot_: takes one snapshot back out with a release, so
 *  whoever sees a bucket drain sees its members' claims; only a drain recomputes the mark. */
void retire_snapshot(snapshot_claim_t &claim) noexcept {
    bucket_word_t const before = snapshots[claim.bucket].fetch_sub(1, std::memory_order_release);
    if ((before & open_snapshots_mask_k) == 1) republish_mark();
}

/** Draws a stamp in one unconditional relaxed add, so a draw never waits and never refuses. Models
 *  @c basic_commit_order::begin_commit. */
int draw_stamp() noexcept { return commits.fetch_add(1, std::memory_order_relaxed) + 1; }

/**
 *  @brief end_commit: the slot waited for, the mark released into it, the watermark walked in stamp
 *      order, and the low-water mark republished where the walk moved it.
 *  @return The mark this commit may prune at, which is zero where its walk moved nothing.
 */
int land_stamp(int stamp) noexcept {
    // The slot still holds `stamp - ring_k`, and overwriting a mark the watermark has not consumed loses it.
    while (stamp - published_stamp.load(std::memory_order_acquire) > ring_k) {}
    landed[stamp % ring_k].store(stamp, std::memory_order_release);
    // Reading the newest, not merely loading: a plain load could miss a neighbour's mark and leave ours behind.
    int seen = published_stamp.fetch_add(0, std::memory_order_acq_rel);
    bool moved = false;
    while (landed[(seen + 1) % ring_k].load(std::memory_order_acquire) == seen + 1) {
        int const before = raise_to(published_stamp, seen + 1);
        moved = moved || before < seen + 1;
        seen = before < seen + 1 ? seen + 1 : before;
    }
    return moved ? republish_mark() : 0;
}

/** Whether @p version is newer than @p found among those @p snapshot covers. */
bool supersedes(int version, int found, int snapshot) noexcept {
    return version_stamp[version] != none_k && version_stamp[version] <= snapshot &&
           (found == none_k || version_stamp[version] > version_stamp[found]);
}

/** Models @c snapshot_store::visible_version_: the newest version the snapshot covers. */
int resolve(int snapshot) noexcept {
    int found = none_k;
    if (supersedes(0, found, snapshot)) found = 0;
    if (supersedes(1, found, snapshot)) found = 1;
    return found;
}

/** Models @c snapshot_store::prune_key_of_: frees each version up to @p mark but its survivor. */
void prune(int mark) noexcept {
    int const survivor = resolve(mark);
    if (version_stamp[0] != none_k && version_stamp[0] <= mark && survivor != 0) version_freed[0] = true;
    if (version_stamp[1] != none_k && version_stamp[1] <= mark && survivor != 1) version_freed[1] = true;
}

/** One commit: draws a stamp, publishes the version under the mutex, lands it, prunes the run. */
void *commit(void *) noexcept {
    int const drawn = draw_stamp();
    partition_mutex.lock(); // publish_under: the version joins the key's run
    version_stamp[versions_written++] = drawn;
    partition_mutex.unlock();
    int const mark = land_stamp(drawn);
    partition_mutex.lock(); // prune_committed: at the mark this commit just computed
    prune(mark);
    partition_mutex.unlock();
    return nullptr;
}

int main() {
    thread_t const committer = spawn(commit);

    // reader_t: the claim joins the census and holds the stamp every read is answered at
    snapshot_claim_t reader;
    int const snapshot = take_snapshot(reader);

#if scenario == reading
#ifdef without_held_claim
    retire_snapshot(reader);
#endif
    // reader_t::find, twice, each under the partition's mutex and neither writing anything
    partition_mutex.lock();
    int const first = resolve(snapshot);
    verify(first != none_k && !version_freed[first]);
    partition_mutex.unlock();
    partition_mutex.lock();
    int const second = resolve(snapshot);
    verify(second == first && !version_freed[second]);
    partition_mutex.unlock();
#ifndef without_held_claim
    retire_snapshot(reader);
#endif
#else
    // transaction_t(store, reader): the transaction counted in the reader's bucket while the reader is in it
    snapshot_claim_t adopting;
#ifdef without_shared_claim
    adopting.bucket = reader.bucket;
    adopting.snapshot = reader.snapshot;
#else
    share_snapshot(reader, adopting);
#endif
    // the reader closes, leaving the transaction to pin the stamp both of them read at
    retire_snapshot(reader);
    partition_mutex.lock();
    int const adopted = resolve(adopting.snapshot);
    verify(adopted != none_k && !version_freed[adopted]);
    partition_mutex.unlock();
#ifndef without_shared_claim
    retire_snapshot(adopting);
#endif
#endif

    join(committer);
    return 0;
}
