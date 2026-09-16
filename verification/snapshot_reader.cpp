/**
 *  @brief GenMC client for the pinned reader of @c include/smashtable/snapshot_store.hpp against
 *      concurrent commits and reclamation, spelled over @c std::atomic words.
 *  @author Ash Vardanian
 *  @file verification/snapshot_reader.cpp
 *  @date September 15, 2026
 *
 *  Two committers write one key under the partition's mutex, each pruning its version run down to the
 *  low-water mark it reads without the clock's mutex; the main thread pins a reader, reads the key
 *  twice, then lets a transaction adopt the stamp and closes the reader before that transaction reads.
 *  @c -Dwithout_lease draws the stamp from the bare watermark, and @c -Dwithout_shared_lease copies it to
 *  the transaction without a claim of its own; each lets a prune free a version still being read. The
 *  same scenarios are @c snapshot_reader.pml.
 */
#include <atomic> // `std::atomic` - the mutexes, the stamp counter, the watermark and the low-water mark

#include "genmc.hpp"

/** @brief A mutex as one word, taken by an acquiring exchange from zero and given back by a releasing store. */
struct spin_mutex_t {
    std::atomic<int> word {0};

    void lock() noexcept {
        int expected = 0;
        while (!word.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed))
            expected = 0;
    }
    void unlock() noexcept { word.store(0, std::memory_order_release); }
};

constexpr int versions_k = 3;
constexpr int none_k = -1;

spin_mutex_t partition_mutex;
spin_mutex_t clock_mutex;
std::atomic<int> commits {0};
std::atomic<int> published_stamp {0};
std::atomic<int> low_water_mark {0};

// The key's version run and the census, each ordered by the mutex that guards it.
int version_stamp[versions_k] = {0, none_k, none_k};
bool version_freed[versions_k] = {false, false, false};
int versions_written = 1;
bool reader_live = false;
int reader_snapshot = 0;
bool adopter_live = false;
int adopter_snapshot = 0;

/** @brief republish_mark_, under the clock's mutex: the oldest claim, or the watermark when nobody reads. */
void republish_mark() noexcept {
    int mark = published_stamp.load(std::memory_order_relaxed);
    if (reader_live) mark = reader_snapshot;
    if (adopter_live && (!reader_live || adopter_snapshot < mark)) mark = adopter_snapshot;
    low_water_mark.store(mark, std::memory_order_relaxed);
}

/** @brief Whether @p version is newer than @p found among those @p snapshot covers. */
bool supersedes(int version, int found, int snapshot) noexcept {
    return version_stamp[version] != none_k && version_stamp[version] <= snapshot &&
           (found == none_k || version_stamp[version] > version_stamp[found]);
}

/**
 *  @brief visible_version_, under the partition's mutex: the newest version the snapshot covers.
 *    Spelled without a loop, since GenMC gives every loop only as many turns as the runner's unroll.
 */
int resolve(int snapshot) noexcept {
    int found = none_k;
    if (supersedes(0, found, snapshot)) found = 0;
    if (supersedes(1, found, snapshot)) found = 1;
    if (supersedes(2, found, snapshot)) found = 2;
    return found;
}

/** @brief prune_key_of_, under the partition's mutex: every version at or below @p mark but its survivor freed. */
void prune(int mark) noexcept {
    int const survivor = resolve(mark);
    if (version_stamp[0] != none_k && version_stamp[0] <= mark && survivor != 0) version_freed[0] = true;
    if (version_stamp[1] != none_k && version_stamp[1] <= mark && survivor != 1) version_freed[1] = true;
    if (version_stamp[2] != none_k && version_stamp[2] <= mark && survivor != 2) version_freed[2] = true;
}

void *commit(void *) noexcept {
    partition_mutex.lock();
    clock_mutex.lock();
    int const drawn = commits.load(std::memory_order_relaxed) + 1;
    commits.store(drawn, std::memory_order_relaxed);
    clock_mutex.unlock();

    version_stamp[versions_written++] = drawn;

    clock_mutex.lock();
    published_stamp.store(drawn, std::memory_order_relaxed);
    republish_mark();
    clock_mutex.unlock();

    prune(low_water_mark.load(std::memory_order_relaxed));
    partition_mutex.unlock();
    return nullptr;
}

int main() {
    thread_t const committers[2] = {spawn(commit), spawn(commit)};

#ifdef without_lease
    int const snapshot = published_stamp.load(std::memory_order_relaxed);
#else
    clock_mutex.lock();
    int const snapshot = published_stamp.load(std::memory_order_relaxed);
    reader_live = true;
    reader_snapshot = snapshot;
    republish_mark();
    clock_mutex.unlock();
#endif

    partition_mutex.lock();
    int const first = resolve(snapshot);
    verify(first != none_k && !version_freed[first]);
    partition_mutex.unlock();
    partition_mutex.lock();
    int const second = resolve(snapshot);
    verify(second == first && !version_freed[second]);
    partition_mutex.unlock();

    clock_mutex.lock();
#ifdef without_shared_lease
    adopter_snapshot = snapshot;
#else
    adopter_live = true;
    adopter_snapshot = snapshot;
#endif
    reader_live = false;
    republish_mark();
    clock_mutex.unlock();

    partition_mutex.lock();
    int const adopted = resolve(adopter_snapshot);
    verify(adopted != none_k && !version_freed[adopted]);
    partition_mutex.unlock();

    join(committers[0]);
    join(committers[1]);
    return 0;
}
