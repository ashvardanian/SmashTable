/**
 *  @brief GenMC client for a window write spanning two partitions of @c partitioned_store under one
 *      stamp, spelled over @c std::atomic words.
 *  @author Ash Vardanian
 *  @file verification/partitioned_erase.cpp
 *  @date September 15, 2026
 *
 *  An eraser holds both partitions, draws one stamp, writes both tombstones under it and moves the
 *  watermark once; a reader draws its snapshot under the clock's mutex and reads each partition under
 *  its mutex. @c -Dwithout_one_stamp draws and publishes a stamp per partition, and the reader sees the
 *  window half erased. The same scenario is @c partitioned_erase.pml.
 */
#include <atomic> // `std::atomic` - the mutexes, the tombstone stamps, the stamp counter and the watermark

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

spin_mutex_t partition_mutexes[2];
spin_mutex_t clock_mutex;
std::atomic<int> tombstones[2] = {0, 0}; // zero while unpublished
std::atomic<int> commits {0};
std::atomic<int> published_stamp {0};

/** @brief begin_commit: the next stamp, under the clock's mutex. */
int draw_stamp() noexcept {
    clock_mutex.lock();
    int const drawn = commits.load(std::memory_order_relaxed) + 1;
    commits.store(drawn, std::memory_order_relaxed);
    clock_mutex.unlock();
    return drawn;
}

/** @brief end_commit: the watermark moves to @p stamp, under the clock's mutex. */
void publish(int stamp) noexcept {
    clock_mutex.lock();
    published_stamp.store(stamp, std::memory_order_relaxed);
    clock_mutex.unlock();
}

void *erase_window(void *) noexcept {
    partition_mutexes[0].lock();
    partition_mutexes[1].lock();
#ifdef without_one_stamp
    for (int partition = 0; partition != 2; ++partition) {
        int const drawn = draw_stamp();
        tombstones[partition].store(drawn, std::memory_order_relaxed);
        publish(drawn);
    }
#else
    int const drawn = draw_stamp();
    for (int partition = 0; partition != 2; ++partition) tombstones[partition].store(drawn, std::memory_order_relaxed);
    publish(drawn);
#endif
    partition_mutexes[1].unlock();
    partition_mutexes[0].unlock();
    return nullptr;
}

int main() {
    thread_t const eraser = spawn(erase_window);

    clock_mutex.lock();
    int const snapshot = published_stamp.load(std::memory_order_relaxed);
    clock_mutex.unlock();

    bool erased[2] = {false, false};
    for (int partition = 0; partition != 2; ++partition) {
        partition_mutexes[partition].lock();
        int const observed = tombstones[partition].load(std::memory_order_relaxed);
        erased[partition] = observed != 0 && observed <= snapshot;
        partition_mutexes[partition].unlock();
    }
    verify(erased[0] == erased[1]);

    join(eraser);
    return 0;
}
