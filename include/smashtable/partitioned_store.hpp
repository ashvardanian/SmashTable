/**
 *  @brief Shards a transactional store across independently locked partitions, so writers touching different keys
 *      rarely contend.
 *  @author Ash Vardanian
 *  @file include/smashtable/partitioned_store.hpp
 *  @date October 16, 2022
 */
#pragma once
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`, `std::uint64_t`

#include <array>       // `std::array`
#include <bit>         // `std::countr_zero`
#include <type_traits> // `std::conditional_t`, `std::void_t`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief What a store keeping no stamps contributes to a shard set, which is nothing at all.
 *    Named so the members below can be declared unconditionally and cost nothing where there is no clock.
 */
struct no_clock_t {
    /** @brief The claim a reader of such a store never takes. */
    using snapshot_lease_t = no_clock_t;
};

/** @brief The clock a store shares with its siblings, or @c no_clock_t when it keeps no stamps. */
template <typename store_type_, typename = void>
struct shared_clock_of {
    using type = no_clock_t;
};
template <typename store_type_>
struct shared_clock_of<store_type_, std::void_t<typename store_type_::clock_t>> {
    using type = typename store_type_::clock_t;
};

/** @brief Moves as many constructed objects out of @p from as @p sequence_ names, into a fixed-size array. */
template <typename type_, std::size_t... sequence_>
constexpr std::array<type_, sizeof...(sequence_)> move_to_array(type_ *from,
                                                                std::index_sequence<sequence_...>) noexcept {
    return {{std::move(from[sequence_])...}};
}

/**
 *  @brief Builds a fixed-size array from @p generator, or nothing at all if any element refuses.
 *
 *  The generator hands back an @c expected per element, and a single failure destroys the prefix
 *  already built, so no half-populated array is ever observable. The scratch buffer the elements are
 *  assembled in is raw storage this function owns, so every element it constructs it also destroys -
 *  a moved-from element still has a destructor to run.
 */
template <typename type_, std::size_t count_, typename generator_type_>
static expected<std::array<type_, count_>> generate_array_safely(generator_type_ &&generator) noexcept {
    alignas(type_) char raw_parts_mem[count_ * sizeof(type_)];
    type_ *raw_parts = reinterpret_cast<type_ *>(raw_parts_mem);
    for (std::size_t partition_index = 0; partition_index != count_; ++partition_index) {

        if (auto new_part = generator(partition_index); new_part)
            new (raw_parts + partition_index) type_(std::move(*new_part));
        else {
            // Destruct all the previous parts.
            for (std::size_t destructed_index = 0; destructed_index != partition_index; ++destructed_index)
                raw_parts[destructed_index].~type_();
            return {};
        }
    }

    std::array<type_, count_> moved = move_to_array(raw_parts, std::make_index_sequence<count_> {});
    for (std::size_t partition_index = 0; partition_index != count_; ++partition_index)
        raw_parts[partition_index].~type_();
    return moved;
}

/**
 *  @brief Hashes inputs to route them into separate sets, which can
 *    be concurrent, or have a separate state-full allocator attached.
 *
 *  @tparam store_type_ Type of the wrapped store, like @c monotonic_store.
 *  @tparam hash_type_ Keys that compare equal must have the same hashes.
 *  @tparam shared_mutex_type_ Mutex type to use for partition locking, like @c std::shared_mutex.
 *  @tparam partitions_count_ Number of partitions to split the store into, default 16.
 *
 *  @warning Every callback runs with a partition lock held, and the range walks hold all of them at
 *    once. The mutex is not recursive, so a callback that calls back into this store - or into
 *    anything that eventually does - deadlocks against itself. Copy out what a callback needs and do
 *    the rest after it returns.
 *  @warning Moving a store leaves every open transaction pointing at the husk, whose partitions are
 *    empty and whose mutexes guard nothing, so commits land nowhere and report success. Move only a
 *    store no transaction is open on and no other thread is touching.
 */
template <typename store_type_, typename hash_type_ = hash<typename store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
class partitioned_store {

  public:
    static constexpr std::size_t partitions_k = partitions_count_;
    using self_t = partitioned_store;
    using hash_t = hash_type_;
    using inner_store_t = store_type_;
    using inner_transaction_t = typename inner_store_t::transaction_t;
    using mutex_t = shared_mutex_type_;
    using shared_lock_t = shared_lock<mutex_t>;
    using unique_lock_t = unique_lock<mutex_t>;

    using mutexes_t = std::array<mutex_t, partitions_k>;
    using partitions_t = std::array<inner_store_t, partitions_k>;
    using partition_transactions_t = std::array<inner_transaction_t, partitions_k>;

    using value_t = typename inner_store_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    /** @brief The clock every partition draws its stamps from, or @c no_clock_t where there are none. */
    using clock_t = typename shared_clock_of<inner_store_t>::type;

    /**
     *  @brief Whether the part decides visibility by stamp and lets a shard set hand every partition
     *    the very same clock, which is what a snapshot spanning partitions is made of.
     */
    static constexpr bool inner_shares_clock_k =
        requires(inner_store_t &store, clock_t &clock, typename inner_store_t::generation_t stamp) {
            store.attach_clock(clock);
            store.transaction_at(stamp, stamp);
        };

    /**
     *  @brief A commit takes and releases one partition's lock at a time, so a reader crossing
     *    partitions can catch a transaction half-applied - unless what a reader sees is decided by a
     *    stamp rather than by what happens to be published when it looks.
     *
     *  A part sharing a clock is stamped, and one stamp spans every partition of a commit: a reader
     *  that fixed its snapshot below that stamp sees none of the commit and one at or above it sees
     *  all of it, whatever order the partitions were written in. Such a part keeps its own promise
     *  whole. A part deciding visibility any other way is capped, and one weaker than
     *  @c read_committed_k stays the answer.
     */
    static constexpr isolation_t isolation_k =
        partitions_k == 1 || inner_shares_clock_k || inner_store_t::isolation_k < isolation_t::read_committed_k
            ? inner_store_t::isolation_k
            : isolation_t::read_committed_k;

    using comparator_t = typename inner_store_t::comparator_t;
    using identifier_t = typename inner_store_t::identifier_t;
    using generation_t = typename inner_store_t::generation_t;

    /**
     *  @brief Whether the partitioned store carries the ordered surface, which an unordered core denies it.
     *    The forwards below are gated on this, so an unordered store loses them at overload resolution
     *    rather than deep inside an instantiation.
     */
    static constexpr bool inner_is_ordered_k =
        requires(inner_store_t &store, identifier_t const &key, no_op_t callback) {
            store.lower_bound(key, callback, callback);
            store.upper_bound(key, callback, callback);
            store.range(key, key, callback);
            store.erase_range(key, key, callback);
        };

    /** @brief Whether a partition can draw members of a range at random. */
    static constexpr bool inner_is_samplable_k =
        requires(inner_store_t const &store, identifier_t const &key, no_op_t callback, std::size_t seen) {
            store.sample_one(key, key, callback, callback);
            store.sample_reservoir(key, key, callback, seen, seen, callback);
        };

    /** @brief Whether an open transaction carries the ordered surface its store does. */
    static constexpr bool inner_transaction_is_ordered_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.upper_bound(key, callback, callback);
        };

    /** @brief Whether a partition enumerates its members with no ordering to walk them in. */
    static constexpr bool inner_enumerates_k =
        requires(inner_store_t const &store, no_op_t callback) { store.for_each(callback); };

    /** @brief Whether a partition reclaims superseded versions on demand. */
    static constexpr bool inner_reclaims_k = requires(inner_store_t &store) { store.vacuum(); };

    /** @brief Whether a partition reclaims a window of the keyspace rather than all of it. */
    static constexpr bool inner_reclaims_range_k =
        requires(inner_store_t &store, identifier_t const &key) { store.vacuum(key, key); };

    /**
     *  @brief Whether a partition answers by ordinal, which only an order-statistics core does.
     *    The forwards below are gated on this, so a core keeping no subtree counts loses them at
     *    overload resolution rather than deep inside an instantiation.
     */
    static constexpr bool inner_is_ranked_k =
        requires(inner_store_t const &store, identifier_t const &key, std::size_t ordinal, no_op_t callback) {
            store.select(ordinal, callback, callback);
            store.rank(key, callback, callback);
        };

    /** @brief Whether a partition rewrites the mapped side of a range in place. */
    static constexpr bool inner_revises_range_k = requires(
        inner_store_t &store, identifier_t const &key, no_op_t callback) { store.update_range(key, key, callback); };

    /** @brief Whether a partition refuses an occupied key rather than writing over it. */
    static constexpr bool inner_refuses_occupied_key_k =
        requires(inner_store_t &store, value_t &&element) { store.insert(std::move(element)); };

    /** @brief Whether that refusal also says which element declined the insert. */
    static constexpr bool inner_reports_occupied_key_k =
        requires(inner_store_t &store, value_t &&element, no_op_t callback) {
            store.insert(std::move(element), callback, callback);
        };

    /** @brief Whether a partition refuses an absent key rather than creating it. */
    static constexpr bool inner_refuses_absent_key_k =
        requires(inner_store_t &store, value_t &&element) { store.update(std::move(element)); };

    /** @brief Whether an open transaction refuses an occupied key rather than writing over it. */
    static constexpr bool inner_transaction_refuses_occupied_key_k =
        requires(inner_transaction_t &transaction, value_t &&element) { transaction.insert(std::move(element)); };

    /** @brief Whether that refusal, inside a transaction, also says which element declined the insert. */
    static constexpr bool inner_transaction_reports_occupied_key_k =
        requires(inner_transaction_t &transaction, value_t &&element, no_op_t callback) {
            transaction.insert(std::move(element), callback, callback);
        };

    /** @brief Whether an open transaction refuses an absent key rather than creating it. */
    static constexpr bool inner_transaction_refuses_absent_key_k =
        requires(inner_transaction_t &transaction, value_t &&element) { transaction.update(std::move(element)); };

    /** @brief Whether a partition erases an open-ended window of the keyspace. */
    static constexpr bool inner_erases_open_range_k =
        requires(inner_store_t &store, identifier_t const &key, no_op_t callback) {
            store.erase_from(key, callback);
            store.erase_up_to(key, callback);
        };

  private:
    /**
     *  @brief A set of partition indices, one bit each, sized to @c partitions_k rather than capped by it.
     *
     *  Iterating set bits rather than scanning every partition is the point: a transaction usually
     *  reaches one or two of them, and only those may be locked.
     */
    struct touched_partitions_t {
        static constexpr std::size_t bits_per_word_k = 64;
        static constexpr std::size_t words_k = (partitions_k + bits_per_word_k - 1) / bits_per_word_k;

        std::array<std::uint64_t, words_k> words {};

        void mark(std::size_t partition_index) noexcept {
            words[partition_index / bits_per_word_k] |= std::uint64_t {1} << (partition_index % bits_per_word_k);
        }
        void clear() noexcept { words.fill(0); }

        /** @brief The lowest marked partition, or @c partitions_k when none is. */
        std::size_t first_set() const noexcept { return scan_from_(0); }

        /** @brief The lowest marked partition above @p partition_index, or @c partitions_k when none is. */
        std::size_t next_set(std::size_t partition_index) const noexcept { return scan_from_(partition_index + 1); }

      private:
        std::size_t scan_from_(std::size_t partition_index) const noexcept {
            for (std::size_t word_index = partition_index / bits_per_word_k; word_index < words_k; ++word_index) {
                std::uint64_t left = words[word_index];
                // Mask off the bits below where this scan starts, only in the word it starts in.
                if (word_index == partition_index / bits_per_word_k) {
                    std::size_t const bit = partition_index % bits_per_word_k;
                    left &= bit == 0 ? ~std::uint64_t {0} : ~std::uint64_t {0} << bit;
                }
                if (left) return word_index * bits_per_word_k + static_cast<std::size_t>(countr_zero(left));
            }
            return partitions_k;
        }
    };

    std::size_t bucket_(identifier_t const &id) const noexcept { return hasher_(id) % partitions_k; }

    /**
     *  @brief Holds every partition for its own lifetime, taking them in ascending index order.
     *
     *  Order is the whole of the deadlock argument: every caller wants all of them, and every caller
     *  asks in this same sequence, so two threads cannot each hold what the other is waiting for.
     *  Acquiring what is available and spinning for the rest has no such argument - two threads that
     *  win disjoint subsets wait on each other for good, since neither gives back what it holds.
     *
     *  @tparam lock_type_ Either @c unique_lock_t or @c shared_lock_t, naming which of the two the
     *    whole array is taken under.
     */
    template <typename lock_type_>
    class every_part_lock {
        static constexpr bool exclusive_k = std::is_same<lock_type_, unique_lock_t>();
        static_assert(exclusive_k || std::is_same<lock_type_, shared_lock_t>());

        mutexes_t &mutexes_;

      public:
        explicit every_part_lock(mutexes_t &mutexes) noexcept : mutexes_(mutexes) {
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                if constexpr (exclusive_k) mutexes_[partition_index].lock();
                else mutexes_[partition_index].lock_shared();
            }
        }
        ~every_part_lock() noexcept {
            for (std::size_t partition_index = partitions_k; partition_index != 0; --partition_index) {
                if constexpr (exclusive_k) mutexes_[partition_index - 1].unlock();
                else mutexes_[partition_index - 1].unlock_shared();
            }
        }
        every_part_lock(every_part_lock const &) = delete;
        every_part_lock &operator=(every_part_lock const &) = delete;
    };

    /**
     *  @brief Holds exactly the partitions a transaction reached, ascending, until told to let go.
     *
     *  A commit spanning partitions has to ask all of them whether it may proceed before any of them
     *  writes, or a refusal from the last one would be reported over writes the first one already
     *  published. Ascending order is the same deadlock argument @c every_part_lock makes, and holding
     *  a subset takes nothing from it: a thread only ever waits on an index above everything it holds.
     */
    class touched_parts_lock {
        mutexes_t *mutexes_;
        touched_partitions_t const &touched_;

      public:
        touched_parts_lock(mutexes_t &mutexes, touched_partitions_t const &touched) noexcept
            : mutexes_(&mutexes), touched_(touched) {
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index))
                (*mutexes_)[partition_index].lock();
        }
        ~touched_parts_lock() noexcept { release(); }
        touched_parts_lock(touched_parts_lock const &) = delete;
        touched_parts_lock &operator=(touched_parts_lock const &) = delete;

        /** @brief Gives every held partition back early, so the work after a commit runs unlocked. */
        void release() noexcept {
            if (!mutexes_) return;
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index))
                (*mutexes_)[partition_index].unlock();
            mutexes_ = nullptr;
        }
    };

    /**
     *  @brief Which partition holds the next key of an ordered walk, when one of them does.
     *    @c index is only meaningful once @c presence says a partition claimed it, which is why the
     *    two travel together rather than as an index with a reserved value.
     */
    struct next_partition_t {
        /** @brief Whether any partition offered a key above the bound. */
        enum class presence_t : std::uint8_t {
            /** @brief Every partition ended before the bound, so the walk is over. */
            none_holds_it_k,
            /** @brief The partition at @c index holds the smallest key seen above the bound. */
            one_holds_it_k,
        };

        /** @brief Whether @c index names a partition at all. */
        presence_t presence = presence_t::none_holds_it_k;
        /** @brief The claiming partition, meaningful only under @c one_holds_it_k. */
        std::size_t index = 0;
    };

    /**
     * @brief Walks around all the parts, trying to perform operations on them, until all the tasks are exhausted.
     */
    template <typename lock_type_, typename partitions_type_, typename mutexes_type_, typename callable_type_>
    static status_t for_all(partitions_type_ &parts, mutexes_type_ &mutexes, callable_type_ &&callable) noexcept {
        status_t status = success_k;
        // Ascending order, one partition at a time, blocking. Taking whichever partitions happen to be
        // free and retrying the rest reads as politer, but it shares no order with `every_part_lock`,
        // and two all-partition operations without a common order are two operations that can wait on
        // each other for good.
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            lock_type_ lock {mutexes[partition_index]};
            status = callable(parts[partition_index]);
            if (failed(status)) return status;
        }
        return status;
    }

    template <typename partitions_type_, typename mutexes_type_, typename comparable_type_,
              typename callback_found_type_, typename callback_missing_type_>
    static void for_all_next_lookups(comparator_t const &comparator, partitions_type_ &parts, mutexes_type_ &mutexes,
                                     comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                     callback_missing_type_ &&callback_missing) noexcept {

        identifier_t smallest_id;

        // One pass over every partition, in ascending order - the same order every other all-partition
        // walk here uses, which is what makes them safe to run against each other. The bound arrives as
        // a const lvalue and reaches all @c partitions_k partitions that way, so nothing can move from it;
        // narrowing it to @c identifier_t first would refuse to compile for any comparable that is not
        // one, and would throw away the heterogeneous lookup the caller asked for.
        auto scan_for_smallest = [&](auto const &bound) noexcept {
            next_partition_t smallest;
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                shared_lock_t lock {mutexes[partition_index]};
                parts[partition_index].upper_bound(bound, [&](value_t const &element) noexcept {
                    if (smallest.presence == next_partition_t::presence_t::one_holds_it_k &&
                        !comparator(mapping_key_or_itself(element), smallest_id))
                        return;
                    smallest_id = identifier_t(element);
                    smallest.presence = next_partition_t::presence_t::one_holds_it_k;
                    smallest.index = partition_index;
                });
            }
            return smallest;
        };

        // Under "Read Committed" isolation the winner may be erased between the scan that found it and
        // the read below, which is the non-repeatable read that level permits. The re-read takes that
        // partition's lock, since losing the entry is allowed but reading it mid-update is not - that
        // would hand the callback a half-destroyed value.
        //
        // A miss then rescans from the missing key rather than from the original bound. Restarting from
        // the same place has no progress guarantee: a writer churning one key just above the cursor
        // makes the scan find it and lose it forever. Each rescan raises the bound strictly - the next
        // pass starts above a key just observed absent - so the walk terminates, and skipping a key
        // reinserted behind the cursor is the same thing this level already permits.
        next_partition_t smallest = scan_for_smallest(comparable);
        while (smallest.presence == next_partition_t::presence_t::one_holds_it_k) {
            bool vanished = false;
            {
                shared_lock_t lock {mutexes[smallest.index]};
                parts[smallest.index].find(smallest_id, callback_found, [&]() noexcept { vanished = true; });
            }
            if (!vanished) return;
            identifier_t const vanished_key = std::move(smallest_id);
            smallest = scan_for_smallest(vanished_key);
        }
        callback_missing();
    }

    /** @brief Whether a merged walk wants the key after the one its step was just handed. */
    enum class merge_control_t : std::uint8_t {
        /** @brief The step wants the next key of the merged order. */
        resume_k,
        /** @brief The step has what it came for, and the walk stops here. */
        halt_k,
    };

    /** @brief Whether a partition still has a key to contribute to a merged walk. */
    enum class front_state_t : std::uint8_t {
        /** @brief Every key this partition holds has already been handed to the step. */
        exhausted_k,
        /** @brief The partition's smallest unvisited key is waiting in the front. */
        holds_a_key_k,
    };

    /**
     *  @brief Walks the merged order of every partition, smallest key first, until @p step halts.
     *
     *  A front per partition, refilled from that partition's own successor once its key is consumed,
     *  which is the same exclusive-successor stepping @c for_all_next_lookups takes - what differs is
     *  that every partition is held for the whole walk rather than one at a time, so the sequence is
     *  one order taken at one moment rather than sixteen probes taken at sixteen.
     *
     *  @param[in] step Receives the partition index and the key, and says whether to carry on.
     *  @warning Every partition is locked shared for the length of the walk, so every writer waits.
     */
    template <typename step_type_>
    void walk_merged_order_(step_type_ &&step) const noexcept
        requires inner_is_ranked_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};

        std::array<identifier_t, partitions_k> fronts;
        std::array<front_state_t, partitions_k> states {};
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index)
            partitions_[partition_index].select(0, [&](value_t const &element) noexcept {
                fronts[partition_index] = identifier_t(element);
                states[partition_index] = front_state_t::holds_a_key_k;
            });

        while (true) {
            std::size_t smallest = partitions_k;
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                if (states[partition_index] != front_state_t::holds_a_key_k) continue;
                if (smallest != partitions_k && !comparator_(fronts[partition_index], fronts[smallest])) continue;
                smallest = partition_index;
            }
            if (smallest == partitions_k) return;
            if (step(smallest, fronts[smallest]) == merge_control_t::halt_k) return;

            // The bound outlives the front it came from, since refilling the front is what overwrites it.
            identifier_t const consumed = std::move(fronts[smallest]);
            states[smallest] = front_state_t::exhausted_k;
            partitions_[smallest].upper_bound(consumed, [&](value_t const &element) noexcept {
                fronts[smallest] = identifier_t(element);
                states[smallest] = front_state_t::holds_a_key_k;
            });
        }
    }

  public:
    class transaction_t {
        friend class partitioned_store;
        partitioned_store &store_;
        partition_transactions_t partitions_;
        /**
         *  @brief Which partitions this transaction reached, by write or by watch, so an untouched
         *    one is never locked.
         *
         *  A watched-but-unwritten partition is marked too: its read has to be validated at stage
         *  time, so it is as much a participant as a written one. One bit each, in as many words as
         *  @c partitions_k needs, and the walks below visit only the bits that are set - a transaction
         *  touching two partitions of a hundred pays for two.
         */
        touched_partitions_t touched_ {};

        /**
         *  @brief The one claim on the snapshot every partition of this transaction reads at.
         *    A part registers nothing of its own, so the reader census counts this transaction once
         *    rather than once per partition, and the low-water mark answers for all of them together.
         */
        ST_NO_UNIQUE_ADDRESS_ mutable typename clock_t::snapshot_lease_t lease_ {};

        /**
         *  @brief The stamp this transaction's own commit published, which it may not read below.
         *    Zero whenever this transaction is already reading at or above whatever it committed.
         */
        mutable generation_t unsettled_stamp_ {0};
        static_assert(std::is_nothrow_move_constructible<inner_transaction_t>());

        /**
         *  @brief Moves onto the snapshot this transaction's own commit created, once that commit is whole.
         *
         *  Deferred rather than done inside the commit, because the watermark is held back by every
         *  commit drawn earlier and still writing itself out: waiting there would make each commit
         *  wait for the slowest commit overlapping it, and a transaction that commits and goes away -
         *  which is nearly every transaction - never wanted the new snapshot at all. So the wait is
         *  paid by the next operation, and only when there is one.
         */
        void settle_snapshot_() const noexcept {
            if constexpr (inner_shares_clock_k) {
                if (unsettled_stamp_ == 0) return;
                clock_t &clock = store_.clock_;
                clock.await_published(static_cast<commit_stamp_t>(unsettled_stamp_));
                generation_t const moved = clock.take_snapshot(lease_);
                // Where a transaction reads is its own state, which the const read paths below own as
                // much as the writing ones do - the partitions themselves are not being written here.
                auto &parts = const_cast<partition_transactions_t &>(partitions_);
                for (inner_transaction_t &part : parts) part.adopt_snapshot(moved);
                unsettled_stamp_ = 0;
            }
        }

        /**
         *  @brief Runs @p callable over every partition, in ascending order, and never stops early.
         *
         *  Stopping on the first refusal would leave the partitions after it untouched while the ones
         *  before it had already acted, and nothing would record which side each fell on. The last
         *  refusal is reported.
         */
        template <typename callable_type_>
        status_t for_parts_(callable_type_ &&callable) noexcept {
            status_t status = success_k;
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                unique_lock_t lock {store_.mutexes_[partition_index]};
                status_t const one = callable(partitions_[partition_index]);
                if (failed(one)) status = one;
            }
            return status;
        }

        /** @brief The same walk, restricted to the partitions this transaction reached. */
        template <typename callable_type_>
        status_t for_touched_parts_(callable_type_ &&callable) noexcept {
            status_t status = success_k;
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index)) {
                unique_lock_t lock {store_.mutexes_[partition_index]};
                status_t const one = callable(partitions_[partition_index]);
                if (failed(one)) status = one;
            }
            return status;
        }

        /**
         *  @brief Publishes every reached partition under one stamp drawn for the whole commit.
         *
         *  The stamp is drawn before the first partition is written and the watermark is only moved
         *  once the last one has been, so a reader either names the stamp and sees the whole commit,
         *  or does not name it and sees none of it - whatever order the partitions were written in,
         *  and however long the walk took. Where this transaction itself reads follows in
         *  @c settle_snapshot_, not here.
         *
         *  Every reached partition is asked whether it may still commit before any of them writes,
         *  and all of them are held while it happens. Asking and writing one partition at a time
         *  cannot report a refusal honestly: a partition refusing after its neighbours have published
         *  would leave the caller retrying a transaction half of which already landed. So the walk
         *  either refuses having written nothing, and the transaction stays staged and retryable, or
         *  it publishes every partition. Only the reached partitions are held, ascending, which is
         *  the order every other multi-partition walk here takes them in.
         */
        status_t commit_under_one_stamp_() noexcept
            requires inner_shares_clock_k
        {
            clock_t &clock = store_.clock_;
            touched_parts_lock held {store_.mutexes_, touched_};
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index))
                if (status_t const refused = partitions_[partition_index].validate_for_commit(); failed(refused))
                    return refused;

            typename clock_t::commit_in_flight_t in_flight;
            clock.begin_commit(in_flight);
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index))
                partitions_[partition_index].publish_under(in_flight.stamp());
            clock.end_commit(in_flight);
            held.release();

            // The claim on the old snapshot goes back at once, so a long-lived transaction committing
            // in a loop stops pinning the very versions it is superseding. Where this transaction
            // reads moves separately, in `settle_snapshot_`, once its own commit is whole - and every
            // entry point below settles before it does anything, so nothing observes the gap.
            [[maybe_unused]] generation_t const released = clock.take_snapshot(lease_);
            unsettled_stamp_ = static_cast<generation_t>(in_flight.stamp());

            // Reclamation runs at the mark this transaction just moved to, so a version this commit
            // superseded is freed rather than pinned by the claim it has already given up.
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index)) {
                unique_lock_t lock {store_.mutexes_[partition_index]};
                partitions_[partition_index].prune_committed();
            }
            touched_.clear();
            return success_k;
        }

      public:
        transaction_t(partitioned_store &db, partition_transactions_t &&unlocked,
                      typename clock_t::snapshot_lease_t &&lease = {}) noexcept
            : store_(db), partitions_(std::move(unlocked)), lease_(std::move(lease)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;

        [[nodiscard]] status_t reset() noexcept {
            settle_snapshot_();
            if constexpr (inner_shares_clock_k) {
                // One snapshot for every partition, drawn once, exactly as opening the transaction did.
                generation_t const snapshot = store_.clock_.take_snapshot(lease_);
                auto status =
                    for_parts_([snapshot](inner_transaction_t &part) noexcept { return part.reset_at(snapshot); });
                touched_.clear();
                return status;
            }
            else {
                // `std::mem_fn(&inner_transaction_t::reset)` is cute... but we don't like heavy includes.
                auto status = for_parts_([](inner_transaction_t &part) noexcept { return part.reset(); });
                // Every partition was visited whatever it answered, and a reset keeps nothing to retry.
                touched_.clear();
                return status;
            }
        }
        [[nodiscard]] status_t rollback() noexcept {
            settle_snapshot_();
            // `std::mem_fn(&inner_transaction_t::rollback)` is cute... but we don't like heavy includes.
            auto status = for_touched_parts_([](inner_transaction_t &part) noexcept { return part.rollback(); });
            // A partition that refused may still hold a reservation, so its bit stays for the next attempt.
            if (succeeded(status)) touched_.clear();
            return status;
        }

        /**
         *  @brief Stages every reached partition, undoing the ones that already landed if a later
         *    partition refuses.
         *
         *  A partial stage is never observable. Partitions are taken ascending, so a group racing for
         *  the same set meets this one in a consistent order; the undo walks the staged prefix and
         *  rolls it back rather than resetting it, which leaves the caller's writes intact for a retry.
         */
        [[nodiscard]] status_t stage() noexcept {
            settle_snapshot_();
            touched_partitions_t staged;
            status_t status = success_k;
            for (std::size_t partition_index = touched_.first_set(); partition_index != partitions_k;
                 partition_index = touched_.next_set(partition_index)) {
                unique_lock_t lock {store_.mutexes_[partition_index]};
                status = partitions_[partition_index].stage();
                if (failed(status)) break;
                staged.mark(partition_index);
            }
            if (succeeded(status)) return status;

            for (std::size_t partition_index = staged.first_set(); partition_index != partitions_k;
                 partition_index = staged.next_set(partition_index)) {
                unique_lock_t lock {store_.mutexes_[partition_index]};
                [[maybe_unused]] status_t const unwound = partitions_[partition_index].rollback();
            }
            return status;
        }
        /**
         *  @brief Publishes every reached partition, and reports the last refusal without stopping.
         *
         *  A commit consumes this transaction's generation in every partition it reaches, so the walk
         *  cannot be resumed - stopping halfway would leave the rest staged with no owner, and clearing
         *  the record only on success would let a retry re-commit a partition that already published.
         */
        [[nodiscard]] status_t commit() noexcept {
            settle_snapshot_();
            if constexpr (inner_shares_clock_k) return commit_under_one_stamp_();
            else {
                auto status = for_touched_parts_([&](inner_transaction_t &part) noexcept { return part.commit(); });
                touched_.clear();
                return status;
            }
        }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(id);
            touched_.mark(partition_index);
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].watch(id);
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[partition_index]};
            partitions_[partition_index].find(std::forward<comparable_type_>(comparable),
                                              std::forward<callback_found_type_>(callback_found),
                                              std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
            return result;
        }

        /** @brief Whether @p comparable is there, including this transaction's own writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].contains(std::forward<comparable_type_>(comparable));
        }

        /**
         *  @brief Finds the member equal to @p comparable and records what it saw into the read set.
         *    The watching counterpart to @c find, and the one that can fail, because a read set is memory.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find_and_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                              callback_missing_type_ &&callback_missing = {}) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].find_and_watch(std::forward<comparable_type_>(comparable),
                                                               std::forward<callback_found_type_>(callback_found),
                                                               std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_is_ordered_k
        {
            settle_snapshot_();
            self_t::for_all_next_lookups(store_.comparator_, partitions_, store_.mutexes_,
                                         std::forward<comparable_type_>(comparable),
                                         std::forward<callback_found_type_>(callback_found),
                                         std::forward<callback_missing_type_>(callback_missing));
        }

        [[nodiscard]] status_t upsert(value_t &&element) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(element));
            touched_.mark(partition_index);
            return partitions_[partition_index].upsert(std::move(element));
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(id);
            touched_.mark(partition_index);
            return partitions_[partition_index].erase(id);
        }

        /**
         *  @brief Stages @p element only if its key is free, refusing rather than writing over it.
         *
         *  One key lives in one partition, so the refusal is decided by the one partition that owns it
         *  and carries the inner store's promise whole - unlike a walk crossing partitions. The lock is
         *  taken, unlike @c upsert: a strict insert reads its partition to learn whether the key is
         *  already there.
         */
        [[nodiscard]] status_t insert(value_t &&element) noexcept
            requires inner_transaction_refuses_occupied_key_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(element));
            touched_.mark(partition_index);
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].insert(std::move(element));
        }

        /**
         *  @brief Stages @p element only if its key is free, and says which branch was taken.
         *  @param[in] callback_inserted Receives the element once staged.
         *  @param[in] callback_existing Receives the element already under the key, which refused the insert.
         */
        template <typename callback_inserted_type_, typename callback_existing_type_>
        [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                      callback_existing_type_ &&callback_existing) noexcept
            requires inner_transaction_reports_occupied_key_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(element));
            touched_.mark(partition_index);
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].insert(std::move(element),
                                                       std::forward<callback_inserted_type_>(callback_inserted),
                                                       std::forward<callback_existing_type_>(callback_existing));
        }

        /** @brief Stages @p element only if its key is already taken, refusing to create one. */
        [[nodiscard]] status_t update(value_t &&element) noexcept
            requires inner_transaction_refuses_absent_key_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_.bucket_(identifier_t(element));
            touched_.mark(partition_index);
            shared_lock_t _ {store_.mutexes_[partition_index]};
            return partitions_[partition_index].update(std::move(element));
        }
    };

  private:
    mutable mutexes_t mutexes_;
    partitions_t partitions_;

    /**
     *  @brief The one clock every partition draws from, so a stamp means the same thing in each.
     *    Empty, and free, for a part that keeps no stamps.
     */
    ST_NO_UNIQUE_ADDRESS_ clock_t clock_ {};

    // Held rather than default-constructed per call: a hasher or comparator carrying state answers
    // differently from a fresh one, so rebuilding either would discard what the store was given.
    ST_NO_UNIQUE_ADDRESS_ hash_t hasher_ {};
    ST_NO_UNIQUE_ADDRESS_ comparator_t comparator_ {};

    friend class transaction_t;

    partitioned_store(partitions_t &&unlocked, hash_t const &hasher = {}, comparator_t const &comparator = {}) noexcept
        : partitions_(std::move(unlocked)), hasher_(hasher), comparator_(comparator) {
        share_clock_with_parts_();
    }
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    partitioned_store &operator=(partitioned_store &&other) noexcept {
        every_part_lock<unique_lock_t> _ {mutexes_};
        partitions_ = std::move(other.partitions_);
        adopt_clock_of_(other);
        hasher_ = other.hasher_;
        comparator_ = other.comparator_;
        return *this;
    }

    /** @brief Points every partition at this store's clock, which is what makes them one snapshot. */
    void share_clock_with_parts_() noexcept {
        if constexpr (inner_shares_clock_k)
            for (inner_store_t &part : partitions_) part.attach_clock(clock_);
    }

    /** @brief Takes over @p other's stamps and readers, then re-points the partitions that came with them. */
    void adopt_clock_of_(partitioned_store &other) noexcept {
        if constexpr (inner_shares_clock_k) clock_.adopt(other.clock_);
        share_clock_with_parts_();
    }

    static expected<partitions_t> new_parts() noexcept {
        return generate_array_safely<inner_store_t, partitions_k>([](std::size_t) { return inner_store_t::make(); });
    }

    static expected<partitions_t> new_parts(comparator_t const &comparator) noexcept {
        // Backends differ in whether they also take an allocator here, so the shape is detected rather
        // than assumed - a tree seeds both policies, a `std::set`-backed store only the comparator.
        return generate_array_safely<inner_store_t, partitions_k>([&](std::size_t) {
            if constexpr (requires { inner_store_t::make(comparator, typename inner_store_t::allocator_t {}); })
                return inner_store_t::make(comparator, typename inner_store_t::allocator_t {});
            else return inner_store_t::make(comparator);
        });
    }

  public:
    partitioned_store() noexcept { share_clock_with_parts_(); }
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    partitioned_store(partitioned_store &&other) noexcept
        : partitions_(std::move(other.partitions_)), hasher_(other.hasher_), comparator_(other.comparator_) {
        adopt_clock_of_(other);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t total = 0;
        every_part_lock<shared_lock_t> _ {mutexes_};
        for (auto const &part : partitions_) total += part.size();
        return total;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] static expected<partitioned_store> make() noexcept {
        expected<partitioned_store> result;
        if (expected<partitions_t> unlocked = new_parts(); unlocked) result = partitioned_store {std::move(*unlocked)};
        return result;
    }

    /**
     *  @brief Builds a store whose every partition shares one comparator and one hasher.
     *  @param[in] comparator The instance every comparison consults, in each partition and across them.
     *  @param[in] hasher The instance that maps an identifier to its partition.
     *  @return Collection instance or empty optional on failure.
     */
    [[nodiscard]] static expected<partitioned_store> make(comparator_t const &comparator,
                                                          hash_t const &hasher) noexcept {
        expected<partitioned_store> result;
        if (expected<partitions_t> unlocked = new_parts(comparator); unlocked)
            result = partitioned_store {std::move(*unlocked), hasher, comparator};
        return result;
    }

    /**
     *  @brief Opens one transaction per partition, or none at all.
     *    Each partition is taken exclusively while its transaction is built, since opening one reads
     *    the live container and draws a fresh generation from it.
     */
    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        // A stamped part draws its snapshot and its generation once, here, and every partition opens
        // on them: reading two partitions at two stamps is what would make the snapshot a lie, and a
        // commit landing between two of these locks is exactly how that would happen.
        typename clock_t::snapshot_lease_t lease;
        [[maybe_unused]] generation_t snapshot = 0;
        [[maybe_unused]] generation_t generation = 0;
        if constexpr (inner_shares_clock_k) {
            snapshot = clock_.take_snapshot(lease);
            generation = clock_.next_generation();
        }

        // Ascending order, one lock at a time, like every other all-partition walk here.
        auto maybe = generate_array_safely<inner_transaction_t, partitions_k>([&](std::size_t partition_index) {
            unique_lock_t lock {mutexes_[partition_index]};
            if constexpr (inner_shares_clock_k)
                return partitions_[partition_index].transaction_at(snapshot, generation);
            else return partitions_[partition_index].transaction();
        });
        if (!maybe) return {};

        return transaction_t(*this, std::move(*maybe), std::move(lease));
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].upsert(std::move(element));
    }

    /**
     *  @brief Removes one element, reporting whether it was there.
     *    Touches only the partition that owns the key, so the rest stay unlocked.
     *  @param[in] id The identifier to remove.
     *  @param[in] callback_found Receives the element that was removed.
     *  @param[in] callback_missing Fires when no such element existed.
     *  @return @c key_not_found_k when absent, so the answer is available without a second probe.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(identifier_t const &id, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        std::size_t partition_index = bucket_(id);
        unique_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].erase(id, std::forward<callback_found_type_>(callback_found),
                                                  std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        // This might be implemented more efficiently, but using
        // a transaction beneath looks like the most straightforward approach.
        auto opened = transaction();
        // A transaction that cannot open failed to allocate its per-partition state; nothing was
        // compared against anything, so this is not a serialization conflict.
        if (!opened) return out_of_memory_heap_k;
        for (; begin != end; ++begin)
            if (auto status = opened->upsert(*begin); failed(status)) return status;
        if (auto status = opened->stage(); failed(status)) return status;
        return opened->commit();
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::size_t partition_index = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[partition_index]};
        partitions_[partition_index].find(std::forward<comparable_type_>(comparable),
                                          std::forward<callback_found_type_>(callback_found),
                                          std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Whether @p comparable is there, asking only the partition that owns it. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        std::size_t partition_index = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].contains(std::forward<comparable_type_>(comparable));
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1u : 0u;
    }

    /** @brief Inserts one element, refusing with @c key_already_exists_k when the key is already present. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[partition_index]};
        return partitions_[partition_index].insert_if_missing(std::move(element));
    }

    /**
     *  @brief Inserts one element, saying which of the two happened rather than leaving it inferred.
     *  @param[in] element The element to insert.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element already present, which is what declined the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    [[nodiscard]] status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                             callback_existing_type_ &&callback_existing) noexcept {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[partition_index]};
        return partitions_[partition_index].insert_if_missing(std::move(element),
                                                              std::forward<callback_inserted_type_>(callback_inserted),
                                                              std::forward<callback_existing_type_>(callback_existing));
    }

    /**
     *  @brief Inserts one element, refusing with the inner store's own status when the key is taken.
     *    One key lives in one partition, so the refusal is decided under a single lock and carries the
     *    inner store's promise whole.
     */
    [[nodiscard]] status_t insert(value_t &&element) noexcept
        requires inner_refuses_occupied_key_k
    {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].insert(std::move(element));
    }

    /**
     *  @brief Inserts one element, saying which of the two happened rather than leaving it inferred.
     *  @param[in] element The element to insert.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element already under the key, which refused the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                  callback_existing_type_ &&callback_existing) noexcept
        requires inner_reports_occupied_key_k
    {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].insert(std::move(element),
                                                   std::forward<callback_inserted_type_>(callback_inserted),
                                                   std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Writes one element only if its key is already taken, refusing to create one. */
    [[nodiscard]] status_t update(value_t &&element) noexcept
        requires inner_refuses_absent_key_k
    {
        std::size_t partition_index = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].update(std::move(element));
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        for (; begin != end; ++begin) {
            value_t element(*begin);
            std::size_t partition_index = bucket_(identifier_t(element));
            unique_lock_t _ {mutexes_[partition_index]};
            if (auto status = partitions_[partition_index].insert_if_missing(std::move(element)); failed(status))
                return status;
        }
        return success_k;
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        for_all_next_lookups(comparator_, partitions_, mutexes_, std::forward<comparable_type_>(comparable),
                             std::forward<callback_found_type_>(callback_found),
                             std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief The first element at or after @p comparable, which may live in any partition.
     *
     *  @warning Two separately locked probes - an exact match, then the successor - with no lock held
     *    across them, so this is not atomic even against a single partition. A key erased between the
     *    two is answered by its successor, and one inserted between them is skipped.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        bool found_exact = false;
        find(
            comparable,
            [&](value_t const &value) noexcept {
                found_exact = true;
                callback_found(value);
            },
            []() noexcept {});
        if (found_exact) return;
        upper_bound(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                    std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
        requires inner_is_ordered_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};
        for (auto const &part : partitions_) part.range(lower, upper, callback);
    }

    /** @brief Erases the half-open range from every partition, reporting the last refusal. */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires inner_is_ordered_k
    {
        every_part_lock<unique_lock_t> _ {mutexes_};
        status_t status = success_k;
        for (auto &part : partitions_)
            if (status_t const one = part.erase_range(lower, upper, callback); failed(one)) status = one;
        return status;
    }

    /** @brief Erases every element at or after @p lower from every partition, reporting the last refusal. */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        every_part_lock<unique_lock_t> _ {mutexes_};
        status_t status = success_k;
        for (auto &part : partitions_)
            if (status_t const one = part.erase_from(lower, callback); failed(one)) status = one;
        return status;
    }

    /** @brief Erases every element before @p upper from every partition, reporting the last refusal. */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        every_part_lock<unique_lock_t> _ {mutexes_};
        status_t status = success_k;
        for (auto &part : partitions_)
            if (status_t const one = part.erase_up_to(upper, callback); failed(one)) status = one;
        return status;
    }

    /**
     *  @brief Rewrites the mapped side of every element in [ @p lower, @p upper ), across all partitions.
     *
     *  Every partition is held exclusively for the length of the walk, as @c erase_range holds them,
     *  so the window is revised as one write rather than sixteen.
     *
     *  @param[in] callback Invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Success, or the last refusal. A partition whose own walk cannot fail always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires inner_revises_range_k
    {
        every_part_lock<unique_lock_t> _ {mutexes_};
        status_t status = success_k;
        for (auto &part : partitions_) {
            // A partition answers either @c status_t or nothing at all, and the wrapper has to name one
            // return type. The wider of the two loses nothing: a walk that cannot refuse always succeeds.
            if constexpr (std::is_void<decltype(part.update_range(lower, upper, callback))>())
                part.update_range(lower, upper, callback);
            else if (status_t const one = part.update_range(lower, upper, callback); failed(one)) status = one;
        }
        return status;
    }

    /**
     *  @brief Hands @p callback every member the store holds, in no particular order.
     *
     *  One partition is locked shared at a time, ascending, as @c sample_reservoir takes them - holding
     *  all sixteen for the length of an enumeration would stall every writer in the store for as long
     *  as the caller takes to consume it.
     *
     *  What that buys and what it costs, stated rather than left to be inferred: a key's partition is
     *  fixed by its hash and never changes, so an element present in the store for the whole walk is
     *  visited @b exactly @b once - it cannot be missed by being moved ahead of the cursor, nor seen
     *  twice by being moved behind it, and the partition holding it is locked while it is read. An
     *  element inserted or erased during the walk @b may @b or @b may @b not be seen, according to
     *  whether its partition had already been visited. The first guarantee is the one a caller
     *  enumerating a container depends on; the second is the one it has to tolerate.
     */
    template <typename callback_type_ = no_op_t>
    void for_each(callback_type_ &&callback) const noexcept
        requires inner_enumerates_k
    {
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            shared_lock_t lock {mutexes_[partition_index]};
            partitions_[partition_index].for_each(callback);
        }
    }

    /**
     *  @brief Frees every version no reader can still reach, one partition at a time.
     *  @return How many versions were reclaimed across all partitions, or why one of them refused.
     *    A partition whose sweep cannot refuse always answers with a count.
     */
    [[nodiscard]] expected<std::size_t> vacuum() noexcept
        requires inner_reclaims_k
    {
        std::size_t reclaimed = 0;
        // Ascending order, one partition at a time, like every other all-partition walk here.
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            unique_lock_t lock {mutexes_[partition_index]};
            auto one = partitions_[partition_index].vacuum();
            if constexpr (std::is_same<decltype(one), std::size_t>()) reclaimed += one;
            else {
                if (!one) return one.status();
                reclaimed += *one;
            }
        }
        return reclaimed;
    }

    /**
     *  @brief Frees the unreachable versions of every key in [ @p lower, @p upper ), one partition at a time.
     *  @return How many versions were reclaimed across all partitions, or why one of them refused.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept
        requires inner_reclaims_range_k
    {
        std::size_t reclaimed = 0;
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            unique_lock_t lock {mutexes_[partition_index]};
            auto one = partitions_[partition_index].vacuum(lower, upper);
            if constexpr (std::is_same<decltype(one), std::size_t>()) reclaimed += one;
            else {
                if (!one) return one.status();
                reclaimed += *one;
            }
        }
        return reclaimed;
    }

    /**
     *  @brief Hands @p callback_found the element at zero-based position @p ordinal of the merged order.
     *
     *  A partition knows only its own ordinals, so the global one is reached by merging the partitions
     *  rather than by descending a subtree count: linear in @p ordinal, not logarithmic in the size.
     *  Every partition is held shared for the length of that merge, so the position is answered against
     *  one order rather than sixteen probes.
     *
     *  @param[in] callback_missing Fires when fewer elements are there. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    void select(std::size_t ordinal, callback_found_type_ &&callback_found,
                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        std::size_t position = 0;
        bool delivered = false;
        walk_merged_order_([&](std::size_t partition_index, identifier_t const &key) noexcept {
            if (position++ != ordinal) return merge_control_t::resume_k;
            partitions_[partition_index].find(key, callback_found, no_op_t {});
            delivered = true;
            return merge_control_t::halt_k;
        });
        if (!delivered) callback_missing();
    }

    /**
     *  @brief Hands @p callback_found how many elements the store orders before @p comparable.
     *    Counted by the same merge @c select walks, so it costs the rank rather than a descent.
     *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        identifier_t const target(comparable);
        std::size_t position = 0;
        bool found = false;
        walk_merged_order_([&](std::size_t, identifier_t const &key) noexcept {
            if (comparator_(key, target)) {
                ++position;
                return merge_control_t::resume_k;
            }
            found = !comparator_(target, key);
            return merge_control_t::halt_k;
        });
        if (found) callback_found(position);
        else callback_missing();
    }

    /**
     *  @brief Hands one element of the range to @p callback, drawn from a single random partition.
     *
     *  @warning The draw is uniform only if every partition holds a similar number of in-range
     *    entries, and it takes one lock. The reservoir overload below walks all of them instead, so a
     *    call site that needs the whole range weighted correctly wants that one.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    void sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                    callback_type_ &&callback) const noexcept
        requires inner_is_samplable_k
    {
        std::size_t partition_index = generator() % partitions_k;
        shared_lock_t _ {mutexes_[partition_index]};
        partitions_[partition_index].sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                std::forward<generator_type_>(generator),
                                                std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Fills @p reservoir with up to @p reservoir_capacity elements of the range, weighting
     *    every partition by how many in-range entries it actually holds.
     *
     *  @warning One partition is locked at a time rather than all of them, so a writer moving an entry
     *    between partitions can have it sampled twice or not at all.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                          std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept
        requires inner_is_samplable_k
    {
        // Ascending order, blocking, like every other all-partition walk here.
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            shared_lock_t lock {mutexes_[partition_index]};
            partitions_[partition_index].sample_reservoir(lower, upper, generator, seen, reservoir_capacity, reservoir);
        }
    }

    /**
     *  @brief Empties every partition in place.
     *
     *  Each partition clears itself rather than being replaced by a fresh one: a replacement leaves an
     *  open transaction referring to contents that are gone, and it would have to rebuild the
     *  comparator and allocator the partition was given.
     */
    [[nodiscard]] status_t clear() noexcept {
        return for_all<unique_lock_t>(partitions_, mutexes_, [](inner_store_t &part) noexcept { return part.clear(); });
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        return for_all<unique_lock_t>(partitions_, mutexes_, [size](inner_store_t &part) noexcept { //
            return part.reserve(size / partitions_k);
        });
    }
};

#pragma region Aliases

/**
 *  @brief A set-shaped store sharded across independently locked partitions, spelled
 *    @c partitioned_set<monotonic_avl_set<key_t>>.
 *    The element type comes from the store being wrapped, so this alias and @c partitioned_map build
 *    the same thing; what they add is the shape at the point of use, which the store families already name.
 */
template <typename set_store_type_, typename hash_type_ = hash<typename set_store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
using partitioned_set = partitioned_store<set_store_type_, hash_type_, shared_mutex_type_, partitions_count_>;

/**
 *  @brief A map-shaped store sharded across independently locked partitions, spelled
 *    @c partitioned_map<monotonic_avl_map<key_t, value_t>>.
 */
template <typename map_store_type_, typename hash_type_ = hash<typename map_store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
using partitioned_map = partitioned_store<map_store_type_, hash_type_, shared_mutex_type_, partitions_count_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
