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
#include <type_traits> // `std::is_same`

#include "shared.hpp"

namespace ashvardanian::smashtable {

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

        auto new_part = generator(partition_index);
        if (!new_part) {
            // The prefix goes back before the reason does, so no half-populated array is observable -
            // and the reason travels, because a default-constructed `expected` says only `unknown_k`.
            for (std::size_t destructed_index = 0; destructed_index != partition_index; ++destructed_index)
                raw_parts[destructed_index].~type_();
            return new_part.status();
        }
        new (raw_parts + partition_index) type_(std::move(*new_part));
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
 *  @warning Every callback runs with at least one partition lock held - the walks that revise or
 *    erase a window hold every partition at once, while the enumerating and sampling walks hold one
 *    at a time, each saying which in its own docblock. The mutex is not recursive, so a callback that
 *    calls back into this store - or into anything that eventually does - deadlocks against itself.
 *    Copy out what a callback needs and do the rest after it returns.
 *  @warning Moving a store leaves every open transaction pointing at the husk, whose partitions are
 *    empty and whose mutexes guard nothing, so commits land nowhere and report success. Move only a
 *    store no transaction is open on and no other thread is touching.
 */
template <typename store_type_, typename hash_type_ = hash<typename store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
class partitioned_store {

  public:
    static constexpr std::size_t partitions_k = partitions_count_;
    using store_t = partitioned_store;
    using hash_t = hash_type_;
    using inner_store_t = store_type_;
    using inner_transaction_t = typename inner_store_t::transaction_t;
    using mutex_t = shared_mutex_type_;
    using shared_lock_t = shared_lock<mutex_t>;
    using unique_lock_t = unique_lock<mutex_t>;

    using mutexes_t = std::array<mutex_t, partitions_k>;
    /** @brief How many times each partition has been written, so a cached read can be told it is stale. */
    using epoch_t = std::size_t;
    using epochs_t = std::array<epoch_t, partitions_k>;
    using partitions_t = std::array<inner_store_t, partitions_k>;
    using partition_transactions_t = std::array<inner_transaction_t, partitions_k>;

    using value_t = typename inner_store_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;

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
     *  whole. A part deciding visibility any other way is capped at @c read_committed_k, which is the
     *  weakest level named and so the floor a cap can reach.
     */
    static constexpr isolation_t isolation_k =
        partitions_k == 1 || inner_shares_clock_k ? inner_store_t::isolation_k : isolation_t::read_committed_k;

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

    /** @brief Whether a partition draws one member of a range at random. */
    static constexpr bool inner_samples_one_k =
        requires(inner_store_t const &store, identifier_t const &key, no_op_t callback) {
            store.sample_one(key, key, callback, callback);
        };

    /** @brief Whether a partition fills a reservoir from a range. */
    static constexpr bool inner_samples_reservoir_k =
        requires(inner_store_t const &store, identifier_t const &key, no_op_t callback, std::size_t seen) {
            store.sample_reservoir(key, key, callback, seen, seen, callback);
        };

    /**
     *  @brief Whether an open transaction can be asked to commit in two steps rather than one.
     *
     *  A commit spanning partitions has to learn that every one of them may proceed before any of
     *  them writes. An engine that only offers a single @c commit decides and writes in the same
     *  call, so the wrapper cannot ask first, and a refusal from a later partition arrives over
     *  writes an earlier one already published.
     */
    static constexpr bool inner_transaction_splits_commit_k = requires(inner_transaction_t &transaction) {
        { transaction.validate_for_commit() } noexcept -> std::same_as<status_t>;
        transaction.publish_under();
    };

    /** @brief Whether an open transaction carries the ordered surface its store does. */
    static constexpr bool inner_transaction_is_ordered_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.upper_bound(key, callback, callback);
        };

    /** @brief Whether the wrapped transaction counts the members matching a key. */
    static constexpr bool inner_transaction_counts_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key) { transaction.count(key); };
    /** @brief Whether the wrapped transaction erases a window of its own keys. */
    static constexpr bool inner_transaction_erases_range_k =
        requires(inner_transaction_t &transaction, identifier_t const &key, no_op_t callback) {
            transaction.erase_range(key, key, callback);
            transaction.erase_from(key, callback);
            transaction.erase_up_to(key, callback);
        };
    /** @brief Whether the wrapped transaction revises a window of its own members. */
    static constexpr bool inner_transaction_revises_range_k =
        requires(inner_transaction_t &transaction, identifier_t const &key, no_op_t callback) {
            transaction.update_range(key, key, callback);
        };
    /** @brief Whether an open transaction answers an inclusive bound, which a merged scan needs to seed from. */
    static constexpr bool inner_transaction_lower_bounds_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.lower_bound(key, callback, callback);
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

    /** @brief Whether a partition answers every member equal to a key rather than only the first. */
    static constexpr bool inner_matches_equals_k = requires(inner_store_t const &store, identifier_t const &key,
                                                            no_op_t callback) { store.equal_range(key, callback); };

    /** @brief Whether a partition keeps superseded versions and can count them. */
    static constexpr bool inner_counts_versions_k = requires(inner_store_t const &store, identifier_t const &key) {
        store.versions_count();
        store.versions_count(key);
    };

    /** @brief Whether a partition names the snapshot no open reader sits below. */
    static constexpr bool inner_marks_low_water_k = requires(inner_store_t const &store) { store.low_water_mark(); };

    /** @brief Whether a partition spells an overwriting write as @c insert_or_assign. */
    static constexpr bool inner_assigns_over_key_k =
        requires(inner_store_t &store, value_t &&element) { store.insert_or_assign(std::move(element)); };

    /** @brief Whether an open transaction reports what it has staged so far. */
    static constexpr bool inner_transaction_reports_changes_k = requires(inner_transaction_t const &transaction) {
        transaction.has_changes();
        transaction.changes_count();
    };

    /** @brief Whether an open transaction refuses an occupied key silently rather than with a status. */
    static constexpr bool inner_transaction_skips_occupied_key_k = requires(
        inner_transaction_t &transaction, value_t &&element) { transaction.insert_if_missing(std::move(element)); };

    /** @brief Whether an open transaction enumerates what its snapshot and its own writes show. */
    static constexpr bool inner_transaction_enumerates_k =
        requires(inner_transaction_t const &transaction, no_op_t callback) { transaction.for_each(callback); };

    /** @brief Whether an open transaction walks a half-open window of the keyspace. */
    static constexpr bool inner_transaction_walks_range_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.range(key, key, callback);
        };

    /** @brief Whether an open transaction answers every member equal to a key. */
    static constexpr bool inner_transaction_matches_equals_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.equal_range(key, callback);
        };

    /** @brief Whether an open transaction sizes its watch list ahead of the writes that fill it. */
    static constexpr bool inner_transaction_reserves_k =
        requires(inner_transaction_t &transaction, std::size_t size) { transaction.reserve(size); };

    /** @brief What an all-partition walk does with a partition that refuses. */
    enum class refusal_policy_t : std::uint8_t {
        /** @brief The walk returns the first refusal, leaving every partition after it untouched. */
        stop_at_first_k,
        /** @brief Every partition is attempted whatever its neighbours answered, and the last refusal is reported. */
        attempt_every_k,
    };

    /**
     *  @brief What a refusal from each all-partition method says about the partitions it did not report on.
     *
     *  Two behaviours live under one type, and the returned status cannot tell them apart: under
     *  @c stop_at_first_k the partitions after the refusal are untouched, while under
     *  @c attempt_every_k every partition was attempted and an unknown subset applied. Naming the
     *  policy per method is what lets a caller — and a test — know which answer it just received.
     */
    static constexpr refusal_policy_t clear_policy_k = refusal_policy_t::stop_at_first_k;
    static constexpr refusal_policy_t reserve_policy_k = refusal_policy_t::stop_at_first_k;
    static constexpr refusal_policy_t erase_range_policy_k = refusal_policy_t::attempt_every_k;
    static constexpr refusal_policy_t erase_from_policy_k = refusal_policy_t::attempt_every_k;
    static constexpr refusal_policy_t erase_up_to_policy_k = refusal_policy_t::attempt_every_k;
    static constexpr refusal_policy_t update_range_policy_k = refusal_policy_t::attempt_every_k;

  private:
    /**
     *  @brief Whether a scan over a set of partition indices stopped on a marked partition.
     */
    enum class marked_presence_t : std::uint8_t {
        /** @brief Nothing was marked at or above where the scan started, so the walk is over. */
        none_marked_k,
        /** @brief The scan stopped on the lowest marked partition at or above where it started. */
        one_marked_k,
    };

    /**
     *  @brief Where a scan over a set of partition indices stopped.
     *    @c index is only meaningful once @c presence says a partition was found, which is why the two
     *    travel together rather than as an index with a reserved value.
     */
    struct marked_partition_t {
        /** @brief Whether @c index names a partition at all. */
        marked_presence_t presence = marked_presence_t::none_marked_k;
        /** @brief The marked partition, meaningful only under @c one_marked_k. */
        std::size_t index = 0;
    };

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

        /** @brief The lowest marked partition, when any of them is marked. */
        marked_partition_t first_marked() const noexcept { return scan_from_(0); }

        /** @brief The lowest marked partition above @p partition_index, when there is one. */
        marked_partition_t next_marked(std::size_t partition_index) const noexcept {
            return scan_from_(partition_index + 1);
        }

      private:
        marked_partition_t scan_from_(std::size_t partition_index) const noexcept {
            for (std::size_t word_index = partition_index / bits_per_word_k; word_index < words_k; ++word_index) {
                std::uint64_t left = words[word_index];
                // Mask off the bits below where this scan starts, only in the word it starts in.
                if (word_index == partition_index / bits_per_word_k) {
                    std::size_t const bit = partition_index % bits_per_word_k;
                    left &= bit == 0 ? ~std::uint64_t {0} : ~std::uint64_t {0} << bit;
                }
                if (left)
                    return {marked_presence_t::one_marked_k,
                            word_index * bits_per_word_k + static_cast<std::size_t>(countr_zero(left))};
            }
            return {};
        }
    };

    /**
     *  @brief Whether a merged walk can hold one key per partition between its steps.
     *
     *  A walk answers in one order across sixteen independently ordered partitions, which it can only
     *  do by remembering the smallest key each one is offering. That remembering is a copy, so a key
     *  that refuses to be copied cannot be walked in merged order - it can still be written, read by
     *  its own key, and enumerated in no particular order, none of which hold a key between steps.
     */
    static constexpr bool identifier_survives_a_walk_k = std::is_copy_constructible_v<identifier_t>;

    /**
     *  @brief Which partition owns @p comparable, hashing what it is handed rather than a copy of it.
     *
     *  A key that is already the identifier is hashed through the reference, so a move-only key reaches
     *  a partition at all and a heavy one is not duplicated per lookup. A comparable of some other type
     *  is converted first, since the hasher is only promised to answer for the identifier and a
     *  differently-typed hash would send a lookup to a partition the key does not live in.
     */
    template <typename comparable_type_>
    std::size_t bucket_(comparable_type_ const &comparable) const noexcept {
        auto const &key = mapping_key_or_itself(comparable);
        using key_t = std::remove_cvref_t<decltype(key)>;
        if constexpr (std::is_same_v<key_t, identifier_t>) return hasher_(key) % partitions_k;
        else return hasher_(identifier_t(key)) % partitions_k;
    }

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

    // A cursor keeps one key per partition between its steps, so it needs to know which partitions a
    // writer touched since it last looked. Counting writes answers that without taking every lock on
    // every step, which is what makes a resumable walk cheaper than probing all sixteen per element.

    /** @brief Records that a partition changed, ordered so its new contents are visible with the count. */
    static void note_written_(epoch_t &epoch) noexcept {
        atomic_ref<epoch_t> counter {epoch};
        counter.store(counter.load(memory_order_relaxed_k) + 1, memory_order_release_k);
    }

    /** @brief The count a reader last saw, paired with whatever it read under it. */
    static epoch_t epoch_seen_(epoch_t const &epoch) noexcept {
        return atomic_ref<epoch_t>(const_cast<epoch_t &>(epoch)).load(memory_order_acquire_k);
    }

    /**
     *  @brief Holds one partition exclusively, and records that it changed when the write is done.
     *
     *  Taking this rather than the bare mutex is what keeps the write count honest: a writer cannot
     *  forget to move it, because moving it is part of giving the lock back. A refused write moves it
     *  too, which costs a reader one re-probe and never costs it a key.
     */
    class writing_part_lock {
        mutex_t &mutex_;
        epoch_t &epoch_;

      public:
        writing_part_lock(mutex_t &mutex, epoch_t &epoch) noexcept : mutex_(mutex), epoch_(epoch) { mutex_.lock(); }
        ~writing_part_lock() noexcept {
            note_written_(epoch_);
            mutex_.unlock();
        }
        writing_part_lock(writing_part_lock const &) = delete;
        writing_part_lock &operator=(writing_part_lock const &) = delete;
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
        epochs_t &epochs_;
        touched_partitions_t const &touched_;

      public:
        touched_parts_lock(mutexes_t &mutexes, epochs_t &epochs, touched_partitions_t const &touched) noexcept
            : mutexes_(&mutexes), epochs_(epochs), touched_(touched) {
            for (marked_partition_t held = touched_.first_marked(); held.presence == marked_presence_t::one_marked_k;
                 held = touched_.next_marked(held.index))
                (*mutexes_)[held.index].lock();
        }
        ~touched_parts_lock() noexcept { release(); }
        touched_parts_lock(touched_parts_lock const &) = delete;
        touched_parts_lock &operator=(touched_parts_lock const &) = delete;

        /** @brief Gives every held partition back early, so the work after a commit runs unlocked. */
        void release() noexcept {
            if (!mutexes_) return;
            for (marked_partition_t held = touched_.first_marked(); held.presence == marked_presence_t::one_marked_k;
                 held = touched_.next_marked(held.index)) {
                note_written_(epochs_[held.index]);
                (*mutexes_)[held.index].unlock();
            }
            mutexes_ = nullptr;
        }
    };

    /** @brief How long an all-partition walk holds the partitions it visits. */
    enum class locking_policy_t : std::uint8_t {
        /** @brief One partition is held while it is visited and given back before the next one is taken. */
        one_at_a_time_k,
        /** @brief Every partition is held for the whole walk, so the visits are one write rather than many. */
        all_at_once_k,
    };

    /** @brief Visits every partition index in ascending order, reporting refusals as @p refusal_ says to. */
    template <refusal_policy_t refusal_, typename step_type_>
    static status_t for_all_indices_(step_type_ &&step) noexcept {
        status_t status = success_k;
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            status_t const one = step(partition_index);
            if (!failed(one)) continue;
            if constexpr (refusal_ == refusal_policy_t::stop_at_first_k) return one;
            else status = one;
        }
        return status;
    }

    /**
     *  @brief Runs @p callable over every partition, under @p locking_, reporting refusals under @p refusal_.
     *
     *  Ascending order either way, blocking. Taking whichever partitions happen to be free and retrying
     *  the rest reads as politer, but it shares no order with @c every_part_lock, and two all-partition
     *  operations without a common order are two operations that can wait on each other for good.
     */
    template <typename lock_type_, locking_policy_t locking_, refusal_policy_t refusal_, typename partitions_type_,
              typename mutexes_type_, typename callable_type_>
    status_t for_all(partitions_type_ &parts, mutexes_type_ &mutexes, callable_type_ &&callable) noexcept {
        constexpr bool writing_k = std::is_same<lock_type_, unique_lock_t>();
        if constexpr (locking_ == locking_policy_t::all_at_once_k) {
            status_t answered = success_k;
            {
                every_part_lock<lock_type_> _ {mutexes};
                answered = for_all_indices_<refusal_>(
                    [&](std::size_t partition_index) noexcept { return callable(parts[partition_index]); });
            }
            // Counted after the whole hold rather than per partition, since nothing may read a front
            // while every partition is held anyway.
            if constexpr (writing_k)
                for (epoch_t &epoch : epochs_) note_written_(epoch);
            return answered;
        }
        else
            return for_all_indices_<refusal_>([&](std::size_t partition_index) noexcept {
                if constexpr (writing_k) {
                    writing_part_lock lock {mutexes[partition_index], epochs_[partition_index]};
                    return callable(parts[partition_index]);
                }
                else {
                    lock_type_ lock {mutexes[partition_index]};
                    return callable(parts[partition_index]);
                }
            });
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

    /** @brief Where a cursor's next step begins, which is a different question before and after the first. */
    enum class cursor_seed_t : std::uint8_t {
        /** @brief No bound was given and nothing has been handed over, so the walk starts at the least key. */
        the_smallest_k,
        /** @brief A bound was given and nothing has been handed over, so that bound is inclusive. */
        the_given_bound_k,
        /** @brief A key has been handed over, so the walk continues strictly above it. */
        past_the_last_k,
    };

    /** @brief One key per partition, the smallest each has left to contribute. */
    using fronts_t = std::array<identifier_t, partitions_k>;
    /** @brief Whether each of those fronts holds anything. */
    using front_states_t = std::array<front_state_t, partitions_k>;

    /** @brief Which partition's front is the smallest, when any of them still holds one. */
    static marked_partition_t smallest_front_(comparator_t const &comparator, fronts_t const &fronts,
                                              front_states_t const &states) noexcept {
        marked_partition_t smallest;
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            if (states[partition_index] != front_state_t::holds_a_key_k) continue;
            if (smallest.presence == marked_presence_t::one_marked_k &&
                !comparator(fronts[partition_index], fronts[smallest.index]))
                continue;
            smallest = {marked_presence_t::one_marked_k, partition_index};
        }
        return smallest;
    }

    /**
     *  @brief Walks the merged order of every partition, smallest key first, until @p step halts.
     *
     *  A front per partition, refilled from that partition's own successor once its key is consumed.
     *  Where the walk begins is @p seed_front's business and nothing else's, which is what lets one
     *  walk answer a bound, a range and an ordinal without knowing which of them it is serving.
     *
     *  Static, so a transaction drives it over its own array of partition transactions rather than
     *  needing a second copy written against those.
     *
     *  @param[in] seed_front Fills one partition's front with the first key that partition contributes.
     *  @param[in] step Receives the partition index and the key, and says whether to carry on.
     *  @warning Every partition must already be held for the length of the walk, and @p step runs
     *    under all of them - so a step reaching back into this store deadlocks against a mutex that
     *    does not recurse.
     */
    template <typename parts_type_, typename seed_type_, typename step_type_>
    static void walk_merged_(comparator_t const &comparator, parts_type_ &parts, seed_type_ &&seed_front,
                             step_type_ &&step) noexcept {
        static_assert(identifier_survives_a_walk_k,
                      "a merged walk remembers one key per partition, so the identifier must be copyable");

        fronts_t fronts;
        front_states_t states {};
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index)
            seed_front(partition_index, [&](value_t const &element) noexcept {
                fronts[partition_index] = identifier_t(element);
                states[partition_index] = front_state_t::holds_a_key_k;
            });

        while (true) {
            marked_partition_t const smallest = smallest_front_(comparator, fronts, states);
            if (smallest.presence != marked_presence_t::one_marked_k) return;
            if (step(smallest.index, fronts[smallest.index]) == merge_control_t::halt_k) return;

            // The bound outlives the front it came from, since refilling the front is what overwrites it.
            identifier_t const consumed = std::move(fronts[smallest.index]);
            states[smallest.index] = front_state_t::exhausted_k;
            [[maybe_unused]] status_t const refilled =
                parts[smallest.index].upper_bound(consumed, [&](value_t const &element) noexcept {
                    fronts[smallest.index] = identifier_t(element);
                    states[smallest.index] = front_state_t::holds_a_key_k;
                });
        }
    }

    /** @brief Whether a partition names its smallest member without being given a bound to beat. */
    static constexpr bool inner_names_its_smallest_k =
        requires(inner_store_t const &store, no_op_t callback) { store.smallest(callback, callback); };

    /**
     *  @brief Seeds every front with the smallest key its partition holds, for a walk with no lower bound.
     *
     *  A store that names its smallest member answers straight away. One that does not, but counts its
     *  members, reaches the same key through its zeroth ordinal - which is why an ordinal walk works at
     *  all over a core that keeps subtree counts and nothing else.
     */
    auto seed_from_the_start_() const noexcept
        requires inner_names_its_smallest_k || inner_is_ranked_k
    {
        return [this](std::size_t partition_index, auto &&fill) noexcept {
            // Only one of these exists on a given inner store, so the other branch has to be discarded
            // rather than merely unevaluated - a ternary would instantiate both.
            if constexpr (inner_names_its_smallest_k) {
                [[maybe_unused]] status_t const seeded = partitions_[partition_index].smallest(fill, no_op_t {});
            }
            else { [[maybe_unused]] status_t const seeded = partitions_[partition_index].select(0, fill, no_op_t {}); }
        };
    }

    /**
     *  @brief Hands @p callback_found the first key of the merged order, wherever @p seed_front starts it.
     *
     *  Every partition is taken once, for the one answer, so the bound is read against a single moment
     *  rather than against sixteen. Under the held locks the winning key cannot be erased between being
     *  chosen and being read, which is what leaves this with no retry to make.
     */
    template <typename parts_type_, typename mutexes_type_, typename seed_type_, typename callback_found_type_,
              typename callback_missing_type_>
    static void first_merged_(comparator_t const &comparator, parts_type_ &parts, mutexes_type_ &mutexes,
                              seed_type_ &&seed_front, callback_found_type_ &&callback_found,
                              callback_missing_type_ &&callback_missing) noexcept {

        every_part_lock<shared_lock_t> _ {mutexes};
        bool delivered = false;
        walk_merged_(comparator, parts, seed_front, [&](std::size_t partition_index, identifier_t const &key) noexcept {
            [[maybe_unused]] status_t const answered = parts[partition_index].find(key, callback_found, no_op_t {});
            delivered = true;
            return merge_control_t::halt_k;
        });
        if (!delivered) callback_missing();
    }

  public:
    /**
     *  @brief A resumable walk of the merged order that holds no lock between its steps.
     *
     *  The position is held by value rather than as an iterator, which is what makes erasing the key
     *  the cursor stands on harmless: the next step asks for the first key at or after it and is
     *  answered by the successor. Each partition's smallest unvisited key is cached beside the write
     *  count it was read under, so a step re-probes only the partitions somebody has written since -
     *  none of them, in a store nobody is writing, which is where the cost goes from sixteen probes
     *  per element to one.
     *
     *  What it promises: no key is handed over twice, nothing fails, and every key present for the
     *  whole walk is handed over exactly once. A key inserted ahead of the cursor is seen, because the
     *  insert moved the count its partition's front was cached under. A key inserted behind it is
     *  missed, which is what walking without holding the store means.
     */
    class ordered_cursor_t {
        static_assert(identifier_survives_a_walk_k,
                      "a cursor remembers one key per partition, so the identifier must be copyable");

        friend class partitioned_store;

        partitioned_store const *store_ {nullptr};
        /** @brief Each partition's smallest key not yet handed over. */
        fronts_t fronts_;
        /** @brief Whether each of those fronts holds anything. */
        front_states_t states_ {};
        /** @brief The write count each front was read under, which is what dates it. */
        epochs_t seen_epochs_ {};
        /** @brief The bound every step is taken from - the one given, then the last key handed over. */
        identifier_t position_;
        /** @brief Where the next step begins, which is not the same question before and after the first. */
        cursor_seed_t seed_ {cursor_seed_t::the_smallest_k};

        explicit ordered_cursor_t(partitioned_store const &store, cursor_seed_t seed) noexcept
            : store_(&store), seed_(seed) {
            // Settled here rather than on the first step, so `exhausted` answers about the store
            // rather than about whether anybody has asked yet - the same moment `visible_cursor_t`
            // settles, and what lets `while (!exhausted()) next(...)` walk the whole order.
            seed_every_front_();
        }

        /** @brief Reads one partition's front at the seed this cursor is standing on. */
        void seed_one_front_(std::size_t partition_index, epoch_t written) noexcept {
            states_[partition_index] = front_state_t::exhausted_k;
            shared_lock_t _ {store_->mutexes_[partition_index]};
            auto fill = [&](value_t const &element) noexcept {
                fronts_[partition_index] = identifier_t(element);
                states_[partition_index] = front_state_t::holds_a_key_k;
            };
            inner_store_t const &part = store_->partitions_[partition_index];
            switch (seed_) {
            case cursor_seed_t::the_smallest_k: store_->seed_from_the_start_()(partition_index, fill); break;
            case cursor_seed_t::the_given_bound_k: {
                [[maybe_unused]] status_t const seeded = part.lower_bound(position_, fill, no_op_t {});
                break;
            }
            case cursor_seed_t::past_the_last_k: {
                [[maybe_unused]] status_t const seeded = part.upper_bound(position_, fill, no_op_t {});
                break;
            }
            }
            seen_epochs_[partition_index] = written;
        }

        /** @brief Reads every partition's front, which is how a cursor starts knowing anything. */
        void seed_every_front_() noexcept {
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index)
                seed_one_front_(partition_index, epoch_seen_(store_->epochs_[partition_index]));
        }

        /**
         *  @brief Re-reads the front of every partition written since this cursor last looked at it.
         *
         *  A front is only stale if somebody wrote its partition, and the constructor has already read
         *  every one of them - so an unchanged count means the cached key still stands.
         */
        void refresh_stale_fronts_() noexcept {
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                epoch_t const written = epoch_seen_(store_->epochs_[partition_index]);
                if (written == seen_epochs_[partition_index]) continue;
                seed_one_front_(partition_index, written);
            }
        }

      public:
        ordered_cursor_t() noexcept = default;

        /** @brief Whether every partition is spent, so the walk has nothing left to hand over. */
        [[nodiscard]] bool exhausted() const noexcept {
            for (front_state_t state : states_)
                if (state == front_state_t::holds_a_key_k) return false;
            return true;
        }

        /**
         *  @brief Hands @p callback_found the next key of the merged order, or reports the walk is over.
         *
         *  One partition is held while its key is read, and none between calls - so a walk may be
         *  suspended anywhere without holding the store, but @p callback_found itself runs under that
         *  one partition's lock and must not write to it.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        void next(callback_found_type_ &&callback_found, callback_missing_type_ &&callback_missing = {}) noexcept {
            while (true) {
                refresh_stale_fronts_();
                marked_partition_t const smallest = smallest_front_(store_->comparator_, fronts_, states_);
                if (smallest.presence != marked_presence_t::one_marked_k) {
                    callback_missing();
                    return;
                }

                identifier_t const chosen = fronts_[smallest.index];
                bool delivered = false;
                {
                    // One hold covers handing the key over and refilling the front behind it, so a step
                    // over a store nobody is writing costs exactly one partition acquisition.
                    shared_lock_t _ {store_->mutexes_[smallest.index]};
                    inner_store_t const &part = store_->partitions_[smallest.index];

                    // Inclusive, so a key erased since it was cached is answered by its successor rather
                    // than by a miss - the same probe covers both, and neither needs a second pass.
                    [[maybe_unused]] status_t const bounded = part.lower_bound(
                        chosen,
                        [&](value_t const &element) noexcept {
                            identifier_t landed(element);
                            fronts_[smallest.index] = landed;
                            if (store_->comparator_(chosen, landed)) return;
                            callback_found(element);
                            position_ = std::move(landed);
                            delivered = true;
                        },
                        [&]() noexcept { states_[smallest.index] = front_state_t::exhausted_k; });
                    seen_epochs_[smallest.index] = epoch_seen_(store_->epochs_[smallest.index]);

                    if (delivered) {
                        states_[smallest.index] = front_state_t::exhausted_k;
                        [[maybe_unused]] status_t const bounded = part.upper_bound(
                            position_,
                            [&](value_t const &element) noexcept {
                                fronts_[smallest.index] = identifier_t(element);
                                states_[smallest.index] = front_state_t::holds_a_key_k;
                            },
                            no_op_t {});
                    }
                }

                if (!delivered) continue; // The front rose or emptied; whichever it was, retry the minimum.
                seed_ = cursor_seed_t::past_the_last_k;
                return;
            }
        }
    };

    /** @brief A walk of every member in ascending order, resumable and holding no lock between steps. */
    [[nodiscard]] ordered_cursor_t cursor() const noexcept
        requires inner_is_ordered_k && (inner_names_its_smallest_k || inner_is_ranked_k)
    {
        return ordered_cursor_t {*this, cursor_seed_t::the_smallest_k};
    }

    /** @brief The same walk, begun at the first member ordered at or after @p from. */
    [[nodiscard]] ordered_cursor_t cursor_from(identifier_t from) const noexcept
        requires inner_is_ordered_k
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_given_bound_k};
        walking.position_ = std::move(from);
        return walking;
    }

    class transaction_t {
        friend class partitioned_store;
        /**
         *  @brief The shard set this transaction reaches through, held by pointer rather than reference.
         *    A reference member deletes the defaulted move assignment, and a transaction that moves but
         *    cannot be move-assigned is one no container can hold.
         */
        partitioned_store *store_;
        /** @brief One inner transaction per partition, all opened on the same snapshot and generation. */
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
                clock_t &clock = store_->clock_;
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
                writing_part_lock lock {store_->mutexes_[partition_index], store_->epochs_[partition_index]};
                status_t const one = callable(partitions_[partition_index]);
                if (failed(one)) status = one;
            }
            return status;
        }

        /** @brief The same walk, restricted to the partitions this transaction reached. */
        template <typename callable_type_>
        status_t for_touched_parts_(callable_type_ &&callable) noexcept {
            status_t status = success_k;
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index)) {
                writing_part_lock lock {store_->mutexes_[reached.index], store_->epochs_[reached.index]};
                status_t const one = callable(partitions_[reached.index]);
                if (failed(one)) status = one;
            }
            return status;
        }

        /**
         *  @brief Publishes every reached partition, having first learned that all of them may.
         *
         *  The engines behind this path stamp their own versions, so there is no shared clock and no
         *  one stamp to draw - but a partition can still refuse after its neighbours have written,
         *  because each re-checks its watches as it commits. Asking all of them while holding all of
         *  them is what keeps a refusal honest.
         */
        status_t commit_together_() noexcept
            requires(!inner_shares_clock_k && inner_transaction_splits_commit_k)
        {
            touched_parts_lock held {store_->mutexes_, store_->epochs_, touched_};
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index))
                if (status_t const refused = partitions_[reached.index].validate_for_commit(); failed(refused))
                    return refused;

            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index))
                partitions_[reached.index].publish_under();
            held.release();

            touched_.clear();
            return success_k;
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
            clock_t &clock = store_->clock_;
            touched_parts_lock held {store_->mutexes_, store_->epochs_, touched_};
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index))
                if (status_t const refused = partitions_[reached.index].validate_for_commit(); failed(refused))
                    return refused;

            typename clock_t::commit_in_flight_t in_flight;
            clock.begin_commit(in_flight);
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index))
                partitions_[reached.index].publish_under(in_flight.stamp());
            clock.end_commit(in_flight);
            held.release();

            // Held is released first on purpose: the wait blocks on another thread finishing its own
            // publication, and a partition lock carried into it would block that thread in turn. What
            // it costs is the tail of whatever older commit is still writing itself out, so it is a
            // property of how much the commits overlap rather than a constant.
            if constexpr (at_least(isolation_k, isolation_t::strict_serializable_k))
                clock.await_published(in_flight.stamp());

            // The claim on the old snapshot goes back at once, so a long-lived transaction committing
            // in a loop stops pinning the very versions it is superseding. Where this transaction
            // reads moves separately, in `settle_snapshot_`, once its own commit is whole - and every
            // entry point below settles before it does anything, so nothing observes the gap.
            [[maybe_unused]] generation_t const released = clock.take_snapshot(lease_);
            unsettled_stamp_ = static_cast<generation_t>(in_flight.stamp());

            // Reclamation runs at the mark this transaction just moved to, so a version this commit
            // superseded is freed rather than pinned by the claim it has already given up.
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index)) {
                writing_part_lock lock {store_->mutexes_[reached.index], store_->epochs_[reached.index]};
                partitions_[reached.index].prune_committed();
            }
            touched_.clear();
            return success_k;
        }

      public:
        transaction_t(partitioned_store &db, partition_transactions_t &&partition_transactions,
                      typename clock_t::snapshot_lease_t &&lease = {}) noexcept
            : store_(&db), partitions_(std::move(partition_transactions)), lease_(std::move(lease)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;

        [[nodiscard]] status_t reset() noexcept {
            settle_snapshot_();
            if constexpr (inner_shares_clock_k) {
                // One snapshot for every partition, drawn once, exactly as opening the transaction did.
                generation_t const snapshot = store_->clock_.take_snapshot(lease_);
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
            for (marked_partition_t reached = touched_.first_marked();
                 reached.presence == marked_presence_t::one_marked_k; reached = touched_.next_marked(reached.index)) {
                writing_part_lock lock {store_->mutexes_[reached.index], store_->epochs_[reached.index]};
                status = partitions_[reached.index].stage();
                if (failed(status)) break;
                staged.mark(reached.index);
            }
            if (succeeded(status)) return status;

            for (marked_partition_t landed = staged.first_marked(); landed.presence == marked_presence_t::one_marked_k;
                 landed = staged.next_marked(landed.index)) {
                writing_part_lock lock {store_->mutexes_[landed.index], store_->epochs_[landed.index]};
                [[maybe_unused]] status_t const unwound = partitions_[landed.index].rollback();
            }
            return status;
        }
        /**
         *  @brief Publishes every reached partition, or none of them.
         *
         *  Both paths ask every reached partition whether it may commit before any of them writes, so
         *  a refusal is reported over nothing and the transaction stays staged and retryable. They
         *  differ only in where the stamp comes from: one clock shared across the partitions, or each
         *  engine stamping its own versions.
         */
        [[nodiscard]] status_t commit() noexcept {
            settle_snapshot_();
            if constexpr (inner_shares_clock_k) return commit_under_one_stamp_();
            else return commit_together_();
        }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(id);
            touched_.mark(partition_index);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].watch(id);
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(comparable);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].find(std::forward<comparable_type_>(comparable),
                                                     std::forward<callback_found_type_>(callback_found),
                                                     std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const looked_up = find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /** @brief Whether @p comparable is there, including this transaction's own writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(comparable);
            shared_lock_t _ {store_->mutexes_[partition_index]};
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
            std::size_t partition_index = store_->bucket_(comparable);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].find_and_watch(std::forward<comparable_type_>(comparable),
                                                               std::forward<callback_found_type_>(callback_found),
                                                               std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief The first member this transaction reads at or after @p comparable, from any partition. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_lower_bounds_k
        {
            settle_snapshot_();
            store_t::first_merged_(
                store_->comparator_, partitions_, store_->mutexes_,
                [&](std::size_t partition_index, auto &&fill) noexcept {
                    [[maybe_unused]] status_t const seeded =
                        partitions_[partition_index].lower_bound(comparable, fill, no_op_t {});
                },
                std::forward<callback_found_type_>(callback_found),
                std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /** @brief The first member this transaction reads strictly after @p comparable, from any partition. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_is_ordered_k
        {
            settle_snapshot_();
            store_t::first_merged_(
                store_->comparator_, partitions_, store_->mutexes_,
                [&](std::size_t partition_index, auto &&fill) noexcept {
                    [[maybe_unused]] status_t const seeded =
                        partitions_[partition_index].upper_bound(comparable, fill, no_op_t {});
                },
                std::forward<callback_found_type_>(callback_found),
                std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /**
         *  @brief Hands @p callback every member of [ @p lower, @p upper ) this transaction reads, in order.
         *
         *  Merged across the partitions rather than concatenated, so a transactional scan answers in the
         *  order an ordered container promises.
         *
         *  @warning Every partition is held shared for the walk, and @p callback runs under all of them.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires inner_transaction_lower_bounds_k && inner_transaction_is_ordered_k
        {
            settle_snapshot_();
            every_part_lock<shared_lock_t> _ {store_->mutexes_};
            store_t::walk_merged_(
                store_->comparator_, partitions_,
                [&](std::size_t partition_index, auto &&fill) noexcept {
                    [[maybe_unused]] status_t const seeded =
                        partitions_[partition_index].lower_bound(lower, fill, no_op_t {});
                },
                [&](std::size_t partition_index, identifier_t const &key) noexcept {
                    if (!store_->comparator_(key, upper)) return merge_control_t::halt_k;
                    [[maybe_unused]] status_t const answered =
                        partitions_[partition_index].find(key, callback, no_op_t {});
                    return merge_control_t::resume_k;
                });
            return success_k;
        }

        [[nodiscard]] status_t upsert(value_t &&element) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(element);
            touched_.mark(partition_index);
            return partitions_[partition_index].upsert(std::move(element));
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(id);
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
            std::size_t partition_index = store_->bucket_(element);
            touched_.mark(partition_index);
            shared_lock_t _ {store_->mutexes_[partition_index]};
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
            std::size_t partition_index = store_->bucket_(element);
            touched_.mark(partition_index);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].insert(std::move(element),
                                                       std::forward<callback_inserted_type_>(callback_inserted),
                                                       std::forward<callback_existing_type_>(callback_existing));
        }

        /** @brief Stages @p element only if its key is already taken, refusing to create one. */
        [[nodiscard]] status_t update(value_t &&element) noexcept
            requires inner_transaction_refuses_absent_key_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(element);
            touched_.mark(partition_index);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].update(std::move(element));
        }

        /** @brief Stages @p element only if its key is free, leaving an incumbent untouched. */
        [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept
            requires inner_transaction_skips_occupied_key_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(element);
            touched_.mark(partition_index);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].insert_if_missing(std::move(element));
        }

        /** @brief Hands @p callback every member equal to @p comparable, which one partition owns outright. */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
            requires inner_transaction_matches_equals_k
        {
            settle_snapshot_();
            std::size_t partition_index = store_->bucket_(comparable);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].equal_range(std::forward<comparable_type_>(comparable),
                                                            std::forward<callback_type_>(callback));
        }

#pragma region Transaction Range Operations

        /** @brief How many members equal @p comparable, which one partition alone can answer. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_counts_k
        {
            settle_snapshot_();
            std::size_t const partition_index = store_->bucket_(comparable);
            shared_lock_t _ {store_->mutexes_[partition_index]};
            return partitions_[partition_index].count(std::forward<comparable_type_>(comparable));
        }

        /**
         *  @brief Copies out the first member at or after @p comparable in the merged order.
         *
         *  Answered through this transaction's own bound rather than one partition's, since the smallest
         *  successor may live in any of them.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_lower_bounds_k && inner_transaction_is_ordered_k
        {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const answered = lower_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /** @brief Copies out the first member after @p comparable in the merged order. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_lower_bounds_k && inner_transaction_is_ordered_k
        {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const answered = upper_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /**
         *  @brief Stages a tombstone for every member in [ @p lower, @p upper ), in every partition.
         *
         *  The window spans the whole keyspace and a key's partition is its hash, so every partition is
         *  asked. Each answers for its own share, and the union is the window.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            settle_snapshot_();
            return for_parts_(
                [&](inner_transaction_t &part) noexcept { return part.erase_range(lower, upper, callback); });
        }

        /** @brief Stages a tombstone for every member at or after @p lower, in every partition. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            settle_snapshot_();
            return for_parts_([&](inner_transaction_t &part) noexcept { return part.erase_from(lower, callback); });
        }

        /** @brief Stages a tombstone for every member before @p upper, in every partition. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            settle_snapshot_();
            return for_parts_([&](inner_transaction_t &part) noexcept { return part.erase_up_to(upper, callback); });
        }

        /** @brief Hands @p callback each member in [ @p lower, @p upper ) to revise, in every partition. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                            callback_type_ &&callback) noexcept
            requires inner_transaction_revises_range_k
        {
            settle_snapshot_();
            return for_parts_(
                [&](inner_transaction_t &part) noexcept { return part.update_range(lower, upper, callback); });
        }

        /** @brief Draws one member uniformly from [ @p lower, @p upper ) of the merged order. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                          callback_type_ &&callback) const noexcept
            requires inner_transaction_lower_bounds_k && inner_transaction_is_ordered_k
        {
            std::size_t counted = 0;
            if (status_t const measured = range(lower, upper, [&](value_t const &) noexcept { ++counted; });
                failed(measured))
                return measured;
            if (!counted) return success_k;

            std::size_t skipped = draw_below(generator, counted);
            bool drawn = false;
            return range(lower, upper, [&](value_t const &value) noexcept {
                if (drawn) return;
                if (skipped) --skipped;
                else {
                    callback(value);
                    drawn = true;
                }
            });
        }

        /** @brief Fills @p reservoir with up to @p capacity members drawn from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename output_iterator_type_ = no_op_t>
        [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                                std::size_t &seen, std::size_t capacity,
                                                output_iterator_type_ &&reservoir) const noexcept
            requires inner_transaction_lower_bounds_k && inner_transaction_is_ordered_k
        {
            static_assert(std::is_nothrow_copy_assignable_v<value_t>,
                          "a reservoir copies into the caller's buffer, so the member must copy without throwing");
            return range(lower, upper, [&](value_t const &value) noexcept {
                if (seen < capacity) reservoir[seen] = value;
                else if (std::size_t const slot = draw_below(generator, seen + 1); slot < capacity)
                    reservoir[slot] = value;
                ++seen;
            });
        }

#pragma endregion Transaction Range Operations

        /** @brief Hands @p callback every member this transaction reads, in no particular order. */
        template <typename callback_type_ = no_op_t>
        [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept
            requires inner_transaction_enumerates_k
        {
            settle_snapshot_();
            for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
                shared_lock_t lock {store_->mutexes_[partition_index]};
                if (status_t const visited = partitions_[partition_index].for_each(callback); failed(visited))
                    return visited;
            }
            return success_k;
        }

        /**
         *  @brief The generation every part of this transaction was opened on.
         *    A shard set draws one generation and opens every partition on it, so any part answers for
         *    all of them; a part keeping its own counter is read from the first partition, which is the
         *    one every other single-answer query here consults.
         */
        [[nodiscard]] generation_t generation() const noexcept { return partitions_[0].generation(); }

        /** @brief Sizes every part's watch list, splitting @p size the way the store splits a reserve. */
        [[nodiscard]] status_t reserve(std::size_t size) noexcept
            requires inner_transaction_reserves_k
        {
            status_t status = success_k;
            for (inner_transaction_t &part : partitions_)
                if (status_t const one = part.reserve(size / partitions_k); failed(one)) status = one;
            return status;
        }

        /** @brief Whether anything is staged in any part, which is transaction-local and needs no lock. */
        [[nodiscard]] bool has_changes() const noexcept
            requires inner_transaction_reports_changes_k
        {
            for (inner_transaction_t const &part : partitions_)
                if (part.has_changes()) return true;
            return false;
        }

        /** @brief How many writes are staged across every part, which is transaction-local and needs no lock. */
        [[nodiscard]] std::size_t changes_count() const noexcept
            requires inner_transaction_reports_changes_k
        {
            std::size_t counted = 0;
            for (inner_transaction_t const &part : partitions_) counted += part.changes_count();
            return counted;
        }
    };

  private:
    mutable mutexes_t mutexes_;
    /** @brief One write count per partition, moved by every exclusive hold of the matching mutex. */
    epochs_t epochs_ {};
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

    partitioned_store(partitions_t &&parts, hash_t const &hasher = {}, comparator_t const &comparator = {}) noexcept
        : partitions_(std::move(parts)), hasher_(hasher), comparator_(comparator) {
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
        if (expected<partitions_t> parts = new_parts(); parts) result = partitioned_store {std::move(*parts)};
        return result;
    }

    /**
     *  @brief Builds a store whose every partition shares one comparator and one hasher.
     *  @param[in] comparator The instance every comparison consults, in each partition and across them.
     *  @param[in] hasher The instance that maps an identifier to its partition.
     *  @return The store, or the status that refused to build one of its partitions.
     */
    [[nodiscard]] static expected<partitioned_store> make(comparator_t const &comparator,
                                                          hash_t const &hasher) noexcept {
        expected<partitioned_store> result;
        if (expected<partitions_t> parts = new_parts(comparator); parts)
            result = partitioned_store {std::move(*parts), hasher, comparator};
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
            writing_part_lock lock {mutexes_[partition_index], epochs_[partition_index]};
            if constexpr (inner_shares_clock_k)
                return partitions_[partition_index].transaction_at(snapshot, generation);
            else return partitions_[partition_index].transaction();
        });
        if (!maybe) return maybe.status();

        return transaction_t(*this, std::move(*maybe), std::move(lease));
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        std::size_t partition_index = bucket_(element);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
        return partitions_[partition_index].upsert(std::move(element));
    }

    /**
     *  @brief Removes one element, reporting whether it was there.
     *    Touches only the partition that owns the key, so the rest stay unlocked.
     *  @param[in] comparable Anything the inner store compares against, convertible to an identifier.
     *  @param[in] callback_found Receives the element that was removed.
     *  @param[in] callback_missing Fires when no such element existed.
     *  @return @c key_not_found_k when absent, so the answer is available without a second probe.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        std::size_t partition_index = bucket_(comparable);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
        return partitions_[partition_index].erase(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_found_type_>(callback_found),
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
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::size_t partition_index = bucket_(comparable);
        shared_lock_t _ {mutexes_[partition_index]};
        [[maybe_unused]] status_t const answered = partitions_[partition_index].find(
            std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
            std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /** @brief Whether @p comparable is there, asking only the partition that owns it. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        std::size_t partition_index = bucket_(comparable);
        shared_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].contains(std::forward<comparable_type_>(comparable));
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

    /** @brief Inserts one element only if its key is absent, leaving an incumbent untouched. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        std::size_t partition_index = bucket_(element);
        writing_part_lock lock {mutexes_[partition_index], epochs_[partition_index]};
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
        std::size_t partition_index = bucket_(element);
        writing_part_lock lock {mutexes_[partition_index], epochs_[partition_index]};
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
        std::size_t partition_index = bucket_(element);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
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
        std::size_t partition_index = bucket_(element);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
        return partitions_[partition_index].insert(std::move(element),
                                                   std::forward<callback_inserted_type_>(callback_inserted),
                                                   std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Writes one element only if its key is already taken, refusing to create one. */
    [[nodiscard]] status_t update(value_t &&element) noexcept
        requires inner_refuses_absent_key_k
    {
        std::size_t partition_index = bucket_(element);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
        return partitions_[partition_index].update(std::move(element));
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        for (; begin != end; ++begin) {
            value_t element(*begin);
            std::size_t partition_index = bucket_(element);
            writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
            if (auto status = partitions_[partition_index].insert_if_missing(std::move(element)); failed(status))
                return status;
        }
        return success_k;
    }

    /** @brief The first element ordered strictly after @p comparable, which may live in any partition. */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        first_merged_(
            comparator_, partitions_, mutexes_,
            [&](std::size_t partition_index, auto &&fill) noexcept {
                [[maybe_unused]] status_t const seeded =
                    partitions_[partition_index].upper_bound(comparable, fill, no_op_t {});
            },
            std::forward<callback_found_type_>(callback_found), std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /**
     *  @brief The first element at or after @p comparable, which may live in any partition.
     *
     *  Every partition is asked for its own bound under one set of locks, so the exact match and the
     *  cross-partition minimum are read at the same moment rather than in two probes a writer can slip
     *  between.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        first_merged_(
            comparator_, partitions_, mutexes_,
            [&](std::size_t partition_index, auto &&fill) noexcept {
                [[maybe_unused]] status_t const seeded =
                    partitions_[partition_index].lower_bound(comparable, fill, no_op_t {});
            },
            std::forward<callback_found_type_>(callback_found), std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /** @brief Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const looked_up = find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /** @brief Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /**
     *  @brief Hands @p callback every member of [ @p lower, @p upper ) in one ascending order.
     *
     *  The partitions are merged rather than concatenated, so the sequence is the one an ordered
     *  container promises whatever the key's hash happened to be. Each element costs a comparison per
     *  partition and one descent to refill the front it came from, against the single iterator step a
     *  concatenated walk would take - the price of the order being right.
     *
     *  @warning Every partition is held shared for the length of the walk, so every writer waits, and
     *    @p callback runs under all of them: a callback reaching back into this store deadlocks.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
        requires inner_is_ordered_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};
        walk_merged_(
            comparator_, partitions_,
            [&](std::size_t partition_index, auto &&fill) noexcept {
                [[maybe_unused]] status_t const seeded =
                    partitions_[partition_index].lower_bound(lower, fill, no_op_t {});
            },
            [&](std::size_t partition_index, identifier_t const &key) noexcept {
                if (!comparator_(key, upper)) return merge_control_t::halt_k;
                [[maybe_unused]] status_t const answered = partitions_[partition_index].find(key, callback, no_op_t {});
                return merge_control_t::resume_k;
            });
        return success_k;
    }

    /**
     *  @brief Erases the half-open range from every partition, attempting all of them whatever one answers.
     *  @return Success, or the last refusal - after which every partition was attempted and which
     *    subset the window was erased from is not reported.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires inner_is_ordered_k
    {
        return for_all<unique_lock_t, locking_policy_t::all_at_once_k, refusal_policy_t::attempt_every_k>(
            partitions_, mutexes_,
            [&](inner_store_t &part) noexcept { return part.erase_range(lower, upper, callback); });
    }

    /**
     *  @brief Erases every element at or after @p lower from every partition, attempting all of them.
     *  @return Success, or the last refusal - after which every partition was attempted and which
     *    subset the tail was erased from is not reported.
     */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        return for_all<unique_lock_t, locking_policy_t::all_at_once_k, refusal_policy_t::attempt_every_k>(
            partitions_, mutexes_, [&](inner_store_t &part) noexcept { return part.erase_from(lower, callback); });
    }

    /**
     *  @brief Erases every element before @p upper from every partition, attempting all of them.
     *  @return Success, or the last refusal - after which every partition was attempted and which
     *    subset the head was erased from is not reported.
     */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        return for_all<unique_lock_t, locking_policy_t::all_at_once_k, refusal_policy_t::attempt_every_k>(
            partitions_, mutexes_, [&](inner_store_t &part) noexcept { return part.erase_up_to(upper, callback); });
    }

    /**
     *  @brief Rewrites the mapped side of every element in [ @p lower, @p upper ), across all partitions.
     *
     *  Every partition is held exclusively for the length of the walk, as @c erase_range holds them,
     *  so the window is revised as one write rather than sixteen.
     *
     *  @param[in] callback Invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Success, or the last refusal - after which every partition was attempted and which
     *    subset the window was revised in is not reported. A partition whose own walk cannot fail
     *    always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires inner_revises_range_k
    {
        return for_all<unique_lock_t, locking_policy_t::all_at_once_k, refusal_policy_t::attempt_every_k>(
            partitions_, mutexes_,
            [&](inner_store_t &part) noexcept { return part.update_range(lower, upper, callback); });
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
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept
        requires inner_enumerates_k
    {
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            shared_lock_t lock {mutexes_[partition_index]};
            if (status_t const visited = partitions_[partition_index].for_each(callback); failed(visited))
                return visited;
        }
        return success_k;
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
            writing_part_lock lock {mutexes_[partition_index], epochs_[partition_index]};
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
            writing_part_lock lock {mutexes_[partition_index], epochs_[partition_index]};
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
    [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                  callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};
        std::size_t position = 0;
        bool delivered = false;
        walk_merged_(comparator_, partitions_, seed_from_the_start_(),
                     [&](std::size_t partition_index, identifier_t const &key) noexcept {
                         if (position++ != ordinal) return merge_control_t::resume_k;
                         [[maybe_unused]] status_t const answered =
                             partitions_[partition_index].find(key, callback_found, no_op_t {});
                         delivered = true;
                         return merge_control_t::halt_k;
                     });
        if (!delivered) callback_missing();
        return success_k;
    }

    /**
     *  @brief Hands @p callback_found how many elements the store orders before @p comparable.
     *    Counted by the same merge @c select walks, so it costs the rank rather than a descent.
     *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};
        std::size_t position = 0;
        bool found = false;
        walk_merged_(comparator_, partitions_, seed_from_the_start_(),
                     [&](std::size_t, identifier_t const &key) noexcept {
                         if (comparator_(key, comparable)) {
                             ++position;
                             return merge_control_t::resume_k;
                         }
                         found = !comparator_(comparable, key);
                         return merge_control_t::halt_k;
                     });
        if (found) callback_found(position);
        else callback_missing();
        return success_k;
    }

    /**
     *  @brief Hands one element of the range to @p callback, drawn from a single random partition.
     *
     *  @warning The draw is uniform only if every partition holds a similar number of in-range
     *    entries, and it takes one lock. The reservoir overload below walks all of them instead, so a
     *    call site that needs the whole range weighted correctly wants that one.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                      callback_type_ &&callback) const noexcept
        requires inner_samples_one_k
    {
        std::size_t partition_index = generator() % partitions_k;
        shared_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].sample_one(
            std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
            std::forward<generator_type_>(generator), std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Fills @p reservoir with up to @p reservoir_capacity elements of the range, weighting
     *    every partition by how many in-range entries it actually holds.
     *
     *  @warning One partition is locked at a time rather than all of them, so a writer moving an entry
     *    between partitions can have it sampled twice or not at all.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                            std::size_t &seen, std::size_t reservoir_capacity,
                                            output_iterator_type_ &&reservoir) const noexcept
        requires inner_samples_reservoir_k
    {
        // Ascending order, blocking, like every other all-partition walk here.
        for (std::size_t partition_index = 0; partition_index != partitions_k; ++partition_index) {
            shared_lock_t lock {mutexes_[partition_index]};
            [[maybe_unused]] status_t const answered = partitions_[partition_index].sample_reservoir(
                lower, upper, generator, seen, reservoir_capacity, reservoir);
        }
        return success_k;
    }

    /**
     *  @brief Empties every partition in place, stopping where one refuses.
     *
     *  Each partition clears itself rather than being replaced by a fresh one: a replacement leaves an
     *  open transaction referring to contents that are gone, and it would have to rebuild the
     *  comparator and allocator the partition was given.
     *
     *  @return Success, or the refusal that stopped the walk - after which the partitions above the
     *    refusing one still hold everything they held.
     */
    [[nodiscard]] status_t clear() noexcept {
        return for_all<unique_lock_t, locking_policy_t::one_at_a_time_k, refusal_policy_t::stop_at_first_k>(
            partitions_, mutexes_, [](inner_store_t &part) noexcept { return part.clear(); });
    }

    /**
     *  @brief Sizes every partition for its share of @p size, stopping where one refuses.
     *  @return Success, or the refusal that stopped the walk - after which the partitions above the
     *    refusing one are sized as they were.
     */
    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        return for_all<unique_lock_t, locking_policy_t::one_at_a_time_k, refusal_policy_t::stop_at_first_k>(
            partitions_, mutexes_, [size](inner_store_t &part) noexcept { return part.reserve(size / partitions_k); });
    }

    /** @brief Hands @p callback every member equal to @p comparable, which one partition owns outright. */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
        requires inner_matches_equals_k
    {
        std::size_t partition_index = bucket_(comparable);
        shared_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].equal_range(std::forward<comparable_type_>(comparable),
                                                        std::forward<callback_type_>(callback));
    }

    /** @brief How many versions every partition holds together, published and staged alike. */
    [[nodiscard]] std::size_t versions_count() const noexcept
        requires inner_counts_versions_k
    {
        std::size_t counted = 0;
        every_part_lock<shared_lock_t> _ {mutexes_};
        for (auto const &part : partitions_) counted += part.versions_count();
        return counted;
    }

    /** @brief How many versions of @p comparable are still held, by the one partition that owns it. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t versions_count(comparable_type_ const &comparable) const noexcept
        requires inner_counts_versions_k
    {
        std::size_t partition_index = bucket_(comparable);
        shared_lock_t _ {mutexes_[partition_index]};
        return partitions_[partition_index].versions_count(comparable);
    }

    /**
     *  @brief The newest snapshot no open transaction sits below, which is what @c vacuum prunes to.
     *    The oldest of the partitions' answers, so no partition prunes past a reader another still
     *    counts - which for partitions sharing one clock is the single answer all of them give.
     */
    [[nodiscard]] generation_t low_water_mark() const noexcept
        requires inner_marks_low_water_k
    {
        every_part_lock<shared_lock_t> _ {mutexes_};
        generation_t oldest = partitions_[0].low_water_mark();
        for (auto const &part : partitions_)
            if (generation_t const one = part.low_water_mark(); one < oldest) oldest = one;
        return oldest;
    }

    /** @brief Writes @p element whether or not its key is taken, spelled as a partition spells it. */
    [[nodiscard]] status_t insert_or_assign(value_t &&element) noexcept
        requires inner_assigns_over_key_k
    {
        std::size_t partition_index = bucket_(element);
        writing_part_lock _ {mutexes_[partition_index], epochs_[partition_index]};
        return partitions_[partition_index].insert_or_assign(std::move(element));
    }
};

#pragma region Aliases

/**
 *  @brief A set-shaped store sharded across independently locked partitions, spelled
 *    @c partitioned_set<monotonic_avl_set<key_t>>.
 *
 *  The wrapper is one class whichever shape it holds, so the shape is a constraint rather than a
 *  second spelling: naming the set alias over a store of mappings has no substitution, and neither
 *  does naming @c partitioned_map over a store of plain keys. Two aliases that accept the same
 *  arguments would name the same type and catch nothing.
 */
template <set_shaped_store set_store_type_, typename hash_type_ = hash<typename set_store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
using partitioned_set = partitioned_store<set_store_type_, hash_type_, shared_mutex_type_, partitions_count_>;

/**
 *  @brief A map-shaped store sharded across independently locked partitions, spelled
 *    @c partitioned_map<monotonic_avl_map<key_t, value_t>>.
 */
template <map_shaped_store map_store_type_, typename hash_type_ = hash<typename map_store_type_::identifier_t>,
          typename shared_mutex_type_ = spin_shared_mutex, std::size_t partitions_count_ = 16>
using partitioned_map = partitioned_store<map_store_type_, hash_type_, shared_mutex_type_, partitions_count_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
