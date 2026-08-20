/**
 *  @brief Multi-version transactional store granting every transaction a snapshot of the store as it
 *    stood when the transaction opened.
 *  @author Ash Vardanian
 *  @file include/smashtable/snapshot_store.hpp
 *  @date August 17, 2026
 *
 *  Can be instantiated with any key-addressable core - AVL trees, weight-balanced trees, open-addressed
 *  tables - and gates its ordered surface behind the cores that supply an ordering. Exception-free,
 *  with all reads delivered through callbacks.
 */
#pragma once
#include <cassert> // `assert`

#include <iterator>    // `std::iterator_traits`
#include <memory>      // `std::allocator_traits`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::exchange`, `std::move`

#include "basic_avl_tree.hpp"
#include "basic_hash_table.hpp"
#include "basic_vector.hpp"
#include "basic_wb_tree.hpp"
#include "shared.hpp"

namespace ashvardanian::smashtable {

#pragma region Liveness Ranking

/** @brief Whether an entry is the newest committed-and-present version of its key, as of right now. */
enum class liveness_t : bool {
    /** @brief Something newer speaks for the key, or this entry says the key was taken away. */
    superseded_k,
    /** @brief This is the version the newest published stamp reads for the key. */
    survivor_k,
};

/**
 *  @brief A stored version carrying the one bit an order-statistic descent reads off the entry.
 *
 *  A rotation recomputes a subtree sum at a moment nobody chose, so the predicate the sum is kept over
 *  has to be answerable from the entry alone rather than from the store around it.
 */
template <typename versioned_type_>
struct live_tagged : versioned_type_ {
    using versioned_type_::versioned_type_;

    live_tagged() = default;
    live_tagged(live_tagged &&) noexcept = default;
    live_tagged &operator=(live_tagged &&) noexcept = default;
    live_tagged(live_tagged const &) = delete;
    live_tagged &operator=(live_tagged const &) = delete;

    /** @brief Whether this entry is the one the newest published stamp reads for its key. */
    liveness_t liveness {liveness_t::superseded_k};
};

/**
 *  @brief Counts one entry per key the newest published stamp reads, for the tree to sum per subtree.
 *    Answers zero for an undecorated element, which is what the bare core the store rebinds from holds.
 */
struct liveness_augmentation_t {
    template <typename fruit_type_>
    static std::size_t augmented_count(fruit_type_ const &fruit) noexcept {
        if constexpr (requires { fruit.liveness; })
            return fruit.liveness == liveness_t::survivor_k ? std::size_t {1} : std::size_t {0};
        else return std::size_t {0};
    }
};

/** @brief A core keeping a second subtree count, which is what a snapshot store's @c select descends on. */
template <typename collection_type_>
concept ranked_by_liveness = requires { typename collection_type_::augmentation_t; } &&
                             !std::is_same_v<typename collection_type_::augmentation_t, no_augmentation_t>;

#pragma endregion Liveness Ranking

#pragma region Snapshot Clock

/**
 *  @brief The stamps and the reader census a family of stamp-based stores keeps in common.
 *
 *  A single store owns one of these and never shares it. A shard set hands the very same instance to
 *  every partition, which is what turns sixteen independently locked stores into one snapshot: a
 *  reader draws one snapshot here and answers every partition at it, and a commit draws one stamp
 *  here and writes it into every partition it touched.
 *
 *  @section snapshot_clock_publication Publication
 *
 *  A stamp is drawn before the versions carrying it are written and the watermark only moves once
 *  they all are, so a snapshot never names a commit that is still writing itself out. Commits may
 *  overlap, so the watermark stops one below the oldest commit still in flight rather than at the
 *  newest one finished - a later commit that finishes first stays invisible until the earlier one
 *  lands, and then both appear together.
 *
 *  @section snapshot_clock_census Census
 *
 *  Readers are listed, not counted: every open snapshot is one node on an intrusive list ordered by
 *  the snapshot it reads at, exactly as commits in flight are ordered by their stamp. A snapshot is
 *  drawn at the watermark, which never moves backwards, so a fresh reader joins at the newest end
 *  and the oldest reader is the head - the minimum comes for free rather than from a remembered mark.
 *
 *  That minimum is what reclamation prunes to, and being global is the point: a partition that pruned
 *  to its own readers would free a version a reader of another partition is still entitled to name.
 *  With the list empty the mark is the watermark itself, which is what a reader arriving now would
 *  open on. A reader that outlives every other one therefore holds retention at its own snapshot and
 *  no lower, and its departure releases everything at once.
 *
 *  @section snapshot_clock_ordering Ordering
 *
 *  The watermark is a plain relaxed atomic, and the guarantee does not rest on it: @c end_commit
 *  stores it and @c take_snapshot and @c low_water_mark read it under @c mutex_, so publication and
 *  the drawing of a snapshot are ordered by that mutex rather than by the atomic. The bare
 *  @c published_stamp accessor reaches only single-key reads outside any transaction, which promise
 *  nothing across keys.
 *
 *  @warning A transactional path answering at @c published_stamp instead of at its own snapshot would
 *    make that relaxed load load-bearing, and would be a real defect on a weakly ordered machine.
 *    Read at a snapshot drawn through @c take_snapshot, or take @c mutex_.
 *
 *  @warning Every entry point here takes one short lock, and none of them calls back into a store, so
 *    the clock is a leaf and may be reached with a store's own lock held. The one exception is
 *    @c await_published, which blocks on another thread finishing its publication and must therefore
 *    be called with no store lock held.
 */
class snapshot_clock_t {

  public:
    /**
     *  @brief One commit that has drawn its stamp and has not finished writing it everywhere yet.
     *    The clock links it in place, so it must outlive the publication - a local of the committing
     *    frame, never a temporary - and it is neither copied nor moved.
     */
    class commit_in_flight_t {
        friend class snapshot_clock_t;

        /** @brief The commit that drew the previous stamp and is also still in flight. */
        commit_in_flight_t *older_ {nullptr};
        /** @brief The commit that drew the next stamp and is also still in flight. */
        commit_in_flight_t *newer_ {nullptr};
        /** @brief The stamp this commit publishes under, meaningless before it is drawn. */
        generation_t stamp_ {0};

      public:
        constexpr commit_in_flight_t() noexcept = default;
        commit_in_flight_t(commit_in_flight_t const &) = delete;
        commit_in_flight_t &operator=(commit_in_flight_t const &) = delete;

        /** @brief The stamp every version of this commit is written under. */
        [[nodiscard]] commit_stamp_t stamp() const noexcept { return static_cast<commit_stamp_t>(stamp_); }
    };

    /**
     *  @brief One reader's claim on a snapshot, which holds the low-water mark at or below it.
     *
     *  While the claim is held the lease is a node on the clock's census list, so it is the reader
     *  itself that pins retention and no separate mark has to be kept in step with it. Moving a lease
     *  splices the new object into the old one's place under the clock's lock, which is what lets a
     *  transaction holding one be moved. The claim is given back once, whenever the lease goes away.
     */
    class snapshot_lease_t {
        friend class snapshot_clock_t;

        /** @brief The clock the claim is registered with, null once it has been given back. */
        snapshot_clock_t *clock_ {nullptr};
        /** @brief The reader on an older snapshot, or null when this is the oldest one open. */
        snapshot_lease_t *older_ {nullptr};
        /** @brief The reader on a newer snapshot, or null when this is the newest one open. */
        snapshot_lease_t *newer_ {nullptr};
        /** @brief The stamp every read under this claim is answered at. */
        generation_t snapshot_ {0};

        /** @brief Takes @p other's place on the census list, leaving @p other holding nothing. */
        void adopt_(snapshot_lease_t &other) noexcept {
            snapshot_ = other.snapshot_;
            clock_ = other.clock_;
            if (clock_) clock_->relink_lease_(other, *this);
        }

      public:
        constexpr snapshot_lease_t() noexcept = default;
        snapshot_lease_t(snapshot_lease_t &&other) noexcept { adopt_(other); }
        snapshot_lease_t &operator=(snapshot_lease_t &&other) noexcept {
            if (this == &other) return *this;
            retire();
            adopt_(other);
            return *this;
        }
        ~snapshot_lease_t() noexcept { retire(); }
        snapshot_lease_t(snapshot_lease_t const &) = delete;
        snapshot_lease_t &operator=(snapshot_lease_t const &) = delete;

        /** @brief The stamp this claim reads at. */
        [[nodiscard]] generation_t snapshot() const noexcept { return snapshot_; }

        /** @brief Whether this claim is still registered, which a moved-from one is not. */
        [[nodiscard]] bool held() const noexcept { return clock_ != nullptr; }

        /** @brief Gives the claim back, letting the low-water mark move past it. Idempotent. */
        void retire() noexcept {
            if (!clock_) return;
            clock_->retire_snapshot_(*this);
            clock_ = nullptr;
        }
    };

  private:
    /** @brief Guards the census and the in-flight list, both of which are read and written together. */
    mutable spin_shared_mutex_t mutex_ {};
    /** @brief Dates transactions rather than their visibility, and is drawn without the lock. */
    alignas(atomic_alignment<generation_t>) generation_t generation_ {0};
    /** @brief The newest stamp handed to a commit, whether or not that commit has landed. */
    generation_t commits_ {0};
    /** @brief The newest stamp a fully written commit left behind, and the only snapshot handed out. */
    alignas(atomic_alignment<generation_t>) generation_t published_stamp_ {0};
    /** @brief The commit with the smallest stamp still writing itself out, which caps the watermark. */
    commit_in_flight_t *oldest_in_flight_ {nullptr};
    /** @brief Where a freshly drawn stamp joins the list, keeping it ordered by stamp. */
    commit_in_flight_t *newest_in_flight_ {nullptr};
    /** @brief The reader on the smallest snapshot, which is what retention is pinned to. */
    snapshot_lease_t *oldest_lease_ {nullptr};
    /** @brief Where a freshly drawn snapshot joins the census, keeping it ordered by snapshot. */
    snapshot_lease_t *newest_lease_ {nullptr};
    /** @brief What the census currently answers, republished under the lock whenever it changes. */
    alignas(atomic_alignment<generation_t>) generation_t low_water_mark_ {0};

    /** @brief Files @p lease at the newest end of the census, where its snapshot belongs. */
    void link_newest_(snapshot_lease_t &lease) noexcept {
        lease.older_ = newest_lease_;
        lease.newer_ = nullptr;
        (newest_lease_ ? newest_lease_->newer_ : oldest_lease_) = &lease;
        newest_lease_ = &lease;
    }

    /** @brief Takes @p lease off the census, leaving it linked to nothing. */
    void unlink_(snapshot_lease_t &lease) noexcept {
        (lease.older_ ? lease.older_->newer_ : oldest_lease_) = lease.newer_;
        (lease.newer_ ? lease.newer_->older_ : newest_lease_) = lease.older_;
        lease.older_ = nullptr;
        lease.newer_ = nullptr;
    }

    /** @brief Republishes the mark from the census head, which is the answer whenever it changes. */
    void republish_mark_() noexcept {
        atomic_store<generation_t>(low_water_mark_,
                                   oldest_lease_ ? oldest_lease_->snapshot_ : atomic_load(published_stamp_));
    }

    /** @brief Puts @p replacement in @p held 's place, for a lease that is being moved. */
    void relink_lease_(snapshot_lease_t &held, snapshot_lease_t &replacement) noexcept {
        unique_lock<spin_shared_mutex_t> _ {mutex_};
        replacement.older_ = held.older_;
        replacement.newer_ = held.newer_;
        (held.older_ ? held.older_->newer_ : oldest_lease_) = &replacement;
        (held.newer_ ? held.newer_->older_ : newest_lease_) = &replacement;
        held.older_ = nullptr;
        held.newer_ = nullptr;
        held.clock_ = nullptr;
    }

    /** @brief Gives one reader's claim back, which only @c snapshot_lease_t is allowed to do. */
    void retire_snapshot_(snapshot_lease_t &lease) noexcept {
        unique_lock<spin_shared_mutex_t> _ {mutex_};
        unlink_(lease);
        republish_mark_();
    }

  public:
    constexpr snapshot_clock_t() noexcept = default;
    snapshot_clock_t(snapshot_clock_t const &) = delete;
    snapshot_clock_t &operator=(snapshot_clock_t const &) = delete;

    /** @brief Hands out the next generation, which dates a transaction rather than its visibility. */
    generation_t next_generation() noexcept { return atomic_add_fetch<generation_t>(generation_, 1); }

    /** @brief The newest stamp every part of which is written, which is what a fresh read answers at. */
    [[nodiscard]] generation_t published_stamp() const noexcept { return atomic_load(published_stamp_); }

    /**
     *  @brief The newest stamp any commit has drawn, landed or not, which is what a reader must
     *    validate against rather than the watermark.
     *
     *  A commit in flight has already written its stamp onto its versions while the watermark still
     *  sits below it, so a validator asking whether anything moved has to compare against what was
     *  drawn. Under a shard set that overlap is ordinary rather than a corner case.
     */
    [[nodiscard]] generation_t drawn_stamp() const noexcept {
        shared_lock<spin_shared_mutex_t> _ {mutex_};
        return commits_;
    }

    /**
     *  @brief Points @p lease at the newest whole stamp, giving back whatever it held first.
     *  @return The snapshot the lease now reads at.
     */
    generation_t take_snapshot(snapshot_lease_t &lease) noexcept {
        unique_lock<spin_shared_mutex_t> _ {mutex_};
        if (lease.clock_) unlink_(lease);
        generation_t const snapshot = atomic_load(published_stamp_);
        lease.clock_ = this;
        lease.snapshot_ = snapshot;
        // The watermark never moves backwards, so the snapshot just drawn is at least as new as every
        // one already on the census and the newest end is where it belongs.
        link_newest_(lease);
        republish_mark_();
        return snapshot;
    }

    /** @brief Draws the stamp @p node publishes under, and records that it has not landed yet. */
    void begin_commit(commit_in_flight_t &node) noexcept {
        unique_lock<spin_shared_mutex_t> _ {mutex_};
        node.stamp_ = ++commits_;
        node.older_ = newest_in_flight_;
        node.newer_ = nullptr;
        (newest_in_flight_ ? newest_in_flight_->newer_ : oldest_in_flight_) = &node;
        newest_in_flight_ = &node;
    }

    /**
     *  @brief Records that every version of @p node is written, and moves the watermark as far as it may go.
     *    Which is one below the oldest commit still in flight, or all the way to the newest stamp drawn
     *    when none is: a commit is whole only once every commit before it is.
     */
    void end_commit(commit_in_flight_t &node) noexcept {
        unique_lock<spin_shared_mutex_t> _ {mutex_};
        (node.older_ ? node.older_->newer_ : oldest_in_flight_) = node.newer_;
        (node.newer_ ? node.newer_->older_ : newest_in_flight_) = node.older_;
        node.older_ = nullptr;
        node.newer_ = nullptr;
        generation_t const whole = oldest_in_flight_ ? oldest_in_flight_->stamp_ - 1 : commits_;
        atomic_store<generation_t>(published_stamp_, whole);
        // A store wakes nobody, so anyone parked in `await_published` sleeps through the very
        // publication it waits for unless the wake goes out here.
        atomic_notify_all(published_stamp_);
        // With nobody reading, the mark is the watermark, so publishing one moves the other.
        republish_mark_();
    }

    /**
     *  @brief Waits until the watermark covers @p stamp, so the committer may read what it just wrote.
     *  @warning Blocks on another thread's publication, so no store lock may be held across it.
     */
    void await_published(commit_stamp_t stamp) noexcept {
        generation_t const wanted = static_cast<generation_t>(stamp);
        // Parked rather than spun: this waits on another thread's commit, which is unbounded, and a
        // spinner would hold a core for the whole of it.
        for (generation_t seen = atomic_load(published_stamp_); seen < wanted; seen = atomic_load(published_stamp_))
            atomic_wait(published_stamp_, seen);
    }

    /**
     *  @brief The oldest snapshot any reader can still name, so everything older is unreachable.
     *    Every arrival and departure republishes it from the census head, so an early reader leaving
     *    while a later one stays open moves it up to that later one rather than leaving it behind.
     */
    [[nodiscard]] generation_t low_water_mark() const noexcept {
        // Read without the lock: the mark is only written under it and never falls, so a stale read is
        // an older mark - keeping versions nobody needs rather than freeing one somebody still names.
        return atomic_load(low_water_mark_);
    }

    /** @brief How many readers currently hold a snapshot, counted off the census. */
    [[nodiscard]] std::size_t open_snapshots() const noexcept {
        shared_lock<spin_shared_mutex_t> _ {mutex_};
        std::size_t counted = 0;
        for (snapshot_lease_t const *lease = oldest_lease_; lease; lease = lease->newer_) ++counted;
        return counted;
    }

    /** @brief Takes over @p other's stamps, for a store that is being moved and has nothing open. */
    void adopt(snapshot_clock_t const &other) noexcept {
        assert(!oldest_lease_ && !other.oldest_lease_ && "a lease names the clock it was drawn from");
        // `generation_` moves atomically because `next_generation` increments it that way; `commits_`
        // is guarded by `mutex_`.
        atomic_store<generation_t>(generation_, atomic_load(other.generation_));
        commits_ = other.commits_;
        atomic_store<generation_t>(published_stamp_, atomic_load(other.published_stamp_));
        atomic_store<generation_t>(low_water_mark_, atomic_load(other.low_water_mark_));
    }
};

#pragma endregion Snapshot Clock

/**
 *  @brief  Transactional store answering every read at one snapshot, over any key-addressable core.
 *
 *  @p isolation_ picks the rung: @c snapshot_k, @c serializable_k or @c strict_serializable_k. Not
 *  thread-safe by itself. Entirely exception-free, with all methods marked @c noexcept.
 *
 *  @section snapshot_store_design_goals Design Goals
 *
 *  A transaction fixes one commit stamp when it opens and answers every read at it, so a value read
 *  twice reads the same both times and a range walked twice admits no key that was not there the
 *  first time. Writes are checked first-committer-wins: a transaction is refused if anything committed
 *  to a key it wrote or watched after its snapshot was taken. At @c snapshot_k only a watched key is
 *  re-checked; from @c serializable_k up every read and every window read is re-checked too.
 *
 *  @c monotonic_store keeps one published version per key and promises only Monotonic Atomic
 *  View. This store keeps every version a live snapshot can still reach, which is what buys the
 *  stronger promise and what the reclamation machinery below has to give back.
 *
 *  @see https://jepsen.io/consistency/models/snapshot-isolation
 *  @see https://jepsen.io/consistency/models/serializable
 *  @see https://jepsen.io/consistency/models/strict-serializable
 *
 *  @section snapshot_store_representation Representation
 *
 *  One entry per @c (key,generation) pair, uniformly for ordered and unordered cores. An ordered core
 *  needs no new comparator - @c versioned_comparator_t already breaks a tie on the generation, so a
 *  key's versions occupy one contiguous run. An unordered core rebinds on @c per_version_equals, which
 *  widens equality to the pair while the hasher keeps peeling to the bare key, so a key's versions
 *  share one probe run. Neither shape carries a chain, so visibility, rollback and reclamation each
 *  have exactly one implementation.
 *
 *  A version's @c generation is half of its key and is never rewritten after insertion: re-keying
 *  would relocate a tree node and could overflow a table, and a commit that can fail is not a commit.
 *  Only @c committed is ever written in place.
 *
 *  @section snapshot_store_reclamation Reclamation
 *
 *  Versions are pruned on write against a low-water mark, and an explicit @c vacuum sweeps the rest.
 *  There is no background thread, no epoch registry and no hidden global state: registering and
 *  retiring a snapshot is linking a node the caller already owns, so a destructor may do it. The mark
 *  is the oldest snapshot on the clock's census, which cannot pass a snapshot somebody still holds and
 *  follows the oldest reader up as readers leave. With nothing open the mark equals the published
 *  stamp, so a key falls back to a single entry on the next commit that touches it.
 *
 *  @section snapshot_store_ranking Ranking
 *
 *  A core that sums a second per-subtree count carries @c select and @c rank, descending on a survivor
 *  tag written into each entry beside its stamp. Those two answer at the newest published stamp and only
 *  there: the tag says "survives now", and one scalar per node cannot also answer for a reader holding an
 *  older snapshot, which is why a transaction's own @c select walks its merged order instead.
 *
 *  @tparam collection_type_ The underlying core, satisfying @c key_addressable_collection and providing
 *    a @c rebind template alias. Cores that also satisfy @c ordered_collection unlock bounds and ranges.
 */
template <typename collection_type_, isolation_t isolation_ = isolation_t::snapshot_k>
class snapshot_store {
    static_assert(isolation_ == isolation_t::snapshot_k || isolation_ == isolation_t::serializable_k ||
                      isolation_ == isolation_t::strict_serializable_k,
                  "this engine answers at a snapshot; the choice is whether reads are validated and waited for");

  public:
#pragma region Type Definitions

    using value_t = typename owned_value_of<collection_type_>::type;
    using value_type = value_t; // ? STL style

    using key_t = typename collection_type_::key_type;
    using key_type = key_t; // ? STL style

    using mapped_t = typename collection_type_::mapped_type;
    using mapped_type = mapped_t; // ? STL style

    using is_associative = typename collection_type_::is_associative;
    using is_transactional = std::true_type;

    /** @brief Every read of a transaction is answered at the stamp the transaction opened on. */
    static constexpr isolation_t isolation_k = isolation_;

    /**
     *  @brief Whether a write made outside a transaction waits for its own publication before returning.
     *
     *  A transaction's own commit waits at every level, so this gates the direct write paths alone.
     *  Without the wait such a write can return while an older commit still writing itself out holds
     *  the watermark below the stamp just drawn, so a transaction opening afterwards reads at a
     *  snapshot that predates it - serializable, but not in real time.
     */
    static constexpr bool awaits_publication_k = isolation_ == isolation_t::strict_serializable_k;

    /** @brief Where stamps and snapshots come from, which a shard set shares across its partitions. */
    using clock_t = snapshot_clock_t;

    using allocator_t = typename collection_type_::allocator_type;

    /** @brief What this core family rebinds on, and how a decorated entry is reached inside it. */
    using storage_shape_t = versioned_storage_for<collection_type_, value_t>;

    using comparator_t = typename storage_shape_t::addressing_source_t;
    using versioning_t = typename storage_shape_t::versioning_t;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using accessed_identifier_t = typename versioning_t::accessed_identifier_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;

    /** @brief Whether the core sums a second per-subtree count, which is what unlocks @c select and @c rank. */
    static constexpr bool ranked_core_k = ranked_by_liveness<collection_type_>;

    /**
     *  @brief One stored version, decorated with a survivor tag only where a core descends on it.
     *    An unranked core keeps the bare entry, so nothing pays for a bit nothing reads.
     */
    using versioned_t = std::conditional_t<ranked_core_k, live_tagged<typename versioning_t::versioned_t>,
                                           typename versioning_t::versioned_t>;
    using versioned_entry_t = versioned_t;

    /** @brief What this store calls itself, so generic code spells an engine and a wrapper alike. */
    using store_t = snapshot_store;

  private:
    /** @brief Names one stored version without owning its key, so a probe costs no copy. */
    using dated_reference_t = dated_identifier<identifier_t const &>;

    /** @brief Every version of every key, one entry per @c (key,generation) pair. */
    using dated_entries_t = typename storage_shape_t::template rebind_dated<versioned_t>;

    /** @brief One transaction's pending writes, at most one per key since all share its generation. */
    using changes_t = typename storage_shape_t::template rebind<versioned_t>;

    using accesses_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<accessed_identifier_t>;
    using accesses_vector_t = basic_vector<accessed_identifier_t, accesses_allocator_t>;

    using changed_identifiers_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_identifiers_vector_t = basic_vector<identifier_t, changed_identifiers_allocator_t>;

    using values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_t>;
    using values_vector_t = basic_vector<value_t, values_allocator_t>;

    /** @brief Whether the core orders its keys, which is what unlocks bounds and ranges. */
    static constexpr bool ordered_core_k = ordered_collection<dated_entries_t>;

    /** @brief How the store names a position inside its versioned core, never handed out. */
    using entry_iterator_t = decltype(std::declval<dated_entries_t const &>().begin());

  public:
#pragma region Visible Cursor

    /**
     *  @brief A resumable walk over the keys one snapshot reads, yielding each key exactly once.
     *
     *  A key's versions occupy one contiguous run ordered by generation rather than by stamp, so the
     *  version a reader sees is whichever of them carries the greatest visible stamp - which no binary
     *  search inside the run can find. The cursor therefore stands on one whole run at a time and steps
     *  to the next, and that is what a callback loop cannot be paused between.
     *
     *  Only ordered cores have a successor to resume from, so the open-addressed core does not offer
     *  this: its probe order is a function of the table's capacity, which any write may change.
     *
     *  @note Every read is delivered through @c peek, so nothing addressing the store escapes. Any
     *    write to the store abandons the cursor.
     */
    class visible_cursor_t {
        static_assert(ordered_core_k, "an unordered core supplies no successor to resume from");

        friend store_t;

        store_t const *store_ {nullptr};
        /** @brief First entry of the run the cursor stands on, @c end() once the walk is over. */
        entry_iterator_t position_;
        /** @brief One past the last entry of that run. */
        entry_iterator_t run_end_;
        /** @brief The version @c snapshot_ reads in that run, null when it reads nothing there. */
        versioned_t const *standing_ {nullptr};
        /** @brief The stamp every key this cursor reports is resolved at. */
        generation_t snapshot_ {0};

        visible_cursor_t(store_t const &store, entry_iterator_t position, generation_t snapshot) noexcept
            : store_(&store), position_(position), run_end_(position), snapshot_(snapshot) {
            settle_();
        }

        /** @brief Delimits the run @c position_ opens and picks the version @c snapshot_ reads in it. */
        void settle_() noexcept {
            auto const finish = store_->entries_.end();
            standing_ = nullptr;
            run_end_ = position_;
            if (position_ == finish) return;

            versioned_t const &head = *position_;
            versioned_t const *newest = nullptr;
            while (run_end_ != finish && store_->same_key_(*run_end_, head)) {
                versioned_t const &version = *run_end_;
                if (visible_at(version.committed, snapshot_) &&
                    (!newest || stamp_of(newest->committed) < stamp_of(version.committed)))
                    newest = &version;
                ++run_end_;
            }
            if (newest && newest->presence == presence_t::present_k) standing_ = newest;
        }

        /** @brief Steps to the next run, whatever that run turns out to hold. */
        void step_() noexcept {
            position_ = run_end_;
            settle_();
        }

        /** @brief Steps until the cursor stands on a readable key or the walk is over. */
        void seek_readable_() noexcept {
            while (!exhausted() && !standing_) step_();
        }

      public:
        /** @brief The stamp every key this cursor reports is resolved at. */
        [[nodiscard]] generation_t snapshot() const noexcept { return snapshot_; }

        /** @brief Whether the walk is over, which is when nothing more is readable. */
        [[nodiscard]] bool exhausted() const noexcept { return position_ == store_->entries_.end(); }

        /**
         *  @brief Hands @p callback_found the key the cursor stands on, or reports the walk is over.
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered once nothing is left. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        void peek(callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");
            if (standing_) callback_found(standing_->payload);
            else callback_missing();
        }

        /** @brief Copies out the key the cursor stands on, or reports that the walk is over. */
        [[nodiscard]] expected<value_t> peek_copy() const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            peek([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /** @brief Steps to the next key this snapshot reads, strictly greater than the current one. */
        void advance() noexcept {
            step_();
            seek_readable_();
        }
    };

#pragma endregion Visible Cursor

    class transaction_t {
        friend store_t;
        store_t *store_ {nullptr};
        changes_t changes_;
        /** @brief Every key this transaction read, and the ends of every window it walked. */
        mutable accesses_vector_t accesses_ {};
        /**
         *  @brief Whether every read this transaction made named a key, and if not, why not.
         *
         *  A validator can only prove a read is untouched if it knows which key was read. Two things
         *  leave it unable to: a walk of the whole keyspace names no key by design, and a read whose
         *  key could not be copied names none by accident. Both refuse the commit, and a reader of a
         *  refusal needs to know which happened.
         */
        enum class read_set_t : std::uint8_t {
            names_every_key_k,
            covers_everything_k,
            unrecorded_k,
        };

        mutable read_set_t read_set_ {read_set_t::names_every_key_k};
        changed_identifiers_vector_t changed_identifiers_ {};
        generation_t generation_ {0};
        /**
         *  @brief This transaction's own claim on its snapshot, empty when a shard set holds one for it.
         *    A partition of a sharded transaction reads at a snapshot somebody else is answering for,
         *    so it registers nothing and retires nothing.
         */
        snapshot_clock_t::snapshot_lease_t lease_ {};
        /** @brief The stamp every read of this transaction is answered at. */
        generation_t snapshot_ {0};
        staging_t staging_ {staging_t::pending_k};

        transaction_t(store_t &store) noexcept
            : store_(&store), changes_(store_t::build_changes_(store.entries_)),
              accesses_(accesses_allocator_t(storage_shape_t::allocator_of(store.entries_))),
              changed_identifiers_(changed_identifiers_allocator_t(storage_shape_t::allocator_of(store.entries_))),
              generation_(store.next_generation_()), snapshot_(store.clock_->take_snapshot(lease_)) {}

        /**
         *  @brief Opens on a snapshot and a generation somebody else drew, for one partition of a
         *    sharded transaction whose other partitions must answer at the very same stamp.
         */
        transaction_t(store_t &store, generation_t snapshot, generation_t generation) noexcept
            : store_(&store), changes_(store_t::build_changes_(store.entries_)),
              accesses_(accesses_allocator_t(storage_shape_t::allocator_of(store.entries_))),
              changed_identifiers_(changed_identifiers_allocator_t(storage_shape_t::allocator_of(store.entries_))),
              generation_(generation), snapshot_(snapshot) {}

        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

        /** @brief Drops this transaction's version of the first @p processed changed identifiers. */
        void unstage_(std::size_t processed) noexcept {
            auto &store = store_ref();
            for (std::size_t index = 0; index != processed; ++index)
                // ! Don't materialize a new copy of the identifier here, use a reference
                [[maybe_unused]]
                bool const dropped = store.drop_version_(changed_identifiers_[index], generation_);
        }

        /**
         *  @brief Records that this transaction read @p comparable, where the level says reads are validated.
         *
         *  Answers where the read happened as well as latching, because the caller may discard the status
         *  and the commit still has to refuse rather than pass a validation it could not perform. The
         *  latch is the safety net; the return is what lets a caller retry the read it just lost.
         */
        template <typename comparable_type_>
        [[nodiscard]] status_t record_read_(comparable_type_ const &comparable) const noexcept {
            if constexpr (!at_least(isolation_k, isolation_t::serializable_k)) return success_k;
            else {
                auto owned = copy_safely<identifier_t>(identifier_t {comparable});
                if (!owned) {
                    read_set_ = read_set_t::unrecorded_k;
                    return owned.status();
                }
                status_t const recorded = accesses_.push_back({std::move(*owned), access_t::read_k});
                if (failed(recorded)) read_set_ = read_set_t::unrecorded_k;
                return recorded;
            }
        }

        /**
         *  @brief Records that this transaction read the window between @p lower and @p upper.
         *
         *  Two entries pushed next to each other, because adjacency is what pairs them. @p ends names
         *  which side, if either, runs off the end of the keyspace - an ordinal read depends on every
         *  key before the one it lands on, and there is no smallest key to name - in which case the
         *  identifier stored on that side is kept only so the pair stays a pair.
         *
         *  @return Success, or the reason the window could not be written down, which latches as well.
         */
        template <typename lower_type_, typename upper_type_>
        [[nodiscard]] status_t record_window_read_(lower_type_ const &lower, upper_type_ const &upper,
                                                   access_t ends) const noexcept {
            if constexpr (!at_least(isolation_k, isolation_t::serializable_k)) return success_k;
            else {
                auto lower_copy = copy_safely<identifier_t>(identifier_t {lower});
                if (!lower_copy) {
                    read_set_ = read_set_t::unrecorded_k;
                    return lower_copy.status();
                }
                auto upper_copy = copy_safely<identifier_t>(identifier_t {upper});
                if (!upper_copy) {
                    read_set_ = read_set_t::unrecorded_k;
                    return upper_copy.status();
                }
                if (status_t const reserved = accesses_.reserve(accesses_.size() + 2); failed(reserved)) {
                    read_set_ = read_set_t::unrecorded_k;
                    return reserved;
                }
                access_t const opens = holds(ends, access_t::from_the_lowest_k)
                                           ? access_t::opens_k | access_t::from_the_lowest_k
                                           : access_t::opens_k;
                access_t const closes = holds(ends, access_t::to_the_highest_k)
                                            ? access_t::closes_k | access_t::to_the_highest_k
                                            : access_t::closes_k;
                [[maybe_unused]] status_t const opened =
                    accesses_.push_back(assume_reserved, {std::move(*lower_copy), opens});
                [[maybe_unused]] status_t const closed =
                    accesses_.push_back(assume_reserved, {std::move(*upper_copy), closes});
                return success_k;
            }
        }

        /**
         *  @brief Records that this transaction read every key, which no pair of bounds can name.
         *    Cannot fail - it names nothing, so there is nothing to allocate - and answers only so that
         *    every recorder reads the same at its call sites.
         */
        [[nodiscard]] status_t record_whole_keyspace_read_() const noexcept {
            if constexpr (at_least(isolation_k, isolation_t::serializable_k))
                read_set_ = read_set_t::covers_everything_k;
            return success_k;
        }

        /**
         *  @brief Whether every key this transaction wrote, and every read it recorded, is untouched
         *    since its snapshot. At @c snapshot_k the recorded reads are the explicit watches and
         *    nothing else; from @c serializable_k up they are every read and every window read.
         *    Asked twice - once when staging, once when publishing - because a commit landing in between
         *    is the only thing that can invalidate a read after it was validated.
         *
         *  Dated against the snapshot rather than matched against the version the read saw, and the same
         *  predicate a written key is checked with. Comparing versions would let a key be inserted and
         *  erased again under a read of its absence: both ends resolve to missing, so the two match
         *  while two commits the transaction never saw sit between them.
         */
        [[nodiscard]] status_t validate_accesses_() const noexcept {
            auto const &store = store_ref();

            // Nothing has been drawn above this transaction's snapshot, so nothing can have moved
            // under it. Compared against the drawn stamp and not the watermark: a commit in flight
            // has already written its stamp onto its versions while the watermark still lags.
            if (store.drawn_stamp_() == snapshot_) return success_k;

            // A read this transaction cannot name cannot be proven untouched, so any commit conflicts.
            // A walk of the whole keyspace is a window nobody can spell, which is the phantom case.
            if (read_set_ == read_set_t::covers_everything_k) return status_t::phantom_conflict_k;
            if (read_set_ == read_set_t::unrecorded_k) return status_t::read_conflict_k;

            // First-committer-wins: an earlier commit keeps the key and the later transaction is
            // turned away rather than overwriting a version it never read.
            for (identifier_t const &identifier : changed_identifiers_)
                if (store.key_changed_since_(identifier, snapshot_)) return status_t::write_conflict_k;

            for (std::size_t index = 0; index != accesses_.size(); ++index) {
                access_t const access = accesses_[index].access;
                if (holds(access, access_t::read_k) && store.key_changed_since_(accesses_[index].identifier, snapshot_))
                    return status_t::read_conflict_k;

                if constexpr (ordered_core_k)
                    if (holds(access, access_t::opens_k)) {
                        assert(index + 1 != accesses_.size() &&
                               holds(accesses_[index + 1].access, access_t::closes_k) &&
                               "a range endpoint is closed by the entry pushed straight after it");
                        access_t const ends = access | accesses_[index + 1].access;
                        if (store.range_changed_since_(accesses_[index].identifier, accesses_[index + 1].identifier,
                                                       snapshot_, ends))
                            return status_t::phantom_conflict_k;
                        ++index;
                    }
            }
            return success_k;
        }

        /**
         *  @brief Drops anything staged and never published, and gives up this transaction's snapshot.
         *
         *  Retiring the snapshot is what lets the low-water mark move again, so an abandoned
         *  transaction cannot pin a version tail forever. A null @c store_ marks a moved-from
         *  transaction, which owns nothing and must undo nothing.
         */
        void unwind_() noexcept {
            if (!store_) return;
            if (staging_ == staging_t::staged_k) unstage_(changed_identifiers_.size());
            lease_.retire();
            staging_ = staging_t::pending_k;
            store_ = nullptr;
        }

        /**
         *  @brief Stages @p versioned under this transaction's generation, recording @p identifier as changed.
         *    Refuses once the transaction is staged: staging reserved and validated exactly the changes
         *    it found, so a later write would publish behind that check or be dropped by @c commit.
         */
        [[nodiscard]] status_t stage_(identifier_t &&identifier, versioned_t &&versioned) noexcept {
            if (staging_ == staging_t::staged_k) return operation_not_permitted_k;
            auto reserve_status = changed_identifiers_.reserve(changed_identifiers_.size() + 1);
            if (failed(reserve_status)) return reserve_status;
            versioned.generation = generation_;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (failed(result)) return out_of_memory_heap_k;
            [[maybe_unused]] status_t const recorded =
                changed_identifiers_.push_back(assume_reserved, std::move(identifier));
            return success_k;
        }

      public:
        /**
         *  @brief Takes over @p other entirely, leaving it owning nothing.
         *    Written out rather than defaulted: a defaulted move would leave both objects pointing at
         *    the store, so the destructor would retire one snapshot twice.
         */
        transaction_t(transaction_t &&other) noexcept
            : store_(std::exchange(other.store_, nullptr)), changes_(std::move(other.changes_)),
              accesses_(std::move(other.accesses_)), changed_identifiers_(std::move(other.changed_identifiers_)),
              generation_(other.generation_), lease_(std::move(other.lease_)), snapshot_(other.snapshot_),
              staging_(std::exchange(other.staging_, staging_t::pending_k)) {}

        transaction_t &operator=(transaction_t &&other) noexcept {
            if (this == &other) return *this;
            unwind_();
            store_ = std::exchange(other.store_, nullptr);
            changes_ = std::move(other.changes_);
            accesses_ = std::move(other.accesses_);
            changed_identifiers_ = std::move(other.changed_identifiers_);
            generation_ = other.generation_;
            lease_ = std::move(other.lease_);
            snapshot_ = other.snapshot_;
            staging_ = std::exchange(other.staging_, staging_t::pending_k);
            return *this;
        }

        ~transaction_t() noexcept { unwind_(); }

        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;

        /** @brief The generation this transaction stamps its staged versions with. */
        [[nodiscard]] generation_t generation() const noexcept { return generation_; }

        /** @brief The commit stamp every read of this transaction is answered at. */
        [[nodiscard]] generation_t snapshot() const noexcept { return snapshot_; }

        /** @brief Whether anything is pending, which after @c stage means nothing rather than nothing written. */
        [[nodiscard]] bool has_changes() const noexcept { return changes_.size() != 0; }

        /** @brief How many keys carry a pending write. */
        [[nodiscard]] std::size_t changes_count() const noexcept { return changes_.size(); }

#pragma region Transaction Modifiers

        /**
         *  @brief Stages an insert, failing when the key is already there for this transaction.
         *  @param[in] value Element to insert, moved into the transaction.
         *  @return Success, @c key_already_exists_k, or an allocation failure.
         */
        [[nodiscard]] status_t insert(value_t &&value) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(value));
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return key_already_exists_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an insert only when the key is absent, treating an existing key as a no-op.
         *  @param[in] value Element to insert, moved into the transaction.
         *  @return Success unless an allocation failed.
         */
        [[nodiscard]] status_t insert_if_missing(value_t &&value) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(value));
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return success_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages a write of @p value, whether or not the key is there.
         *  @param[in] value Element to write, moved into the transaction.
         *  @return Success unless an allocation failed.
         */
        [[nodiscard]] status_t upsert(value_t &&value) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_identifier) return out_of_memory_heap_k;
            versioned_t versioned(std::move(value));
            versioned.presence = presence_t::present_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        /**
         *  @brief Stages a write of @p value, failing when the key is not there for this transaction.
         *  @param[in] value Element to write, moved into the transaction.
         *  @return Success, @c key_not_found_k, or an allocation failure.
         */
        [[nodiscard]] status_t update(value_t &&value) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(value));
            if (!key_is_present) return key_is_present.status();
            if (!*key_is_present) return key_not_found_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an erase of @p identifier as a tombstone, which only a commit turns into an absence.
         *  @return @c key_not_found_k when this transaction reads no such key, so a caller need not look first.
         *
         *  The look happens whether or not callbacks were passed, so the answer never depends on how the
         *  call was spelled. At @c serializable_k it records a read, which a blind delete did not.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t erase(identifier_t const &identifier, callback_found_type_ &&callback_found = {},
                                     callback_missing_type_ &&callback_missing = {}) noexcept {
            bool present = false;
            status_t const looked_up = find(
                identifier,
                [&](value_t const &element) noexcept {
                    present = true;
                    callback_found(element);
                },
                no_op_t {});
            if (failed(looked_up)) return looked_up;
            if (!present) {
                callback_missing();
                return key_not_found_k;
            }

            // The tombstone owns its own identifier and the changed list owns another, so a move-only
            // key needs two safe copies rather than one copy and one implicit one.
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) return out_of_memory_heap_k;
            auto maybe_payload = copy_safely<identifier_t>(identifier);
            if (!maybe_payload) return out_of_memory_heap_k;

            versioned_t versioned(value_t {std::move(*maybe_payload)});
            versioned.presence = presence_t::erased_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        /**
         *  @brief Sizes the access list up front, so a later read or watch has room already.
         *
         *  Under @c serializable_k a read records itself, so this bounds nothing on its own - it is a
         *  hint sized to what the caller expects to touch, not a promise that recording cannot fail.
         */
        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return accesses_.reserve(size); }

#pragma endregion Transaction Modifiers

#pragma region Transaction Lookup

        /**
         *  @brief Records what this transaction's snapshot resolves @p identifier to, so a later commit can
         *    refuse if anything published over it in the meantime.
         *  @param[in] identifier Identifier to watch, borrowed and copied into the read set - the read
         *    set outlives the call, and a caller's identifier is never consumed by a read.
         *  @return Success unless the read set could not grow, or the identifier could not be copied.
         *
         *  Latches as well as answering, for the reason @c record_read_ gives: a caller may fold the
         *  status into a first-failure and carry on, and the commit still has to refuse rather than pass
         *  a validation it could not perform. At @c snapshot_k this is the only read validation there is.
         */
        [[nodiscard]] status_t watch(identifier_t const &identifier) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) {
                read_set_ = read_set_t::unrecorded_k;
                return maybe_identifier.status();
            }
            status_t const recorded = accesses_.push_back({std::move(*maybe_identifier), access_t::read_k});
            if (failed(recorded)) read_set_ = read_set_t::unrecorded_k;
            return recorded;
        }

        /** @brief Records @p versioned as the version this transaction read of its own key. */
        [[nodiscard]] status_t watch(versioned_t const &versioned) noexcept {
            // Through `copy_safely` rather than a braced `identifier_t`, so a move-only identifier
            // compiles here and an identifier that allocates reports instead of throwing.
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(versioned.payload));
            if (!maybe_identifier) {
                read_set_ = read_set_t::unrecorded_k;
                return maybe_identifier.status();
            }
            status_t const recorded = accesses_.push_back({std::move(*maybe_identifier), access_t::read_k});
            if (failed(recorded)) read_set_ = read_set_t::unrecorded_k;
            return recorded;
        }

        /**
         *  @brief Finds a member equal to @p comparable and records the read at every level.
         *    What @c find does on its own only from @c serializable_k up, so this is the call a
         *    @c snapshot_k transaction needs; it can fail, because a read set is memory.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find_and_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                              callback_missing_type_ &&callback_missing = {}) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(comparable);
            if (!maybe_identifier) return maybe_identifier.status();
            status_t const recorded = watch(*maybe_identifier);
            if (failed(recorded)) return recorded;
            return find(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                        std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Finds a member @b equal to @p comparable, this transaction's own writes included.
         *    From @c serializable_k up the read records itself; at @c snapshot_k it does not, so a
         *    caller wanting it validated at commit calls @c watch or @c find_and_watch.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            status_t const recorded = record_read_(comparable);
            if (auto iterator = changes_.find(comparable); iterator != changes_.end()) {
                if ((*iterator).presence == presence_t::present_k) callback_found((*iterator).payload);
                else callback_missing();
                return recorded;
            }
            store_ref().find_at_(std::forward<comparable_type_>(comparable), snapshot_,
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
            return recorded;
        }

        /** @brief Copies out the member equal to @p comparable, this transaction's own writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            status_t const looked_up = find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            if (failed(looked_up)) return looked_up;
            return result;
        }

        /** @brief Whether a member equal to @p comparable exists, this transaction's own writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            bool present = false;
            status_t const answered = find(
                std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; },
                no_op_t {});
            if (failed(answered)) return answered;
            return present;
        }

        /**
         *  @brief Finds the first member @b greater or equal to @p comparable at this snapshot.
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_core_k
        {
            bool landed = false;
            status_t recorded = success_k;
            bounded_(
                changes_.lower_bound(comparable), store_ref().visible_at_(comparable, snapshot_),
                [&](value_t const &value) noexcept {
                    landed = true;
                    identifier_t const &key = mapping_key_or_itself<value_t>(value);
                    recorded = first_failure(recorded, record_window_read_(comparable, key, access_t::none_k));
                    recorded = first_failure(recorded, record_read_(key));
                    callback_found(value);
                },
                [&]() noexcept { callback_missing(); });
            // Nothing at or above the bound, so the read depended on everything above it.
            if (!landed)
                recorded =
                    first_failure(recorded, record_window_read_(comparable, comparable, access_t::to_the_highest_k));
            return recorded;
        }

        /**
         *  @brief Hands @p callback_found the smallest member this snapshot reads, staged writes included.
         *
         *  The unbounded case of @c lower_bound, which is what a merged walk over several stores needs to
         *  open with: it asks for a first key rather than an ordinal, so a core keeping no subtree counts
         *  can answer it.
         *
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when nothing is readable. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                        callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_core_k
        {
            bool landed = false;
            status_t recorded = success_k;
            bounded_(
                changes_.begin(), store_ref().visible_from_(store_ref().entries_.begin(), snapshot_),
                [&](value_t const &value) noexcept {
                    landed = true;
                    identifier_t const &key = mapping_key_or_itself<value_t>(value);
                    recorded = first_failure(recorded, record_window_read_(key, key, access_t::from_the_lowest_k));
                    recorded = first_failure(recorded, record_read_(key));
                    callback_found(value);
                },
                [&]() noexcept { callback_missing(); });
            // Nothing readable at all, so the answer rests on the whole keyspace being empty.
            if (!landed) recorded = first_failure(recorded, record_whole_keyspace_read_());
            return recorded;
        }

        /** @brief Finds the first member @b strictly greater than @p comparable at this snapshot. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_core_k
        {
            bool landed = false;
            status_t recorded = success_k;
            bounded_(
                changes_.upper_bound(comparable),
                store_ref().visible_from_(store_ref().entries_.upper_bound(comparable), snapshot_),
                [&](value_t const &value) noexcept {
                    landed = true;
                    identifier_t const &key = mapping_key_or_itself<value_t>(value);
                    recorded = first_failure(recorded, record_window_read_(comparable, key, access_t::none_k));
                    recorded = first_failure(recorded, record_read_(key));
                    callback_found(value);
                },
                [&]() noexcept { callback_missing(); });
            if (!landed)
                recorded =
                    first_failure(recorded, record_window_read_(comparable, comparable, access_t::to_the_highest_k));
            return recorded;
        }

        /**
         *  @brief Walks [ @p lower, @p upper ) at this snapshot, this transaction's own writes included.
         *
         *  Repeating the walk inside one transaction yields the same keys, whatever committed between.
         *  The two sides are merged rather than concatenated, so the output is sorted even where a
         *  staged key falls between two committed ones, and a staged key masks the committed version
         *  of itself.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            status_t const recorded = record_window_read_(lower, upper, access_t::none_k);
            auto const ordering = changes_.key_comp();
            merge_from_(std::forward<lower_type_>(lower), [&](versioned_t const &version) noexcept {
                if (!ordering.per_key_compare(version, upper)) return probe_control_t::halt_k;
                if (version.presence == presence_t::present_k) callback(version.payload);
                return probe_control_t::resume_k;
            });
            return recorded;
        }

        /**
         *  @brief Hands @p callback every member at or after @p lower, with no upper end.
         *    Records a window running to the highest key, so a commit into it is a phantom.
         */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range_from(lower_type_ &&lower, callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            status_t const recorded = record_window_read_(lower, lower, access_t::to_the_highest_k);
            merge_from_(std::forward<lower_type_>(lower), [&](versioned_t const &version) noexcept {
                if (version.presence == presence_t::present_k) callback(version.payload);
                return probe_control_t::resume_k;
            });
            return recorded;
        }

        /**
         *  @brief Hands @p callback every member before @p upper, @p upper excluded, with no lower end.
         *    Records a window running from the lowest key, so a caller never has to name a layout's floor
         *    to say "everything below this".
         */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range_up_to(upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            status_t const recorded = record_window_read_(upper, upper, access_t::from_the_lowest_k);
            auto const ordering = changes_.key_comp();
            merge_all_([&](versioned_t const &version) noexcept {
                if (!ordering.per_key_compare(version, upper)) return probe_control_t::halt_k;
                if (version.presence == presence_t::present_k) callback(version.payload);
                return probe_control_t::resume_k;
            });
            return recorded;
        }

        /**
         *  @brief Hands @p callback every key this transaction reads, in whatever order the core holds them.
         *
         *  The one walk an unordered core can offer, so it promises no ordering even where the core has
         *  one. Every key a @c find of this transaction would answer with at the moment of the call is
         *  visited @b exactly @b once - its own staged writes included, its own tombstones and every version
         *  its snapshot cannot read excluded. Nothing may write to the transaction or its store while the
         *  walk runs.
         *
         *  @param[in] callback Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         */
        template <typename callback_type_ = no_op_t>
        [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
            static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                          "callback must be noexcept invocable with value_t const &");

            status_t const recorded = record_whole_keyspace_read_();
            if constexpr (ordered_core_k) {
                merge_all_([&](versioned_t const &version) noexcept {
                    if (version.presence == presence_t::present_k) callback(version.payload);
                    return probe_control_t::resume_k;
                });
            }
            else {
                for (auto staged = changes_.begin(); staged != changes_.end(); ++staged)
                    if ((*staged).presence == presence_t::present_k) callback((*staged).payload);

                // A key this transaction wrote is answered from its own version above, so the committed
                // side skips whatever `changes_` already speaks for and never emits a key twice.
                store_ref().for_each_at_(snapshot_, [&](value_t const &value) noexcept {
                    if (changes_.find(mapping_key_or_itself<value_t>(value)) == changes_.end()) callback(value);
                });
            }
            return recorded;
        }

        /**
         *  @brief Finds every member equal to @p comparable, which for a unique-key store is one or none.
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback Callback receiving each match. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
            return find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), no_op_t {});
        }

        /**
         *  @brief Hands @p callback_found the @p ordinal -th smallest key this transaction can see.
         *    Linear in the keys it walks past, since the staged writes are sorted into the committed
         *    ones rather than counted anywhere.
         *
         *  @param[in] ordinal Zero-based position among the keys this transaction reads.
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when fewer keys are readable. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                      callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_core_k
        {
            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            bool found = false;
            status_t recorded = success_k;
            merge_all_([&](versioned_t const &version) noexcept {
                if (version.presence != presence_t::present_k) return probe_control_t::resume_k;
                if (ordinal != 0) {
                    --ordinal;
                    return probe_control_t::resume_k;
                }
                identifier_t const &key = mapping_key_or_itself<value_t>(version.payload);
                // An ordinal is decided by how many keys precede it, so the window is everything
                // below the one it lands on - a key appearing above cannot move it.
                recorded = first_failure(recorded, record_window_read_(key, key, access_t::from_the_lowest_k));
                recorded = first_failure(recorded, record_read_(key));
                callback_found(version.payload);
                found = true;
                return probe_control_t::halt_k;
            });
            if (!found) {
                recorded = first_failure(recorded, record_whole_keyspace_read_());
                callback_missing();
            }
            return recorded;
        }

        /**
         *  @brief Hands @p callback_found how many keys this transaction reads before @p comparable.
         *    Linear in the keys before it, for the same reason @c select is.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive a @c std::size_t. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if the key is not readable. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_core_k
        {
            static_assert(is_safe_callback_for<callback_found_type_, std::size_t>,
                          "callback_found must be noexcept invocable with std::size_t");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            expected<bool> const key_is_present = contains(comparable);
            if (!key_is_present) return key_is_present.status();
            if (!*key_is_present) {
                callback_missing();
                return success_k;
            }
            status_t const recorded = record_window_read_(comparable, comparable, access_t::from_the_lowest_k);
            auto const ordering = changes_.key_comp();
            std::size_t counted = 0;
            merge_all_([&](versioned_t const &version) noexcept {
                if (!ordering.per_key_compare(version, comparable)) return probe_control_t::halt_k;
                if (version.presence == presence_t::present_k) ++counted;
                return probe_control_t::resume_k;
            });
            callback_found(counted);
            return recorded;
        }

#pragma region Transaction Range Operations

        /** @brief How many members equal @p comparable, which for a unique key is nought or one. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
            expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
            if (!present) return present.status();
            return *present ? std::size_t {1} : std::size_t {0};
        }

        /** @brief Copies out the first member at or after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
            requires ordered_core_k
        {
            expected<value_t> result {status_t::key_not_found_k};
            status_t const answered = lower_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            if (failed(answered)) return answered;
            return result;
        }

        /** @brief Copies out the first member after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
            requires ordered_core_k
        {
            expected<value_t> result {status_t::key_not_found_k};
            status_t const answered = upper_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            if (failed(answered)) return answered;
            return result;
        }

        /**
         *  @brief Stages a tombstone for every member this transaction reads in [ @p lower, @p upper ).
         *
         *  The window is walked first and staged afterwards, because staging a tombstone changes what the
         *  merged walk answers and a walk revising itself would skip its own neighbours.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires ordered_core_k
        {
            return erase_walked_(
                [&](auto &&step) noexcept {
                    return range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), step);
                },
                std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member at or after @p lower, @p lower included. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept
            requires ordered_core_k
        {
            return erase_walked_(
                [&](auto &&step) noexcept {
                    status_t const recorded = record_window_read_(lower, lower, access_t::to_the_highest_k);
                    merge_from_(lower, [&](versioned_t const &version) noexcept {
                        if (version.presence == presence_t::present_k) step(version.payload);
                        return probe_control_t::resume_k;
                    });
                    return recorded;
                },
                std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member before @p upper, @p upper excluded. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires ordered_core_k
        {
            return erase_walked_(
                [&](auto &&step) noexcept {
                    status_t const recorded = record_window_read_(upper, upper, access_t::from_the_lowest_k);
                    auto const ordering = changes_.key_comp();
                    merge_all_([&](versioned_t const &version) noexcept {
                        if (!ordering.per_key_compare(version, upper)) return probe_control_t::halt_k;
                        if (version.presence == presence_t::present_k) step(version.payload);
                        return probe_control_t::resume_k;
                    });
                    return recorded;
                },
                std::forward<callback_type_>(callback));
        }

        /**
         *  @brief Hands @p callback each member in [ @p lower, @p upper ) to revise, and stages the result.
         *
         *  Collected before revising for the same reason the erasing walks collect: a staged write moves
         *  what the merged walk answers underneath itself.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                            callback_type_ &&callback) noexcept
            requires is_mapping<value_t> && ordered_core_k
        {
            values_vector_t revised(values_allocator_t(storage_shape_t::allocator_of(store_ref().entries_)));
            status_t collecting = success_k;
            [[maybe_unused]] status_t const walked = range(
                std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), [&](value_t const &value) noexcept {
                    if (failed(collecting)) return;
                    auto duplicate = copy_safely(value);
                    if (!duplicate) {
                        collecting = duplicate.status();
                        return;
                    }
                    if (status_t const kept = revised.push_back(std::move(*duplicate)); failed(kept)) collecting = kept;
                });
            if (failed(collecting)) return collecting;

            for (std::size_t index = 0; index != revised.size(); ++index) {
                value_t &revision = revised[index];
                callback(revision.key, revision.mapped);
                if (status_t const staged = upsert(std::move(revision)); failed(staged)) return staged;
            }
            return success_k;
        }

        /** @brief Draws one member uniformly from [ @p lower, @p upper ), this transaction's writes included. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                          callback_type_ &&callback) const noexcept
            requires ordered_core_k
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
            requires ordered_core_k
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
      private:
        /**
         *  @brief Hands @p callback every version this transaction reads, in key order, from the two
         *    positions onward, a staged write masking the committed version of its own key.
         *
         *  Tombstones are handed over too: where a walk stops is a question about keys, so the caller
         *  filters on presence after it has decided whether the key is still in its window.
         */
        template <typename staged_iterator_type_, typename callback_type_>
        void merge_(staged_iterator_type_ staged, visible_cursor_t committed, callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            auto const ordering = changes_.key_comp();
            while (staged != changes_.end() || !committed.exhausted()) {
                if (committed.exhausted()) {
                    if (callback(*staged) == probe_control_t::halt_k) return;
                    ++staged;
                    continue;
                }
                versioned_t const &visible = *committed.standing_;
                if (staged == changes_.end()) {
                    if (callback(visible) == probe_control_t::halt_k) return;
                    committed.advance();
                    continue;
                }
                versioned_t const &pending = *staged;
                if (ordering.per_key_compare(visible, pending)) {
                    if (callback(visible) == probe_control_t::halt_k) return;
                    committed.advance();
                    continue;
                }
                // A staged write speaks for its key, so the committed version of it is dropped rather
                // than reported alongside - including when the staging is a tombstone.
                if (!ordering.per_key_compare(pending, visible)) committed.advance();
                if (callback(pending) == probe_control_t::halt_k) return;
                ++staged;
            }
        }

        /**
         *  @brief Walks a window, hands every member to @p callback, and stages a tombstone for each.
         *
         *  The walk finishes before the first tombstone is staged, because a staged write changes what
         *  the merged view answers and a walk revising itself would step over its own neighbours.
         */
        template <typename walk_type_, typename callback_type_>
        [[nodiscard]] status_t erase_walked_(walk_type_ &&walk, callback_type_ &&callback) noexcept
            requires ordered_core_k
        {
            changed_identifiers_vector_t doomed(
                changed_identifiers_allocator_t(storage_shape_t::allocator_of(store_ref().entries_)));
            status_t collecting = success_k;
            [[maybe_unused]] status_t const walked = walk([&](value_t const &value) noexcept {
                if (failed(collecting)) return;
                auto owned = copy_safely<identifier_t>(identifier_t {mapping_key_or_itself<value_t>(value)});
                if (!owned) {
                    collecting = owned.status();
                    return;
                }
                callback(value);
                if (status_t const kept = doomed.push_back(std::move(*owned)); failed(kept)) collecting = kept;
            });
            if (failed(collecting)) return collecting;

            for (std::size_t index = 0; index != doomed.size(); ++index)
                if (status_t const staged = erase(doomed[index]); failed(staged)) return staged;
            return success_k;
        }

        /** @brief Merges both sides from the first key not less than @p comparable. */
        template <typename comparable_type_, typename callback_type_>
        void merge_from_(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            merge_(changes_.lower_bound(comparable),
                   store_ref().visible_at_(std::forward<comparable_type_>(comparable), snapshot_),
                   std::forward<callback_type_>(callback));
        }

        /** @brief Merges both sides from the smallest key either of them holds. */
        template <typename callback_type_>
        void merge_all_(callback_type_ &&callback) const noexcept
            requires ordered_core_k
        {
            merge_(changes_.begin(), store_ref().visible_from_(store_ref().entries_.begin(), snapshot_),
                   std::forward<callback_type_>(callback));
        }

        /** @brief Reports whichever of the staged and the committed candidate comes first. */
        template <typename staged_iterator_type_, typename callback_found_type_, typename callback_missing_type_>
        void bounded_(staged_iterator_type_ staged, visible_cursor_t committed, callback_found_type_ &&callback_found,
                      callback_missing_type_ &&callback_missing) const noexcept
            requires ordered_core_k
        {
            while (!committed.exhausted() &&
                   changes_.find(mapping_key_or_itself<value_t>(committed.standing_->payload)) != changes_.end())
                committed.advance();

            while (staged != changes_.end() && (*staged).presence == presence_t::erased_k) ++staged;

            if (committed.exhausted() && staged == changes_.end()) callback_missing();
            else if (committed.exhausted()) callback_found((*staged).payload);
            else if (staged == changes_.end()) callback_found(committed.standing_->payload);
            else {
                auto const ordering = changes_.key_comp();
                value_t const &visible = committed.standing_->payload;
                if (ordering.per_key_compare(visible, (*staged).payload)) callback_found(visible);
                else callback_found((*staged).payload);
            }
        }

      public:
#pragma endregion Transaction Lookup

#pragma region Transaction Lifecycle
        /**
         *  @brief Moves every pending write into the store, invisible, and reserves everything a
         *    commit would otherwise have to allocate.
         *  @return Success; @c write_conflict_k, @c read_conflict_k or @c phantom_conflict_k naming
         *    which check turned this transaction away; or an allocation failure that leaves the store
         *    untouched.
         */
        [[nodiscard]] status_t stage() noexcept {
            if (staging_ == staging_t::staged_k) return operation_not_permitted_k;
            auto &store = store_ref();
            auto const prepaid = storage_shape_t::prepare(store.entries_, changed_identifiers_.size());
            if (failed(prepaid)) return prepaid;
            if (status_t const validated = validate_accesses_(); failed(validated)) return validated;

            // Every change needs somewhere to land before any of them moves, or a transaction could
            // run out of memory with half of itself already in the store. So this pass files a
            // key-only placeholder under this generation for each written key, and only then does
            // the second pass move the payloads onto slots already paid for, where it cannot fail.
            for (std::size_t reserved = 0; reserved != changed_identifiers_.size(); ++reserved) {
                auto reserved_identifier = copy_safely<identifier_t>(changed_identifiers_[reserved]);
                if (!reserved_identifier) {
                    unstage_(reserved);
                    return out_of_memory_heap_k;
                }
                versioned_t placeholder(value_t {std::move(*reserved_identifier)});
                placeholder.generation = generation_;
                placeholder.presence = presence_t::present_k;
                if (failed(storage_shape_t::upsert(store.entries_, std::move(placeholder)))) {
                    unstage_(reserved);
                    return out_of_memory_heap_k;
                }
            }

            for (auto staged = changes_.begin(); staged != changes_.end(); ++staged) {
                versioned_t &version = store_t::mutable_ref_(*staged);
                [[maybe_unused]] status_t const moved = storage_shape_t::upsert(store.entries_, std::move(version));
                assert(succeeded(moved) && "the placeholder pass already paid for this slot");
            }

            changes_.clear();
            staging_ = staging_t::staged_k;
            return success_k;
        }

        /**
         *  @brief Publishes every staged version under one stamp.
         *
         *  The second phase allocates nothing and cannot fail, and atomicity is the ordering of the
         *  two steps below: nothing stamped is reachable until the published stamp advances, because
         *  no snapshot handed out so far can name the new stamp.
         */
        [[nodiscard]] status_t commit() noexcept {
            if (status_t const permitted = validate_for_commit(); failed(permitted)) return permitted;
            publish_under();
            return success_k;
        }

        /**
         *  @brief Draws this store's stamp and publishes every staged version under it, refusing nothing.
         *
         *  The no-stamp half of the split, so a caller spanning several stores can ask all of them
         *  through @c validate_for_commit and only then tell each to write. The sibling engines keeping
         *  no clock spell it the same way and order their own versions; this one draws a stamp first.
         *
         *  @warning Only ever called after @c validate_for_commit answered success, with nothing since.
         */
        void publish_under() noexcept {
            auto &store = store_ref();

            snapshot_clock_t::commit_in_flight_t in_flight;
            store.clock_->begin_commit(in_flight);
            publish_under(in_flight.stamp());
            store.clock_->end_commit(in_flight);

            // The snapshot moves to the stamp just published, so this transaction reads its own
            // writes and the mark is free to follow it once every other reader has left. A commit
            // drawn before this one and still writing itself out holds the watermark below both, so
            // the wait is what makes reading one's own writes whole rather than partial.
            store.clock_->await_published(in_flight.stamp());
            snapshot_ = store.clock_->take_snapshot(lease_);
            prune_committed();
        }

        /**
         *  @brief Whether this transaction may still publish what it staged, refusing before it writes anything.
         *
         *  Asked again here because staging only proves the keys were free when they were staged, and
         *  a commit landing in between is the one thing that can invalidate that. Nothing is written
         *  until it answers, so a refusal leaves the transaction staged and retryable - and a caller
         *  spreading one commit across several stores asks every one of them before writing any.
         *
         *  @return Success; @c write_conflict_k, @c read_conflict_k or @c phantom_conflict_k naming
         *    which check turned this transaction away; or @c operation_not_permitted_k when nothing
         *    was staged.
         */
        [[nodiscard]] status_t validate_for_commit() const noexcept {
            if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
            return validate_accesses_();
        }

        /**
         *  @brief Makes every staged version visible under @p stamp, which cannot fail and cannot refuse.
         *
         *  The second half of a commit, split out because a caller spanning several stores has to know
         *  that once the first of them writes, none of the rest can turn back: @c validate_for_commit
         *  answered for all of them first, and every version this writes was reserved by @c stage.
         *
         *  @warning Only ever called after @c validate_for_commit answered success, with nothing since.
         */
        void publish_under(commit_stamp_t stamp) noexcept {
            assert(staging_ == staging_t::staged_k && "publishing what was never staged");
            store_ref().stamp_under_(changed_identifiers_.data(), changed_identifiers_.size(), generation_, stamp);
            staging_ = staging_t::pending_k;
        }

        /** @brief Answers every later read at @p snapshot, which a sharded transaction drew for all its parts. */
        void adopt_snapshot(generation_t snapshot) noexcept { snapshot_ = snapshot; }

        /** @brief Frees whatever the commit just published left unreachable, and forgets the keys it named. */
        void prune_committed() noexcept {
            auto &store = store_ref();
            store.prune_each_(changed_identifiers_.data(), changed_identifiers_.size());
            changed_identifiers_.clear();
        }

        /**
         *  @brief Pulls every staged version back into the transaction, leaving it retryable.
         *    The watches are kept, since a read concern outlives the write that failed on it.
         *  @return Success, @c operation_not_permitted_k when nothing was staged, @c consistency_k when
         *    a staged version went missing before the rollback reached it, or an allocation failure.
         */
        [[nodiscard]] status_t rollback() noexcept {
            if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
            auto &store = store_ref();
            status_t result = success_k;

            // The recovered versions carry the generation they were staged under, and this
            // transaction is about to take a new one, so each is re-stamped while it is out of any
            // container. The changed identifiers are kept: these keys are still going to be written.
            generation_t const resumed = store.next_generation_();
            for (identifier_t const &identifier : changed_identifiers_) {
                versioned_t recovered;
                if (!store.recover_version_(identifier, generation_, recovered)) {
                    result = status_t::consistency_k;
                    continue;
                }
                recovered.generation = resumed;
                if (failed(storage_shape_t::upsert(changes_, std::move(recovered)))) result = out_of_memory_heap_k;
            }

            staging_ = staging_t::pending_k;
            generation_ = resumed;
            return result;
        }

        /** @brief Discards everything staged and pending, and takes a fresh snapshot. */
        [[nodiscard]] status_t reset() noexcept { return reset_at(store_ref().clock_->take_snapshot(lease_)); }

        /** @brief The same, at a snapshot a sharded transaction drew once for every one of its parts. */
        [[nodiscard]] status_t reset_at(generation_t snapshot) noexcept {
            auto &store = store_ref();
            if (staging_ == staging_t::staged_k) unstage_(changed_identifiers_.size());

            accesses_.clear();
            changes_.clear();
            changed_identifiers_.clear();
            // A read that named no key refuses every commit that saw a newer stamp. Clearing the reads
            // without clearing that refusal leaves a transaction which can never commit again.
            read_set_ = read_set_t::names_every_key_k;
            staging_ = staging_t::pending_k;
            generation_ = store.next_generation_();
            snapshot_ = snapshot;
            return success_k;
        }

#pragma endregion Transaction Lifecycle
    };

#pragma region Publication

    /**
     *  @brief A group of writes over many keys that becomes visible under one stamp.
     *
     *  Publishing key by key draws a stamp per key, so a reader that opens between two of them sees
     *  the operation half applied. This files every version invisibly first and stamps them all at
     *  once, which is the same two-step shape @c transaction_t::commit has, without the validation a
     *  store-level write never had to do.
     *
     *  Anything staged and not published is dropped when the publication goes away, so a failure part
     *  way through a group leaves the store exactly as it was.
     */
    class publication_t {
        friend store_t;

        store_t *store_ {nullptr};
        changed_identifiers_vector_t staged_identifiers_;
        /** @brief The generation every version staged by this publication carries. */
        generation_t generation_ {0};

        explicit publication_t(store_t &store) noexcept
            : store_(&store),
              staged_identifiers_(changed_identifiers_allocator_t(storage_shape_t::allocator_of(store.entries_))),
              generation_(store.next_generation_()) {}

        /** @brief Stages @p versioned invisibly under this publication's generation. */
        [[nodiscard]] status_t stage_(identifier_t &&identifier, versioned_t &&versioned) noexcept {
            if (!store_) return operation_not_permitted_k;
            if (status_t const reserved = staged_identifiers_.reserve(staged_identifiers_.size() + 1); failed(reserved))
                return reserved;
            if (failed(storage_shape_t::prepare(store_->entries_, 1))) return out_of_memory_heap_k;
            versioned.generation = generation_;
            if (failed(storage_shape_t::upsert(store_->entries_, std::move(versioned)))) return out_of_memory_heap_k;
            [[maybe_unused]] status_t const recorded =
                staged_identifiers_.push_back(assume_reserved, std::move(identifier));
            return success_k;
        }

      public:
        publication_t(publication_t &&other) noexcept
            : store_(std::exchange(other.store_, nullptr)), staged_identifiers_(std::move(other.staged_identifiers_)),
              generation_(other.generation_) {}

        publication_t &operator=(publication_t &&other) noexcept {
            if (this == &other) return *this;
            rollback();
            store_ = std::exchange(other.store_, nullptr);
            staged_identifiers_ = std::move(other.staged_identifiers_);
            generation_ = other.generation_;
            return *this;
        }

        ~publication_t() noexcept { rollback(); }

        publication_t(publication_t const &) = delete;
        publication_t &operator=(publication_t const &) = delete;

        /** @brief How many keys are staged and waiting for a stamp. */
        [[nodiscard]] std::size_t staged_count() const noexcept { return staged_identifiers_.size(); }

        /** @brief Sizes the group up front, so a later @c upsert cannot run out of memory. */
        [[nodiscard]] status_t reserve(std::size_t count) noexcept {
            if (!store_) return operation_not_permitted_k;
            if (status_t const reserved = staged_identifiers_.reserve(count); failed(reserved)) return reserved;
            return storage_shape_t::prepare(store_->entries_, count);
        }

        /**
         *  @brief Files a write of @p value, invisible until @c publish stamps the group.
         *  @param[in] value Element to write, moved into the store.
         *  @return Success or an allocation failure that leaves the group as it was.
         */
        [[nodiscard]] status_t upsert(value_t &&value) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_identifier) return out_of_memory_heap_k;
            versioned_t versioned(std::move(value));
            versioned.presence = presence_t::present_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        /**
         *  @brief Files a tombstone over @p identifier, invisible until @c publish stamps the group.
         *    A tombstone rather than a detached entry, so a snapshot older than the group keeps reading
         *    the version it opened on.
         */
        [[nodiscard]] status_t erase(identifier_t const &identifier) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) return out_of_memory_heap_k;
            auto maybe_payload = copy_safely<identifier_t>(identifier);
            if (!maybe_payload) return out_of_memory_heap_k;

            versioned_t versioned(value_t {std::move(*maybe_payload)});
            versioned.presence = presence_t::erased_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        /**
         *  @brief Makes every staged version visible under one stamp, then prunes what the group masked.
         *    The publication stays usable afterwards and starts a fresh group on the next write.
         */
        [[nodiscard]] status_t publish() noexcept {
            if (!store_) return operation_not_permitted_k;
            if (staged_identifiers_.size() == 0) return success_k;

            store_->stamp_as_one_commit_(staged_identifiers_.data(), staged_identifiers_.size(), generation_);
            store_->prune_each_(staged_identifiers_.data(), staged_identifiers_.size());
            staged_identifiers_.clear();
            generation_ = store_->next_generation_();
            return success_k;
        }

        /** @brief Drops every version staged and not yet published, leaving the store as it was. */
        void rollback() noexcept {
            if (!store_) return;
            for (identifier_t const &identifier : staged_identifiers_) [[maybe_unused]]
                bool const dropped = store_->drop_version_(identifier, generation_);
            staged_identifiers_.clear();
        }
    };

#pragma endregion Publication

  private:
    dated_entries_t entries_;
    /** @brief The clock this store keeps for itself, and the one it uses until somebody attaches another. */
    snapshot_clock_t owned_clock_ {};
    /** @brief Where every stamp and every snapshot comes from, which a shard set repoints at one it shares. */
    snapshot_clock_t *clock_ {&owned_clock_};
    /** @brief Keys whose newest published version says they are there. */
    std::size_t live_count_ {0};

    friend class transaction_t;

#pragma region Stamps and Snapshots

    /**
     *  @brief Points this store's clock where @p other's pointed, and takes its counters when they were its own.
     *    A store sharing somebody's clock simply keeps sharing it; one that owned its clock copies the
     *    counters over, since the object the old one lives in is about to stop speaking for anything.
     */
    void adopt_clock_of_(snapshot_store &other) noexcept {
        if (other.clock_ != &other.owned_clock_) { clock_ = other.clock_; }
        else {
            owned_clock_.adopt(other.owned_clock_);
            clock_ = &owned_clock_;
        }
    }

    /** @brief Hands out the next generation, which dates a transaction rather than its visibility. */
    generation_t next_generation_() noexcept { return clock_->next_generation(); }

    /** @brief The newest stamp every part of which is written, which is what a fresh read answers at. */
    [[nodiscard]] generation_t published_stamp_() const noexcept { return clock_->published_stamp(); }

    /** @brief The newest stamp drawn, which is what a validator compares against. */
    [[nodiscard]] generation_t drawn_stamp_() const noexcept { return clock_->drawn_stamp(); }

    /** @brief The stamp as a plain number, which is how two versions are ordered by recency. */
    static constexpr generation_t stamp_of(commit_stamp_t stamp) noexcept { return static_cast<generation_t>(stamp); }

    /** @brief The newest snapshot no reader can be sitting below, so everything older is unreachable. */
    [[nodiscard]] generation_t low_water_mark_() const noexcept { return clock_->low_water_mark(); }

#pragma endregion Stamps and Snapshots

#pragma region Version Access

    /** @brief A writable reference to a stored element, which every core hands out as immutable. */
    template <typename element_type_>
    static element_type_ &mutable_ref_(element_type_ const &element) noexcept {
        return const_cast<element_type_ &>(element);
    }

    /** @brief Whether two stored objects carry the same bare key, ignoring the generation half. */
    template <typename first_type_, typename second_type_>
    [[nodiscard]] bool same_key_(first_type_ const &first, second_type_ const &second) const noexcept {
        if constexpr (ordered_core_k) {
            auto const ordering = entries_.key_comp();
            return !ordering.per_key_compare(first, second) && !ordering.per_key_compare(second, first);
        }
        else { return entries_.key_eq().equals(identifier_of(first), identifier_of(second)); }
    }

    /**
     *  @brief Applies @p visitor to every stored version of @p comparable, in no particular order.
     *    An ordered core walks the key's contiguous run; an unordered one walks its probe run, which
     *    holds every version of the key because only equality was widened, never the hash.
     */
    template <typename comparable_type_, typename visitor_type_>
    void visit_versions_(comparable_type_ const &comparable, visitor_type_ &&visitor) const noexcept {
        if constexpr (ordered_core_k) {
            for (auto cursor = entries_.lower_bound(comparable);
                 cursor != entries_.end() && same_key_(*cursor, comparable); ++cursor)
                visitor(*cursor);
        }
        else {
            entries_.probe_to_visit(comparable, [&](auto const &slot) noexcept {
                visitor(slot.key());
                return probe_control_t::resume_k;
            });
        }
    }

    /**
     *  @brief The version @p snapshot sees, tombstones included; null when the key did not exist for it.
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] snapshot The stamp the reader holds.
     */
    template <typename comparable_type_>
    [[nodiscard]] versioned_t const *visible_version_(comparable_type_ const &comparable,
                                                      generation_t snapshot) const noexcept {
        versioned_t const *newest = nullptr;
        visit_versions_(comparable, [&](versioned_t const &version) noexcept {
            if (!visible_at(version.committed, snapshot)) return;
            if (!newest || stamp_of(newest->committed) < stamp_of(version.committed)) newest = &version;
        });
        return newest;
    }

    /**
     *  @brief The version @p snapshot may see the payload of, which excludes a committed tombstone.
     *    Every surface that hands a payload out asks this instead, so the rule that a tombstone is
     *    not an element lives in one place, while a watch still gets to observe the erasure.
     */
    template <typename comparable_type_>
    [[nodiscard]] versioned_t const *readable_version_(comparable_type_ const &comparable,
                                                       generation_t snapshot) const noexcept {
        versioned_t const *visible = visible_version_(comparable, snapshot);
        return visible && visible->presence == presence_t::present_k ? visible : nullptr;
    }

    /** @brief The version of @p comparable carrying @p generation, which only its own writer names. */
    template <typename comparable_type_>
    [[nodiscard]] versioned_t const *dated_version_(comparable_type_ const &comparable,
                                                    generation_t generation) const noexcept {
        versioned_t const *found = nullptr;
        visit_versions_(comparable, [&](versioned_t const &version) noexcept {
            if (version.generation == generation) found = &version;
        });
        return found;
    }

    /** @brief Whether anything published over @p comparable after @p snapshot was taken. */
    template <typename comparable_type_>
    [[nodiscard]] bool key_changed_since_(comparable_type_ const &comparable, generation_t snapshot) const noexcept {
        bool changed = false;
        visit_versions_(comparable, [&](versioned_t const &version) noexcept {
            if (version.committed == commit_stamp_t::uncommitted_k) return;
            if (!visible_at(version.committed, snapshot)) changed = true;
        });
        return changed;
    }

    /**
     *  @brief Whether anything published a version of any key in [ @p lower, @p upper ) after @p snapshot.
     *
     *  A phantom is an @b entry in the window, whatever key it carries and whatever masks it, so this
     *  scans the dated core over the window rather than resolving what the window reads. Answering the
     *  latter is per-key work the question does not need, and it would miss a key whose newest version
     *  is a tombstone - an erase publishes one rather than leaving a hole, which is exactly the commit
     *  a range read has to notice.
     *
     *  Pruning cannot hide a conflict here: reclamation never passes an open lease, so no version
     *  stamped above a live reader's snapshot is reclaimable while that reader holds it.
     *
     *  @param[in] ends Which sides of the window run off the end, in which case the matching bound is
     *    ignored - an ordinal read starts before every key, and a bound read that landed on nothing
     *    runs past the last one.
     */
    template <typename lower_type_, typename upper_type_>
    [[nodiscard]] bool range_changed_since_(lower_type_ const &lower, upper_type_ const &upper, generation_t snapshot,
                                            access_t ends = access_t::none_k) const noexcept
        requires ordered_core_k
    {
        auto const ordering = entries_.key_comp();
        bool const runs_to_the_highest = holds(ends, access_t::to_the_highest_k);
        auto cursor = holds(ends, access_t::from_the_lowest_k) ? entries_.begin() : entries_.lower_bound(lower);
        for (; cursor != entries_.end(); ++cursor) {
            versioned_t const &version = *cursor;
            if (!runs_to_the_highest && !ordering.per_key_compare(version, upper)) break;
            if (version.committed == commit_stamp_t::uncommitted_k) continue;
            if (!visible_at(version.committed, snapshot)) return true;
        }
        return false;
    }

    /** @brief Hands @p callback the value @p snapshot reads for @p comparable, or reports absence. */
    template <typename comparable_type_, typename callback_found_type_, typename callback_missing_type_>
    void find_at_(comparable_type_ &&comparable, generation_t snapshot, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing) const noexcept {
        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        versioned_t const *readable = readable_version_(comparable, snapshot);
        if (readable) callback_found(readable->payload);
        else callback_missing();
    }

    /**
     *  @brief Hands @p callback the value @p snapshot reads for every key, in whatever order the core holds.
     *
     *  A key's versions are one contiguous run only on an ordered core, so the unordered walk reports an
     *  entry when it @b is the version its own key resolves to, which names every readable key once and
     *  no unreadable one at all.
     */
    template <typename callback_type_>
    void for_each_at_(generation_t snapshot, callback_type_ &&callback) const noexcept {
        if constexpr (ordered_core_k) {
            walk_visible_keys_(
                entries_.begin(), [](versioned_t const &) noexcept { return true; }, snapshot,
                [&](value_t const &value) noexcept {
                    callback(value);
                    return probe_control_t::resume_k;
                });
        }
        else {
            for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor) {
                versioned_t const &version = *cursor;
                if (readable_version_(identifier_of(version), snapshot) == &version) callback(version.payload);
            }
        }
    }

#pragma endregion Version Access

#pragma region Ordered Walks

    /**
     *  @brief Hands @p callback the value @p snapshot reads for each key from @p cursor onward,
     *    stopping when @p within rejects a key or the callback halts.
     *
     *  The run-by-run stepping lives in @c visible_cursor_t, so the push-style walk and the resumable
     *  one can never disagree about which version of a key a snapshot reads. @p within is asked of the
     *  run's first entry, which lets a bounded walk stop without stepping past its window.
     */
    template <typename within_type_, typename callback_type_>
    void walk_visible_keys_(entry_iterator_t start, within_type_ &&within, generation_t snapshot,
                            callback_type_ &&callback) const noexcept
        requires ordered_core_k
    {
        for (visible_cursor_t cursor(*this, start, snapshot); !cursor.exhausted(); cursor.step_()) {
            if (!within(*cursor.position_)) break;
            if (!cursor.standing_) continue;
            if (callback(cursor.standing_->payload) == probe_control_t::halt_k) break;
        }
    }

    /** @brief A cursor settled on the first key at or after @p start that @p snapshot reads. */
    [[nodiscard]] visible_cursor_t visible_from_(entry_iterator_t start, generation_t snapshot) const noexcept
        requires ordered_core_k
    {
        visible_cursor_t cursor(*this, start, snapshot);
        cursor.seek_readable_();
        return cursor;
    }

    /** @brief A cursor settled on the first key not less than @p comparable that @p snapshot reads. */
    template <typename comparable_type_>
    [[nodiscard]] visible_cursor_t visible_at_(comparable_type_ &&comparable, generation_t snapshot) const noexcept
        requires ordered_core_k
    {
        return visible_from_(entries_.lower_bound(std::forward<comparable_type_>(comparable)), snapshot);
    }

    /** @brief Hands @p callback every value in [ @p lower, @p upper ) as @p snapshot reads them. */
    template <typename lower_type_, typename upper_type_, typename callback_type_>
    void range_at_(lower_type_ &&lower, upper_type_ &&upper, generation_t snapshot,
                   callback_type_ &&callback) const noexcept
        requires ordered_core_k
    {
        auto const ordering = entries_.key_comp();
        walk_visible_keys_(
            entries_.lower_bound(std::forward<lower_type_>(lower)),
            [&](versioned_t const &head) noexcept { return ordering.per_key_compare(head, upper); }, snapshot,
            std::forward<callback_type_>(callback));
    }

    /** @brief Hands @p callback every value at or after @p lower as @p snapshot reads them. */
    template <typename lower_type_, typename callback_type_>
    void range_from_at_(lower_type_ &&lower, generation_t snapshot, callback_type_ &&callback) const noexcept
        requires ordered_core_k
    {
        walk_visible_keys_(
            entries_.lower_bound(std::forward<lower_type_>(lower)),
            []([[maybe_unused]] versioned_t const &head) noexcept { return true; }, snapshot,
            std::forward<callback_type_>(callback));
    }

    /** @brief Hands @p callback every value before @p upper as @p snapshot reads them. */
    template <typename upper_type_, typename callback_type_>
    void range_up_to_at_(upper_type_ &&upper, generation_t snapshot, callback_type_ &&callback) const noexcept
        requires ordered_core_k
    {
        auto const ordering = entries_.key_comp();
        walk_visible_keys_(
            entries_.begin(), [&](versioned_t const &head) noexcept { return ordering.per_key_compare(head, upper); },
            snapshot, std::forward<callback_type_>(callback));
    }

#pragma endregion Ordered Walks

#pragma region Version Lifecycle

    /** @brief Builds the per-transaction staging storage, which keys on the bare key alone. */
    static changes_t build_changes_(dated_entries_t const &entries) noexcept {
        using changes_allocator_t = typename changes_t::allocator_type;
        if constexpr (ordered_core_k) {
            return changes_t(entries.key_comp(), changes_allocator_t(storage_shape_t::allocator_of(entries)));
        }
        else {
            using equality_t = typename storage_shape_t::equality_t;
            return changes_t(entries.hash_function(), equality_t(entries.key_eq().equals),
                             changes_allocator_t(storage_shape_t::allocator_of(entries)));
        }
    }

    /** @brief Builds the versioned storage, which keys on the key @b and the generation. */
    static dated_entries_t build_entries_(comparator_t const &source, allocator_t const &allocator) noexcept {
        using entries_allocator_t = typename dated_entries_t::allocator_type;
        using addressing_t = typename storage_shape_t::addressing_t;
        if constexpr (ordered_core_k) { return dated_entries_t(addressing_t(source), entries_allocator_t(allocator)); }
        else {
            using dated_equality_t = typename storage_shape_t::dated_equality_t;
            return dated_entries_t(addressing_t {}, dated_equality_t(source), entries_allocator_t(allocator));
        }
    }

    /** @brief Drops the version of @p identifier carrying @p generation, which only its own writer may do. */
    [[nodiscard]] bool drop_version_(identifier_t const &identifier, generation_t generation) noexcept {
        return entries_.erase(dated_reference_t {identifier, generation});
    }

    /**
     *  @brief Takes the version of @p identifier carrying @p generation out, moving its payload to @p destination.
     *  @return Whether a version was there to take.
     *  @note The entry leaves the index before anything is moved out of it, since the key the index
     *    addresses it by lives inside the payload and a moved-from key neither compares nor hashes.
     */
    [[nodiscard]] bool recover_version_(identifier_t const &identifier, generation_t generation,
                                        versioned_t &destination) noexcept {
        dated_reference_t const dated {identifier, generation};
        if constexpr (ordered_core_k) {
            auto extracted = entries_.extract(dated);
            if (!extracted.node_ptr_) return false;
            destination = std::move(extracted.node_ptr_->fruit);
            return true;
        }
        else {
            auto found = entries_.find(dated);
            if (found == entries_.end()) return false;
            destination = std::move(mutable_ref_(*found));
            entries_.erase(found);
            return true;
        }
    }

    /**
     *  @brief Stamps the version of @p identifier carrying @p generation as published under @p stamp.
     *  @return Whether the version was still there to publish.
     */
    [[nodiscard]] bool publish_version_(identifier_t const &identifier, generation_t generation,
                                        commit_stamp_t stamp) noexcept {
        versioned_t const *staged = dated_version_(identifier, generation);
        if (!staged) return false;
        versioned_t const *const previously_visible = readable_version_(identifier, latest_snapshot_k);
        mutable_ref_(*staged).committed = stamp;
        versioned_t const *const now_visible = readable_version_(identifier, latest_snapshot_k);
        live_count_ +=
            static_cast<std::size_t>(now_visible != nullptr) - static_cast<std::size_t>(previously_visible != nullptr);
        mark_liveness_(previously_visible, now_visible);
        return true;
    }

    /**
     *  @brief Moves the survivor tag from the version a key read before to the one it reads now,
     *    repairing the counts above both.
     *
     *  The two arguments are the same type and read in time order - what the key resolved to before
     *  this publish, then what it resolves to after - so swapping them tags the wrong entry as the
     *  survivor and leaves the rank index counting a version no reader can reach.
     */
    void mark_liveness_([[maybe_unused]] versioned_t const *previously_visible,
                        [[maybe_unused]] versioned_t const *now_visible) noexcept {
        if constexpr (ranked_core_k) {
            if (previously_visible == now_visible) return;
            if (previously_visible) {
                mutable_ref_(*previously_visible).liveness = liveness_t::superseded_k;
                refresh_tag_(*previously_visible);
            }
            if (now_visible) {
                mutable_ref_(*now_visible).liveness = liveness_t::survivor_k;
                refresh_tag_(*now_visible);
            }
        }
    }

    /** @brief Repairs the subtree counts on the path to @p version after its tag flipped in place. */
    void refresh_tag_(versioned_t const &version) noexcept
        requires ranked_core_k
    {
        [[maybe_unused]] bool const repaired =
            entries_.refresh_augmentation(dated_reference_t {identifier_of(version), version.generation});
        assert(repaired && "the entry whose tag flipped was found through the very same index");
    }

    /**
     *  @brief Writes @p stamp onto every version of @p identifiers carrying @p generation.
     *
     *  One stamp for the whole group is the whole point: a reader's snapshot either names that stamp
     *  or does not, so it observes all of the group or none of it, never a prefix. The watermark is
     *  left alone, so a caller spreading one commit over several stores stamps each of them and
     *  publishes once, when the last has been written.
     *
     *  @warning Every named version was filed under @p generation by whoever staged it and nobody
     *    else may drop it, so a missing one is a defect rather than an outcome a caller can act on.
     */
    void stamp_under_(identifier_t const *identifiers, std::size_t count, generation_t generation,
                      commit_stamp_t stamp) noexcept {
        for (std::size_t index = 0; index != count; ++index) {
            [[maybe_unused]] bool const published = publish_version_(identifiers[index], generation, stamp);
            assert(published && "a staged version went missing between staging and its stamp");
        }
    }

    /**
     *  @brief Stamps a group that lives entirely in this store, drawing its stamp and publishing it.
     *
     *  This is the path a direct write and a publication both take, so it is where they inherit whether
     *  the level waits for its own publication before answering its caller.
     */
    void stamp_as_one_commit_(identifier_t const *identifiers, std::size_t count, generation_t generation) noexcept {
        snapshot_clock_t::commit_in_flight_t in_flight;
        clock_->begin_commit(in_flight);
        stamp_under_(identifiers, count, generation, in_flight.stamp());
        clock_->end_commit(in_flight);
        if constexpr (awaits_publication_k) clock_->await_published(in_flight.stamp());
    }

    /** @brief Frees every version of every identifier in @p identifiers that no snapshot can still reach. */
    void prune_each_(identifier_t const *identifiers, std::size_t count) noexcept {
        generation_t const mark = low_water_mark_();
        for (std::size_t index = 0; index != count; ++index) {
            [[maybe_unused]] std::size_t const reclaimed = prune_key_(identifiers[index], mark);
        }
    }

    /** @brief Whether @p version is unreachable from every snapshot at or below @p mark. */
    static bool unreachable_(versioned_t const &version, versioned_t const *survivor, generation_t mark) noexcept {
        if (!visible_at(version.committed, mark)) return false;
        // Anything older than the survivor is masked by it, and a surviving tombstone answers the
        // same question absence answers, so neither is worth an entry.
        return &version != survivor || version.presence == presence_t::erased_k;
    }

    /**
     *  @brief Frees every version sharing @p anchor's key that no snapshot at or below @p mark reaches.
     *    @p anchor is judged last, so the key the walk is addressed by outlives every erasure but its own.
     *  @return How many versions were reclaimed.
     */
    std::size_t prune_key_of_(versioned_t const &anchor, generation_t mark) noexcept {
        identifier_t const &key = identifier_of(anchor);
        versioned_t const *const survivor = visible_version_(key, mark);
        std::size_t reclaimed = 0;

        if constexpr (ordered_core_k) {
            auto run_end = entries_.lower_bound(key);
            while (run_end != entries_.end() && same_key_(*run_end, anchor)) ++run_end;

            auto scan = entries_.lower_bound(key);
            auto anchor_position = entries_.end();
            while (scan != run_end) {
                if (&*scan == &anchor) {
                    anchor_position = scan;
                    ++scan;
                    continue;
                }
                if (!unreachable_(*scan, survivor, mark)) {
                    ++scan;
                    continue;
                }
                scan = entries_.erase(scan).next;
                ++reclaimed;
            }
            if (anchor_position != entries_.end() && unreachable_(anchor, survivor, mark)) {
                [[maybe_unused]] auto const removed = entries_.erase(anchor_position);
                ++reclaimed;
            }
        }
        else {
            entries_.probe_to_visit(key, [&](auto const &slot) noexcept {
                versioned_t const &version = slot.key();
                if (&version != &anchor && unreachable_(version, survivor, mark)) {
                    // The doomed version is its own search key, so the probe never consults an
                    // identifier that the erasure is about to destroy.
                    [[maybe_unused]] bool const erased = entries_.erase(mutable_ref_(version));
                    ++reclaimed;
                }
                return probe_control_t::resume_k;
            });
            if (unreachable_(anchor, survivor, mark)) {
                [[maybe_unused]] bool const erased = entries_.erase(mutable_ref_(anchor));
                ++reclaimed;
            }
        }
        return reclaimed;
    }

    /** @brief Frees every version of @p identifier that no snapshot at or below @p mark reaches. */
    template <typename comparable_type_>
    std::size_t prune_key_(comparable_type_ const &comparable, generation_t mark) noexcept {
        versioned_t const *any = nullptr;
        visit_versions_(comparable, [&](versioned_t const &version) noexcept {
            if (!any) any = &version;
        });
        return any ? prune_key_of_(*any, mark) : 0;
    }

    /**
     *  @brief Publishes a tombstone over every key @p walk offers, all under one stamp.
     *
     *  @param[in] walk Invoked with a collector that takes one @c value_t const & per key to erase.
     *  @param[in] callback Callback handed each element about to be tombstoned. Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store exactly as it was.
     *
     *  The keys are copied out before anything is staged: the walk that produces them reads the very
     *  index the tombstones are about to be staged into.
     */
    template <typename walk_type_, typename callback_type_>
    [[nodiscard]] status_t tombstone_walked_(walk_type_ &&walk, callback_type_ &&callback) noexcept
        requires ordered_core_k
    {
        changed_identifiers_vector_t doomed(changed_identifiers_allocator_t(storage_shape_t::allocator_of(entries_)));
        status_t collecting = success_k;
        walk([&](value_t const &value) noexcept {
            if (failed(collecting)) return probe_control_t::halt_k;
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_identifier) {
                collecting = out_of_memory_heap_k;
                return probe_control_t::halt_k;
            }
            if (status_t const kept = doomed.push_back(std::move(*maybe_identifier)); failed(kept)) {
                collecting = kept;
                return probe_control_t::halt_k;
            }
            callback(value);
            return probe_control_t::resume_k;
        });
        if (failed(collecting)) return collecting;
        if (doomed.size() == 0) return success_k;

        auto opened = publication();
        if (!opened) return out_of_memory_heap_k;
        if (status_t const reserved = opened->reserve(doomed.size()); failed(reserved)) return reserved;
        for (identifier_t const &identifier : doomed)
            if (status_t const staged = opened->erase(identifier); failed(staged)) return staged;
        return opened->publish();
    }

#pragma endregion Version Lifecycle

#pragma endregion Type Definitions

  public:
#pragma region Constructors and Assignment

    snapshot_store() noexcept : entries_(build_entries_(comparator_t {}, allocator_t {})) {}

    /** @brief Seeds the underlying core's allocator, which a stateful allocator needs. */
    explicit snapshot_store(allocator_t const &allocator) noexcept
        : entries_(build_entries_(comparator_t {}, allocator)) {}

    /** @brief Seeds the addressing as well, which a comparator carrying state or a dispatch pointer needs. */
    snapshot_store(comparator_t const &comparator, allocator_t const &allocator = {}) noexcept
        : entries_(build_entries_(comparator, allocator)) {}

    /**
     *  @brief Takes over @p other's versions and its place on the clock.
     *    A store sharing a clock keeps pointing at it; one keeping its own copies the counters across,
     *    since the moved-from object the old clock lives in is about to stop speaking for anything.
     */
    snapshot_store(snapshot_store &&other) noexcept
        : entries_(std::move(other.entries_)), live_count_(other.live_count_) {
        adopt_clock_of_(other);
    }

    snapshot_store &operator=(snapshot_store &&other) noexcept {
        if (this == &other) return *this;
        entries_ = std::move(other.entries_);
        adopt_clock_of_(other);
        live_count_ = other.live_count_;
        return *this;
    }

    ~snapshot_store() noexcept { entries_.clear(); }

    snapshot_store(snapshot_store const &) = delete;
    snapshot_store &operator=(snapshot_store const &) = delete;

    /** @brief Builds a store around an allocator, reported as a result rather than thrown. */
    [[nodiscard]] static expected<store_t> make(allocator_t const &allocator = {}) noexcept {
        return store_t {allocator};
    }

    /** @brief Builds a store around a specific comparator, for comparators that carry state. */
    [[nodiscard]] static expected<store_t> make(comparator_t const &comparator, allocator_t const &allocator) noexcept {
        return store_t {comparator, allocator};
    }

#pragma endregion Constructors and Assignment

#pragma region Capacity

    /** @brief How many keys the newest published version says are there. */
    [[nodiscard]] std::size_t size() const noexcept { return live_count_; }

    /** @brief Whether no key is currently readable. */
    [[nodiscard]] bool empty() const noexcept { return live_count_ == 0; }

    /** @brief How many versions the store holds across every key, published and staged alike. */
    [[nodiscard]] std::size_t versions_count() const noexcept { return entries_.size(); }

    /** @brief How many versions of @p comparable the store still holds. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t versions_count(comparable_type_ const &comparable) const noexcept {
        std::size_t counted = 0;
        visit_versions_(comparable, [&](versioned_t const &) noexcept { ++counted; });
        return counted;
    }

    /** @brief The newest stamp a fully written commit left behind. */
    [[nodiscard]] generation_t published_stamp() const noexcept { return published_stamp_(); }

    /** @brief The newest snapshot no open transaction sits below, which is what @c vacuum prunes to. */
    [[nodiscard]] generation_t low_water_mark() const noexcept { return low_water_mark_(); }

    /** @brief How many transactions currently hold a snapshot. */
    [[nodiscard]] std::size_t open_snapshots() const noexcept { return clock_->open_snapshots(); }

    /** @brief Whether a member equal to @p comparable is readable now. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        return readable_version_(comparable, published_stamp_()) != nullptr;
    }

    /** @brief How many members equal @p comparable, which for a unique-key store is zero or one. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

#pragma endregion Capacity

#pragma region Transaction Management

    /** @brief Opens a transaction, fixing the snapshot every one of its reads will be answered at. */
    [[nodiscard]] expected<transaction_t> transaction() noexcept { return transaction_t {*this}; }

    /**
     *  @brief Opens one part of a sharded transaction at a @p snapshot and @p generation drawn elsewhere.
     *
     *  Registers nothing with the clock: the owner of the snapshot holds the one claim that answers
     *  for every part, so a claim per part would only make the low-water mark count the same reader
     *  sixteen times.
     */
    [[nodiscard]] expected<transaction_t> transaction_at(generation_t snapshot, generation_t generation) noexcept {
        return transaction_t {*this, snapshot, generation};
    }

    /**
     *  @brief Draws every stamp and every snapshot from @p clock rather than from this store's own.
     *
     *  This is what makes a set of stores one snapshot: they share a stamp counter, a watermark and a
     *  reader census, so a stamp drawn for a commit spanning them means the same thing in each, and no
     *  one of them prunes a version a reader of another still names.
     *
     *  @warning @p clock has to be the clock that issued whatever stamps this store already holds,
     *    which for a fresh store is vacuously true, and no reader may be open - its claim would be
     *    left behind on the clock being dropped.
     */
    void attach_clock(snapshot_clock_t &clock) noexcept {
        assert(clock_->open_snapshots() == 0 && "a reader would leave its claim on the clock being dropped");
        clock_ = &clock;
    }

    /** @brief The clock this store draws from, its own until one is attached. */
    [[nodiscard]] snapshot_clock_t &clock() noexcept { return *clock_; }
    [[nodiscard]] snapshot_clock_t const &clock() const noexcept { return *clock_; }

#pragma endregion Transaction Management

#pragma region Lookup

    /**
     *  @brief Finds a member @b equal to @p comparable as of the newest published commit.
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {
        find_at_(std::forward<comparable_type_>(comparable), published_stamp_(),
                 std::forward<callback_found_type_>(callback_found),
                 std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /** @brief Copies out the member equal to @p comparable, or reports why it could not. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const looked_up = find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(looked_up)) return looked_up;
        return result;
    }

    /**
     *  @brief Hands @p callback_found the smallest member the newest published commit shows.
     *
     *  The unbounded case of @c lower_bound, and the one a merged walk over several stores opens with:
     *  it asks for a first key rather than an ordinal, so a core keeping no subtree counts can answer.
     *
     *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when nothing is readable. Must be @c noexcept.
     */
    /**
     *  @brief How this store orders its keys, so a bounded walk can stop without guessing.
     *    The one ordering every read here descends on; a caller comparing keys any other way is
     *    ordering them differently from the store that holds them.
     */
    [[nodiscard]] comparator_t key_comp() const noexcept { return entries_.key_comp().comparator; }

    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_core_k
    {
        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");
        visible_keys().peek(std::forward<callback_found_type_>(callback_found),
                            std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /**
     *  @brief Removes the smallest member and hands it over, or reports the store is empty.
     *  @return @c key_not_found_k when nothing was there, so emptiness needs no second probe.
     *
     *  The removal publishes a tombstone rather than dropping the version, so a store drained by
     *  popping keeps its versions until a @c vacuum reclaims them.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t pop_smallest(callback_found_type_ &&callback_found = {},
                                        callback_missing_type_ &&callback_missing = {}) noexcept
        requires ordered_core_k && std::is_copy_constructible_v<identifier_t>
    {
        identifier_t doomed;
        bool found = false;
        status_t const looked_up = smallest(
            [&](value_t const &value) noexcept {
                doomed = identifier_t(value);
                found = true;
            },
            no_op_t {});
        if (failed(looked_up)) return looked_up;
        if (!found) {
            callback_missing();
            return status_t::key_not_found_k;
        }
        return erase(doomed, std::forward<callback_found_type_>(callback_found),
                     std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Finds the first member @b greater or equal to @p comparable. */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_core_k
    {
        value_t const *found = nullptr;
        walk_visible_keys_(
            entries_.lower_bound(std::forward<comparable_type_>(comparable)),
            [](versioned_t const &) noexcept { return true; }, published_stamp_(),
            [&](value_t const &value) noexcept {
                found = &value;
                return probe_control_t::halt_k;
            });
        if (found) callback_found(*found);
        else callback_missing();
        return success_k;
    }

    /** @brief Finds the first member @b strictly greater than @p comparable. */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_core_k
    {
        value_t const *found = nullptr;
        walk_visible_keys_(
            entries_.upper_bound(std::forward<comparable_type_>(comparable)),
            [](versioned_t const &) noexcept { return true; }, published_stamp_(),
            [&](value_t const &value) noexcept {
                found = &value;
                return probe_control_t::halt_k;
            });
        if (found) callback_found(*found);
        else callback_missing();
        return success_k;
    }

    /** @brief Copies out the first member not less than @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires ordered_core_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(bounded)) return bounded;
        return result;
    }

    /** @brief Copies out the first member greater than @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires ordered_core_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const bounded = upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(bounded)) return bounded;
        return result;
    }

    /** @brief A resumable walk over every key the newest published commit shows. */
    [[nodiscard]] visible_cursor_t visible_keys() const noexcept
        requires ordered_core_k
    {
        return visible_from_(entries_.begin(), published_stamp_());
    }

    /** @brief A resumable walk starting at the first key not less than @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] visible_cursor_t visible_keys_from(comparable_type_ &&comparable) const noexcept
        requires ordered_core_k
    {
        return visible_at_(std::forward<comparable_type_>(comparable), published_stamp_());
    }

    /** @brief Hands @p callback every member in [ @p lower, @p upper ) as of the newest published commit. */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper,
                                 callback_type_ &&callback = {}) const noexcept
        requires ordered_core_k
    {
        range_at_(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), published_stamp_(),
                  [&](value_t const &value) noexcept {
                      callback(value);
                      return probe_control_t::resume_k;
                  });
        return success_k;
    }

    /**
     *  @brief Finds every member equal to @p comparable, which for a unique-key store is one or none.
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback Callback receiving each match. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        return find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), no_op_t {});
    }

#pragma endregion Lookup

#pragma region Order Statistics

    /**
     *  @brief Hands @p callback_found the @p ordinal -th smallest key the newest published commit shows.
     *
     *  Answered at the newest published stamp and nowhere else: the subtree counts are kept over a tag
     *  reading "survives now", and one scalar per node cannot also answer for a reader holding an older
     *  snapshot. A transaction that needs its own snapshot's ordinal walks instead.
     *
     *  @param[in] ordinal Zero-based position among the keys the store currently shows.
     *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when fewer keys are readable. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                  callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ranked_core_k
    {
        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto const *node = entries_.select_augmented(ordinal);
        if (node) callback_found(node->fruit.payload);
        else callback_missing();
        return success_k;
    }

    /**
     *  @brief Hands @p callback_found how many keys the store shows before @p comparable.
     *    Answered at the newest published stamp only, for the reason @c select is.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c std::size_t. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if the key is not readable. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ranked_core_k
    {
        static_assert(is_safe_callback_for<callback_found_type_, std::size_t>,
                      "callback_found must be noexcept invocable with std::size_t");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        versioned_t const *readable = readable_version_(comparable, published_stamp_());
        if (!readable) {
            callback_missing();
            return success_k;
        }
        // Dated at a generation no version can carry, so the descent orders the target strictly before
        // every version of its own key rather than stopping inside that key's run.
        callback_found(entries_.rank_augmented(dated_reference_t {identifier_of(*readable), 0}));
        return success_k;
    }

    /** @brief How many keys the augmented counts say are readable, which is what @c select indexes. */
    [[nodiscard]] std::size_t ranked_size() const noexcept
        requires ranked_core_k
    {
        return entries_.augmented_size();
    }

#pragma endregion Order Statistics

#pragma region Modifiers

    /**
     *  @brief Publishes @p value under a stamp of its own, then prunes what that hid.
     *  @param[in] value Element to write, moved into the store.
     *  @return Success or an allocation failure that leaves the store untouched.
     */
    [[nodiscard]] status_t upsert(value_t &&value) noexcept {
        auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
        if (!maybe_identifier) return out_of_memory_heap_k;
        versioned_t versioned(std::move(value));
        versioned.presence = presence_t::present_k;
        return publish_directly_(*maybe_identifier, std::move(versioned));
    }

    /** @brief Publishes @p value only if the key is absent, reporting a clash rather than hiding it. */
    [[nodiscard]] status_t insert(value_t &&value) noexcept {
        expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(value));
        if (!key_is_present) return key_is_present.status();
        if (*key_is_present) return key_already_exists_k;
        return upsert(std::move(value));
    }

    /**
     *  @brief Publishes @p value only if the key is absent, reporting which of the two outcomes occurred.
     *    Declining to overwrite is a success, so the status alone cannot distinguish the cases and a
     *    caller that needs to know would otherwise pay for a membership probe of its own.
     *
     *  @param[in] value Element to write, moved into the store when the key is absent.
     *  @param[in] callback_inserted Invoked with the published element after a fresh insert. Must be @c noexcept.
     *  @param[in] callback_existing Invoked with the element already published, which keeps its value.
     *    Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store untouched.
     *  @note Both the probe and the report are answered at the newest published stamp, so an element a
     *    still-open transaction staged is neither a clash nor what @p callback_inserted hands back.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
    [[nodiscard]] status_t insert_if_missing(value_t &&value, callback_inserted_type_ &&callback_inserted = {},
                                             callback_existing_type_ &&callback_existing = {}) noexcept {
        static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                      "callback_existing must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                      "callback_inserted must be noexcept invocable with value_t const &");

        if (versioned_t const *present = readable_version_(mapping_key_or_itself<value_t>(value), published_stamp_())) {
            callback_existing(present->payload);
            return success_k;
        }

        if constexpr (std::is_same_v<std::remove_cvref_t<callback_inserted_type_>, no_op_t>)
            return upsert(std::move(value));

        // The identifier has to be taken before the move, or the lookup below searches by a
        // moved-from key - which compares wrongly rather than failing loudly.
        auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
        if (!maybe_identifier) return out_of_memory_heap_k;
        versioned_t versioned(std::move(value));
        versioned.presence = presence_t::present_k;
        if (status_t const published = publish_directly_(*maybe_identifier, std::move(versioned)); failed(published))
            return published;
        if (versioned_t const *stored = readable_version_(*maybe_identifier, published_stamp_()))
            callback_inserted(stored->payload);
        return success_k;
    }

    /** @brief Publishes @p value only if the key is already there. */
    [[nodiscard]] status_t update(value_t &&value) noexcept {
        expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(value));
        if (!key_is_present) return key_is_present.status();
        if (!*key_is_present) return key_not_found_k;
        return upsert(std::move(value));
    }

    /**
     *  @brief Publishes [ @p first, @p last ) under one stamp, refusing the group if any key is there.
     *
     *  @return Success, @c key_already_exists_k, @c write_conflict_k or @c read_conflict_k when another
     *    writer took one of these keys while the group was being staged, or an allocation failure.
     *    Nothing is published unless all of it is.
     *
     *  @note The clash is probed at the newest published stamp while the group is staged against the
     *    snapshot the call opened on, so a key this very range inserted earlier is not seen as a clash.
     *    Duplicates inside the range therefore collapse onto one version rather than being refused.
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        return commit_each_(first, last, [this](transaction_t &staging, value_t &&candidate) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(candidate));
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return key_already_exists_k;
            return staging.insert_if_missing(std::move(candidate));
        });
    }

    /**
     *  @brief Publishes [ @p first, @p last ) under one stamp, skipping the keys already there.
     *  @return Success, @c write_conflict_k or @c read_conflict_k when another writer took one of these
     *    keys, or an allocation failure. Nothing is published unless all of it is.
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert_if_missing(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        return commit_each_(first, last, [](transaction_t &staging, value_t &&candidate) noexcept {
            return staging.insert_if_missing(std::move(candidate));
        });
    }

    /**
     *  @brief Publishes [ @p first, @p last ) under one stamp, whether or not the keys are there.
     *  @return Success, @c write_conflict_k when another writer took one of these keys - this path
     *    reads none of them - or an allocation failure. Nothing is published unless all of it is.
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t upsert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        return commit_each_(first, last, [](transaction_t &staging, value_t &&candidate) noexcept {
            return staging.upsert(std::move(candidate));
        });
    }

    /**
     *  @brief Publishes [ @p first, @p last ) under one stamp, refusing the group if any key is absent.
     *  @return Success, @c key_not_found_k, @c write_conflict_k or @c read_conflict_k when another
     *    writer took one of these keys, or an allocation failure. Nothing is published unless all of
     *    it is.
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t update(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        return commit_each_(first, last, [](transaction_t &staging, value_t &&candidate) noexcept {
            return staging.update(std::move(candidate));
        });
    }

    /**
     *  @brief Publishes a tombstone over @p comparable, which the mark then reclaims when it may.
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback receiving the erased element. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return @c key_not_found_k if nothing was readable, otherwise success.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        versioned_t const *readable = readable_version_(comparable, published_stamp_());
        if (!readable) {
            callback_missing();
            return status_t::key_not_found_k;
        }
        callback_found(readable->payload);

        auto maybe_identifier = copy_safely<identifier_t>(identifier_of(*readable));
        if (!maybe_identifier) return out_of_memory_heap_k;
        auto maybe_payload = copy_safely<identifier_t>(*maybe_identifier);
        if (!maybe_payload) return out_of_memory_heap_k;

        versioned_t tombstone(value_t {std::move(*maybe_payload)});
        tombstone.presence = presence_t::erased_k;
        return publish_directly_(*maybe_identifier, std::move(tombstone));
    }

    /** @brief Grows the core to hold @p size more versions, so later writes cannot run out of memory. */
    [[nodiscard]] status_t reserve(std::size_t size) noexcept { return storage_shape_t::prepare(entries_, size); }

    /**
     *  @brief Opens a group of writes over many keys that will become visible under one stamp.
     *    A write of one key needs none of this - @c upsert and @c erase already draw a stamp each.
     */
    [[nodiscard]] expected<publication_t> publication() noexcept { return publication_t {*this}; }

    /**
     *  @brief Drops every version of every key, published, staged and historical alike.
     *
     *  @return @c operation_not_permitted_k while any transaction still holds a snapshot.
     *
     *  Refused rather than documented as a hole: a reader that opened before the call would otherwise
     *  watch its own versions vanish, which is the single promise this store exists to make, and
     *  @c live_count_ would be reset while an open transaction can still publish over it. Hiding every
     *  key from future readers without disturbing the current ones is what a tombstone per key does,
     *  which is @c publication_t, not this.
     *
     *  @note The generation and stamp counters keep running. Rewinding either would hand a future
     *    transaction a number an open one already carries, and both are compared by value.
     */
    [[nodiscard]] status_t clear() noexcept {
        if (clock_->open_snapshots() != 0) return operation_not_permitted_k;
        entries_.clear();
        live_count_ = 0;
        return success_k;
    }

#pragma endregion Modifiers

#pragma region Enumeration

    /**
     *  @brief Hands @p callback every key the newest published commit shows, in the core's own order.
     *
     *  The one walk an unordered core can offer, so it promises no ordering even where the core has one.
     *  Every key a @c find would answer with at the moment of the call is visited @b exactly @b once - a
     *  committed tombstone and every version no commit has published yet are both left out. Nothing may
     *  write to the store while the walk runs.
     *
     *  @param[in] callback Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
     */
    template <typename callback_type_ = no_op_t>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
        static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                      "callback must be noexcept invocable with value_t const &");
        for_each_at_(published_stamp_(), std::forward<callback_type_>(callback));
        return success_k;
    }

#pragma endregion Enumeration

#pragma region Range Operations

    /**
     *  @brief Publishes a tombstone over every key in [ @p lower, @p upper ), all under one stamp.
     *
     *  One stamp for the whole window is the point: a reader either names it or does not, so the window
     *  is never observed half erased, and a snapshot older than the stamp keeps reading every key it
     *  opened on. A tombstone per key rather than a detached run, for that same reason.
     *
     *  @param[in] lower Lower bound, inclusive.
     *  @param[in] upper Upper bound, exclusive.
     *  @param[in] callback Callback handed each element about to be tombstoned. Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store exactly as it was.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires ordered_core_k
    {
        return tombstone_walked_(
            [&](auto &&collect) noexcept {
                range_at_(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), published_stamp_(),
                          collect);
            },
            std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Publishes a tombstone over every key at or after @p lower, all under one stamp.
     *    One stamp for the whole window, for the reason @c erase_range() takes one.
     *
     *  @param[in] lower Lower bound, inclusive - a key equal to it is erased, which is the same end
     *    @c erase_range() includes.
     *  @param[in] callback Callback handed each element about to be tombstoned. Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store exactly as it was.
     */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires ordered_core_k
    {
        return tombstone_walked_(
            [&](auto &&collect) noexcept {
                range_from_at_(std::forward<lower_type_>(lower), published_stamp_(), collect);
            },
            std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Publishes a tombstone over every key before @p upper, all under one stamp.
     *    One stamp for the whole window, for the reason @c erase_range() takes one.
     *
     *  @param[in] upper Upper bound, exclusive - a key equal to it is kept, which is the same end
     *    @c erase_range() excludes.
     *  @param[in] callback Callback handed each element about to be tombstoned. Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store exactly as it was.
     */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires ordered_core_k
    {
        return tombstone_walked_(
            [&](auto &&collect) noexcept {
                range_up_to_at_(std::forward<upper_type_>(upper), published_stamp_(), collect);
            },
            std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Publishes a revised version of every key in [ @p lower, @p upper ), all under one stamp.
     *
     *  @p callback is handed a copy of each value rather than the stored one: under snapshot isolation
     *  the visible version is exactly what every open reader is reading, so mutating it in place would
     *  rewrite their history. That copy is also why this can run out of memory where its siblings cannot.
     *
     *  @param[in] lower Lower bound, inclusive.
     *  @param[in] upper Upper bound, exclusive.
     *  @param[in] callback Callback invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Success, or an allocation failure that leaves the store exactly as it was.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_mapping<value_t> && ordered_core_k
    {
        values_vector_t revised(values_allocator_t(storage_shape_t::allocator_of(entries_)));
        status_t collecting = success_k;
        [[maybe_unused]] status_t const walked = range(
            std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), [&](value_t const &value) noexcept {
                if (failed(collecting)) return;
                auto duplicate = copy_safely(value);
                if (!duplicate) {
                    collecting = duplicate.status();
                    return;
                }
                if (status_t const kept = revised.push_back(std::move(*duplicate)); failed(kept)) collecting = kept;
            });
        if (failed(collecting)) return collecting;
        if (revised.size() == 0) return success_k;

        auto opened = publication();
        if (!opened) return out_of_memory_heap_k;
        if (status_t const reserved = opened->reserve(revised.size()); failed(reserved)) return reserved;
        for (std::size_t index = 0; index != revised.size(); ++index) {
            value_t &revision = revised[index];
            callback(revision.key, revision.mapped);
            if (status_t const staged = opened->upsert(std::move(revision)); failed(staged)) return staged;
        }
        return opened->publish();
    }

#pragma endregion Range Operations

#pragma region Sampling

    /**
     *  @brief Draws one key uniformly from [ @p lower, @p upper ), handing it to @p callback.
     *
     *  Two passes over the visible walk rather than one tree descent, which is what snapshot isolation
     *  costs here: every version of a key is its own node, so a descent weighted by subtree size would
     *  draw a key in proportion to how much history it still carries. The walk knows which single
     *  version each key shows, so counting it is the only unbiased draw available.
     *
     *  @param[in] lower Lower bound, inclusive.
     *  @param[in] upper Upper bound, exclusive.
     *  @param[inout] generator Random number generator, such as @c std::mt19937.
     *  @param[in] callback Callback handed the drawn element. Must be @c noexcept. Never invoked when
     *    the window shows nothing.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                      callback_type_ &&callback) const noexcept
        requires ordered_core_k
    {
        std::size_t visible_count = 0;
        [[maybe_unused]] status_t const walked =
            range(lower, upper, [&](value_t const &) noexcept { ++visible_count; });
        if (visible_count == 0) return success_k;

        std::size_t matches_to_skip = draw_below(generator, visible_count);
        bool drawn = false;
        return range(lower, upper, [&](value_t const &element) noexcept {
            if (drawn) return;
            if (matches_to_skip) --matches_to_skip;
            else {
                callback(element);
                drawn = true;
            }
        });
    }

    /**
     *  @brief Samples up to @p reservoir_capacity keys uniformly from [ @p lower, @p upper ).
     *
     *  Unbiased over the keys the store shows rather than over the versions it holds, since the walk it
     *  draws from is the visible one.
     *
     *  @param[in] lower Lower bound, inclusive.
     *  @param[in] upper Upper bound, exclusive.
     *  @param[inout] generator Random number generator, such as @c std::mt19937.
     *  @param[inout] seen How many keys have been offered so far, which may span several calls.
     *  @param[in] reservoir_capacity How many samples the buffer holds.
     *  @param[out] reservoir Random-access iterator to that buffer.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                            std::size_t &seen, std::size_t reservoir_capacity,
                                            output_iterator_type_ &&reservoir) const noexcept
        requires ordered_core_k
    {
        static_assert(std::is_nothrow_copy_assignable_v<value_t>,
                      "sampling copies each drawn element into the caller's buffer, so that copy must not throw");
        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        return range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                     [&](value_t const &value) noexcept {
                         if (seen < reservoir_capacity) { reservoir[seen] = value; }
                         else {
                             auto const slot_to_replace = draw_below(generator, seen + 1);
                             if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = value;
                         }
                         ++seen;
                     });
    }

#pragma endregion Sampling

#pragma region Vacuuming

    /**
     *  @brief Frees every version no open transaction can still reach.
     *
     *  Only reachability is reclaimed, never anything observable: the version each snapshot reads
     *  stays, and what goes is a version some newer commit already masked, or a tombstone whose
     *  erasure predates every live snapshot - which answers exactly what absence answers.
     *
     *  @return How many versions were reclaimed, or the reason the sweep could not run. Never a bare
     *    zero standing for both, which is why the count travels inside @c expected.
     */
    [[nodiscard]] expected<std::size_t> vacuum() noexcept {
        if constexpr (ordered_core_k) {
            return vacuum_runs_(entries_.begin(), [](versioned_t const &) noexcept { return true; });
        }
        else {
            // Pruning a key is idempotent, so meeting the same key once per version it still carries
            // costs a probe and changes nothing after the first pass over it.
            generation_t const mark = low_water_mark_();
            std::size_t reclaimed = 0;
            for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor)
                reclaimed += prune_key_of_(*cursor, mark);
            return reclaimed;
        }
    }

    /**
     *  @brief Frees every unreachable version of every key ordered in [ @p lower, @p upper ).
     *
     *  The window is closed on whole runs: a run is judged by its first entry, which is where a bare
     *  key's @c lower_bound lands, so no key is left with half its history swept.
     *
     *  @return How many versions were reclaimed, or the reason the sweep could not run.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept
        requires ordered_core_k
    {
        auto const ordering = entries_.key_comp();
        return vacuum_runs_(entries_.lower_bound(std::forward<lower_type_>(lower)),
                            [&](versioned_t const &head) noexcept { return ordering.per_key_compare(head, upper); });
    }

#pragma endregion Vacuuming

  private:
    /**
     *  @brief Frees every unreachable version, run by run, from @p start until @p within rejects a run.
     *    @p within is asked of the run's first entry, before anything inside that run is erased.
     *  @return How many versions were reclaimed.
     */
    template <typename mutable_iterator_type_, typename within_type_>
    std::size_t vacuum_runs_(mutable_iterator_type_ start, within_type_ &&within) noexcept
        requires ordered_core_k
    {
        generation_t const mark = low_water_mark_();
        std::size_t reclaimed = 0;
        auto cursor = start;
        while (cursor != entries_.end()) {
            versioned_t const &head = *cursor;
            if (!within(head)) break;

            versioned_t const *newest = nullptr;
            auto run_end = cursor;
            while (run_end != entries_.end() && same_key_(*run_end, head)) {
                versioned_t const &version = *run_end;
                if (visible_at(version.committed, mark) &&
                    (!newest || stamp_of(newest->committed) < stamp_of(version.committed)))
                    newest = &version;
                ++run_end;
            }

            auto scan = cursor;
            while (scan != run_end) {
                if (!unreachable_(*scan, newest, mark)) {
                    ++scan;
                    continue;
                }
                scan = entries_.erase(scan).next;
                ++reclaimed;
            }
            cursor = run_end;
        }
        return reclaimed;
    }

    /**
     *  @brief Stages every element of [ @p first, @p last ) into one transaction and commits it.
     *    @p stage_one decides what each element means, which is the only difference between the
     *    strict, lenient, overwriting and updating bulk modifiers.
     */
    template <typename input_iterator_type_, typename stage_one_type_>
    [[nodiscard]] status_t commit_each_(input_iterator_type_ first, input_iterator_type_ last,
                                        stage_one_type_ &&stage_one) noexcept {
        if (first == last) return success_k;
        auto opened = transaction();
        if (!opened) return out_of_memory_heap_k;

        for (; first != last; ++first)
            if (status_t const staged = stage_one(*opened, value_t(*first)); failed(staged)) return staged;
        if (status_t const staged = opened->stage(); failed(staged)) return staged;
        return opened->commit();
    }

    /**
     *  @brief Publishes @p versioned under a fresh stamp and prunes what it masked.
     *    Direct writes take a generation of their own, so they never collide with a staged version
     *    an open transaction still owns.
     */
    [[nodiscard]] status_t publish_directly_(identifier_t const &identifier, versioned_t &&versioned) noexcept {
        generation_t const generation = next_generation_();
        versioned.generation = generation;
        if (failed(storage_shape_t::prepare(entries_, 1))) return out_of_memory_heap_k;
        if (failed(storage_shape_t::upsert(entries_, std::move(versioned)))) return out_of_memory_heap_k;

        stamp_as_one_commit_(&identifier, 1, generation);
        prune_each_(&identifier, 1);
        return success_k;
    }
};

#pragma region Aliases

/**
 *  @brief Snapshot-isolated transactional set backed by an AVL tree.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using snapshot_avl_set = snapshot_store<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief Snapshot-isolated transactional map backed by an AVL tree.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using snapshot_avl_map =
    snapshot_store<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief Snapshot-isolated transactional set backed by a weight-balanced tree.
 *    The only alias carrying @c select and @c rank, since only this core sums a second per-subtree count.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using snapshot_wb_set =
    snapshot_store<basic_wb_tree<value_type_, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/**
 *  @brief Snapshot-isolated transactional map backed by a weight-balanced tree.
 *    The only alias carrying @c select and @c rank, since only this core sums a second per-subtree count.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using snapshot_wb_map = snapshot_store<
    basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/**
 *  @brief Snapshot-isolated transactional set backed by an open-addressed hash table.
 *    Point access only - no bounds or ranges, as the core supplies no ordering.
 *
 *  @tparam key_type_ Type of elements stored in the set.
 *  @tparam hasher_type_ Hasher for placing elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam equals_type_ Equality for resolving collisions. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for the table's slabs, defaults to @c std::allocator.
 */
template <typename key_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = std::allocator<std::byte>>
using snapshot_hash_set = snapshot_store<basic_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>>;

/**
 *  @brief Snapshot-isolated transactional map backed by an open-addressed hash table.
 *    Point access only - no bounds or ranges, as the core supplies no ordering.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam hasher_type_ Hasher for placing keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam equals_type_ Equality for resolving collisions. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for the table's slabs, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename hasher_type_ = default_hash_t,
          typename equals_type_ = equal_to_t, typename allocator_type_ = std::allocator<std::byte>>
using snapshot_hash_map =
    snapshot_store<basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>>;

/**
 *  @brief Snapshot isolation with the read set validated at commit as well as the write set.
 *
 *  That is serializability: every committed history is equivalent to some serial one. A transaction
 *  commit here already waits for its own publication, so what the strict rung adds reaches only a
 *  sharded commit and a write made outside a transaction.
 */
template <typename collection_type_>
using serializable_store = snapshot_store<collection_type_, isolation_t::serializable_k>;

/** @brief Serializable transactional set backed by an AVL tree. */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using serializable_avl_set = serializable_store<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

/** @brief Serializable transactional map backed by an AVL tree. */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using serializable_avl_map =
    serializable_store<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/** @brief Serializable transactional set backed by a weight-balanced tree, so it answers ordinals too. */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using serializable_wb_set =
    serializable_store<basic_wb_tree<value_type_, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/** @brief Serializable transactional map backed by a weight-balanced tree, so it answers ordinals too. */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using serializable_wb_map = serializable_store<
    basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/** @brief Serializable transactional set backed by an open-addressed table, so it keeps no ordering. */
template <typename value_type_, typename hasher_type_ = hash<value_type_>, typename equals_type_ = equal_to_t,
          typename allocator_type_ = std::allocator<std::byte>>
using serializable_hash_set =
    serializable_store<basic_hash_table<value_type_, hasher_type_, equals_type_, allocator_type_>>;

/** @brief Serializable transactional map backed by an open-addressed table, so it keeps no ordering. */
template <typename key_type_, typename value_type_, typename hasher_type_ = hash<key_type_>,
          typename equals_type_ = equal_to_t, typename allocator_type_ = std::allocator<std::byte>>
using serializable_hash_map =
    serializable_store<basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>>;

/**
 *  @brief Serializable, and ordered in real time as well: a transaction opening after a commit returned
 *    sees that commit.
 *
 *  Refuses exactly what @c serializable_store refuses - the two differ only in when a commit becomes
 *  visible. A sharded commit and a write made outside a transaction wait for their own publication
 *  here, which a transaction commit does at either level, and the wait costs however long an older
 *  overlapping commit takes to finish writing itself out.
 */
template <typename collection_type_>
using strict_serializable_store = snapshot_store<collection_type_, isolation_t::strict_serializable_k>;

/** @brief Strictly serializable transactional set backed by an AVL tree. */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using strict_serializable_avl_set =
    strict_serializable_store<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

/** @brief Strictly serializable transactional map backed by an AVL tree. */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using strict_serializable_avl_map =
    strict_serializable_store<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/** @brief Strictly serializable transactional set backed by a weight-balanced tree, so it answers ordinals. */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using strict_serializable_wb_set =
    strict_serializable_store<basic_wb_tree<value_type_, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/** @brief Strictly serializable transactional map backed by a weight-balanced tree, so it answers ordinals. */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using strict_serializable_wb_map = strict_serializable_store<
    basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_, liveness_augmentation_t>>;

/** @brief Strictly serializable transactional set backed by an open-addressed table, so it keeps no ordering. */
template <typename value_type_, typename hasher_type_ = hash<value_type_>, typename equals_type_ = equal_to_t,
          typename allocator_type_ = std::allocator<std::byte>>
using strict_serializable_hash_set =
    strict_serializable_store<basic_hash_table<value_type_, hasher_type_, equals_type_, allocator_type_>>;

/** @brief Strictly serializable transactional map backed by an open-addressed table, so it keeps no ordering. */
template <typename key_type_, typename value_type_, typename hasher_type_ = hash<key_type_>,
          typename equals_type_ = equal_to_t, typename allocator_type_ = std::allocator<std::byte>>
using strict_serializable_hash_map = strict_serializable_store<
    basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
