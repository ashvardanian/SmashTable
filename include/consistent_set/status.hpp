#pragma once
#include <cstdint>      //
#include <system_error> // `ENOMEM`

namespace av {

enum consistent_set_errc_t {
    success_k = 0,
    unknown_k = -1,

    consistency_k,
    transaction_not_recoverable_k = ENOTRECOVERABLE,
    sequence_number_overflow_k = EOVERFLOW,

    out_of_memory_heap_k = ENOMEM,
    out_of_memory_arena_k = ENOBUFS,
    out_of_memory_disk_k = ENOSPC,

    invalid_argument_k = EINVAL,
    operation_in_progress_k = EINPROGRESS,
    operation_not_permitted_k = EPERM,
    operation_not_supported_k = EOPNOTSUPP,
    operation_would_block_k = EWOULDBLOCK,
    operation_canceled_k = ECANCELED,

    connection_broken_k = EPIPE,
    connection_aborted_k = ECONNABORTED,
    connection_already_in_progress_k = EALREADY,
    connection_refused_k = ECONNREFUSED,
    connection_reset_k = ECONNRESET,

};

/**
 * @brief Wraps error-codes into bool-convertible conditions.
 * @see @c consistent_set_errc_t.
 */
struct consistent_set_status_t {
    consistent_set_errc_t errc = consistent_set_errc_t::success_k;
    constexpr operator bool() const noexcept { return errc == consistent_set_errc_t::success_k; }
};

struct no_op_t {
    constexpr void operator()() const noexcept {}
    template <typename at>
    constexpr void operator()(at&&) const noexcept {}
};

template <typename element_at>
struct copy_to_gt {
    element_at& target;
    template <typename at>
    void operator()(at&& source) const noexcept {
        target = std::forward<at>(source);
    }
};

template <typename element_at>
copy_to_gt<element_at> copy_to(element_at& element) noexcept {
    return {element};
}

template <typename element_at, typename comparator_at>
struct element_versioning_gt {

    using element_t = element_at;
    using comparator_t = comparator_at;

    using identifier_t = typename comparator_t::value_type;
    using generation_t = std::int64_t;
    using status_t = consistent_set_status_t;

    static_assert(!std::is_reference<element_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_copy_constructible<identifier_t>(), "To WATCH, the ID must be safe to copy.");
    static_assert(std::is_nothrow_default_constructible<element_t>(), "We need an empty state.");
    static_assert(std::is_nothrow_move_constructible<element_t>() && std::is_nothrow_move_assignable<element_t>(),
                  "To make all the methods `noexcept`, the moves must be safe too.");

    struct dated_identifier_t {
        identifier_t id;
        generation_t generation {0};
    };

    struct watch_t {
        generation_t generation {0};
        bool deleted {false};

        bool operator==(watch_t const& watch) const noexcept {
            return watch.deleted == deleted && watch.generation == generation;
        }
        bool operator!=(watch_t const& watch) const noexcept {
            return watch.deleted != deleted || watch.generation != generation;
        }
    };

    struct watched_identifier_t {
        identifier_t id;
        watch_t watch;
    };

    struct entry_t {
        mutable element_t element;
        mutable generation_t generation {0};
        mutable bool deleted {false};
        mutable bool visible {true};

        entry_t() = default;
        entry_t(entry_t&&) noexcept = default;
        entry_t& operator=(entry_t&&) noexcept = default;
        entry_t(entry_t const&) noexcept = delete;
        entry_t& operator=(entry_t const&) noexcept = delete;
        entry_t(element_t&& element) noexcept : element(std::move(element)) {}

        operator element_t const&() const& noexcept { return element; }
        bool operator==(watch_t const& watch) const noexcept {
            return watch.deleted == deleted && watch.generation == generation;
        }
        bool operator!=(watch_t const& watch) const noexcept {
            return watch.deleted != deleted || watch.generation != generation;
        }
    };

    template <typename at>
    constexpr static bool knows_generation() {
        using t = std::remove_reference_t<at>;
        return std::is_same<t, entry_t>() || std::is_same<t, dated_identifier_t>();
    }

    struct entry_comparator_t {
        using is_transparent = void;

        template <typename at>
        decltype(auto) comparable(at const& object) const noexcept {
            using t = std::remove_reference_t<at>;
            if constexpr (std::is_same<t, entry_t>())
                return (element_t const&)object.element;
            else if constexpr (std::is_same<t, dated_identifier_t>())
                return (identifier_t const&)object.id;
            else
                return (t const&)object;
        }

        template <typename first_at, typename second_at>
        bool dated_compare(first_at const& a, second_at const& b) const noexcept {
            comparator_t less;
            auto a_less_b = less(comparable(a), comparable(b));
            auto b_less_a = less(comparable(b), comparable(a));
            return !a_less_b && !b_less_a ? a.generation < b.generation : a_less_b;
        }

        template <typename first_at, typename second_at>
        bool native_compare(first_at const& a, second_at const& b) const noexcept {
            return comparator_t {}(comparable(a), comparable(b));
        }

        template <typename first_at, typename second_at>
        bool less(first_at const& a, second_at const& b) const noexcept {
            using first_t = std::remove_reference_t<first_at>;
            using second_t = std::remove_reference_t<second_at>;
            if constexpr (knows_generation<first_t>() && knows_generation<second_t>())
                return dated_compare(a, b);
            else
                return native_compare(a, b);
        }

        template <typename first_at, typename second_at>
        bool operator()(first_at const& a, second_at const& b) const noexcept {
            return less(a, b);
        }

        template <typename first_at, typename second_at>
        bool same(first_at const& a, second_at const& b) const noexcept {
            return !less(a, b) && !less(b, a);
        }
    };
};

} // namespace av