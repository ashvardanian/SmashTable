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

} // namespace av