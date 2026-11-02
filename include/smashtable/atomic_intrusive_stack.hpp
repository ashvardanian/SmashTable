/**
 *  @file include/smashtable/atomic_intrusive_stack.hpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief A stack any thread pushes records onto and one thread drains whole, in one pointer word.
 *
 *  Records link through their own @c next pointers, so a push allocates nothing. A push lands with
 *  a releasing compare-exchange and the drain with an acquiring exchange to null, which hands the
 *  taker every record and every link written before it was pushed. Nothing pops a single record,
 *  so a record cannot leave and come back between a pusher's read and its exchange, and the
 *  compare-exchange has no ABA to guard against.
 */
#pragma once
#include <concepts> // `std::same_as`

#include "smashtable/shared.hpp"

namespace ashvardanian::smashtable {

/** A record naming the one pushed before it through a @c next pointer of its own type. */
template <typename record_type_>
concept pointer_linked = requires(record_type_ &record) {
    { record.next } -> std::same_as<record_type_ *&>;
};

/**
 *  @brief The newest of a stack of records linked newest first, pushed from any thread and taken
 *      whole by one.
 *
 *  @tparam record_type_ The record, linked through its @c next pointer.
 *  @tparam atomic_reference_ The reference the top word is owned through for one operation.
 */
template <pointer_linked record_type_, template <typename> class atomic_reference_ = atomic_ref>
class atomic_intrusive_stack {
  public:
    using record_t = record_type_;

  private:
    record_t *newest_ {nullptr};

  public:
    /** Pushes @p record, which no other thread may reach until it is taken. */
    void push(record_t &record) noexcept { push_chain(record, record); }

    /** Pushes the records linked from @p newest down to @p oldest, whose @c next is overwritten,
     *  in one step, so the taker meets them together and in that order. */
    void push_chain(record_t &newest, record_t &oldest) noexcept {
        atomic_reference_<record_t *> top(newest_);
        record_t *observed = top.load(memory_order_relaxed_k);
        // A plain exchange would publish the chain before its last link points at what was below.
        do oldest.next = observed;
        while (!top.compare_exchange_weak(observed, &newest, memory_order_release_k, memory_order_relaxed_k));
    }

    /** Takes every record pushed so far, the newest first and linked through @c next, or null. */
    [[nodiscard]] record_t *take_all() noexcept {
        static_assert(sizeof(atomic_intrusive_stack) == sizeof(record_t *), "one pointer word");
        return atomic_reference_<record_t *>(newest_).exchange(nullptr, memory_order_acquire_k);
    }
};

} // namespace ashvardanian::smashtable
