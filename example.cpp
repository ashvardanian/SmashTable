/**
 *  @brief Compile-time tour of the API every container family shares.
 *  @author Ash Vardanian
 *  @file example.cpp
 *  @date October 16, 2022
 *
 *  Instantiating @c api() over each container is the point: the store, both trees and both
 *  thread-safety wrappers have to answer the same calls, so a surface that drifts between them
 *  fails here rather than in a binding. Nothing is asserted, and nothing needs to be.
 */
#include <smashtable/locked_collection.hpp>
#include <smashtable/partitioned_collection.hpp>
#include <smashtable/transactional_store.hpp>
#include <smashtable/transactional_std_store.hpp>

#define macro_concat_(prefix, suffix) prefix##suffix
#define macro_concat(prefix, suffix) macro_concat_(prefix, suffix)
#define _ [[maybe_unused]] auto macro_concat(_, __LINE__)

using namespace ashvardanian::smashtable;

template <typename container_type_>
void api() {
    using value_t = typename container_type_::value_t;
    using identifier_t = typename container_type_::identifier_t;

    // Head state
    auto container = *container_type_::make();
    _ = container.upsert(value_t {});
    container.find(identifier_t {}, [](value_t const &) noexcept {}, []() noexcept {});
    container.upper_bound(identifier_t {}, [](value_t const &) noexcept {}, []() noexcept {});
    container.range(identifier_t {}, identifier_t {}, [](value_t const &) noexcept {});
    container.erase_range(identifier_t {}, identifier_t {}, [](value_t const &) noexcept {});
    _ = container.clear();
    _ = container.size();

    // Transactions
    auto transaction = *container.transaction();
    _ = transaction.upsert(value_t {});
    _ = transaction.watch(identifier_t {});
    _ = transaction.erase(identifier_t {});
    transaction.find(identifier_t {}, [](value_t const &) noexcept {}, []() noexcept {});
    transaction.upper_bound(identifier_t {}, [](value_t const &) noexcept {}, []() noexcept {});
    _ = transaction.stage();
    _ = transaction.rollback();
    _ = transaction.commit();
    _ = transaction.reset();

    // Machine Learning
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    container.sample_range( //
        identifier_t {}, identifier_t {}, random_generator, [](value_t const &) noexcept {});

    std::size_t count_seen = 0;
    std::array<value_t, 16> reservoir;
    container.sample_range( //
        identifier_t {}, identifier_t {}, random_generator, count_seen, reservoir.size(), reservoir.data());

    // Exports
    value_t result;
    container.find(identifier_t {}, copy_to(result), no_op_fn_t {});
}

using pair_t = mapping<std::size_t, std::size_t>;

struct pair_compare_t {
    using is_transparent = void;
    using value_type = std::size_t;
    bool operator()(pair_t const &a, pair_t const &b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t const &b) const noexcept { return a < b.key; }
    bool operator()(pair_t const &a, std::size_t b) const noexcept { return a.key < b; }
    bool operator()(std::size_t a, std::size_t b) const noexcept { return a < b; }
};

int main() {

    using stl_t = transactional_std_store<pair_t, pair_compare_t>;
    api<stl_t>();
    api<locked_collection<stl_t>>();
    api<partitioned_collection<stl_t>>();

    using avl_t = transactional_avl_set<pair_t, pair_compare_t>;
    api<avl_t>();
    api<locked_collection<avl_t>>();
    api<partitioned_collection<avl_t>>();

    return 0;
}
