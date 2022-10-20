#include <consistent_set/consistent_set.hpp>
#include <consistent_set/consistent_avl.hpp>
#include <consistent_set/versioning_avl.hpp>
#include <consistent_set/locked.hpp>
#include <consistent_set/partitioned.hpp>

#define macro_concat_(prefix, suffix) prefix##suffix
#define macro_concat(prefix, suffix) macro_concat_(prefix, suffix)
#define _ [[maybe_unused]] auto macro_concat(_, __LINE__)

using namespace av;

template <typename container_at>
void api() {
    using element_t = typename container_at::element_t;
    using identifier_t = typename container_at::identifier_t;

    // Head state
    auto container = *container_at::make();
    _ = container.upsert(element_t {});
    _ = container.find(
        identifier_t {},
        [](element_t const&) noexcept {},
        []() noexcept {});
    _ = container.upper_bound(
        identifier_t {},
        [](element_t const&) noexcept {},
        []() noexcept {});
    _ = container.equal_range(identifier_t {}, [](element_t const&) noexcept {});
    _ = container.erase_equal_range(identifier_t {}, [](element_t const&) noexcept {});
    _ = container.clear();
    _ = container.size();

    // Transactions
    auto txn = *container.transaction();
    _ = txn.upsert(element_t {});
    _ = txn.watch(identifier_t {});
    _ = txn.erase(identifier_t {});
    _ = txn.find(
        identifier_t {},
        [](element_t const&) noexcept {},
        []() noexcept {});
    _ = txn.upper_bound(
        identifier_t {},
        [](element_t const&) noexcept {},
        []() noexcept {});
    _ = txn.stage();
    _ = txn.rollback();
    _ = txn.commit();
    _ = txn.reset();

    // Exports
    element_t result;
    _ = container.find(identifier_t {}, copy_to(result), no_op_t {});
}

struct pair_t {
    std::size_t key;
    std::size_t value;

    pair_t(std::size_t key = 0, std::size_t value = 0) noexcept : key(key), value(value) {}
    explicit operator std::size_t() const noexcept { return key; }
};

struct pair_compare_t {
    using value_type = std::size_t;
    bool operator()(pair_t a, pair_t b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t b) const noexcept { return a < b.key; }
    bool operator()(pair_t a, std::size_t b) const noexcept { return a.key < b; }
};

int main() {

    using stl_t = consistent_set_gt<pair_t, pair_compare_t>;
    api<stl_t>();
    api<locked_gt<stl_t>>();
    api<partitioned_gt<stl_t>>();

    // using avl_t = consistent_avl_gt<pair_t, pair_compare_t>;
    // api<avl_t>();
    // api<locked_gt<avl_t>>();
    // api<partitioned_gt<avl_t>>();

    // using mvcc_t = consistent_set_gt<pair_t, pair_compare_t>;
    // api<mvcc_t>();
    // api<locked_gt<mvcc_t>>();
    // api<partitioned_gt<mvcc_t>>();

    return 0;
}