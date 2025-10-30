#include <smashtable/transactional_std_store.hpp>
#include <smashtable/transactional_binary_tree.hpp>
#include <smashtable/locked_collection.hpp>
#include <smashtable/partitioned_collection.hpp>

#define macro_concat_(prefix, suffix) prefix##suffix
#define macro_concat(prefix, suffix) macro_concat_(prefix, suffix)
#define _ [[maybe_unused]] auto macro_concat(_, __LINE__)

using namespace ashvardanian::smashtable;

template <typename container_type_>
void api() {
    using element_t = typename container_type_::element_t;
    using identifier_t = typename container_type_::identifier_t;

    // Head state
    auto container = *container_type_::make();
    _ = container.upsert(element_t {});
    container.find(identifier_t {}, [](element_t const &) noexcept {}, []() noexcept {});
    container.upper_bound(identifier_t {}, [](element_t const &) noexcept {}, []() noexcept {});
    container.range(identifier_t {}, identifier_t {}, [](element_t const &) noexcept {});
    container.erase_range(identifier_t {}, identifier_t {}, [](element_t const &) noexcept {});
    _ = container.clear();
    _ = container.size();

    // Transactions
    auto txn = *container.transaction();
    _ = txn.upsert(element_t {});
    _ = txn.watch(identifier_t {});
    _ = txn.erase(identifier_t {});
    txn.find(identifier_t {}, [](element_t const &) noexcept {}, []() noexcept {});
    txn.upper_bound(identifier_t {}, [](element_t const &) noexcept {}, []() noexcept {});
    _ = txn.stage();
    _ = txn.rollback();
    _ = txn.commit();
    _ = txn.reset();

    // Machine Learning
    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    container.sample_range( //
        identifier_t {}, identifier_t {}, random_generator, [](element_t const &) noexcept {});

    std::size_t count_seen = 0;
    std::array<element_t, 16> reservoir;
    container.sample_range( //
        identifier_t {}, identifier_t {}, random_generator, count_seen, reservoir.size(), reservoir.data());

    // Exports
    element_t result;
    container.find(identifier_t {}, copy_to(result), no_op_fn_t {});
}

using pair_t = mapping<std::size_t, std::size_t>;

struct pair_compare_t {
    using value_type = std::size_t;
    bool operator()(pair_t const &a, pair_t const &b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t const &b) const noexcept { return a < b.key; }
    bool operator()(pair_t const &a, std::size_t b) const noexcept { return a.key < b; }
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
