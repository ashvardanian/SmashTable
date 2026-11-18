/**
 *  @file bench/isolation.cpp
 *  @author Ash Vardanian
 *  @date November 17, 2026
 *  @brief Benchmark of what each isolation level costs one read-modify-write transaction, from
 *      every thread asked for, over the same partitioned AVL core.
 *
 *  Each transaction reads a key, writes it back incremented and commits, retrying from a fresh
 *  snapshot until a commit lands, so only landed commits count and every refused one is reported
 *  as an abort. Every thread draws keys uniformly from the same distinct keys, each from a stream
 *  of its own. The settings it reads are listed in CONTRIBUTING.md.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <span>   // `std::span`
#include <string> // `std::string`

#include <forkunion.hpp>

#include "smashtable/basic_vector.hpp"
#include "smashtable/monotonic_store.hpp"
#include "smashtable/partitioned_store.hpp"
#include "smashtable/snapshot_store.hpp"

#include "harness.hpp"

namespace fu = ashvardanian::forkunion;

namespace ashvardanian::smashtable::bench {

using key_t = std::uint64_t;

/** The mutex and the commit order with every shared word owned through ForkUnion's reference for
 *  this build, and every wait handed to its waiter. */
using forkunion_mutex_t = spin_shared_mutex<fu::preferred_yield_t, fu::preferred_atomic_ref>;
using forkunion_order_t =
    basic_commit_order<16, 4, order_sharing_t::shared_k, fu::preferred_yield_t, fu::preferred_atomic_ref>;

static_assert(extended_atomic_ref<fu::preferred_atomic_ref<std::uint64_t>, std::uint64_t>,
              "a reference short of the concept would pass as a ForkUnion row over the fallback loops");
static_assert(extended_atomic_ref<fu::preferred_atomic_ref<std::uint32_t>, std::uint32_t>,
              "the mutex word is narrower than a stamp, and must reach the same verbs");

/** A partitioned map locked, ordered and waited on through ForkUnion, whichever store it shards. */
template <typename store_type_>
using forkunion_partitioned_map = partitioned_map<store_type_, hash<key_t>, forkunion_mutex_t, 16, forkunion_order_t>;

using counted_allocator_t = counting_allocator<mapping<key_t, key_t>>;
using monotonic_map_t = monotonic_avl_map<key_t, key_t, less_t, counted_allocator_t>;
using snapshot_map_t = snapshot_avl_map<key_t, key_t, less_t, counted_allocator_t>;
using serializable_map_t = serializable_avl_map<key_t, key_t, less_t, counted_allocator_t>;
using strict_serializable_map_t = strict_serializable_avl_map<key_t, key_t, less_t, counted_allocator_t>;

/** The monotonic store drawing its generations and stamps through ForkUnion's reference as well. */
using forkunion_monotonic_map_t =
    monotonic_avl_map<key_t, key_t, less_t, counted_allocator_t, fu::preferred_atomic_ref>;

template <typename store_type_>
double bench_increments(environment_t const &env, std::string const &name, std::span<key_t const> keys,
                        double baseline) {
    if (!env.settings.regex_filter.selects(name)) return 0;
    store_type_ store;
    for (key_t const key : keys)
        if (failed(store.upsert({key, 0}))) {
            print_line(stdout, "{:<64} skipped: needs memory for {} keys", name, keys.size());
            return 0;
        }
    double const loaded_bytes_per_key = live_bytes_per_key(keys.size());
    return run(env, name, baseline, [&](loop_t &loop) noexcept {
        split_mix_generator_t generator {stream_key(env.settings.seed, name, loop.thread())};
        auto transaction = store.transaction();
        if (!transaction) return loop.skip(name_of(transaction.status()));
        std::size_t aborts = 0, calls = 0;
        for (std::size_t call : loop) {
            if (call == 0) aborts = 0;
            calls = call + 1;
            key_t const key = keys[generator.below(keys.size())];
            std::size_t attempts = 0;
            status_t const committed = commit_with_retries(*transaction, ~std::size_t {0}, [&](auto &writes) noexcept {
                ++attempts;
                key_t count = 0;
                status_t const read = writes.find(key, [&](auto const &entry) noexcept { count = entry.mapped; });
                return failed(read) ? read : writes.upsert({key, count + 1});
            });
            if (failed(committed)) return loop.skip(name_of(committed));
            aborts += attempts - 1;
        }
        loop.rate("aborts", calls ? static_cast<double>(aborts) / calls : 0);
        loop.counter("loaded bytes/key", loaded_bytes_per_key);
        loop.counter("timed bytes/key", live_bytes_per_key(keys.size()));
    });
}

void bench_isolation(environment_t const &env) {
    basic_vector<key_t> keys;
    if (failed(keys.resize(env.settings.keys)))
        return print_line(stdout, "{:<64} skipped: needs memory for {} keys", "isolation", env.settings.keys);
    // SplitMix's finalizer and an odd multiplier are both bijections, so every key is distinct.
    split_mix_generator_t generator {stream_key(env.settings.seed, "isolation")};
    key_t const stride = generator() | 1;
    for (std::size_t index = 0; index != keys.size(); ++index) keys[index] = split_mix_bits(index) * stride;
    std::span<key_t const> const key_span(keys.data(), keys.size());

    print_line(stdout, "\nRead, increment and commit one of {} keys, the speed-up against monotonic last",
               env.settings.keys);
    double const monotonic = bench_increments<partitioned_map<monotonic_map_t>>(
        env, "isolation.partitioned_map<monotonic_avl_map>", key_span, 0);
    bench_increments<forkunion_partitioned_map<forkunion_monotonic_map_t>>(
        env, "isolation.partitioned_map<monotonic_avl_map,forkunion>", key_span, monotonic);
    bench_increments<partitioned_map<snapshot_map_t>>(env, "isolation.partitioned_map<snapshot_avl_map>", key_span,
                                                      monotonic);
    bench_increments<forkunion_partitioned_map<snapshot_map_t>>(
        env, "isolation.partitioned_map<snapshot_avl_map,forkunion>", key_span, monotonic);
    bench_increments<partitioned_map<serializable_map_t>>(env, "isolation.partitioned_map<serializable_avl_map>",
                                                          key_span, monotonic);
    bench_increments<forkunion_partitioned_map<serializable_map_t>>(
        env, "isolation.partitioned_map<serializable_avl_map,forkunion>", key_span, monotonic);
    bench_increments<partitioned_map<strict_serializable_map_t>>(
        env, "isolation.partitioned_map<strict_serializable_avl_map>", key_span, monotonic);
    bench_increments<forkunion_partitioned_map<strict_serializable_map_t>>(
        env, "isolation.partitioned_map<strict_serializable_avl_map,forkunion>", key_span, monotonic);
}

} // namespace ashvardanian::smashtable::bench

using namespace ashvardanian::smashtable::bench;

int main(int argc, char **argv) {
    environment_t const env {read_bench_settings(argc, argv), probe_machine()};
    print(env.machine);
    print(env.settings);
    bench_isolation(env);
    return 0;
}
