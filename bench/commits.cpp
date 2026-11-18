/**
 *  @file bench/commits.cpp
 *  @author Ash Vardanian
 *  @date November 17, 2026
 *  @brief Benchmark of one update spanning two containers per commit, from every thread asked
 *      for: two maps behind one mutex against two partitioned stores joined by a group.
 *
 *  Both sides keep the pair consistent for a reader, the mutex by excluding it and the group by
 *  publishing both stores under one stamp of the order they share. Every thread updates keys
 *  drawn uniformly from the same distinct keys, each from a stream of its own. The settings it
 *  reads are listed in CONTRIBUTING.md.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <map>    // `std::map`
#include <mutex>  // `std::mutex`, `std::lock_guard`
#include <span>   // `std::span`
#include <string> // `std::string`

#include <forkunion.hpp>

#include "smashtable/basic_vector.hpp"
#include "smashtable/partitioned_store.hpp"
#include "smashtable/snapshot_store.hpp"

#include "harness.hpp"

namespace fu = ashvardanian::forkunion;

namespace ashvardanian::smashtable::bench {

using key_t = std::uint64_t;
using store_t = partitioned_map<snapshot_avl_map<key_t, key_t>>;

/** The mutex and the commit order with every shared word owned through ForkUnion's reference for
 *  this build, and every wait handed to its waiter. */
using forkunion_mutex_t = spin_shared_mutex<fu::preferred_yield_t, fu::preferred_atomic_ref>;
using forkunion_order_t =
    basic_commit_order<16, 4, order_sharing_t::shared_k, fu::preferred_yield_t, fu::preferred_atomic_ref>;

static_assert(extended_atomic_ref<fu::preferred_atomic_ref<std::uint64_t>, std::uint64_t>,
              "a reference short of the concept would pass as a ForkUnion row over the fallback loops");
static_assert(extended_atomic_ref<fu::preferred_atomic_ref<std::uint32_t>, std::uint32_t>,
              "the mutex word is narrower than a stamp, and must reach the same verbs");

/** Two partitioned stores locked, ordered and waited on through ForkUnion. */
using forkunion_store_t =
    partitioned_map<snapshot_avl_map<key_t, key_t>, hash<key_t>, forkunion_mutex_t, 16, forkunion_order_t>;

double bench_std_maps(environment_t const &env, std::span<key_t const> keys) {
    std::string const name = "commits.std_maps";
    if (!env.settings.regex_filter.selects(name)) return 0;
    std::map<key_t, key_t> forward, backward;
    std::mutex mutex;
    for (key_t const key : keys) forward[key] = 0, backward[key] = 0;
    return run(env, name, 0, [&](loop_t &loop) noexcept {
        split_mix_generator_t generator {stream_key(env.settings.seed, name, loop.thread())};
        for (std::size_t call : loop) {
            key_t const key = keys[generator.below(keys.size())];
            std::lock_guard<std::mutex> lock {mutex};
            forward.find(key)->second = call, backward.find(key)->second = call;
        }
    });
}

template <typename store_type_>
void bench_transaction_group(environment_t const &env, std::string const &name, std::span<key_t const> keys,
                             double std_maps) {
    if (!env.settings.regex_filter.selects(name)) return;
    typename store_type_::order_t order;
    store_type_ forward {order}, backward {order};
    for (key_t const key : keys)
        if (failed(forward.upsert({key, 0})) || failed(backward.upsert({key, 0})))
            return print_line(stdout, "{:<64} skipped: needs memory for {} keys", name, keys.size());
    run(env, name, std_maps, [&](loop_t &loop) noexcept {
        split_mix_generator_t generator {stream_key(env.settings.seed, name, loop.thread())};
        auto group = make_transaction_group(forward, backward);
        if (!group) return loop.skip(name_of(group.status()));
        std::size_t aborts = 0, calls = 0;
        for (std::size_t call : loop) {
            if (call == 0) aborts = 0;
            calls = call + 1;
            key_t const key = keys[generator.below(keys.size())];
            status_t committed = success_k;
            while (conflicted(committed = group->commit_with([&](auto &forward_writes, auto &backward_writes) noexcept {
                status_t const filed = forward_writes.upsert({key, call});
                return failed(filed) ? filed : backward_writes.upsert({key, call});
            })))
                ++aborts;
            if (failed(committed)) return loop.skip(name_of(committed));
        }
        loop.rate("aborts", calls ? static_cast<double>(aborts) / calls : 0);
    });
}

void bench_commits(environment_t const &env) {
    basic_vector<key_t> keys;
    if (failed(keys.resize(env.settings.keys)))
        return print_line(stdout, "{:<64} skipped: needs memory for {} keys", "commits", env.settings.keys);
    // SplitMix's finalizer and an odd multiplier are both bijections, so every key is distinct.
    split_mix_generator_t generator {stream_key(env.settings.seed, "commits")};
    key_t const stride = generator() | 1;
    for (std::size_t index = 0; index != keys.size(); ++index) keys[index] = split_mix_bits(index) * stride;
    std::span<key_t const> const key_span(keys.data(), keys.size());

    print_line(stdout,
               "\nOne update across two containers per commit over {} keys, the speed-up against std::maps last",
               env.settings.keys);
    double const std_maps = bench_std_maps(env, key_span);
    bench_transaction_group<store_t>(env, "commits.transaction_group<partitioned_map<snapshot_avl_map>>", key_span,
                                     std_maps);
    bench_transaction_group<forkunion_store_t>(
        env, "commits.transaction_group<partitioned_map<snapshot_avl_map>,forkunion>", key_span, std_maps);
}

} // namespace ashvardanian::smashtable::bench

using namespace ashvardanian::smashtable::bench;

int main(int argc, char **argv) {
    environment_t const env {read_bench_settings(argc, argv), probe_machine()};
    print(env.machine);
    print(env.settings);
    bench_commits(env);
    return 0;
}
