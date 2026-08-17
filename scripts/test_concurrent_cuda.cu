/**
 *  @brief Device-side suites for the pinned lock-free hash table, over one table in managed memory.
 *  @author Ash Vardanian
 *  @file scripts/test_concurrent_cuda.cu
 *  @date August 17, 2026
 *
 *  @section test_concurrent_cuda_what_is_tested What Is Tested
 *
 *  The point is not that @c concurrent_hash_table compiles for a device - it is that one table, in one
 *  allocation, stays correct while a kernel and the host both operate on it. So every suite writes
 *  through one side and verifies through the other, and the last one has both running at once.
 *
 *  @section test_concurrent_cuda_helpers What The Caller Supplies
 *
 *  The library's defaults reach @c std::hash and @c std::allocator, neither of which a device has, so
 *  the helpers below stand in. They are ordinary @c constexpr callables rather than annotated device
 *  functions, which is the same rule the headers follow.
 *
 *  @section test_concurrent_cuda_scheduling Hardware Requirements
 *
 *  The slot spin needs independent thread scheduling, so @c sm_70 and newer. The mixed host-and-device
 *  suite additionally needs concurrent managed access, which the runtime is asked about rather than
 *  assumed - a device without it skips that one suite instead of faulting.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::printf`

#include <thread> // `std::thread`
#include <vector> // `std::vector`

#include <smashtable/concurrent_hash_table.hpp>

#include "test.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Device Helpers

/** @brief Aborts the process when a CUDA call reports a failure, naming the call site. */
#define st_verify_cuda_(call)                                                                                 \
    do {                                                                                                      \
        cudaError_t const error = (call);                                                                     \
        if (error != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA failure: %s, %s, %s:%d\n", #call, cudaGetErrorString(error), __FILE__, \
                         __LINE__);                                                                           \
            std::abort();                                                                                     \
        }                                                                                                     \
    } while (0)

/** @brief The table's key. A distinct type, so a key and a value cannot be swapped at a call site. */
enum class user_id_t : std::uint64_t {};

/** @brief The table's mapped value, naming the one state a failed lookup reports. */
enum class session_id_t : std::uint64_t {
    /** @brief No session, which @c session_of never produces and so cannot collide with a real one. */
    missing_k = 0,
};

/**
 *  @brief The session every suite stores under @p user, so a mispaired key and value cannot pass.
 *  @note Users are numbered from one, which is what keeps the result clear of @c missing_k.
 */
constexpr session_id_t session_of(user_id_t user) noexcept {
    return static_cast<session_id_t>(static_cast<std::uint64_t>(user) * 3ull + 1ull);
}

/**
 *  @brief Multiply-shift hash over the key, replacing the @c std::hash the library defaults to.
 *  @note Its return type is what the table adopts as the slot-offset type.
 */
struct user_id_hash_t {
    constexpr std::uint64_t operator()(user_id_t user) const noexcept {
        std::uint64_t mixed = static_cast<std::uint64_t>(user);
        mixed *= 0x9e3779b97f4a7c15ull;
        mixed ^= mixed >> 31;
        return mixed;
    }
};

/** @brief Equality over the key, replacing @c std::equal_to. */
struct user_id_equals_t {
    constexpr bool operator()(user_id_t first, user_id_t second) const noexcept { return first == second; }
};

/**
 *  @brief Hands the table memory both sides can address.
 *  @note Only ever called from the host, since @c hash_storage::make runs there.
 */
struct managed_allocator_t {
    std::byte *allocate(std::size_t bytes) noexcept {
        void *pointer = nullptr;
        return cudaMallocManaged(&pointer, bytes) == cudaSuccess ? static_cast<std::byte *>(pointer) : nullptr;
    }
    void deallocate(std::byte *pointer, std::size_t) noexcept { st_verify_cuda_(cudaFree(pointer)); }
};

using table_t = concurrent_hash_map<user_id_t, session_id_t, user_id_hash_t, user_id_equals_t, managed_allocator_t>;
using storage_t = table_t::storage_type;

/**
 *  @brief Objects in managed memory, which the host and a kernel address alike.
 *    Both constructors construct and the destructor destroys, so the table's storage is released
 *    through the same path an array of keys is.
 */
template <typename element_type_>
struct managed {
    element_type_ *data {};
    std::size_t count {};

    /** @brief An array of @p elements value-initialized objects. */
    explicit managed(std::size_t elements) noexcept : count(elements) {
        data = allocate_(elements);
        for (std::size_t index = 0; index != elements; ++index) new (data + index) element_type_ {};
    }

    /** @brief One object, moved in from @p value. */
    explicit managed(element_type_ &&value) noexcept : count(1) {
        data = allocate_(1);
        new (data) element_type_(std::move(value));
    }

    managed(managed const &) = delete;
    managed &operator=(managed const &) = delete;

    ~managed() noexcept {
        for (std::size_t index = 0; index != count; ++index) data[index].~element_type_();
        st_verify_cuda_(cudaFree(data));
    }

    element_type_ &operator[](std::size_t index) const noexcept { return data[index]; }
    element_type_ *operator->() const noexcept { return data; }

  private:
    static element_type_ *allocate_(std::size_t elements) noexcept {
        void *raw = nullptr;
        st_verify_cuda_(cudaMallocManaged(&raw, elements * sizeof(element_type_)));
        return static_cast<element_type_ *>(raw);
    }
};

/** @brief A pinned table of @p slots slots, the object and its storage both in managed memory. */
static managed<table_t> make_table(std::size_t slots) noexcept {
    storage_t storage = storage_t::make(hash_slots_count_t::from_slots(slots), managed_allocator_t {});
    st_verify_(storage.is_allocated() && "Managed storage allocation failed");
    return managed<table_t>(table_t::adopt(std::move(storage)));
}

#pragma endregion Device Helpers

#pragma region Kernels

__global__ void insert_kernel(table_t *table, user_id_t const *users, std::size_t count) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        [[maybe_unused]] status_t const status = table->emplace(users[index], session_of(users[index]));
    }
}

__global__ void find_kernel(table_t *table, user_id_t const *users, std::size_t count, session_id_t *found) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x) {
        session_id_t session = session_id_t::missing_k;
        table->find(users[index], [&](auto const &slot) noexcept { session = slot.value(); });
        found[index] = session;
    }
}

__global__ void erase_kernel(table_t *table, user_id_t const *users, std::size_t count) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
        [[maybe_unused]]
        bool const erased = table->erase(users[index]);
}

/** @brief Threads per block, one warp's multiple, so a block covers whole buckets. */
constexpr unsigned threads_per_block_k = 128;

/** @brief Blocks needed to give @p count elements one thread each, never zero. */
constexpr unsigned blocks_for(std::size_t count) noexcept {
    unsigned const blocks = static_cast<unsigned>((count + threads_per_block_k - 1) / threads_per_block_k);
    return blocks ? blocks : 1u;
}

/** @brief Runs a key-only kernel over @p users and waits for it. */
static void launch_over(void (*kernel)(table_t *, user_id_t const *, std::size_t), managed<table_t> const &table,
                        managed<user_id_t> const &users, std::size_t count) noexcept {
    kernel<<<blocks_for(count), threads_per_block_k>>>(table.data, users.data, count);
    st_verify_cuda_(cudaGetLastError());
    st_verify_cuda_(cudaDeviceSynchronize());
}

#pragma endregion Kernels

#pragma region Suites

/** @brief Keys per suite - enough to span many buckets, few enough to verify one by one on the host. */
constexpr std::size_t keys_count_k = 20000;

/** @brief Fills @p users with the @p count identifiers starting at @p first. */
static void number_users(managed<user_id_t> const &users, std::size_t count, std::uint64_t first) noexcept {
    for (std::size_t index = 0; index != count; ++index) users[index] = static_cast<user_id_t>(first + index);
}

/** @brief Verifies on the host that @p user maps to its expected session. */
static void verify_present(managed<table_t> const &table, user_id_t user) noexcept {
    session_id_t session = session_id_t::missing_k;
    bool const found = table->find(user, [&](auto const &slot) noexcept { session = slot.value(); });
    st_verify_(found && "Key missing from the table");
    st_verify_eq_(session, session_of(user));
}

/** @brief Tests that keys a kernel inserted are all visible to the host, with the right values */
static void cuda_device_inserts_host_reads() {
    std::size_t const count = keys_count_k;
    managed<table_t> table = make_table(count * 2);
    managed<user_id_t> users(count);
    number_users(users, count, 1);

    launch_over(insert_kernel, table, users, count);

    st_verify_eq_(table->size(), count);
    for (std::size_t index = 0; index != count; ++index) verify_present(table, users[index]);
}

/** @brief Tests that keys the host inserted are all visible to a kernel, with the right values */
static void cuda_host_inserts_device_reads() {
    std::size_t const count = keys_count_k;
    managed<table_t> table = make_table(count * 2);
    managed<user_id_t> users(count);
    managed<session_id_t> found(count);
    number_users(users, count, 1);
    for (std::size_t index = 0; index != count; ++index) {
        [[maybe_unused]] status_t const status = table->emplace(users[index], session_of(users[index]));
        st_verify_(succeeded(status) && "Host insert into a pinned table failed");
    }

    find_kernel<<<blocks_for(count), threads_per_block_k>>>(table.data, users.data, count, found.data);
    st_verify_cuda_(cudaGetLastError());
    st_verify_cuda_(cudaDeviceSynchronize());

    for (std::size_t index = 0; index != count; ++index) st_verify_eq_(found[index], session_of(users[index]));
}

/** @brief Tests that a device erase tombstones exactly its own keys, leaving the rest findable */
static void cuda_device_insert_find_erase_cycle() {
    std::size_t const count = keys_count_k;
    managed<table_t> table = make_table(count * 2);
    managed<user_id_t> users(count);
    number_users(users, count, 1);

    launch_over(insert_kernel, table, users, count);
    st_verify_eq_(table->size(), count);

    // Erase the first half, which keeps the two ranges contiguous and the arithmetic obvious.
    std::size_t const erased_count = count / 2;
    launch_over(erase_kernel, table, users, erased_count);

    st_verify_eq_(table->size(), count - erased_count);
    st_verify_eq_(table->deleted_count(), erased_count);

    for (std::size_t index = 0; index != erased_count; ++index)
        st_verify_(!table->contains(users[index]) && "Erased key still present");
    for (std::size_t index = erased_count; index != count; ++index) verify_present(table, users[index]);
}

/**
 *  @brief Tests that host threads and a kernel inserting disjoint ranges into one table both land.
 *  @note Skipped on a device without concurrent managed access, where touching the allocation from the
 *    host while a kernel runs is not merely slow but a fault.
 */
static void cuda_host_and_device_insert_together() {
    std::size_t const count = keys_count_k;
    int concurrent_managed_access = 0;
    st_verify_cuda_(cudaDeviceGetAttribute(&concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, 0));
    if (!concurrent_managed_access) {
        print_line(stdout, "  (skipped: device lacks concurrent managed access)");
        return;
    }

    managed<table_t> table = make_table(count * 4);
    managed<user_id_t> device_users(count);
    managed<user_id_t> host_users(count);
    // The host's range starts past the device's, so a key found later names the side that wrote it.
    number_users(device_users, count, 1);
    number_users(host_users, count, count + 1);

    insert_kernel<<<blocks_for(count), threads_per_block_k>>>(table.data, device_users.data, count);
    st_verify_cuda_(cudaGetLastError());

    unsigned const host_threads_count = 4;
    std::vector<std::thread> host_threads;
    for (unsigned thread_index = 0; thread_index != host_threads_count; ++thread_index)
        host_threads.emplace_back([&, thread_index] {
            for (std::size_t index = thread_index; index < count; index += host_threads_count) {
                user_id_t const user = host_users[index];
                [[maybe_unused]] status_t const status = table->emplace(user, session_of(user));
            }
        });
    for (std::thread &host_thread : host_threads) host_thread.join();
    st_verify_cuda_(cudaDeviceSynchronize());

    st_verify_eq_(table->size(), count * 2);
    for (std::size_t index = 0; index != count; ++index) {
        verify_present(table, device_users[index]);
        verify_present(table, host_users[index]);
    }
}

#pragma endregion Suites

int main() {
    install_test_signal_handlers();

    // A build machine with `nvcc` need not have a device, and neither need a CI runner, so an absent
    // or too-old device is a skip rather than a failure.
    int devices_count = 0;
    if (cudaGetDeviceCount(&devices_count) != cudaSuccess || devices_count == 0) {
        print_line(stdout, "No CUDA device visible - skipping.");
        return 0;
    }

    cudaDeviceProp properties {};
    st_verify_cuda_(cudaGetDeviceProperties(&properties, 0));
    if (properties.major < 7) {
        print_line(stdout, "Device {} is sm_{}{}; the slot spin needs sm_70 or newer - skipping.", properties.name,
                   properties.major, properties.minor);
        return 0;
    }
    print_line(stdout, "Running on {} (sm_{}{})", properties.name, properties.major, properties.minor);

    char const *const filter = std::getenv("SMASHTABLE_FILTER");
    std::size_t failures = 0;

    failures += run_test(filter, "cuda.device_inserts_host_reads", cuda_device_inserts_host_reads);
    failures += run_test(filter, "cuda.host_inserts_device_reads", cuda_host_inserts_device_reads);
    failures += run_test(filter, "cuda.device_insert_find_erase_cycle", cuda_device_insert_find_erase_cycle);
    failures += run_test(filter, "cuda.host_and_device_insert_together", cuda_host_and_device_insert_together);

    return report_test_failures(failures);
}
