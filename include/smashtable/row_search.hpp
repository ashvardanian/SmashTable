/**
 *  @file include/smashtable/row_search.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Row searchers over ordered keys: one kit per microarchitecture, chosen once and then
 *      called directly. The rows themselves, and the keys in them, are @c row_layout.hpp .
 *
 *  @section row_search_kits Kits
 *
 *  A kit is a struct of static @c count_below overloads, one per key type, each answering how many
 *  keys of a row order below a wanted key. For a sorted row that count is the lower bound, and
 *  counts of consecutive rows add up to a rank, so the answer serves an implicit B-tree, a B+ tree
 *  and a sorted leaf alike.
 *
 *  Every kit the compiler can emit is compiled into the same artifact, each region under its own
 *  target attributes, so no global instruction-set flag is needed. @c detect_row_kit probes the
 *  processor, and @c visit_row_kit hands the chosen kit to a callback as a type: a structure
 *  instantiated over that type calls its kit directly, with no table and no branch per key.
 *
 *  @section row_search_layout Row Layout
 *
 *  A row is a @c std::span whose extent is the row width: static where the medium fixes it, which
 *  lets the kit unroll its loop, and dynamic otherwise. A 16-byte key is @c key128_t, two
 *  big-endian words ordered as @c memcmp. Split rows keep high and low words in separate columns,
 *  so the low column is read only where a high word ties; interleaved rows keep whole @c key128_t
 *  pairs for containers that hand out keys.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`, `offsetof`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`, `std::int64_t`

#include <bit>         // `std::popcount`
#include <concepts>    // `std::same_as`
#include <limits>      // `std::numeric_limits`
#include <span>        // `std::span`
#include <type_traits> // `std::type_identity_t`

#include "row_layout.hpp"

namespace ashvardanian::smashtable {

#pragma region Kit Selection

/** The kits, one enumerator per microarchitecture, newest instruction sets after the baseline. */
enum class row_kit_t : std::uint8_t {
    serial_k,
    haswell_k,
    skylake_k,
    icelake_k,
    neon_k,
    sve_k,
    rvv_k,
    v128_k,
};

/** Every kit in enumerator order, for a benchmark or a log that walks them all. */
inline constexpr row_kit_t every_row_kit_k[] = {row_kit_t::serial_k,  row_kit_t::haswell_k, row_kit_t::skylake_k,
                                                row_kit_t::icelake_k, row_kit_t::neon_k,    row_kit_t::sve_k,
                                                row_kit_t::rvv_k,     row_kit_t::v128_k};

/** The enumerator's own name without its suffix, for a benchmark or a log. */
[[nodiscard]] constexpr char const *name_of(row_kit_t kit) noexcept {
    switch (kit) {
    case row_kit_t::serial_k: return "serial";
    case row_kit_t::haswell_k: return "haswell";
    case row_kit_t::skylake_k: return "skylake";
    case row_kit_t::icelake_k: return "icelake";
    case row_kit_t::neon_k: return "neon";
    case row_kit_t::sve_k: return "sve";
    case row_kit_t::rvv_k: return "rvv";
    case row_kit_t::v128_k: return "v128";
    }
    return "unrecognized";
}

/** Whether this build carries @p kit, whatever the processor running it. */
[[nodiscard]] constexpr bool row_kit_compiled(row_kit_t kit) noexcept {
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return SMASHTABLE_TARGET_HASWELL;
    case row_kit_t::skylake_k: return SMASHTABLE_TARGET_SKYLAKE;
    case row_kit_t::icelake_k: return SMASHTABLE_TARGET_ICELAKE;
    case row_kit_t::neon_k: return SMASHTABLE_TARGET_NEON;
    case row_kit_t::sve_k: return SMASHTABLE_TARGET_SVE;
    case row_kit_t::rvv_k: return SMASHTABLE_TARGET_RVV;
    case row_kit_t::v128_k: return SMASHTABLE_TARGET_V128;
    }
    return false;
}

/** Whether this build carries @p kit and the running processor and operating system can execute it.
 *  Probes the processor on every call, so ask once at open or construction and keep the answer. */
#if SMASHTABLE_ARCH_X86_64_ && (defined(__GNUC__) || defined(__clang__))
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    __builtin_cpu_init();
    bool const runs_haswell = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
                              __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
    bool const runs_skylake = runs_haswell && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
                              __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    bool const runs_icelake = runs_skylake && __builtin_cpu_supports("avx512vpopcntdq");
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return SMASHTABLE_TARGET_HASWELL && runs_haswell;
    case row_kit_t::skylake_k: return SMASHTABLE_TARGET_SKYLAKE && runs_skylake;
    case row_kit_t::icelake_k: return SMASHTABLE_TARGET_ICELAKE && runs_icelake;
    default: return false;
    }
}
#elif SMASHTABLE_ARCH_X86_64_ && defined(_MSC_VER)
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 1);
    bool const has_os_saves = ((registers[2] >> 27) & 1) != 0;
    bool const has_popcnt = ((registers[2] >> 23) & 1) != 0;
    unsigned long long const saved_state = has_os_saves ? _xgetbv(0) : 0;
    __cpuidex(registers, 7, 0);
    auto const has_leaf7 = [&registers](int bit) noexcept { return ((registers[1] >> bit) & 1) != 0; };
    bool const runs_haswell = has_popcnt && (saved_state & 0x6) == 0x6 && has_leaf7(3) && has_leaf7(5) && has_leaf7(8);
    bool const runs_skylake = runs_haswell && (saved_state & 0xE6) == 0xE6 && has_leaf7(16) && has_leaf7(17) &&
                              has_leaf7(30) && has_leaf7(31);
    bool const runs_icelake = runs_skylake && ((registers[2] >> 14) & 1) != 0; // `AVX512_VPOPCNTDQ` sits in ECX
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return SMASHTABLE_TARGET_HASWELL && runs_haswell;
    case row_kit_t::skylake_k: return SMASHTABLE_TARGET_SKYLAKE && runs_skylake;
    case row_kit_t::icelake_k: return SMASHTABLE_TARGET_ICELAKE && runs_icelake;
    default: return false;
    }
}
#elif (SMASHTABLE_ARCH_ARM64_ || SMASHTABLE_ARCH_RISCV64_) && (defined(__linux__) || defined(__FreeBSD__))
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    unsigned long capabilities = 0; // `AT_HWCAP` carries the same bits on both kernels
#if defined(__FreeBSD__)
    elf_aux_info(AT_HWCAP, &capabilities, sizeof(capabilities));
#else
    capabilities = getauxval(AT_HWCAP);
#endif
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::neon_k: return SMASHTABLE_TARGET_NEON && (capabilities & (1ul << 1)) != 0; // `HWCAP_ASIMD`
    case row_kit_t::sve_k: return SMASHTABLE_TARGET_SVE && (capabilities & (1ul << 22)) != 0;  // `HWCAP_SVE`
    case row_kit_t::rvv_k: return SMASHTABLE_TARGET_RVV && (capabilities & (1ul << ('V' - 'A'))) != 0;
    default: return false;
    }
}
#elif SMASHTABLE_ARCH_ARM64_
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    // Advanced SIMD is the AArch64 baseline; SVE is only probed where the kernel reports it.
    return kit == row_kit_t::serial_k || (kit == row_kit_t::neon_k && SMASHTABLE_TARGET_NEON);
}
#elif SMASHTABLE_TARGET_V128
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    // A module carrying SIMD only loads where the engine runs it, so building it in is the proof.
    return kit == row_kit_t::serial_k || kit == row_kit_t::v128_k;
}
#else
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept { return kit == row_kit_t::serial_k; }
#endif

/** The newest kit this build carries and the processor can run, falling back to @c serial_k. */
[[nodiscard]] inline row_kit_t detect_row_kit() noexcept {
    for (row_kit_t const kit : {row_kit_t::icelake_k, row_kit_t::skylake_k, row_kit_t::haswell_k, row_kit_t::sve_k,
                                row_kit_t::neon_k, row_kit_t::rvv_k, row_kit_t::v128_k})
        if (row_kit_supported(kit)) return kit;
    return row_kit_t::serial_k;
}

#pragma endregion Kit Selection

#pragma region Serial Kit

/** The reference kit every kit must agree with, the one each finishes a partial chunk with. */
struct serial_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::serial_k;
    static constexpr bool compiled_k = true;

    /** Keys one @c mask_equal compares, its answer in the low bits of a 32-bit mask. */
    static constexpr std::size_t mask_equal_width_k = 16;

    /** How many of @p keys order below @p wanted; for a sorted row, the lower bound. */
    template <typename key_type_, std::size_t extent_>
        requires row_searchable_key<key_type_>
    [[nodiscard]] static constexpr std::size_t count_below(std::span<key_type_ const, extent_> keys,
                                                           std::type_identity_t<key_type_> wanted) noexcept {
        std::size_t below = 0;
        for (key_type_ const key : keys) below += static_cast<std::size_t>(key < wanted);
        return below;
    }

    /** How many keys of a split row order below @p wanted, reading a low word only where its high
     *  word ties. */
    template <std::size_t extent_>
    [[nodiscard]] static constexpr std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                           std::span<std::uint64_t const, extent_> low_words,
                                                           key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        std::size_t below = 0;
        for (std::size_t index = 0; index < high_words.size(); ++index)
            below += static_cast<std::size_t>(high_words[index] < wanted.high ||
                                              (high_words[index] == wanted.high && low_words[index] < wanted.low));
        return below;
    }

    /** Which of @p keys equal @p wanted, one bit per key, the lowest for the first: a hash table's
     *  window, where slots nobody filled compare too and the caller masks them off. */
    template <typename word_type_>
        requires std::same_as<word_type_, std::uint32_t> || std::same_as<word_type_, std::uint64_t>
    [[nodiscard]] static constexpr std::uint32_t mask_equal(std::span<word_type_ const, mask_equal_width_k> keys,
                                                            word_type_ wanted) noexcept {
        std::uint32_t matched = 0;
        for (std::size_t index = 0; index != keys.size(); ++index)
            matched |= static_cast<std::uint32_t>(keys[index] == wanted) << index;
        return matched;
    }

    /**
     *  @brief Where the @p remaining -th set bit after set bit @p sampled of @p words sits,
     *      counting @p sampled as the zeroth: the scan an Elias-Fano select runs from its sample.
     *
     *  The bit must exist and @p remaining must be positive. Kits that scan several words per
     *  instruction may read up to seven words past the one holding the answer, which the sequence
     *  keeps as padding.
     */
    [[nodiscard]] static std::uint64_t select_set_bit(std::span<std::uint64_t const> words, std::uint64_t sampled,
                                                      std::size_t remaining) noexcept {
        std::size_t word_index = static_cast<std::size_t>(sampled / 64);
        std::uint64_t word = words[word_index] & ((~std::uint64_t {0} << (sampled % 64)) << 1);
        while (true) {
            std::size_t const ones = static_cast<std::size_t>(popcount(word));
            if (remaining <= ones) return word_index * 64 + select_in_word(word, remaining - 1);
            remaining -= ones;
            word = words[++word_index];
        }
    }

    /** The offset of the @p rank -th set bit of @p word, counted from zero, found by halving. */
    [[nodiscard]] static constexpr unsigned select_in_word(std::uint64_t word, std::size_t rank) noexcept {
        unsigned offset = 0;
        for (unsigned half = 32; half; half >>= 1) {
            std::uint64_t const lower = word & ((std::uint64_t {1} << half) - 1);
            std::size_t const ones = static_cast<std::size_t>(popcount(lower));
            if (rank < ones) {
                word = lower;
                continue;
            }
            rank -= ones;
            word >>= half;
            offset += half;
        }
        return offset;
    }
};

#pragma endregion Serial Kit

#pragma region Haswell Kit

#if SMASHTABLE_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "popcnt")
#endif

/** AVX2 over four 64-bit or eight 32-bit keys per load, unsigned order by flipping the sign bit. */
struct haswell_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = 16;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        __m256i const sign = _mm256_set1_epi32(std::numeric_limits<std::int32_t>::min());
        __m256i const threshold = _mm256_xor_si256(_mm256_set1_epi32(static_cast<std::int32_t>(wanted)), sign);
        std::size_t below = 0;
        std::size_t offset = 0;
        for (; offset + 8 <= keys.size(); offset += 8) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(keys.data() + offset)));
            __m256i const flipped = _mm256_xor_si256(loaded, sign);
            int const mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(threshold, flipped)));
            below += static_cast<std::size_t>(popcount(static_cast<unsigned>(mask)));
        }
        return below + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        std::size_t below = 0;
        std::size_t offset = 0;
        __m256i const threshold = _mm256_set1_epi32(wanted);
        for (; offset + 8 <= keys.size(); offset += 8) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(keys.data() + offset)));
            int const mask = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(threshold, loaded)));
            below += static_cast<std::size_t>(popcount(static_cast<unsigned>(mask)));
        }
        return below + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                 std::uint64_t wanted) noexcept {
        return count_below_words_(keys, wanted, std::uint64_t {1} << 63);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                 std::int64_t wanted) noexcept {
        std::span<std::uint64_t const, extent_> const words(
            static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data())), keys.size());
        return count_below_words_(words, static_cast<std::uint64_t>(wanted), 0);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<key128_t const, extent_> keys, key128_t wanted) noexcept {
        __m256i const sign = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
        // Lanes run low to high, so a two-key load reads high, low, high, low.
        __m256i const threshold = _mm256_xor_si256(
            _mm256_set_epi64x(static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high),
                              static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high)),
            sign);
        std::size_t below = 0;
        std::size_t offset = 0;
        for (; offset + 2 <= keys.size(); offset += 2) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(keys.data() + offset)));
            __m256i const flipped = _mm256_xor_si256(loaded, sign);
            unsigned const below_words =
                static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(threshold, flipped))));
            unsigned const tied_words =
                static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(threshold, flipped))));
            below += static_cast<std::size_t>(popcount((below_words | (tied_words & (below_words >> 1))) & 0b0101u));
        }
        return below + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                 std::span<std::uint64_t const, extent_> low_words,
                                                 key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        __m256i const sign = _mm256_set1_epi64x(std::numeric_limits<std::int64_t>::min());
        __m256i const high_threshold =
            _mm256_xor_si256(_mm256_set1_epi64x(static_cast<std::int64_t>(wanted.high)), sign);
        __m256i const low_threshold = _mm256_xor_si256(_mm256_set1_epi64x(static_cast<std::int64_t>(wanted.low)), sign);
        std::size_t const whole = high_words.size() - high_words.size() % 4;
        __m256i accumulated = _mm256_setzero_si256();
        __m256i tied = _mm256_setzero_si256();
        for (std::size_t offset = 0; offset < whole; offset += 4) {
            __m256i const high = flipped_load_(high_words.data() + offset, sign);
            accumulated = _mm256_sub_epi64(accumulated, _mm256_cmpgt_epi64(high_threshold, high));
            tied = _mm256_or_si256(tied, _mm256_cmpeq_epi64(high_threshold, high));
        }
        // One branch per row rather than per load, since a present key ties exactly once at an unpredictable spot.
        if (!_mm256_testz_si256(tied, tied))
            for (std::size_t offset = 0; offset < whole; offset += 4) {
                __m256i const high_tied =
                    _mm256_cmpeq_epi64(high_threshold, flipped_load_(high_words.data() + offset, sign));
                __m256i const low_below =
                    _mm256_cmpgt_epi64(low_threshold, flipped_load_(low_words.data() + offset, sign));
                accumulated = _mm256_sub_epi64(accumulated, _mm256_and_si256(high_tied, low_below));
            }
        return sum_lanes_(accumulated) +
               serial_row_kit_t::count_below(high_words.subspan(whole), low_words.subspan(whole), wanted);
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint32_t const, mask_equal_width_k> keys,
                                                  std::uint32_t wanted) noexcept {
        __m256i const needle = _mm256_set1_epi32(static_cast<std::int32_t>(wanted));
        std::uint32_t matched = 0;
        for (std::size_t offset = 0; offset != keys.size(); offset += 8) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(keys.data() + offset)));
            matched |=
                static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(loaded, needle))))
                << offset;
        }
        return matched;
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint64_t const, mask_equal_width_k> keys,
                                                  std::uint64_t wanted) noexcept {
        __m256i const needle = _mm256_set1_epi64x(static_cast<std::int64_t>(wanted));
        std::uint32_t matched = 0;
        for (std::size_t offset = 0; offset != keys.size(); offset += 4) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(keys.data() + offset)));
            matched |=
                static_cast<std::uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(loaded, needle))))
                << offset;
        }
        return matched;
    }

  private:
    template <std::size_t extent_>
    static std::size_t count_below_words_(std::span<std::uint64_t const, extent_> words, std::uint64_t wanted,
                                          std::uint64_t flip) noexcept {
        __m256i const flipper = _mm256_set1_epi64x(static_cast<std::int64_t>(flip));
        __m256i const threshold = _mm256_set1_epi64x(static_cast<std::int64_t>(wanted ^ flip));
        __m256i accumulated = _mm256_setzero_si256();
        std::size_t offset = 0;
        for (; offset + 4 <= words.size(); offset += 4) {
            __m256i const loaded =
                _mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(words.data() + offset)));
            accumulated =
                _mm256_sub_epi64(accumulated, _mm256_cmpgt_epi64(threshold, _mm256_xor_si256(loaded, flipper)));
        }
        std::size_t below = sum_lanes_(accumulated);
        for (; offset < words.size(); ++offset)
            below += static_cast<std::size_t>(static_cast<std::int64_t>(words[offset] ^ flip) <
                                              static_cast<std::int64_t>(wanted ^ flip));
        return below;
    }

    static __m256i flipped_load_(std::uint64_t const *words, __m256i sign) noexcept {
        return _mm256_xor_si256(_mm256_loadu_si256(static_cast<__m256i const *>(static_cast<void const *>(words))),
                                sign);
    }

    static std::size_t sum_lanes_(__m256i lanes) noexcept {
        __m128i const folded = _mm_add_epi64(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
        return static_cast<std::size_t>(_mm_cvtsi128_si64(folded) + _mm_extract_epi64(folded, 1));
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct haswell_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::haswell_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Haswell Kit

#pragma region Skylake Kit

#if SMASHTABLE_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi", "bmi2", "popcnt")
#endif

/** AVX-512 over eight 64-bit or sixteen 32-bit keys per load, masked loads over a scalar tail. */
struct skylake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = 16;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        __m512i const threshold = _mm512_set1_epi32(static_cast<std::int32_t>(wanted));
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += 16) {
            std::size_t const remaining = keys.size() - offset;
            __mmask16 const present = remaining >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << remaining) - 1u);
            __m512i const loaded = _mm512_maskz_loadu_epi32(present, keys.data() + offset);
            below += static_cast<std::size_t>(
                popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epu32_mask(present, threshold, loaded))));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        __m512i const threshold = _mm512_set1_epi32(wanted);
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += 16) {
            std::size_t const remaining = keys.size() - offset;
            __mmask16 const present = remaining >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << remaining) - 1u);
            __m512i const loaded = _mm512_maskz_loadu_epi32(present, keys.data() + offset);
            below += static_cast<std::size_t>(
                popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epi32_mask(present, threshold, loaded))));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                 std::uint64_t wanted) noexcept {
        __m512i const threshold = _mm512_set1_epi64(static_cast<std::int64_t>(wanted));
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += 8) {
            __mmask8 const present = present_words_(keys.size() - offset);
            __m512i const loaded = _mm512_maskz_loadu_epi64(present, keys.data() + offset);
            below += static_cast<std::size_t>(
                popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epu64_mask(present, threshold, loaded))));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                 std::int64_t wanted) noexcept {
        __m512i const threshold = _mm512_set1_epi64(wanted);
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += 8) {
            __mmask8 const present = present_words_(keys.size() - offset);
            __m512i const loaded = _mm512_maskz_loadu_epi64(present, keys.data() + offset);
            below += static_cast<std::size_t>(
                popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epi64_mask(present, threshold, loaded))));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<key128_t const, extent_> keys, key128_t wanted) noexcept {
        __m512i const threshold =
            _mm512_set_epi64(static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high),
                             static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high),
                             static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high),
                             static_cast<std::int64_t>(wanted.low), static_cast<std::int64_t>(wanted.high));
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += 4) {
            __mmask8 const present = present_words_(2 * smaller_of<std::size_t>(keys.size() - offset, 4));
            __m512i const loaded = _mm512_maskz_loadu_epi64(present, keys.data() + offset);
            unsigned const below_words = _mm512_mask_cmpgt_epu64_mask(present, threshold, loaded);
            unsigned const tied_words = _mm512_mask_cmpeq_epu64_mask(present, threshold, loaded);
            below += static_cast<std::size_t>(popcount((below_words | (tied_words & (below_words >> 1))) & 0x55u));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                 std::span<std::uint64_t const, extent_> low_words,
                                                 key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        __m512i const high_threshold = _mm512_set1_epi64(static_cast<std::int64_t>(wanted.high));
        __m512i const low_threshold = _mm512_set1_epi64(static_cast<std::int64_t>(wanted.low));
        std::size_t below = 0;
        unsigned tied = 0;
        for (std::size_t offset = 0; offset < high_words.size(); offset += 8) {
            __mmask8 const present = present_words_(high_words.size() - offset);
            __m512i const high = _mm512_maskz_loadu_epi64(present, high_words.data() + offset);
            below += static_cast<std::size_t>(
                popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epu64_mask(present, high_threshold, high))));
            tied |= _mm512_mask_cmpeq_epu64_mask(present, high_threshold, high);
        }
        // One branch per row, and the second pass loads only the low words whose high words tie.
        if (tied != 0)
            for (std::size_t offset = 0; offset < high_words.size(); offset += 8) {
                __mmask8 const present = present_words_(high_words.size() - offset);
                __m512i const high = _mm512_maskz_loadu_epi64(present, high_words.data() + offset);
                __mmask8 const high_tied = _mm512_mask_cmpeq_epu64_mask(present, high_threshold, high);
                __m512i const low = _mm512_maskz_loadu_epi64(high_tied, low_words.data() + offset);
                below += static_cast<std::size_t>(
                    popcount(static_cast<unsigned>(_mm512_mask_cmpgt_epu64_mask(high_tied, low_threshold, low))));
            }
        return below;
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint32_t const, mask_equal_width_k> keys,
                                                  std::uint32_t wanted) noexcept {
        return _mm512_cmpeq_epi32_mask(_mm512_loadu_si512(keys.data()),
                                       _mm512_set1_epi32(static_cast<std::int32_t>(wanted)));
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint64_t const, mask_equal_width_k> keys,
                                                  std::uint64_t wanted) noexcept {
        __m512i const needle = _mm512_set1_epi64(static_cast<std::int64_t>(wanted));
        std::uint32_t const low = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(keys.data()), needle);
        std::uint32_t const high = _mm512_cmpeq_epi64_mask(_mm512_loadu_si512(keys.data() + 8), needle);
        return low | (high << 8);
    }

  private:
    static __mmask8 present_words_(std::size_t remaining) noexcept {
        return remaining >= 8 ? __mmask8(0xFF) : __mmask8((1u << remaining) - 1u);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct skylake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::skylake_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Skylake Kit

#pragma region Ice Lake Kit

#if SMASHTABLE_TARGET_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                           \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vpopcntdq,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vpopcntdq", "bmi", "bmi2", "popcnt")
#endif

/** The Skylake kit plus @c VPOPCNTQ, counting eight words of a bit vector per instruction, and a
 *  @c PDEP every such core runs in hardware, unlike the microcoded one of the Zen 2 parts the
 *  Haswell kit also serves. */
struct icelake_row_kit_t : skylake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::icelake_k;

    [[nodiscard]] static std::uint64_t select_set_bit(std::span<std::uint64_t const> words, std::uint64_t sampled,
                                                      std::size_t remaining) noexcept {
        std::size_t word_index = static_cast<std::size_t>(sampled / 64);
        std::uint64_t const first = words[word_index] & ((~std::uint64_t {0} << (sampled % 64)) << 1);
        std::size_t const first_ones = static_cast<std::size_t>(popcount(first));
        if (remaining <= first_ones) return word_index * 64 + select_in_word(first, remaining - 1);
        remaining -= first_ones;
        ++word_index;
        // Shifting the counts up one, two and four lanes, three adds leave inclusive prefix sums.
        __m512i const by_one = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0);
        __m512i const by_two = _mm512_set_epi64(5, 4, 3, 2, 1, 0, 0, 0);
        __m512i const by_four = _mm512_set_epi64(3, 2, 1, 0, 0, 0, 0, 0);
        while (true) {
            __m512i sums = _mm512_popcnt_epi64(_mm512_loadu_si512(words.data() + word_index));
            sums = _mm512_add_epi64(sums, _mm512_maskz_permutexvar_epi64(0xFE, by_one, sums));
            sums = _mm512_add_epi64(sums, _mm512_maskz_permutexvar_epi64(0xFC, by_two, sums));
            sums = _mm512_add_epi64(sums, _mm512_maskz_permutexvar_epi64(0xF0, by_four, sums));
            __mmask8 const reached =
                _mm512_cmpge_epu64_mask(sums, _mm512_set1_epi64(static_cast<std::int64_t>(remaining)));
            if (reached) {
                unsigned const lane = static_cast<unsigned>(countr_zero(reached));
                std::size_t const before = lane ? lane_of_(sums, lane - 1) : 0;
                return (word_index + lane) * 64 + select_in_word(words[word_index + lane], remaining - before - 1);
            }
            remaining -= lane_of_(sums, 7);
            word_index += 8;
        }
    }

    /** The offset of the @p rank -th set bit of @p word, counted from zero, found by @c PDEP. */
    [[nodiscard]] static unsigned select_in_word(std::uint64_t word, std::size_t rank) noexcept {
        return static_cast<unsigned>(_tzcnt_u64(_pdep_u64(std::uint64_t {1} << rank, word)));
    }

  private:
    static std::size_t lane_of_(__m512i lanes, unsigned lane) noexcept {
        __m512i const moved = _mm512_permutexvar_epi64(_mm512_set1_epi64(lane), lanes);
        return static_cast<std::size_t>(_mm_cvtsi128_si64(_mm512_castsi512_si128(moved)));
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct icelake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::icelake_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Ice Lake Kit

#pragma region NEON Kit

#if SMASHTABLE_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Advanced SIMD over two 64-bit or four 32-bit keys per load, deinterleaving keys as it loads. */
struct neon_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::neon_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = 16;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        uint32x4_t const threshold = vdupq_n_u32(wanted);
        uint32x4_t accumulated = vdupq_n_u32(0);
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            accumulated = vsubq_u32(accumulated, vcltq_u32(vld1q_u32(keys.data() + offset), threshold));
        return vaddvq_u32(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        int32x4_t const threshold = vdupq_n_s32(wanted);
        uint32x4_t accumulated = vdupq_n_u32(0);
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            accumulated = vsubq_u32(accumulated, vcltq_s32(vld1q_s32(keys.data() + offset), threshold));
        return vaddvq_u32(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                 std::uint64_t wanted) noexcept {
        uint64x2_t const threshold = vdupq_n_u64(wanted);
        uint64x2_t accumulated = vdupq_n_u64(0);
        std::size_t offset = 0;
        for (; offset + 2 <= keys.size(); offset += 2)
            accumulated = vsubq_u64(accumulated, vcltq_u64(vld1q_u64(keys.data() + offset), threshold));
        return vaddvq_u64(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                 std::int64_t wanted) noexcept {
        int64x2_t const threshold = vdupq_n_s64(wanted);
        uint64x2_t accumulated = vdupq_n_u64(0);
        std::size_t offset = 0;
        for (; offset + 2 <= keys.size(); offset += 2)
            accumulated = vsubq_u64(accumulated, vcltq_s64(vld1q_s64(keys.data() + offset), threshold));
        return vaddvq_u64(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<key128_t const, extent_> keys, key128_t wanted) noexcept {
        uint64x2_t const high_threshold = vdupq_n_u64(wanted.high);
        uint64x2_t const low_threshold = vdupq_n_u64(wanted.low);
        uint64x2_t accumulated = vdupq_n_u64(0);
        std::size_t offset = 0;
        for (; offset + 2 <= keys.size(); offset += 2) {
            std::uint64_t const *const words =
                static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data() + offset));
            uint64x2_t const first_key = vld1q_u64(words);
            uint64x2_t const second_key = vld1q_u64(words + 2);
            uint64x2_t const high = vuzp1q_u64(first_key, second_key);
            uint64x2_t const high_tied = vceqq_u64(high, high_threshold);
            uint64x2_t const low_below =
                vandq_u64(high_tied, vcltq_u64(vuzp2q_u64(first_key, second_key), low_threshold));
            accumulated = vsubq_u64(accumulated, vorrq_u64(vcltq_u64(high, high_threshold), low_below));
        }
        return vaddvq_u64(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                 std::span<std::uint64_t const, extent_> low_words,
                                                 key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        uint64x2_t const high_threshold = vdupq_n_u64(wanted.high);
        uint64x2_t const low_threshold = vdupq_n_u64(wanted.low);
        std::size_t const whole = high_words.size() - high_words.size() % 2;
        uint64x2_t accumulated = vdupq_n_u64(0);
        uint64x2_t tied = vdupq_n_u64(0);
        for (std::size_t offset = 0; offset < whole; offset += 2) {
            uint64x2_t const high = vld1q_u64(high_words.data() + offset);
            accumulated = vsubq_u64(accumulated, vcltq_u64(high, high_threshold));
            tied = vorrq_u64(tied, vceqq_u64(high, high_threshold));
        }
        // One branch per row rather than per load, since a present key ties exactly once at an unpredictable spot.
        if (vmaxvq_u32(vreinterpretq_u32_u64(tied)) != 0)
            for (std::size_t offset = 0; offset < whole; offset += 2) {
                uint64x2_t const high_tied = vceqq_u64(vld1q_u64(high_words.data() + offset), high_threshold);
                uint64x2_t const low_below = vcltq_u64(vld1q_u64(low_words.data() + offset), low_threshold);
                accumulated = vsubq_u64(accumulated, vandq_u64(high_tied, low_below));
            }
        return vaddvq_u64(accumulated) +
               serial_row_kit_t::count_below(high_words.subspan(whole), low_words.subspan(whole), wanted);
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint32_t const, mask_equal_width_k> keys,
                                                  std::uint32_t wanted) noexcept {
        uint32x4_t const needle = vdupq_n_u32(wanted);
        std::uint32_t const *const words = keys.data();
        uint16x8_t const low = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(words), needle)),
                                            vmovn_u32(vceqq_u32(vld1q_u32(words + 4), needle)));
        uint16x8_t const high = vcombine_u16(vmovn_u32(vceqq_u32(vld1q_u32(words + 8), needle)),
                                             vmovn_u32(vceqq_u32(vld1q_u32(words + 12), needle)));
        return mask_of_lanes_(vcombine_u8(vmovn_u16(low), vmovn_u16(high)));
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint64_t const, mask_equal_width_k> keys,
                                                  std::uint64_t wanted) noexcept {
        uint64x2_t const needle = vdupq_n_u64(wanted);
        uint8x8_t halves[2];
        for (std::size_t half = 0; half != 2; ++half) {
            std::uint64_t const *const words = keys.data() + half * 8;
            uint32x4_t const low = vcombine_u32(vmovn_u64(vceqq_u64(vld1q_u64(words), needle)),
                                                vmovn_u64(vceqq_u64(vld1q_u64(words + 2), needle)));
            uint32x4_t const high = vcombine_u32(vmovn_u64(vceqq_u64(vld1q_u64(words + 4), needle)),
                                                 vmovn_u64(vceqq_u64(vld1q_u64(words + 6), needle)));
            halves[half] = vmovn_u16(vcombine_u16(vmovn_u32(low), vmovn_u32(high)));
        }
        return mask_of_lanes_(vcombine_u8(halves[0], halves[1]));
    }

    /** Two words per @c CNT, which AArch64 only offers on vector registers, so a scalar popcount
     *  pays a register move each way. */
    [[nodiscard]] static std::uint64_t select_set_bit(std::span<std::uint64_t const> words, std::uint64_t sampled,
                                                      std::size_t remaining) noexcept {
        std::size_t word_index = static_cast<std::size_t>(sampled / 64);
        std::uint64_t const first = words[word_index] & ((~std::uint64_t {0} << (sampled % 64)) << 1);
        std::size_t const first_ones = vaddv_u8(vcnt_u8(vcreate_u8(first)));
        if (remaining <= first_ones) return word_index * 64 + serial_row_kit_t::select_in_word(first, remaining - 1);
        remaining -= first_ones;
        ++word_index;
        while (true) {
            uint8x16_t const bytes = vcntq_u8(vreinterpretq_u8_u64(vld1q_u64(words.data() + word_index)));
            uint64x2_t const counts = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(bytes)));
            std::size_t const low = vgetq_lane_u64(counts, 0);
            if (remaining <= low)
                return word_index * 64 + serial_row_kit_t::select_in_word(words[word_index], remaining - 1);
            std::size_t const both = low + vgetq_lane_u64(counts, 1);
            if (remaining <= both)
                return (word_index + 1) * 64 +
                       serial_row_kit_t::select_in_word(words[word_index + 1], remaining - low - 1);
            remaining -= both;
            word_index += 2;
        }
    }

  private:
    /** Packs sixteen all-ones or all-zeros byte lanes into 16 bits; NEON has no movemask. */
    static std::uint32_t mask_of_lanes_(uint8x16_t lanes) noexcept {
        static constexpr std::uint8_t weights_k[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        uint8x16_t const weighted = vandq_u8(lanes, vld1q_u8(weights_k));
        return static_cast<std::uint32_t>(vaddv_u8(vget_low_u8(weighted)) | vaddv_u8(vget_high_u8(weighted)) << 8);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct neon_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::neon_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion NEON Kit

#pragma region SVE Kit

#if SMASHTABLE_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** Scalable vectors sized to whatever the processor offers, a loop predicate over a scalar tail. */
struct sve_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::sve_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = neon_row_kit_t::mask_equal_width_k;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += svcntw()) {
            svbool_t const present = svwhilelt_b32_u64(offset, keys.size());
            svuint32_t const loaded = svld1_u32(present, keys.data() + offset);
            below += svcntp_b32(present, svcmplt_n_u32(present, loaded, wanted));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += svcntw()) {
            svbool_t const present = svwhilelt_b32_u64(offset, keys.size());
            svint32_t const loaded = svld1_s32(present, keys.data() + offset);
            below += svcntp_b32(present, svcmplt_n_s32(present, loaded, wanted));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                 std::uint64_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += svcntd()) {
            svbool_t const present = svwhilelt_b64_u64(offset, keys.size());
            svuint64_t const loaded = svld1_u64(present, keys.data() + offset);
            below += svcntp_b64(present, svcmplt_n_u64(present, loaded, wanted));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                 std::int64_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += svcntd()) {
            svbool_t const present = svwhilelt_b64_u64(offset, keys.size());
            svint64_t const loaded = svld1_s64(present, keys.data() + offset);
            below += svcntp_b64(present, svcmplt_n_s64(present, loaded, wanted));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<key128_t const, extent_> keys, key128_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0; offset < keys.size(); offset += svcntd()) {
            svbool_t const present = svwhilelt_b64_u64(offset, keys.size());
            svuint64x2_t const columns =
                svld2_u64(present, static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data() + offset)));
            svuint64_t const high = svget2_u64(columns, 0);
            svbool_t const high_tied = svcmpeq_n_u64(present, high, wanted.high);
            svbool_t const low_below = svcmplt_n_u64(high_tied, svget2_u64(columns, 1), wanted.low);
            below += svcntp_b64(present, svorr_b_z(present, svcmplt_n_u64(present, high, wanted.high), low_below));
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                 std::span<std::uint64_t const, extent_> low_words,
                                                 key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        std::size_t below = 0;
        bool has_ties = false;
        for (std::size_t offset = 0; offset < high_words.size(); offset += svcntd()) {
            svbool_t const present = svwhilelt_b64_u64(offset, high_words.size());
            svuint64_t const high = svld1_u64(present, high_words.data() + offset);
            below += svcntp_b64(present, svcmplt_n_u64(present, high, wanted.high));
            has_ties |= svptest_any(present, svcmpeq_n_u64(present, high, wanted.high));
        }
        // One branch per row, and the second pass loads only the low words whose high words tie.
        if (has_ties)
            for (std::size_t offset = 0; offset < high_words.size(); offset += svcntd()) {
                svbool_t const present = svwhilelt_b64_u64(offset, high_words.size());
                svbool_t const high_tied =
                    svcmpeq_n_u64(present, svld1_u64(present, high_words.data() + offset), wanted.high);
                svuint64_t const low = svld1_u64(high_tied, low_words.data() + offset);
                below += svcntp_b64(present, svcmplt_n_u64(high_tied, low, wanted.low));
            }
        return below;
    }

    /** A predicate has no cheap move into a general register, so a window takes the NEON form
     *  every SVE core carries. */
    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint32_t const, mask_equal_width_k> keys,
                                                  std::uint32_t wanted) noexcept {
        return neon_row_kit_t::mask_equal(keys, wanted);
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint64_t const, mask_equal_width_k> keys,
                                                  std::uint64_t wanted) noexcept {
        return neon_row_kit_t::mask_equal(keys, wanted);
    }

    [[nodiscard]] static std::uint64_t select_set_bit(std::span<std::uint64_t const> words, std::uint64_t sampled,
                                                      std::size_t remaining) noexcept {
        return neon_row_kit_t::select_set_bit(words, sampled, remaining);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct sve_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::sve_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion SVE Kit

#pragma region RVV Kit

#if SMASHTABLE_TARGET_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#endif

/** RISC-V vectors at whatever length the processor grants per step, with no tail. */
struct rvv_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = 16;

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                                           std::uint32_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0, length = 0; offset < keys.size(); offset += length) {
            length = __riscv_vsetvl_e32m1(keys.size() - offset);
            vuint32m1_t const loaded = __riscv_vle32_v_u32m1(keys.data() + offset, length);
            below += __riscv_vcpop_m_b32(__riscv_vmsltu_vx_u32m1_b32(loaded, wanted, length), length);
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                                           std::int32_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0, length = 0; offset < keys.size(); offset += length) {
            length = __riscv_vsetvl_e32m1(keys.size() - offset);
            vint32m1_t const loaded = __riscv_vle32_v_i32m1(keys.data() + offset, length);
            below += __riscv_vcpop_m_b32(__riscv_vmslt_vx_i32m1_b32(loaded, wanted, length), length);
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                                           std::uint64_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0, length = 0; offset < keys.size(); offset += length) {
            length = __riscv_vsetvl_e64m1(keys.size() - offset);
            vuint64m1_t const loaded = __riscv_vle64_v_u64m1(keys.data() + offset, length);
            below += __riscv_vcpop_m_b64(__riscv_vmsltu_vx_u64m1_b64(loaded, wanted, length), length);
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                                           std::int64_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0, length = 0; offset < keys.size(); offset += length) {
            length = __riscv_vsetvl_e64m1(keys.size() - offset);
            vint64m1_t const loaded = __riscv_vle64_v_i64m1(keys.data() + offset, length);
            below += __riscv_vcpop_m_b64(__riscv_vmslt_vx_i64m1_b64(loaded, wanted, length), length);
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<key128_t const, extent_> keys,
                                                                           key128_t wanted) noexcept {
        std::size_t below = 0;
        for (std::size_t offset = 0, length = 0; offset < keys.size(); offset += length) {
            length = __riscv_vsetvl_e64m1(keys.size() - offset);
            vuint64m1x2_t const columns = __riscv_vlseg2e64_v_u64m1x2(
                static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data() + offset)), length);
            vuint64m1_t const high = __riscv_vget_v_u64m1x2_u64m1(columns, 0);
            vuint64m1_t const low = __riscv_vget_v_u64m1x2_u64m1(columns, 1);
            vbool64_t const high_tied = __riscv_vmseq_vx_u64m1_b64(high, wanted.high, length);
            vbool64_t const low_below =
                __riscv_vmand_mm_b64(high_tied, __riscv_vmsltu_vx_u64m1_b64(low, wanted.low, length), length);
            vbool64_t const high_below = __riscv_vmsltu_vx_u64m1_b64(high, wanted.high, length);
            below += __riscv_vcpop_m_b64(__riscv_vmor_mm_b64(high_below, low_below, length), length);
        }
        return below;
    }

    template <std::size_t extent_>
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::size_t count_below(
        std::span<std::uint64_t const, extent_> high_words, std::span<std::uint64_t const, extent_> low_words,
        key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        std::size_t below = 0;
        std::size_t ties = 0;
        for (std::size_t offset = 0, length = 0; offset < high_words.size(); offset += length) {
            length = __riscv_vsetvl_e64m1(high_words.size() - offset);
            vuint64m1_t const high = __riscv_vle64_v_u64m1(high_words.data() + offset, length);
            below += __riscv_vcpop_m_b64(__riscv_vmsltu_vx_u64m1_b64(high, wanted.high, length), length);
            ties += __riscv_vcpop_m_b64(__riscv_vmseq_vx_u64m1_b64(high, wanted.high, length), length);
        }
        // One branch per row rather than per step, and low words are read only for a row where a high word ties.
        if (ties != 0)
            for (std::size_t offset = 0, length = 0; offset < high_words.size(); offset += length) {
                length = __riscv_vsetvl_e64m1(high_words.size() - offset);
                vuint64m1_t const high = __riscv_vle64_v_u64m1(high_words.data() + offset, length);
                vuint64m1_t const low = __riscv_vle64_v_u64m1(low_words.data() + offset, length);
                vbool64_t const high_tied = __riscv_vmseq_vx_u64m1_b64(high, wanted.high, length);
                vbool64_t const low_below = __riscv_vmsltu_vx_u64m1_b64(low, wanted.low, length);
                below += __riscv_vcpop_m_b64(__riscv_vmand_mm_b64(high_tied, low_below, length), length);
            }
        return below;
    }

    /** One compare over the whole window: the V extension's 128-bit floor fits it in a register
     *  group, and the mask is stored as the bytes the caller wants. */
    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::uint32_t mask_equal(
        std::span<std::uint32_t const, mask_equal_width_k> keys, std::uint32_t wanted) noexcept {
        std::size_t const length = __riscv_vsetvl_e32m4(keys.size());
        vbool8_t const equal = __riscv_vmseq_vx_u32m4_b8(__riscv_vle32_v_u32m4(keys.data(), length), wanted, length);
        std::uint8_t bits[2] = {0, 0};
        __riscv_vsm_v_b8(bits, equal, length);
        return static_cast<std::uint32_t>(bits[0] | bits[1] << 8);
    }

    [[nodiscard]] SMASHTABLE_RVV_ATTRIBUTE_ static std::uint32_t mask_equal(
        std::span<std::uint64_t const, mask_equal_width_k> keys, std::uint64_t wanted) noexcept {
        std::size_t const length = __riscv_vsetvl_e64m8(keys.size());
        vbool8_t const equal = __riscv_vmseq_vx_u64m8_b8(__riscv_vle64_v_u64m8(keys.data(), length), wanted, length);
        std::uint8_t bits[2] = {0, 0};
        __riscv_vsm_v_b8(bits, equal, length);
        return static_cast<std::uint32_t>(bits[0] | bits[1] << 8);
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct rvv_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::rvv_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion RVV Kit

#pragma region V128 Kit

#if SMASHTABLE_TARGET_V128

/** WebAssembly SIMD over two 64-bit or four 32-bit keys per load; unsigned 64-bit order by flipping
 *  the sign bit, since the instruction set compares 64-bit lanes only as signed. */
struct v128_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::v128_k;
    static constexpr bool compiled_k = true;
    static constexpr std::size_t mask_equal_width_k = 16;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        v128_t const threshold = wasm_u32x4_splat(wanted);
        v128_t accumulated = wasm_i32x4_splat(0);
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            accumulated = wasm_i32x4_sub(accumulated, wasm_u32x4_lt(wasm_v128_load(keys.data() + offset), threshold));
        return sum_u32_lanes_(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        v128_t const threshold = wasm_i32x4_splat(wanted);
        v128_t accumulated = wasm_i32x4_splat(0);
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            accumulated = wasm_i32x4_sub(accumulated, wasm_i32x4_lt(wasm_v128_load(keys.data() + offset), threshold));
        return sum_u32_lanes_(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
                                                 std::uint64_t wanted) noexcept {
        return count_below_words_(keys, wanted, std::uint64_t {1} << 63);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
                                                 std::int64_t wanted) noexcept {
        std::span<std::uint64_t const, extent_> const words(
            static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data())), keys.size());
        return count_below_words_(words, static_cast<std::uint64_t>(wanted), 0);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<key128_t const, extent_> keys, key128_t wanted) noexcept {
        std::uint64_t const sign_bit = std::uint64_t {1} << 63;
        v128_t const sign = wasm_u64x2_splat(sign_bit);
        v128_t const high_threshold = wasm_u64x2_splat(wanted.high ^ sign_bit);
        v128_t const low_threshold = wasm_u64x2_splat(wanted.low ^ sign_bit);
        v128_t accumulated = wasm_i64x2_splat(0);
        std::size_t offset = 0;
        for (; offset + 2 <= keys.size(); offset += 2) {
            std::uint64_t const *const words =
                static_cast<std::uint64_t const *>(static_cast<void const *>(keys.data() + offset));
            v128_t const first_key = wasm_v128_load(words);
            v128_t const second_key = wasm_v128_load(words + 2);
            v128_t const high = wasm_v128_xor(wasm_i64x2_shuffle(first_key, second_key, 0, 2), sign);
            v128_t const low = wasm_v128_xor(wasm_i64x2_shuffle(first_key, second_key, 1, 3), sign);
            v128_t const low_below =
                wasm_v128_and(wasm_i64x2_eq(high, high_threshold), wasm_i64x2_lt(low, low_threshold));
            accumulated = wasm_i64x2_sub(accumulated, wasm_v128_or(wasm_i64x2_lt(high, high_threshold), low_below));
        }
        return sum_u64_lanes_(accumulated) + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint64_t const, extent_> high_words,
                                                 std::span<std::uint64_t const, extent_> low_words,
                                                 key128_t wanted) noexcept {
        assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
        std::uint64_t const sign_bit = std::uint64_t {1} << 63;
        v128_t const sign = wasm_u64x2_splat(sign_bit);
        v128_t const high_threshold = wasm_u64x2_splat(wanted.high ^ sign_bit);
        v128_t const low_threshold = wasm_u64x2_splat(wanted.low ^ sign_bit);
        std::size_t const whole = high_words.size() - high_words.size() % 2;
        v128_t accumulated = wasm_i64x2_splat(0);
        v128_t tied = wasm_i64x2_splat(0);
        for (std::size_t offset = 0; offset < whole; offset += 2) {
            v128_t const high = wasm_v128_xor(wasm_v128_load(high_words.data() + offset), sign);
            accumulated = wasm_i64x2_sub(accumulated, wasm_i64x2_lt(high, high_threshold));
            tied = wasm_v128_or(tied, wasm_i64x2_eq(high, high_threshold));
        }
        // One branch per row, not per load: a present key ties once, at an unpredictable spot.
        if (wasm_v128_any_true(tied))
            for (std::size_t offset = 0; offset < whole; offset += 2) {
                v128_t const high = wasm_v128_xor(wasm_v128_load(high_words.data() + offset), sign);
                v128_t const low = wasm_v128_xor(wasm_v128_load(low_words.data() + offset), sign);
                v128_t const low_below =
                    wasm_v128_and(wasm_i64x2_eq(high, high_threshold), wasm_i64x2_lt(low, low_threshold));
                accumulated = wasm_i64x2_sub(accumulated, low_below);
            }
        return sum_u64_lanes_(accumulated) +
               serial_row_kit_t::count_below(high_words.subspan(whole), low_words.subspan(whole), wanted);
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint32_t const, mask_equal_width_k> keys,
                                                  std::uint32_t wanted) noexcept {
        v128_t const needle = wasm_u32x4_splat(wanted);
        std::uint32_t matched = 0;
        for (std::size_t offset = 0; offset != keys.size(); offset += 4)
            matched |= wasm_i32x4_bitmask(wasm_i32x4_eq(wasm_v128_load(keys.data() + offset), needle)) << offset;
        return matched;
    }

    [[nodiscard]] static std::uint32_t mask_equal(std::span<std::uint64_t const, mask_equal_width_k> keys,
                                                  std::uint64_t wanted) noexcept {
        v128_t const needle = wasm_u64x2_splat(wanted);
        std::uint32_t matched = 0;
        for (std::size_t offset = 0; offset != keys.size(); offset += 2)
            matched |= wasm_i64x2_bitmask(wasm_i64x2_eq(wasm_v128_load(keys.data() + offset), needle)) << offset;
        return matched;
    }

  private:
    template <std::size_t extent_>
    static std::size_t count_below_words_(std::span<std::uint64_t const, extent_> words, std::uint64_t wanted,
                                          std::uint64_t flip) noexcept {
        v128_t const flipper = wasm_u64x2_splat(flip);
        v128_t const threshold = wasm_u64x2_splat(wanted ^ flip);
        v128_t accumulated = wasm_i64x2_splat(0);
        std::size_t offset = 0;
        for (; offset + 2 <= words.size(); offset += 2) {
            v128_t const flipped = wasm_v128_xor(wasm_v128_load(words.data() + offset), flipper);
            accumulated = wasm_i64x2_sub(accumulated, wasm_i64x2_lt(flipped, threshold));
        }
        std::size_t below = sum_u64_lanes_(accumulated);
        for (; offset < words.size(); ++offset)
            below += static_cast<std::size_t>(static_cast<std::int64_t>(words[offset] ^ flip) <
                                              static_cast<std::int64_t>(wanted ^ flip));
        return below;
    }

    static std::size_t sum_u32_lanes_(v128_t lanes) noexcept {
        return static_cast<std::size_t>(wasm_u32x4_extract_lane(lanes, 0)) + wasm_u32x4_extract_lane(lanes, 1) +
               wasm_u32x4_extract_lane(lanes, 2) + wasm_u32x4_extract_lane(lanes, 3);
    }

    static std::size_t sum_u64_lanes_(v128_t lanes) noexcept {
        return static_cast<std::size_t>(wasm_u64x2_extract_lane(lanes, 0) + wasm_u64x2_extract_lane(lanes, 1));
    }
};
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of
 *  preprocessor branches. */
struct v128_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::v128_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion V128 Kit

#pragma region Kit Dispatch

/** A kit this build carries, ranking every key type in both 16-byte layouts and masking its
 *  @c mask_equal_width_k of equal 4- or 8-byte keys at once, at most a 32-bit mask's worth. */
template <typename kit_type_>
concept row_kit = kit_type_::compiled_k && (kit_type_::mask_equal_width_k <= 32) &&
                  requires(std::span<std::uint32_t const> narrow_keys, std::span<std::int32_t const> signed_narrow_keys,
                           std::span<std::uint64_t const> words, std::span<std::int64_t const> signed_keys,
                           std::span<key128_t const> wide_keys,
                           std::span<std::uint32_t const, kit_type_::mask_equal_width_k> narrow_window,
                           std::span<std::uint64_t const, kit_type_::mask_equal_width_k> window) {
                      { kit_type_::mask_equal(narrow_window, std::uint32_t {}) } -> std::same_as<std::uint32_t>;
                      { kit_type_::mask_equal(window, std::uint64_t {}) } -> std::same_as<std::uint32_t>;
                      { kit_type_::count_below(narrow_keys, std::uint32_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(signed_narrow_keys, std::int32_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(words, std::uint64_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(signed_keys, std::int64_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(wide_keys, key128_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(words, words, key128_t {}) } -> std::same_as<std::size_t>;
                  };

/** A kit with a select scan of its own, several words per instruction; the others take the serial
 *  kit's, which every Elias-Fano sequence can fall back on. */
template <typename kit_type_>
concept select_kit =
    row_kit<kit_type_> && requires(std::span<std::uint64_t const> words, std::uint64_t sampled, std::size_t remaining) {
        { kit_type_::select_set_bit(words, sampled, remaining) } -> std::same_as<std::uint64_t>;
    };

/** The newest kit the compiler's flags promise, for code selecting at compile time, not at open. */
#if SMASHTABLE_TARGET_ICELAKE && defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && \
    defined(__AVX512DQ__) && defined(__AVX512VPOPCNTDQ__) && defined(__BMI2__)
using native_row_kit_t = icelake_row_kit_t;
#elif SMASHTABLE_TARGET_SKYLAKE && defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && \
    defined(__AVX512DQ__)
using native_row_kit_t = skylake_row_kit_t;
#elif SMASHTABLE_TARGET_HASWELL && defined(__AVX2__) && defined(__BMI2__)
using native_row_kit_t = haswell_row_kit_t;
#elif SMASHTABLE_TARGET_SVE && defined(__ARM_FEATURE_SVE)
using native_row_kit_t = sve_row_kit_t;
#elif SMASHTABLE_TARGET_NEON
using native_row_kit_t = neon_row_kit_t;
#elif SMASHTABLE_TARGET_RVV && defined(__riscv_v)
using native_row_kit_t = rvv_row_kit_t;
#elif SMASHTABLE_TARGET_V128
using native_row_kit_t = v128_row_kit_t;
#else
using native_row_kit_t = serial_row_kit_t;
#endif

/**
 *  @brief Calls @p callback with an instance of the kit @p kit names, or of the serial kit when
 *      this build lacks it. The callback is instantiated once per compiled kit, so a structure it
 *      builds calls that kit with no dispatch.
 *
 *  @warning Visiting a kit the processor cannot run faults on its first search; pass a kit
 *      @c row_kit_supported accepted, such as the one @c detect_row_kit returns.
 */
template <typename callback_type_>
constexpr decltype(auto) visit_row_kit(row_kit_t kit, callback_type_ &&callback) noexcept {
    switch (kit) {
    case row_kit_t::haswell_k:
        if constexpr (haswell_row_kit_t::compiled_k) return callback(haswell_row_kit_t {});
        break;
    case row_kit_t::skylake_k:
        if constexpr (skylake_row_kit_t::compiled_k) return callback(skylake_row_kit_t {});
        break;
    case row_kit_t::icelake_k:
        if constexpr (icelake_row_kit_t::compiled_k) return callback(icelake_row_kit_t {});
        break;
    case row_kit_t::neon_k:
        if constexpr (neon_row_kit_t::compiled_k) return callback(neon_row_kit_t {});
        break;
    case row_kit_t::sve_k:
        if constexpr (sve_row_kit_t::compiled_k) return callback(sve_row_kit_t {});
        break;
    case row_kit_t::rvv_k:
        if constexpr (rvv_row_kit_t::compiled_k) return callback(rvv_row_kit_t {});
        break;
    case row_kit_t::v128_k:
        if constexpr (v128_row_kit_t::compiled_k) return callback(v128_row_kit_t {});
        break;
    case row_kit_t::serial_k: break;
    }
    return callback(serial_row_kit_t {});
}

#pragma endregion Kit Dispatch

#pragma region Bounds and Ranks

/** The key right after @p key, or @p key itself when nothing orders after it. */
template <typename key_type_>
    requires row_searchable_key<key_type_>
[[nodiscard]] constexpr key_type_ successor_or_self(key_type_ key) noexcept {
    if constexpr (std::same_as<key_type_, key128_t>) {
        if (key.low != std::numeric_limits<std::uint64_t>::max()) return key128_t {key.high, key.low + 1};
        if (key.high != std::numeric_limits<std::uint64_t>::max()) return key128_t {key.high + 1, 0};
        return key;
    }
    else return key == std::numeric_limits<key_type_>::max() ? key : static_cast<key_type_>(key + 1);
}

/** How many of @p keys order at or below @p wanted; for a sorted row, the upper bound. */
template <row_kit row_kit_type_, typename key_type_, std::size_t extent_>
    requires row_searchable_key<key_type_>
[[nodiscard]] std::size_t count_not_above(std::span<key_type_ const, extent_> keys,
                                          std::type_identity_t<key_type_> wanted) noexcept {
    key_type_ const successor = successor_or_self(wanted);
    return successor == wanted ? keys.size() : row_kit_type_::count_below(keys, successor);
}

/** How many keys of a split row order at or below @p wanted. */
template <row_kit row_kit_type_, std::size_t extent_>
[[nodiscard]] std::size_t count_not_above(std::span<std::uint64_t const, extent_> high_words,
                                          std::span<std::uint64_t const, extent_> low_words, key128_t wanted) noexcept {
    key128_t const successor = successor_or_self(wanted);
    return successor == wanted ? high_words.size() : row_kit_type_::count_below(high_words, low_words, successor);
}

/** The lower bound of @p wanted in @p sorted of any length. A row of up to about 256 bytes of keys
 *  goes to the kit whole; a longer one is first narrowed to such a window by branchless halving.
 *  Exact only when @p sorted is sorted. */
template <row_kit row_kit_type_, typename key_type_, std::size_t extent_>
    requires row_searchable_key<key_type_>
[[nodiscard]] std::size_t count_below_sorted(std::span<key_type_ const, extent_> sorted,
                                             std::type_identity_t<key_type_> wanted) noexcept {
    std::size_t const window = 256 / sizeof(key_type_);
    if (sorted.size() <= window) return row_kit_type_::count_below(sorted, wanted);
    std::size_t low = 0;
    std::size_t length = sorted.size();
    while (length > window) {
        std::size_t const half = length / 2;
        low += sorted[low + half - 1] < wanted ? half : 0;
        length -= half;
    }
    return low + row_kit_type_::count_below(sorted.subspan(low, length), wanted);
}

/** The lower bound of @p wanted in a sorted split row of any length, by 256 bytes of high words. */
template <row_kit row_kit_type_, std::size_t extent_>
[[nodiscard]] std::size_t count_below_sorted(std::span<std::uint64_t const, extent_> high_words,
                                             std::span<std::uint64_t const, extent_> low_words,
                                             key128_t wanted) noexcept {
    assert(high_words.size() == low_words.size() && "a split row has one low word per high word");
    std::size_t const window = 256 / sizeof(std::uint64_t);
    if (high_words.size() <= window) return row_kit_type_::count_below(high_words, low_words, wanted);
    std::size_t low = 0;
    std::size_t length = high_words.size();
    while (length > window) {
        std::size_t const half = length / 2;
        std::size_t const probe = low + half - 1;
        low += key128_t {high_words[probe], low_words[probe]} < wanted ? half : 0;
        length -= half;
    }
    return low + row_kit_type_::count_below(high_words.subspan(low, length), low_words.subspan(low, length), wanted);
}

/** The lower bound of @p wanted in the sorted row at @p row, laid out by @p format_type_ and
 *  searched by @p row_kit_type_. */
template <typename format_type_, row_kit row_kit_type_>
[[nodiscard]] std::size_t count_below_in_row(typename format_type_::word_t const *row,
                                             typename format_type_::key_t wanted) noexcept {
    if constexpr (format_type_::keys_are_split_k)
        return count_below_sorted<row_kit_type_>(
            std::span<std::uint64_t const, format_type_::keys_per_row_k>(row, format_type_::keys_per_row_k),
            std::span<std::uint64_t const, format_type_::keys_per_row_k>(row + format_type_::keys_per_row_k,
                                                                         format_type_::keys_per_row_k),
            wanted);
    else
        return count_below_sorted<row_kit_type_>(
            std::span<typename format_type_::key_t const, format_type_::keys_per_row_k>(row,
                                                                                        format_type_::keys_per_row_k),
            wanted);
}

#pragma endregion Bounds and Ranks

} // namespace ashvardanian::smashtable
