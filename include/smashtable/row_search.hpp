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

/** The kits, one enumerator per microarchitecture, newest instruction sets after their baselines. */
enum class row_kit_t : std::uint8_t {
    serial_k,
    haswell_k,
    skylake_k,
    neon_k,
    sve_k,
    rvv_k,
};

/** The enumerator's own name without its suffix, for a benchmark or a log. */
[[nodiscard]] constexpr char const *name_of(row_kit_t kit) noexcept {
    switch (kit) {
    case row_kit_t::serial_k: return "serial";
    case row_kit_t::haswell_k: return "haswell";
    case row_kit_t::skylake_k: return "skylake";
    case row_kit_t::neon_k: return "neon";
    case row_kit_t::sve_k: return "sve";
    case row_kit_t::rvv_k: return "rvv";
    }
    return "unrecognized";
}

/** Whether this build carries @p kit, whatever the processor running it. */
[[nodiscard]] constexpr bool row_kit_compiled(row_kit_t kit) noexcept {
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return ST_TARGET_HASWELL;
    case row_kit_t::skylake_k: return ST_TARGET_SKYLAKE;
    case row_kit_t::neon_k: return ST_TARGET_NEON;
    case row_kit_t::sve_k: return ST_TARGET_SVE;
    case row_kit_t::rvv_k: return ST_TARGET_RVV;
    }
    return false;
}

/** Whether this build carries @p kit and the running processor and operating system can execute it.
 *  Probes the processor on every call, so a caller asks once, at open or construction, and keeps
 *  the answer. */
#if ST_TARGET_X8664_ && (defined(__GNUC__) || defined(__clang__))
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    __builtin_cpu_init();
    bool const runs_haswell = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
                              __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("popcnt");
    bool const runs_skylake = runs_haswell && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
                              __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return ST_TARGET_HASWELL && runs_haswell;
    case row_kit_t::skylake_k: return ST_TARGET_SKYLAKE && runs_skylake;
    default: return false;
    }
}
#elif ST_TARGET_X8664_ && defined(_MSC_VER)
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
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::haswell_k: return ST_TARGET_HASWELL && runs_haswell;
    case row_kit_t::skylake_k: return ST_TARGET_SKYLAKE && runs_skylake;
    default: return false;
    }
}
#elif ST_TARGET_ARM64_ && defined(__linux__)
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    unsigned long const capabilities = getauxval(AT_HWCAP);
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::neon_k: return ST_TARGET_NEON && (capabilities & (1ul << 1)) != 0; // `HWCAP_ASIMD`
    case row_kit_t::sve_k: return ST_TARGET_SVE && (capabilities & (1ul << 22)) != 0;  // `HWCAP_SVE`
    default: return false;
    }
}
#elif ST_TARGET_ARM64_
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    // Advanced SIMD is the AArch64 baseline; SVE is only probed where the kernel reports it.
    return kit == row_kit_t::serial_k || (kit == row_kit_t::neon_k && ST_TARGET_NEON);
}
#elif ST_TARGET_RISCV64_ && defined(__linux__)
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept {
    unsigned long const capabilities = getauxval(AT_HWCAP);
    switch (kit) {
    case row_kit_t::serial_k: return true;
    case row_kit_t::rvv_k: return ST_TARGET_RVV && (capabilities & (1ul << ('V' - 'A'))) != 0;
    default: return false;
    }
}
#else
[[nodiscard]] inline bool row_kit_supported(row_kit_t kit) noexcept { return kit == row_kit_t::serial_k; }
#endif

/** The newest kit this build carries and the running processor executes, falling back to @c serial_k. */
[[nodiscard]] inline row_kit_t detect_row_kit() noexcept {
    for (row_kit_t const kit :
         {row_kit_t::skylake_k, row_kit_t::haswell_k, row_kit_t::sve_k, row_kit_t::neon_k, row_kit_t::rvv_k})
        if (row_kit_supported(kit)) return kit;
    return row_kit_t::serial_k;
}

#pragma endregion Kit Selection

#pragma region Serial Kit

/** The reference kit every other kit must agree with, and the one each of them finishes a partial chunk with. */
struct serial_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::serial_k;
    static constexpr bool compiled_k = true;

    /** How many of @p keys order below @p wanted; for a sorted row, the lower bound. */
    template <typename key_type_, std::size_t extent_>
        requires row_searchable_key<key_type_>
    [[nodiscard]] static constexpr std::size_t count_below(std::span<key_type_ const, extent_> keys,
                                                           std::type_identity_t<key_type_> wanted) noexcept {
        std::size_t below = 0;
        for (key_type_ const key : keys) below += static_cast<std::size_t>(key < wanted);
        return below;
    }

    /** How many keys of a split row order below @p wanted, reading a low word only where its high word ties. */
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
};

#pragma endregion Serial Kit

#pragma region Haswell Kit

#if ST_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "popcnt")
#endif

/** AVX2 over four 64-bit or eight 32-bit keys per load, unsigned order taken by flipping the sign bit. */
struct haswell_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::haswell_k;
    static constexpr bool compiled_k = true;

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

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of preprocessor
 * branches. */
struct haswell_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::haswell_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Haswell Kit

#pragma region Skylake Kit

#if ST_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "bmi", "bmi2", "popcnt")
#endif

/** AVX-512 over eight 64-bit or sixteen 32-bit keys per load, with masked loads in place of a scalar tail. */
struct skylake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::skylake_k;
    static constexpr bool compiled_k = true;

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

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of preprocessor
 * branches. */
struct skylake_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::skylake_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Skylake Kit

#pragma region Neon Kit

#if ST_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Advanced SIMD over two 64-bit or four 32-bit keys per load, deinterleaving 16-byte keys on the load. */
struct neon_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::neon_k;
    static constexpr bool compiled_k = true;

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
                                                 std::uint32_t wanted) noexcept {
        uint32x4_t const threshold = vdupq_n_u32(wanted);
        std::size_t below = 0;
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            below += vaddvq_u32(vshrq_n_u32(vcltq_u32(vld1q_u32(keys.data() + offset), threshold), 31));
        return below + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
    }

    template <std::size_t extent_>
    [[nodiscard]] static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
                                                 std::int32_t wanted) noexcept {
        int32x4_t const threshold = vdupq_n_s32(wanted);
        std::size_t below = 0;
        std::size_t offset = 0;
        for (; offset + 4 <= keys.size(); offset += 4)
            below += vaddvq_u32(vshrq_n_u32(vcltq_s32(vld1q_s32(keys.data() + offset), threshold), 31));
        return below + serial_row_kit_t::count_below(keys.subspan(offset), wanted);
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
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of preprocessor
 * branches. */
struct neon_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::neon_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Neon Kit

#pragma region Sve Kit

#if ST_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/** Scalable vectors at whatever length the processor offers, with a loop predicate in place of a scalar tail. */
struct sve_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::sve_k;
    static constexpr bool compiled_k = true;

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
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of preprocessor
 * branches. */
struct sve_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::sve_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Sve Kit

#pragma region Rvv Kit

#if ST_TARGET_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#endif

/** RISC-V vectors at whatever length the processor grants per step, with no tail. */
struct rvv_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::rvv_k;
    static constexpr bool compiled_k = true;

    template <std::size_t extent_>
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::uint32_t const, extent_> keys,
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
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::int32_t const, extent_> keys,
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
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::uint64_t const, extent_> keys,
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
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<std::int64_t const, extent_> keys,
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
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(std::span<key128_t const, extent_> keys,
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
    [[nodiscard]] ST_TARGET_RVV_ATTRIBUTE_ static std::size_t count_below(
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
};

#if defined(__clang__)
#pragma clang attribute pop
#endif
#else

/** The kit this build cannot carry, kept so @c visit_row_kit stays one switch rather than a run of preprocessor
 * branches. */
struct rvv_row_kit_t {
    static constexpr row_kit_t kit_k = row_kit_t::rvv_k;
    static constexpr bool compiled_k = false;
};
#endif

#pragma endregion Rvv Kit

#pragma region Kit Dispatch

/** A kit this build carries, answering every key type and both 16-byte layouts. */
template <typename kit_type_>
concept row_kit = kit_type_::compiled_k &&
                  requires(std::span<std::uint32_t const> narrow_keys, std::span<std::int32_t const> signed_narrow_keys,
                           std::span<std::uint64_t const> words, std::span<std::int64_t const> signed_keys,
                           std::span<key128_t const> wide_keys) {
                      { kit_type_::count_below(narrow_keys, std::uint32_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(signed_narrow_keys, std::int32_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(words, std::uint64_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(signed_keys, std::int64_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(wide_keys, key128_t {}) } -> std::same_as<std::size_t>;
                      { kit_type_::count_below(words, words, key128_t {}) } -> std::same_as<std::size_t>;
                  };

/** The newest kit the compiler's own flags promise, for code that selects at compile time rather than at open. */
#if ST_TARGET_SKYLAKE && defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
using native_row_kit_t = skylake_row_kit_t;
#elif ST_TARGET_HASWELL && defined(__AVX2__) && defined(__BMI2__)
using native_row_kit_t = haswell_row_kit_t;
#elif ST_TARGET_SVE && defined(__ARM_FEATURE_SVE)
using native_row_kit_t = sve_row_kit_t;
#elif ST_TARGET_NEON
using native_row_kit_t = neon_row_kit_t;
#elif ST_TARGET_RVV && defined(__riscv_v)
using native_row_kit_t = rvv_row_kit_t;
#else
using native_row_kit_t = serial_row_kit_t;
#endif

/**
 *  Calls @p callback with an instance of the kit @p kit names, or of the serial kit when this build
 *  lacks it. The callback is instantiated once per compiled kit, so a structure it builds calls
 *  that kit with no dispatch.
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
    case row_kit_t::neon_k:
        if constexpr (neon_row_kit_t::compiled_k) return callback(neon_row_kit_t {});
        break;
    case row_kit_t::sve_k:
        if constexpr (sve_row_kit_t::compiled_k) return callback(sve_row_kit_t {});
        break;
    case row_kit_t::rvv_k:
        if constexpr (rvv_row_kit_t::compiled_k) return callback(rvv_row_kit_t {});
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

/** The lower bound of @p wanted in a sorted split row of any length, windowed by 256 bytes of high words. */
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

/** The lower bound of @p wanted in the sorted row at @p row, laid out by @p format_type_ and searched by
 *  @p row_kit_type_. */
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
