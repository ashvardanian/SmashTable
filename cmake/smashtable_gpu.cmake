# The CUDA language, its architectures, and the CUDA test.

# The CUDA leg is opt-out rather than opt-in, but only materializes where a device compiler exists.
# `check_language` finding one says nothing about a device being present, so the binary itself
# skips when `cudaGetDeviceCount` comes back empty - that is what keeps a GPU-less runner green.
include(CheckLanguage)
check_language(CUDA)
# `check_language` shadows a cached `CMAKE_CUDA_HOST_COMPILER` with its own result; drop it to see the cache again.
unset(CMAKE_CUDA_HOST_COMPILER)

if (SMASHTABLE_CUDA_ARCH AND CMAKE_CUDA_COMPILER)
    # Decide the architectures before `enable_language(CUDA)`, which otherwise fills them in from
    # the compiler's own default of `sm_52` - too old to assemble the scoped atomics the slot
    # protocol is built on, so it fails in `ptxas` rather than at runtime. `all` is the release set, PTX
    # at `compute_75`, which every later GPU runs, so it needs no device; `native` suits anyone holding
    # one, and it or a named code compiles as SASS alone.
    if (SMASHTABLE_CUDA_ARCH STREQUAL "all")
        set(CMAKE_CUDA_ARCHITECTURES 75-virtual)
    elseif (SMASHTABLE_CUDA_ARCH STREQUAL "native")
        set(CMAKE_CUDA_ARCHITECTURES native)
    elseif (SMASHTABLE_CUDA_ARCH MATCHES "^[0-9]+[af]?$")
        set(CMAKE_CUDA_ARCHITECTURES ${SMASHTABLE_CUDA_ARCH}-real)
    else ()
        message(
            FATAL_ERROR "SMASHTABLE_CUDA_ARCH takes all, native or one code like 100f, not `${SMASHTABLE_CUDA_ARCH}`"
        )
    endif ()
    if (CMAKE_VERSION VERSION_LESS 3.31 AND CMAKE_CUDA_ARCHITECTURES MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()

    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "SmashTable needs CUDA 13 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_executable(smashtable_test_concurrent_cuda test/atomic_cuda.cu)
    target_include_directories(smashtable_test_concurrent_cuda PRIVATE test/)
    target_link_libraries(smashtable_test_concurrent_cuda PRIVATE smashtable::header)

    # `--expt-relaxed-constexpr` is what makes the library's `constexpr` surface device-callable;
    # without it every probe, lock and unlock is a host function a kernel may not call.
    target_compile_options(
        smashtable_test_concurrent_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
    )
    if (MSVC)
        target_compile_options(
            smashtable_test_concurrent_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
        )
    endif ()
    message(STATUS "smashtable: CUDA test enabled for ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

if (TARGET smashtable_test_concurrent_cuda)
    add_test(NAME test_atomic_cuda COMMAND smashtable_test_concurrent_cuda)
endif ()
