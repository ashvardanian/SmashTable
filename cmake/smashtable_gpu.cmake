# The CUDA language, its architectures, and the CUDA test.

# The CUDA leg is opt-out rather than opt-in, but only materializes where a device compiler exists.
# `check_language` finding one says nothing about a device being present, so the binary itself
# skips when `cudaGetDeviceCount` comes back empty - that is what keeps a GPU-less runner green.
include(CheckLanguage)
check_language(CUDA)
# `check_language` shadows a cached `CMAKE_CUDA_HOST_COMPILER` with its own result; drop it to see the cache again.
unset(CMAKE_CUDA_HOST_COMPILER)

if (SMASHTABLE_BUILD_CUDA AND CMAKE_CUDA_COMPILER)
    # Decide the architectures before `enable_language(CUDA)`, which otherwise fills them in from
    # the compiler's own default of `sm_52` - too old to assemble the scoped atomics the slot
    # protocol is built on, so it fails in `ptxas` rather than at runtime. The default is the list every
    # sibling library builds, so it needs no device; `native` suits anyone holding one.
    set(SMASHTABLE_CUDA_ARCHITECTURES
        "75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
        CACHE STRING "GPU architectures baked into the CUDA test"
    )
    if (CMAKE_VERSION VERSION_LESS 3.31 AND SMASHTABLE_CUDA_ARCHITECTURES MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES "${SMASHTABLE_CUDA_ARCHITECTURES}")

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
