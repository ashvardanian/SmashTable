# SmashTable is a header-only C++20 library, so CMake is here to build the tests and the example,
# to export a `find_package` target, and to arm the repository's Git hooks. Use like:
#
#   cmake -B build_release -D CMAKE_BUILD_TYPE=Release
#   cmake --build build_release
#   ctest --test-dir build_release --output-on-failure
#
# CMake options:
#
#   -D CMAKE_BUILD_TYPE=<Release|Debug|RelWithDebInfo>  (default: Release, as the main project)
#   -D SMASHTABLE_BUILD_TESTS=<ON|OFF>                  (default: ON as the main project, else OFF)
#   -D SMASHTABLE_INSTALL=<ON|OFF>                      (default: ON as the main project, else OFF)
#   -D SMASHTABLE_WERROR=<ON|OFF>                       (default: OFF)
#   -D SMASHTABLE_BUILD_CUDA=<ON|OFF>                   (default: ON as the main project, else OFF)
#   -D CMAKE_CUDA_ARCHITECTURES=<list>                  (default: native; name a list when building GPU-less)
#
# Targets:
#
#   * smashtable::smashtable - the INTERFACE library consumers link, carrying the include path and
#     `cxx_std_20`. Spelled the same through `add_subdirectory` and `find_package(smashtable)`.
#   * smashtable_warnings - INTERFACE flags for our own tests and example, never for consumers.
#   * smashtable_example - compile-time tour instantiating the shared API over every container family.
#   * smashtable_test_std_set, _avl_tree, _wb_tree, _partitioned - one binary per ordered container
#     family running the same suites, so a behavioural difference between families surfaces as a
#     failure. Registered with CTest under the shorter `test_*` names, and each honors
#     `SMASHTABLE_FILTER`.
#   * smashtable_test_hash_table - the unordered family, running the hash-specific suites of
#     `scripts/test_unordered.hpp` instead, including the threaded ones.
#   * smashtable_test_monotonic_hash - the transactional store over the open-addressed table,
#     running only the point-access suites an unordered core can answer.
#   * smashtable_test_concurrent_cuda - the pinned lock-free table driven from kernels and from the
#     host over one managed allocation. Built only where `nvcc` is visible, and skipping at runtime
#     when no device is, so a GPU-less machine still configures and tests green.
#
# The Python extension is not built here: `setup.py` compiles the translation units under
# `python/` directly, as a wheel build has no use for the test and install machinery below.
cmake_minimum_required(VERSION 3.21)

# Leverage `ccache` where it is installed, which turns a rebuild of an unchanged translation unit into a
# lookup. This tree compiles 16 of them, several taking tens of seconds, so the cache earns its keep.
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif ()
project(
    smashtable
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Safer associative containers with DBMS-like transactions in C++"
    HOMEPAGE_URL "https://github.com/ashvardanian/SmashTable"
)

set(SMASHTABLE_IS_MAIN_PROJECT ${PROJECT_IS_TOP_LEVEL})

# Only steer global state when we own the build; a consumer's standard and output paths are theirs.
if (SMASHTABLE_IS_MAIN_PROJECT)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # An unset build type means no optimization flags at all on single-config generators, which is a surprising default
    # for a header-only library whose tests spend their time inside inlined templates.
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE
            Release
            CACHE STRING "Build type" FORCE
        )
    endif ()
endif ()

include(GNUInstallDirs)

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (SMASHTABLE_IS_MAIN_PROJECT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _smashtable_hooks_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (_smashtable_hooks_status EQUAL 0)
        message(STATUS "smashtable: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# Header-only library
add_library(smashtable INTERFACE)
add_library(smashtable::smashtable ALIAS smashtable)

target_include_directories(
    smashtable INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(smashtable INTERFACE cxx_std_20)

# Compiler warnings for tests/examples
add_library(smashtable_warnings INTERFACE)

option(SMASHTABLE_WERROR "Treat compiler warnings as errors" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # GCC warns about the `#pragma region` markers Clang and GCC both ignore. `-Wshadow` is not here:
    # the two compilers disagree about what it means, and GCC counts a constructor parameter named after
    # the member it initializes, which is the idiom rather than a defect.
    target_compile_options(smashtable_warnings INTERFACE -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    if (SMASHTABLE_WERROR)
        target_compile_options(smashtable_warnings INTERFACE -Werror)
    endif ()
elseif (MSVC)
    # Without this branch MSVC compiles at its default warning level and `SMASHTABLE_WERROR` silently does nothing.
    # `/permissive-` buys two-phase lookup, `/Zc:preprocessor` buys the `__VA_OPT__` the assertion macros are built
    # on, and `/bigobj` is cheap insurance: the partitioned suite already emits 2k sections against a 64k ceiling.
    target_compile_options(smashtable_warnings INTERFACE /W4 /permissive- /utf-8 /Zc:preprocessor /bigobj)
    # C4324 fires on the over-alignment the alignment tests ask for on purpose. C4702 judges reachability per
    # instantiation after inlining, so the dead arm of an `if constexpr` and the `return` past a test callback
    # that asserts it can never fire both trip it; GCC and Clang judge the template and say nothing.
    # C4456-C4459 are the shadowing family, left off here for the same reason `-Wshadow` is left off above.
    target_compile_options(smashtable_warnings INTERFACE /wd4324 /wd4702 /wd4456 /wd4457 /wd4458 /wd4459)
    if (SMASHTABLE_WERROR)
        target_compile_options(smashtable_warnings INTERFACE /WX)
    endif ()
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(smashtable_warnings INTERFACE -fmax-errors=1)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(smashtable_warnings INTERFACE -ferror-limit=1)
endif ()

# Tests and examples
option(SMASHTABLE_BUILD_TESTS "Build tests and examples" ${SMASHTABLE_IS_MAIN_PROJECT})
option(SMASHTABLE_BUILD_CUDA "Build the CUDA test when a device compiler is visible" ${SMASHTABLE_IS_MAIN_PROJECT})

if (SMASHTABLE_BUILD_TESTS)
    add_executable(smashtable_example example.cpp)
    target_link_libraries(smashtable_example PRIVATE smashtable::smashtable smashtable_warnings)

    # Individual test executables for faster incremental builds
    add_executable(smashtable_test_std_set scripts/test_std_set.cpp)
    target_include_directories(smashtable_test_std_set PRIVATE scripts/)
    target_link_libraries(smashtable_test_std_set PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_avl_tree scripts/test_avl_tree.cpp)
    target_include_directories(smashtable_test_avl_tree PRIVATE scripts/)
    target_link_libraries(smashtable_test_avl_tree PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_wb_tree scripts/test_wb_tree.cpp)
    target_include_directories(smashtable_test_wb_tree PRIVATE scripts/)
    target_link_libraries(smashtable_test_wb_tree PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_partitioned scripts/test_partitioned.cpp)
    target_include_directories(smashtable_test_partitioned PRIVATE scripts/)
    target_link_libraries(smashtable_test_partitioned PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_hash_table scripts/test_hash_table.cpp)
    target_include_directories(smashtable_test_hash_table PRIVATE scripts/)
    target_link_libraries(smashtable_test_hash_table PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_monotonic_hash scripts/test_monotonic_hash.cpp)
    target_include_directories(smashtable_test_monotonic_hash PRIVATE scripts/)
    target_link_libraries(smashtable_test_monotonic_hash PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_snapshot_store scripts/test_snapshot_store.cpp)
    target_include_directories(smashtable_test_snapshot_store PRIVATE scripts/)
    target_link_libraries(smashtable_test_snapshot_store PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_shared scripts/test_shared.cpp)
    target_include_directories(smashtable_test_shared PRIVATE scripts/)
    target_link_libraries(smashtable_test_shared PRIVATE smashtable::smashtable smashtable_warnings)

    add_executable(smashtable_test_vector scripts/test_vector.cpp)
    target_include_directories(smashtable_test_vector PRIVATE scripts/)
    target_link_libraries(smashtable_test_vector PRIVATE smashtable::smashtable smashtable_warnings)

    # The CUDA leg is opt-out rather than opt-in, but only materializes where a device compiler exists.
    # `check_language` finding one says nothing about a device being present, so the binary itself
    # skips when `cudaGetDeviceCount` comes back empty - that is what keeps a GPU-less runner green.
    include(CheckLanguage)
    check_language(CUDA)

    if (SMASHTABLE_BUILD_CUDA AND CMAKE_CUDA_COMPILER)
        # Decide the architectures before `enable_language(CUDA)`, which otherwise fills them in from
        # the compiler's own default of `sm_52` - too old to assemble the scoped atomics the slot
        # protocol is built on, so it fails in `ptxas` rather than at runtime. `native` suits anyone
        # holding a device; a builder without one, CI included, names its own list.
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif ()

        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_executable(smashtable_test_concurrent_cuda scripts/test_atomic_cuda.cu)
        target_include_directories(smashtable_test_concurrent_cuda PRIVATE scripts/)
        target_link_libraries(smashtable_test_concurrent_cuda PRIVATE smashtable::smashtable)

        # `--expt-relaxed-constexpr` is what makes the library's `constexpr` surface device-callable;
        # without it every probe, lock and unlock is a host function a kernel may not call.
        target_compile_options(
            smashtable_test_concurrent_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        message(STATUS "smashtable: CUDA test enabled for ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()

    enable_testing()
    add_test(NAME test_std_set COMMAND smashtable_test_std_set)
    add_test(NAME test_avl_tree COMMAND smashtable_test_avl_tree)
    add_test(NAME test_wb_tree COMMAND smashtable_test_wb_tree)
    add_test(NAME test_partitioned COMMAND smashtable_test_partitioned)
    add_test(NAME test_hash_table COMMAND smashtable_test_hash_table)
    add_test(NAME test_monotonic_hash COMMAND smashtable_test_monotonic_hash)
    add_test(NAME test_snapshot_store COMMAND smashtable_test_snapshot_store)
    add_test(NAME test_shared COMMAND smashtable_test_shared)
    add_test(NAME test_vector COMMAND smashtable_test_vector)
    if (TARGET smashtable_test_concurrent_cuda)
        add_test(NAME test_atomic_cuda COMMAND smashtable_test_concurrent_cuda)
    endif ()
endif ()

option(SMASHTABLE_INSTALL "Install headers and CMake package files" ${SMASHTABLE_IS_MAIN_PROJECT})

if (SMASHTABLE_INSTALL)
    # Installation
    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.hpp"
    )

    install(
        TARGETS smashtable
        EXPORT smashtableTargets
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        EXPORT smashtableTargets
        FILE smashtableTargets.cmake
        NAMESPACE smashtable::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
    )

    # Package config for find_package(smashtable)
    include(CMakePackageConfigHelpers)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/smashtableConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
        NO_CHECK_REQUIRED_COMPONENTS_MACRO
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
    )
endif ()
