# SmashTable is a header-only C++20 library, so CMake is here to build the tests, the example and the
# Python extension, to export a `find_package` target, and to arm the repository's Git hooks. Use like:
#
#   cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D SMASHTABLE_BUILD_TEST=ON
#   cmake --build build_release
#   ctest --test-dir build_release --output-on-failure
#
# CMake options:
#
#   -D CMAKE_BUILD_TYPE=<Release|Debug|RelWithDebInfo>  (default: Release, as the main project)
#   -D SMASHTABLE_BUILD_TEST=<ON|OFF>                   (default: OFF, the presets switch it on)
#   -D SMASHTABLE_INSTALL=<ON|OFF>                      (default: ON as the main project, else OFF)
#   -D SMASHTABLE_WERROR=<ON|OFF>                       (default: OFF)
#   -D SMASHTABLE_BUILD_SANITIZERS=<off|address|thread> (default: off, address also arms UndefinedBehaviorSanitizer)
#   -D SMASHTABLE_BUILD_CUDA=<ON|OFF>                   (default: OFF, the `cuda` preset switches it on)
#   -D SMASHTABLE_BUILD_BENCH=<ON|OFF>                  (default: OFF; CTest never runs the benchmarks)
#   -D SMASHTABLE_BUILD_PYTHON=<ON|OFF>                 (default: OFF, `scikit-build-core` switches it on)
#   -D SMASHTABLE_CUDA_ARCHITECTURES=<list>             (default: one cubin per device family plus a PTX tail)
#
# Targets:
#
#   * smashtable::header - the INTERFACE library consumers link, carrying the include path and
#     `cxx_std_20`. Spelled the same through `add_subdirectory` and `find_package(smashtable)`.
#   * smashtable_warnings - INTERFACE flags for our own tests and example, never for consumers.
#   * smashtable_example - compile-time tour instantiating the shared API over every container family.
#   * smashtable_test_std_set, _avl_tree, _wb_tree, _partitioned - one binary per ordered container
#     family running the same suites, so a behavioural difference between families surfaces as a
#     failure. Registered with CTest under the shorter `test_*` names, and each honors
#     `SMASHTABLE_FILTER`.
#   * smashtable_test_hash_table - the unordered family, running the hash-specific suites of
#     `test/unordered.hpp` instead, including the threaded ones.
#   * smashtable_test_monotonic_hash - the transactional store over the open-addressed table,
#     running only the point-access suites an unordered core can answer.
#   * smashtable_test_row_search, _flat_set, _ring, _elias_fano - the ordered kits: every compiled row kit the
#     processor runs against the serial kit, the static B-tree and S+ tree against a sorted array, the micro-set
#     through each kit, the ring, and the Elias-Fano sequence and packed array against their source arrays.
#   * smashtable_bench_row_search - row kits per row width, and both immutable layouts against a binary
#     search, printing a machine block. Needs `-D SMASHTABLE_BUILD_BENCH=ON` and is never registered
#     with CTest, so no suite pays for compiling it.
#   * smashtable_test_concurrent_cuda - the pinned lock-free table driven from kernels and from the
#     host over one managed allocation. Built only where `nvcc` is visible, and skipping at runtime
#     when no device is, so a GPU-less machine still configures and tests green.
#   * smashtable_python - the CPython extension over the translation units under `python/`, which
#     `scikit-build-core` builds and packages from `pyproject.toml`.
cmake_minimum_required(VERSION 3.28)

# Embedded debug information avoids depending on a running MSVC PDB server during probes.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    if (NOT DEFINED CMAKE_MSVC_DEBUG_INFORMATION_FORMAT)
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
    endif ()
endif ()

# Leverage `ccache` where it is installed, which turns a rebuild of an unchanged translation unit into a
# lookup. This tree compiles 16 of them, several taking tens of seconds, so the cache earns its keep.
find_program(SMASHTABLE_CCACHE_PROGRAM ccache)
if (SMASHTABLE_CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${SMASHTABLE_CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SMASHTABLE_CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${SMASHTABLE_CCACHE_PROGRAM}")
endif ()
project(
    smashtable
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Safer associative containers with DBMS-like transactions in C++"
    HOMEPAGE_URL "https://github.com/ashvardanian/SmashTable"
)

# Only steer global state when we own the build; a consumer's standard and output paths are theirs.
if (PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    # No target uses C++20 modules, and a Clang installed without `clang-scan-deps` would fail every C++ unit.
    set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    # An unset build type means no optimization flags at all on single-config generators, which is a surprising default
    # for a header-only library whose tests spend their time inside inlined templates.
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release)
    endif ()
endif ()

include(GNUInstallDirs)

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE smashtable_hooks_status_
        OUTPUT_QUIET ERROR_QUIET
    )
    if (smashtable_hooks_status_ EQUAL 0)
        message(STATUS "smashtable: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# Header-only library
add_library(smashtable_header INTERFACE)
add_library(smashtable::header ALIAS smashtable_header)
set_target_properties(smashtable_header PROPERTIES EXPORT_NAME header)

target_include_directories(
    smashtable_header INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(smashtable_header INTERFACE cxx_std_20)

include(cmake/smashtable_flags.cmake)

# Tests and examples
option(SMASHTABLE_BUILD_TEST "Build tests and examples" OFF)
option(SMASHTABLE_BUILD_CUDA "Build the CUDA test when a device compiler is visible" OFF)
option(SMASHTABLE_BUILD_BENCH "Compile the native benchmarks, which CTest never runs" OFF)
option(SMASHTABLE_BUILD_PYTHON "Compile the CPython extension; `scikit-build-core` switches it on" OFF)

if (SMASHTABLE_BUILD_TEST)
    add_executable(smashtable_example example.cpp)
    target_link_libraries(smashtable_example PRIVATE smashtable::header smashtable_warnings)

    enable_testing()

    # One target per entry point, each its own translation unit so an incremental build stays small.
    foreach (
        subject_ IN
        ITEMS std_set
              avl_tree
              wb_tree
              partitioned
              hash_table
              monotonic_hash
              snapshot_store
              shared
              vector
              row_search
              flat_set
              ring
              elias_fano
    )
        add_executable(smashtable_test_${subject_} test/${subject_}.cpp)
        target_include_directories(smashtable_test_${subject_} PRIVATE test/)
        target_link_libraries(smashtable_test_${subject_} PRIVATE smashtable::header smashtable_warnings)
        add_test(NAME test_${subject_} COMMAND smashtable_test_${subject_})
    endforeach ()

    # No benchmark is registered with CTest, so no suite pays for one.
    if (SMASHTABLE_BUILD_BENCH)
        foreach (subject_ IN ITEMS row_search)
            add_executable(smashtable_bench_${subject_} bench/${subject_}.cpp)
            target_include_directories(smashtable_bench_${subject_} PRIVATE bench/)
            target_link_libraries(smashtable_bench_${subject_} PRIVATE smashtable::header smashtable_warnings)
        endforeach ()
    endif ()

    include(cmake/smashtable_gpu.cmake)
endif ()

# The `smashtable` extension module, installed at the wheel's root beside its stub and typing marker.
if (SMASHTABLE_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
    python_add_library(
        smashtable_python
        MODULE
        WITH_SOABI
        python/module.cpp
        python/container.cpp
        python/shared.cpp
        python/store_ops_hashed.cpp
        python/store_ops_sorted.cpp
        python/transactions.cpp
    )
    set_target_properties(smashtable_python PROPERTIES OUTPUT_NAME smashtable CXX_VISIBILITY_PRESET hidden)
    target_link_libraries(smashtable_python PRIVATE smashtable_header smashtable_warnings)
    install(
        TARGETS smashtable_python
        DESTINATION .
        COMPONENT smashtable_python
    )
    install(
        FILES python/smashtable.pyi python/py.typed
        DESTINATION .
        COMPONENT smashtable_python
    )
endif ()

include(cmake/smashtable_install.cmake)
