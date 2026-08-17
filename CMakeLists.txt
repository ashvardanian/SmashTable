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
#   * smashtable_test_transactional_hash - the transactional store over the open-addressed table,
#     running only the point-access suites an unordered core can answer.
#
# The Python extension is not built here: `setup.py` compiles the translation units under
# `python/smashtable/` directly, as a wheel build has no use for the test and install machinery below.
cmake_minimum_required(VERSION 3.21)
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

# `#pragma region` markers are ignored by Clang and GCC, but GCC still warns about them.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(smashtable_warnings INTERFACE -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    if (SMASHTABLE_WERROR)
        target_compile_options(smashtable_warnings INTERFACE -Werror)
    endif ()
elseif (MSVC)
    # Without a branch here MSVC compiles at its default warning level and `SMASHTABLE_WERROR` silently does nothing, so
    # the Windows leg of CI reports green on code the other two legs reject. `/permissive-` matters for a template-heavy
    # header: it turns on two-phase name lookup, which is what GCC and Clang already do.
    target_compile_options(smashtable_warnings INTERFACE /W4 /permissive- /utf-8)
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

    add_executable(smashtable_test_transactional_hash scripts/test_transactional_hash.cpp)
    target_include_directories(smashtable_test_transactional_hash PRIVATE scripts/)
    target_link_libraries(smashtable_test_transactional_hash PRIVATE smashtable::smashtable smashtable_warnings)

    enable_testing()
    add_test(NAME test_std_set COMMAND smashtable_test_std_set)
    add_test(NAME test_avl_tree COMMAND smashtable_test_avl_tree)
    add_test(NAME test_wb_tree COMMAND smashtable_test_wb_tree)
    add_test(NAME test_partitioned COMMAND smashtable_test_partitioned)
    add_test(NAME test_hash_table COMMAND smashtable_test_hash_table)
    add_test(NAME test_transactional_hash COMMAND smashtable_test_transactional_hash)
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
