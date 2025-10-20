cmake_minimum_required(VERSION 3.15)
project(
    smashtable
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(GNUInstallDirs)

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

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(smashtable_warnings INTERFACE -Wunknown-attributes -ferror-limit=1)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(smashtable_warnings INTERFACE -pedantic -Werror -Wfatal-errors)
endif ()

# Tests and examples
option(SMASHTABLE_BUILD_TESTS "Build tests and examples" ON)

if (SMASHTABLE_BUILD_TESTS)
    include(FetchContent)

    set(INSTALL_GTEST
        OFF
        CACHE BOOL "" FORCE
    )
    set(BUILD_GMOCK
        OFF
        CACHE BOOL "" FORCE
    )
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE
    )

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )

    FetchContent_MakeAvailable(googletest)

    add_executable(smashtable_example example.cpp)
    target_link_libraries(smashtable_example PRIVATE smashtable::smashtable smashtable_warnings GTest::gtest)

    add_executable(smashtable_test test.cpp)
    target_link_libraries(smashtable_test PRIVATE smashtable::smashtable smashtable_warnings GTest::gtest)

    enable_testing()
    add_test(NAME smashtable_tests COMMAND smashtable_test)
endif ()

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
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/smashtableConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/smashtableConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smashtable
)
