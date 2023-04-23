cmake_minimum_required(VERSION 3.15)
project(ucset)

set(CMAKE_CXX_STANDARD 17)

add_library(ucset INTERFACE)
target_include_directories(ucset INTERFACE include)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)
include_directories(${googletest_SOURCE_DIR}/googletest/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunknown-attributes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ferror-limit=1")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
add_executable(example example.cpp)
target_link_libraries(example gtest)

add_executable(test test.cpp)
target_link_libraries(test gtest)