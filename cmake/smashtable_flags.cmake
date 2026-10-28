# Sanitizers, and the warning flags SmashTable's own targets take through `smashtable_warnings`.

# Compiler warnings for tests/examples
add_library(smashtable_warnings INTERFACE)

option(SMASHTABLE_WERROR "Treat compiler warnings as errors" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # GCC warns about the `#pragma region` markers Clang and GCC both ignore. `-Wshadow` is not here:
    # the two compilers disagree about what it means, and GCC counts a constructor parameter named after
    # the member it initializes, which is the idiom rather than a defect.
    target_compile_options(
        smashtable_warnings INTERFACE -Wall -Wextra -pedantic -Wno-unknown-pragmas
                                      $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wdocumentation>
    )
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

# Sanitizers instrument every target this directory builds, and never a parent project's own targets.
set(SMASHTABLE_BUILD_SANITIZERS
    "off"
    CACHE STRING "Sanitizers for SmashTable's own targets: off, address with undefined, or thread"
)
set_property(CACHE SMASHTABLE_BUILD_SANITIZERS PROPERTY STRINGS off address thread)
if (SMASHTABLE_BUILD_SANITIZERS STREQUAL "address" AND MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/fsanitize=address>")
elseif (SMASHTABLE_BUILD_SANITIZERS STREQUAL "address")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>")
    add_link_options("$<$<LINK_LANGUAGE:CXX>:-fsanitize=address,undefined>")
elseif (SMASHTABLE_BUILD_SANITIZERS STREQUAL "thread" AND NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread;-fno-omit-frame-pointer>")
    add_link_options("$<$<LINK_LANGUAGE:CXX>:-fsanitize=thread>")
elseif (NOT SMASHTABLE_BUILD_SANITIZERS STREQUAL "off")
    message(FATAL_ERROR "SMASHTABLE_BUILD_SANITIZERS is off, address or thread, the last with Clang or GCC, "
                        "not '${SMASHTABLE_BUILD_SANITIZERS}'"
    )
endif ()
