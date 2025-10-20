# Contributing to SmashTable

## Compiling C++ Code

Release build:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -B build_release
cmake --build build_release --config Release --target smashtable_test --parallel
build_release/smashtable_test
```

Debug build:

```bash
cmake -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
build_debug/smashtable_test
```

### Running Tests

Run all tests:

```bash
build_debug/smashtable_test
```

Run specific test suites:

```bash
build_debug/smashtable_test --gtest_filter="upsert_and_find*"
build_debug/smashtable_test --gtest_filter="transaction_*"
build_debug/smashtable_test --gtest_filter="*with_threads"
```

Brief output:

```bash
build_debug/smashtable_test --gtest_brief=1
```

List available tests:

```bash
build_debug/smashtable_test --gtest_list_tests
```

## Code and Documentation Styling Guidelines

Code is formatted automatically using `clang-format` with the configuration specified in `.clang-format`.
CMake is formatted using `cmake-format` with the configuration specified in `.cmake-format.py`.
Please ensure your code adheres to this style before submitting a pull request.

Format C++ files:

```bash
clang-format -i include/smashtable/*.hpp
clang-format -i test.cpp example.cpp
```

Format CMake files:

```bash
cmake-format -i CMakeLists.txt
```

Check formatting without modifying:

```bash
clang-format --dry-run --Werror include/smashtable/*.hpp
```

Documentation is trickier.
All docstrings must use Doxygen-style comments.
Here are a couple of examples.
For minimalistic single-line descriptions:

```cpp
/** @brief Brief one-line description of the function or class. */
```

For more complex functions with many parameters, use the following template:


```cpp
/**
 *  @brief Some function similar to STL's @c std::map::try_emplace().
 *  @see https://en.cppreference.com/w/cpp/container/map/try_emplace.html
 *  @sa @c insert_if_missing() provides the same functionality under a less ambiguous name.
 *  @param[in] key Object comparable and convertible to @c element_t.
 *  @return pair<iterator, bool> Pair consisting of an iterator to the inserted or existing element.
 *  @retval second True if a new element was inserted, false if an existing element was found.
 */
```

For multi-line descriptions, use 4 spaces for continuation indents:

```cpp
/**
 *  @brief Erases all elements in the range [first, last) using const_iterators.
 *    Unlike STL, returns both the iterator following the last erased element and a status code.
 *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
 *
 *  @param[in] first Beginning of range to erase.
 *  @param[in] last End of range to erase (not erased).
 *  @return erase_result_t Contains iterator equal to @p last and status of the operation.
 *    Returns first error encountered, or success if all elements erased.
 */
```

Use `@see` for external references, `@sa` for internal cross-references, `@note` for attention points.
Use `@retval` for enumerated return values.
Mark class template parameters with `@tparam`.
Mark function parameters with direction indicators: `@param[in]`, `@param[out]`, or `@param[inout]`.
Mention them with `@p`.
Mark class/type names and method names with `@c`.
Put examples or multi-token code snippets in `@code` ... `@endcode` blocks.
Use `@b` for bold text and `@a` for italics.
Keep whitespaces on both sides of tags - `[ @p example]` not `[@p example]`.
