# Contributing to SmashTable

## Compiling C++ Code

Release build:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -B build_release
cmake --build build_release --config Release --parallel
build_release/smashtable_test_avl_tree
```

Debug build for the test suite:

```bash
cmake -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug --target smashtable_test_avl_tree
build_debug/smashtable_test_avl_tree
```

### Running Tests

Every container family runs the same suites, so `ctest` covers all four binaries:

```bash
ctest --test-dir build_debug --output-on-failure
```

Run one binary directly, or narrow it with a substring matched against `suite.name`:

```bash
build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=transactional_consistency build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=basic_ops.insertion build_debug/smashtable_test_wb_tree
```

Assertions abort on the first failure and print the expression, file and line, so a run reports one defect rather than a list.

### Before Opening a Pull Request

CI builds with warnings as errors on both compilers, and checks that every header compiles as the
first thing a translation unit sees. Both are worth reproducing locally:

```bash
cmake -B build_strict -DCMAKE_BUILD_TYPE=Debug -DSMASHTABLE_WERROR=ON
cmake --build build_strict && ctest --test-dir build_strict --output-on-failure

for header in include/smashtable/*.hpp; do
    printf '#include <%s>\nint main() { return 0; }\n' "${header#include/}" \
        | c++ -std=c++20 -fsyntax-only -Wall -Wextra -Iinclude -x c++ - || echo "FAILED $header"
done
```

The lock-free paths carry a third check, since a data race in them is invisible to the address and behaviour sanitizers:

```bash
cmake -B build_tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=thread"
cmake --build build_tsan && ctest --test-dir build_tsan --output-on-failure
```

## Compiling Python Bindings

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.
Still, you need a virtual environment, and it's recommended to use `uv` to create one.

```bash
uv venv --python 3.14t                  # a free-threading build is the interesting one to test against
source .venv/bin/activate               # to activate the virtual environment
uv pip install setuptools wheel         # to pull the build tools
uv pip install --group test             # to pull the test tools
uv pip install -e . --force-reinstall   # to build locally from source
pytest test/
```

The extension requires 3.12 or later, where a module can declare per-interpreter GIL support.
On a free-threading interpreter, run the suite under contention, since exercising such a build single-threaded proves nothing about it:

```bash
pytest test/ --parallel-threads=4 --iterations=2
python -c "import sys, smashtable; assert not sys._is_gil_enabled()"
```

## Code Styling Guidelines

Internal `private` data and functions should be suffixed with an underscore (`_`).
Avoid obvious inline comments.
Prefer full words over abbreviations (e.g., `iterator` instead of `iter`, `element` instead of `elem`, `transaction` instead of `tx`, etc.).
Code is formatted automatically using `clang-format` with the configuration specified in `.clang-format`.
It's not all-mighty, so avoid the following anti-patterns:

```cpp
if constexpr (condition) {
    // Some comment on a separate line
    one_expression();
}
else {
    // Large, multi-line comment ...
    // continued here
    another_expression();
}
```

Should be written as:

```cpp
if constexpr (condition) one_expression(); // Short inline comment
// Large, multi-line comment ...
// continued here
else another_expression(); 
```

Please, avoid generic variable names that may lead to confusion when debugging, especially in algebraic data types that immediately spread across the repo.
Typical examples are `value`, `item`, `obj`, `data`, `entry`, etc.
For example, in this codebase:

- `mapping` replaces `std::pair` for key-value pairs in associative containers (maps). It has a `key` member for the lack of a better name, but the second one isn't a `value` - it's in the `mapped` variable.
- `expected` replaces `std::optional` augmenting the semantics with error codes. It has an `outcome` always initialized member instead of a generic `value` or `object`, often nested inside some union for uninitialized states.

## Documentation Styling Guidelines

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
