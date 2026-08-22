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

Every container family runs the same suites, so one `ctest` run covers every binary at once:

```bash
ctest --test-dir build_debug --output-on-failure
```

Run one binary directly, or steer it with the two environment variables the harness reads:

```bash
build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=transactional_consistency build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=basic_ops.insertion build_debug/smashtable_test_wb_tree
SMASHTABLE_SEED=1234 build_debug/smashtable_test_wb_tree
```

`SMASHTABLE_FILTER` keeps the tests whose `suite.name` contains the substring and announces every one it skips.
A filter matching nothing fails the binary rather than reporting an empty run as green, so a typo is loud instead of reassuring.

`SMASHTABLE_SEED` is the seed every randomized suite draws from, so a failing run names the sequence that produced it.
Unset means a fixed default, which keeps CI and an unattended build deterministic.
Hunting for a rare defect is therefore a loop in the shell rather than an edit to the source:

```bash
for seed in $(seq 1 64); do SMASHTABLE_SEED=$seed build_debug/smashtable_test_avl_tree || break; done
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

`SMASHTABLE_TESTS_SEED` pins the seed the Python suite draws from, which is otherwise taken at random per run.
Either way pytest prints it in its own header, so a failing run is reproducible by copying the number back:

```bash
SMASHTABLE_TESTS_SEED=42 pytest test/
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

Format the C++ sources — the headers, the CPython bindings, the suites and the example:

```bash
clang-format -i include/smashtable/*.hpp python/*.hpp python/*.cpp
clang-format -i scripts/*.hpp scripts/*.cpp scripts/*.cu example.cpp
```

Format CMake files:

```bash
cmake-format -i CMakeLists.txt
```

Check formatting without modifying, which is what the pre-commit hook does to the staged bytes:

```bash
clang-format --dry-run --Werror include/smashtable/*.hpp python/*.hpp python/*.cpp \
    scripts/*.hpp scripts/*.cpp scripts/*.cu example.cpp
```

Python sources are `black` and `ruff` clean at the 120-column width `pyproject.toml` sets:

```bash
black test/ python/ example.py && ruff check test/ python/ example.py
```

Documentation is trickier.
Every docstring is a Doxygen block, and the voice the headers are converging on is short: `@brief` says what the thing does, a line of prose names the guarantee and the scope it holds over, and the block stops there.
A tag earns its place only by carrying something the signature cannot.

```cpp
/**
 *  @brief Hands @p callback_found the element at zero-based position @p ordinal.
 *  @param[in] callback_missing Fires when fewer elements are there. Must be @c noexcept.
 *
 *  Exact at the newest commit, where the descent reads one augmented count per node. A reader at an
 *  older snapshot is answered by a merged walk instead, since one count per node cannot answer for
 *  an unbounded parameter.
 */
```

The dialect to avoid restates the signature and never reaches the guarantee:

```cpp
/**
 *  @brief Selects an element by ordinal.
 *  @param[in] ordinal The ordinal to select.
 *  @param[in] callback_found The callback for the found case.
 *  @param[in] callback_missing The callback for the missing case.
 *  @return status_t The status code of the operation.
 *  @retval success_k The element was found.
 *  @retval key_not_found_k The element was not found.
 *  @note Attention: the callbacks must be noexcept.
 */
```

What falls out of the difference:

- `@brief` on every symbol, and on most of them nothing else.
  A one-liner is a whole docblock: `/** @brief What it does. */`.
- Prose sits after the tags, below a blank line, and names the guarantee together with the scope it holds over.
  Write the scope in wherever a sibling path in the same file would make the sentence false — an isolation level, a partition, a phase of the commit.
- `@param` only where a parameter carries a constraint the signature does not, such as a callback that must be `noexcept` or a bound that is exclusive.
  It always carries a direction — `@param[in]`, `@param[out]` or `@param[inout]` — since the pre-commit hook rejects a bare one.
- `@return` names what the answer means rather than its type.
  `@retval` is unused here: a return with named outcomes is one `@return` sentence listing them.
- `@warning` for the edge a caller can fall off, such as a lock held across a callback or a moved-from store whose open transactions land nowhere.
  There is no `@note`: an attention point either belongs in the prose or does not belong.
- `@tparam` for template parameters, `@p` to mention a parameter, `@c` for a type or method name, `@b` for bold and `@a` for italics.
  Backticks and Markdown emphasis print literally through Doxygen, and the hook rejects both inside a block.
- `@see` for an external reference, `@sa` for one inside the repository, `@code` and `@endcode` around a multi-token snippet, and `@section` with an anchor before its title.
- Continuation lines indent 4 spaces, and a tag keeps whitespace on both sides — `[ @p lower, @p upper )` rather than `[@p lower, @p upper)`.
- Document every member of a type or none of them, and never with a trailing `//!<`.
  The shared explanation belongs in the type's own docblock, where one sentence covers what a column of markers would repeat.
