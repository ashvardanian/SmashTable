# Contributing to SmashTable

## Compiling C++ Code

The presets in `CMakePresets.json` are the configurations CI builds, and `cmake --list-presets` shows the ones this machine can run.
Machine-specific settings, like a compiler path, belong in an untracked `CMakeUserPresets.json`.

Release build:

```bash
cmake --preset release
cmake --build --preset release
build_release/smashtable_test_avl_tree
```

Debug build for the test suite:

```bash
cmake --preset debug
cmake --build --preset debug --target smashtable_test_avl_tree
build_debug/smashtable_test_avl_tree
```

### Running Tests

Every container family runs the same suites, so one `ctest` run covers every binary at once:

```bash
ctest --preset debug
```

Run one binary directly, or steer it with the two environment variables the harness reads:

```bash
build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=transactional_consistency build_debug/smashtable_test_avl_tree
SMASHTABLE_FILTER=basic_ops.insertion build_debug/smashtable_test_wb_tree
SMASHTABLE_SEED=1234 build_debug/smashtable_test_wb_tree
```

`SMASHTABLE_FILTER` is an ECMAScript regex searched for in each test's `suite.name`, so only the matching tests run, and the binary announces every one it skips.
A pattern that does not compile matches as a plain substring instead.
A filter matching nothing fails the binary rather than reporting an empty run as green, so a typo is loud instead of reassuring.

`SMASHTABLE_SEED` is the seed every randomized suite draws from, so a failing run names the sequence that produced it.
It defaults to 42, which keeps CI and an unattended build deterministic, and `random` draws a fresh one; either way the binary prints it on its `- Seed:` line.
Hunting for a rare defect is therefore a loop in the shell rather than an edit to the source:

```bash
for seed in $(seq 1 64); do SMASHTABLE_SEED=$seed build_debug/smashtable_test_avl_tree || break; done
```

A failed check prints the expression, file and line, then reports its test as failed and continues with the next one, so a run lists every defect it reaches and exits with 1 at the end.
Beneath each failure the binary prints a `rerun:` line with the seed and a filter that selects only the failing test:

```text
  rerun: SMASHTABLE_SEED=42 SMASHTABLE_FILTER='^basic_ops.insertion_patterns$' build_debug/smashtable_test_wb_tree
```

A check inside a `noexcept` callback or a spawned thread cannot leave its test, so it terminates the binary instead, as a crash does.
The backtrace follows, and the last test started on stdout is the one that died.

### Before Opening a Pull Request

CI builds with warnings as errors on both compilers, and checks that every header compiles as the first thing a translation unit sees.
Both are worth reproducing locally, and every preset already sets `SMASHTABLE_WERROR`:

```bash
cmake --workflow --preset debug

for header in include/smashtable/*.hpp; do
    printf '#include <%s>\nint main() { return 0; }\n' "${header#include/}" \
        | c++ -std=c++20 -fsyntax-only -Wall -Wextra -Iinclude -x c++ - || echo "FAILED $header"
done
```

The lock-free paths carry a third check, since a data race in them is invisible to the address and behaviour sanitizers:

```bash
cmake --workflow --preset tsan
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

`SMASHTABLE_SEED` pins the Python suite's seed too, with the same default of 42, and `random` again draws a fresh one per run.
Either way pytest prints it in its own header, so a failing run is reproducible by copying the number back:

```bash
SMASHTABLE_SEED=random pytest test/
SMASHTABLE_SEED=1234 pytest test/
```

### Model Checking

The commit protocols are also checked as Promela models under Spin, and two of them as GenMC clients over `std::atomic`; the README's Model Checking section says what each model covers.
The memory model comes from ForkUnion's `verification/`, checked out beside this repository, which is what CI does:

```bash
./verification/check.sh
```

Every `verify` line names a model, the verdict expected of it and its defines, and the run fails if a deliberately broken variant passes.

### Git Hooks

Configuring the CMake build points `core.hooksPath` at `.githooks`, so the pre-commit checks run on every commit from then on.
Every check reads only the added lines of a staged file, so legacy code migrates on its own schedule.
After touching a check, run its fixtures, which feed known-good and known-bad files through the real hooks and assert that a bad one trips exactly the rule it was written for:

```bash
.githooks/selftest
```

## Code Styling Guidelines

A batch operation applies wholly or not at all.
That holds against a refused allocation and against an element refusing its own copy alike, so a range modifier builds every element outside the destination and absorbs them in one step that cannot fail part-way.
A method that cannot honour this says so in its own docblock, the way a bounded ring says it takes what fits.
Never duplicate a batch element with `value_t(*first)`: that expression cannot report a refusal and will not compile over a move-only element, so route it through `stage_each`, which copies through `copy_safely` and moves an rvalue.
The `batches_atomically` concept checks the shape; the rollback itself is pinned by `test_batch_atomicity.hpp`, which refuses the allocator at every point a batch asks for memory.

Internal `private` data and functions should be suffixed with an underscore (`_`).

Every all-caps name starts with the full project name, `SMASHTABLE_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                       | Form                       | Example                     |
| ---------------------------- | -------------------------- | --------------------------- |
| ISA tier, backend, GPU layer | `SMASHTABLE_TARGET_<TIER>` | `SMASHTABLE_TARGET_HASWELL` |
| Architecture fact            | `SMASHTABLE_ARCH_<ARCH>_`  | `SMASHTABLE_ARCH_X86_64_`   |

Architectures are spelled `X86_64`, `X86_32`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`.
Every name in these families is always defined, as 0 or 1, and tested with `#if`, never with `defined(...)`.

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
clang-format -i test/*.hpp test/*.cpp test/*.cu bench/*.hpp bench/*.cpp example.cpp
```

Format CMake files:

```bash
cmake-format -i CMakeLists.txt
```

Check formatting without modifying, which is what the pre-commit hook does to the staged bytes:

```bash
clang-format --dry-run --Werror include/smashtable/*.hpp python/*.hpp python/*.cpp \
    test/*.hpp test/*.cpp test/*.cu bench/*.hpp bench/*.cpp example.cpp
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

Prose wraps at 100 columns and code at 120, so a comment block stays a narrower column than the code beneath it; `.editorconfig` carries both.

What falls out of the difference:

- `@brief` is a separator, not a label: it earns its place above a body paragraph or a tag section, and nowhere else.
  A lone paragraph carries no tag and opens on its own brace — `/** What it does. */`, or `/** What it does, at length,
  wrapping onto a second line. */` — which is what the hook's `check_lone_brief` and `check_docblock_brackets` ask for.
  A summary runs at most three lines; past that the remainder is a body paragraph below a blank comment line.
- Prose sits after the tags, below a blank line, and names the guarantee together with the scope it holds over.
  Write the scope in wherever a sibling path in the same file would make the sentence false — an isolation level, a partition, a phase of the commit.
- `@param` only where a parameter carries a constraint the signature does not, such as a callback that must be `noexcept` or a bound that is exclusive.
  It always carries a direction — `@param[in]`, `@param[out]` or `@param[inout]` — since the pre-commit hook rejects a bare one.
- `@return` names what the answer means rather than its type.
  `@retval` is unused here: a return with named outcomes is one `@return` sentence listing them.
- `@warning` for the edge a caller can fall off, such as a lock held across a callback or a moved-from store whose open transactions land nowhere.
  `@note` for an aside a reader needs but can act on without alarm — the cost of a walk, what a callback answering `walk_control_t` may do, which isolation levels a sentence holds for.
- `@tparam` for template parameters, `@p` to mention a parameter, `@c` for a type or method name, `@b` for bold and `@a` for italics.
  Backticks and Markdown emphasis print literally through Doxygen, and the hook rejects both inside a block.
- `@see` for an external reference, `@sa` for one inside the repository, `@code` and `@endcode` around a multi-token snippet, and `@section` with an anchor before its title.
- A tag's continuation lines indent 6 spaces; a body paragraph's sit unindented at ` *  `.
  A tag keeps whitespace on both sides — `[ @p lower, @p upper )` rather than `[@p lower, @p upper)` — and never glues to the character before it.
- Document every member of a type or none of them, with a hovering `/** … */` and never a trailing `//!<` or a bare `//`.
  The shared explanation belongs in the type's own docblock, where one sentence covers what a column of markers would repeat.
