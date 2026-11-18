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
uv pip install scikit-build-core        # to pull the build tools
uv pip install --group test             # to pull the test tools
uv pip install -e . --force-reinstall   # to build locally from source
python -X faulthandler -m pytest -x     # how the CI runs it
```

The extension requires 3.12 or later, where a module can declare per-interpreter GIL support.
On a free-threading interpreter, run the suite under contention, since exercising such a build single-threaded proves nothing about it:

```bash
pytest test/ --parallel-threads=4 --iterations=2
python -c "import sys, smashtable; assert not sys._is_gil_enabled()"
```

The settings both suites read are listed under [Environment](#environment).

### Model Checking

The commit protocols are also checked as Promela models under Spin, and two of them as GenMC clients over `std::atomic`; `verification/README.md` says what each model covers.

```bash
./verification/check.sh                          # everything; GenMC is skipped when absent
./verification/check.sh transaction_group/two_pass.pml   # one scenario
```

Every scenario and client names the verdicts expected of it in its own `@verify` lines, and the run fails if a deliberately broken variant passes.

### Git Hooks

Configuring the CMake build points `core.hooksPath` at `.githooks`, so the pre-commit checks run on every commit from then on.
Every check reads only the added lines of a staged file, so legacy code migrates on its own schedule.
After touching a check, run its fixtures, which feed known-good and known-bad files through the real hooks and assert that a bad one trips exactly the rule it was written for:

```bash
.githooks/selftest
```

## Test Suites

### Principles

Tests reach the library through its public surface only: the headers under `include/smashtable/` and the `smashtable` Python module.
Every container family runs the same suites, one binary per family, so a behavioural difference between two families surfaces as a failure rather than a surprise.
The suites shared between families live in `test/*.hpp` as `check_*` bodies over one container type, and each family's file instantiates them in its own cases.
Oracles never compile out, so a test checks with `expect_true` and its siblings, never `assert`, and every test unit opens with `#undef NDEBUG // ! A test's oracle must stay live in every build`.
The benchmark never includes `test/`, each tree keeps its own harness, and neither uses a test or benchmark framework.
The harness owns stdout, while test bodies write their diagnostics to stderr.
Settings are read once, and a value that does not parse prints one line naming it and exits with 1.
Skips are honest: a filter that selects no case fails the run, and a case that cannot run here says why.

A threaded suite keeps no atomic tally of its own: each worker counts into its own slot, and the main thread sums the slots after the join.
The one exception is the harness's single mutable global, `test_run`, the run in progress: its settings, the case under way, the count of failed checks with the last one's file and line, and the passed, failed and skipped cases.
A check fires wherever the code under test calls back - inside a `noexcept` visitor, an allocator, a comparator or a worker - where no slot is owned and no join follows, so the failed-check count is atomic, and `run_test_case` diffs it around each case rather than threading a tally through every call.
A debugger reads it from a core dump as `p 'ashvardanian::smashtable::test::test_run'`, and the crash handlers print the case in progress before the backtrace.

### Running

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Each binary is one family or one kit: `smashtable_test_std_set`, `_avl_tree`, `_wb_tree` and `_partitioned` for the ordered stores, `_hash_table` and `_monotonic_hash` for the unordered ones, `_snapshot_store` for the stamped stores, `_shared` for the vocabulary, locks and transaction groups, `_vector`, `_ring`, `_pools` and `_intrusive_stack` for the building blocks, and `_row_search`, `_flat_set`, `_hash_index` and `_elias_fano` for the row kits.
A run prints the library and the machine, then its settings, one verdict line per case, and a summary:

```text
SmashTable 1.0.0
- Compiled for: serial,neon,sve
- This machine: serial,neon,sve
- Dispatches to: sve
- Seed: 42, scale: 1, filter: none
- test_kit(row_kit_t::serial_k) ... ok (0.02 s)
- test_kit(row_kit_t::haswell_k) ... skipped: needs row_kit_t::haswell_k
7 passed, 0 failed, 6 skipped, seed 42
```

A failed check prints where it failed, then its details four spaces in - both operands for `expect_equal`, then the message when one was given - and the case goes on, so one run lists every mismatch it reaches.
A case with a failed check prints its verdict and a `rerun:` line on stderr, naming the binary, the seed, and a filter that selects only that case, then the run moves on:

```text
  check failed at /path/to/SmashTable/test/ring.cpp:28
    expected success_k, got invalid_argument_k
- test_capacity_rules ... FAILED: 1 failed check(s)
  rerun: build_debug/smashtable_test_ring --seed 42 --filter '^test_capacity_rules$'
```

The run exits with 1 when a case failed, a check failed outside any case, or the filter selected nothing, and with 3 when `SMASHTABLE_NO_SKIPS=1` refused a skip.
`SMASHTABLE_TRAP_ON_FAILURE=1` raises `SIGTRAP` at the first failed check instead, which stops a debugger there, or dumps a core without one:

```bash
SMASHTABLE_TRAP_ON_FAILURE=1 SMASHTABLE_FILTER='^test_capacity_rules$' gdb -ex run --args build_debug/smashtable_test_ring
```

A crash stops the run with `*** SIGSEGV in case '<label>' - backtrace follows ***` and a backtrace where the C library has one.

The Python suite runs as the CI runs it, after installing the `test` group, and on a free-threaded interpreter once more under contention, as [Compiling Python Bindings](#compiling-python-bindings) shows.
The header pytest prints names the module it imported, `SmashTable <version> from <path>`, and ends with the same `- Seed: 42, scale: 1, filter: none` line as the C++ runner.

### Environment

Each setting is an environment variable, and in the C++ runners and the benchmarks a `--name` flag of the same name overrides it, as `--seed 7` or `--filter='^avl_tree\.'`:

| Variable                     | Flag                | Default    | Meaning                                                                                |
| :--------------------------- | :------------------ | :--------- | :------------------------------------------------------------------------------------- |
| `SMASHTABLE_SEED`            | `--seed`            | `42`       | Any unsigned 64-bit seed, or `random` to draw one, which a run prints                  |
| `SMASHTABLE_FILTER`          | `--filter`          | unset      | Regex searched in C++ case labels or benchmark rows                                    |
| `SMASHTABLE_SCALE`           | `--scale`           | `1`        | C++ only: multiplier for the rounds of the randomized and threaded suites, like `0.1`  |
| `SMASHTABLE_TRAP_ON_FAILURE` | `--trap-on-failure` | `0`        | C++ only: raises `SIGTRAP` at the first failed check, to stop there under a debugger   |
| `SMASHTABLE_NO_SKIPS`        | `--no-skips`        | `0`        | C++ only: exits with 3 when any case skipped, for runners that must cover everything   |
| `SMASHTABLE_WARMUP`          | `--warmup`          | `1s`       | Benchmark only: untimed run ahead of each row, like `200ms`, `10s` or `2m`             |
| `SMASHTABLE_TIME_LIMIT`      | `--time-limit`      | `10s`      | Benchmark only: timed run of each row, like `200ms`, `10s` or `2m`                     |
| `SMASHTABLE_ROWS`            | `--rows`            | `4096`     | Benchmark only: sorted rows per row-kit benchmark                                      |
| `SMASHTABLE_KEYS`            | `--keys`            | `16777216` | Benchmark only: keys per layout or store benchmark                                     |
| `SMASHTABLE_QUERIES`         | `--queries`         | `1048576`  | Benchmark only: queries drawn per benchmark, searched in turn                          |
| `SMASHTABLE_THREADS`         | `--threads`         | `1`        | Benchmark only: threads timing each row at once, `0` for every hardware thread         |

A switch is off when unset, empty or `0`, and on otherwise, and a count of zero is refused.
Python reads only the variables, beside `pytest -k` and pytest's own options, and prints the scale it read without sizing anything by it yet.

`SMASHTABLE_SEED` keys every case's own stream by the case's label, as `stream_key(seed, name, index)`, identical in C++ and Python.
A case's inputs therefore never depend on which cases ran before it, so `SMASHTABLE_SEED` together with `SMASHTABLE_FILTER` is a faithful reproduction recipe, and hunting for a rare defect is a loop in the shell rather than an edit to the source:

```bash
SMASHTABLE_SEED=random build_debug/smashtable_test_avl_tree                          # A fresh seed, which the run prints
SMASHTABLE_SCALE=10 build_debug/smashtable_test_snapshot_store --filter='^test_random_'  # Ten times the fuzz rounds
for seed in $(seq 1 64); do build_debug/smashtable_test_snapshot_store --seed $seed || break; done
python -X faulthandler -m pytest -x -k "SortedMap and str"                             # pytest selects by keyword
```

### Writing a Test

A C++ case is a `void` function, or one returning `case_result_t` when it may skip, that `main` runs as `run_test_case("<label>", function, arguments...)`.
It takes no settings or context, as the run is `test_run`, and it skips only by returning `case_result_t::skipped(reason)`, with a reason like `needs row_kit_t::haswell_k` or `no GPU device`.
A label spells the code it runs, so it greps to its source: the test function as written, with its template arguments, in call notation when `main` passes a runtime choice, as `test_watch_on_erased_key_can_commit`, `test_reader_holds_its_stamp<strict_serializable_avl_map_t>` and `test_kit(row_kit_t::neon_k)`.
So `SMASHTABLE_FILTER='^test_reader_holds_its_stamp<'` runs one behaviour over every store, and `SMASHTABLE_FILTER='row_kit_t::neon_k'` one kit.
Every `main` is the same three steps, as `start_test_run` prints the machine before the settings:

```cpp
int main(int argc, char **argv) {
    start_test_run(argc, argv);
    run_test_case("test_kit(row_kit_t::serial_k)", test_kit, row_kit_t::serial_k);
    return finish_test_run();
}
```

It checks with functions from `test/harness.hpp`, each of which counts a failed check at the caller's line, lets the case go on, and returns whether it held:

- `expect_true(held)`, for any condition;
- `expect_equal(actual, wanted)`, which compares whole numbers across signedness and prints both operands, naming a status rather than numbering it.

Both take an optional message last, a format string and its arguments, as `expect_equal(set.size(), count, "after {} inserts", count)`.
Where the step after a failed check would index or dereference past what it checked, the case stops on the check's answer, as in `if (!expect_equal(made.status(), success_k)) return;`.
A case draws from `split_mix_generator_t generator {case_stream_key()}`, keyed by the run's seed and the case's label, and sizes its rounds through `scaled_iterations(baseline)`, which scales by `SMASHTABLE_SCALE` and floors at 1.
A wait polls through `spin_until(predicate)`, which gives up after `wait_limit_k`, so a hang fails the case rather than the CI job, as in `expect_true(spin_until([&] { return ready.load(); }))`.
The allocators a case starves are `stateful_allocator`, over an `allocation_ledger_t` counting `allocations`, `deallocations`, `live_bytes` and `refusals` and refusing from the `refuse_from`-th request, and `refusing_allocator`, which refuses them all.

A Python test is `test_<subject>_<behaviour>`, read as a sentence, with a one-line docstring saying what holds.
It draws from the `seed` or `rng` fixture, never from a literal seed.
The seed is `stream_key(SMASHTABLE_SEED, name, step)`, where `name` is the test function and its parameters and `step` is the pytest-repeat step.
Axis names follow one vocabulary - `container_class`, `key_type`, `value_type`, `value_mode`, `count`, `isolation` and `sharing` - parametrized directly, with the matrices in `test/base.py` named in upper snake case, like `ALL_CLASSES` and `COUNTS`.
Marks sit above the parametrizes, which run in that order, and parametrize names are tuples.
`MAX_CASES_PER_TEST` in `test/base.py` caps how many cases one test function expands to, and raising it is a decision for review.
A test that needs a free-threaded interpreter takes the `@requires_free_threading` mark of `test/base.py`, and anything only a run can find out skips inside a fixture.
`test/harness.hpp`, `bench/harness.hpp`, `test/base.py` and `test/conftest.py` each hold a pinned `Shared` region, which the pre-commit hook keeps identical to its canonical copy, so edit around it.

### Benchmarks

`bench/harness.hpp` carries the same pinned settings readers as `test/harness.hpp`, so its variables parse, refuse and take `--name` flags the same way, under the same `SMASHTABLE_` prefix.
Each workload is its own target over that harness, built on request and never run by CTest:

- `smashtable_bench_commits`: one update across two containers per commit, two `std::map`s behind one `std::mutex` against two partitioned snapshot stores joined by a `transaction_group`, then a `std::unordered_map` and a `std::map` behind one `std::mutex` against a partitioned snapshot hash map and AVL map joined the same way.
- `smashtable_bench_isolation`: a read-modify-write transaction at each isolation level over one partitioned AVL core, retried until it lands, reporting the aborts it met.
- `smashtable_bench_ranking`: the row kits within one row of each width, and the immutable B-tree and S+ tree against a binary search among all the keys, for `u32`, `u64` and both `key128` draws.

```bash
cmake --preset release -D SMASHTABLE_BUILD_BENCH=ON
cmake --build --preset release --target smashtable_bench_commits smashtable_bench_isolation smashtable_bench_ranking
build_release/smashtable_bench_commits --keys=1048576 --threads=16
build_release/smashtable_bench_isolation --keys=1048576 --threads=16 --filter=snapshot
build_release/smashtable_bench_ranking --filter='layouts\.u64\.' --time-limit=200ms
```

A run prints the same machine header as the tests, its `- Name: value` settings, and one row per benchmark, named in the label grammar as `commits.<container>`, `isolation.<store>`, `ranking.rows.<bytes>b.<keys>.<row layout>.<kit>` and `ranking.layouts.<keys>.<bytes>b.<tree>.<kit>`, which is what `SMASHTABLE_FILTER` matches.
Above one thread, each thread times the row on a loop of its own and draws from a stream of its own, `stream_key(seed, label, thread)`, and the row sums their calls and rates.
Every store and standard container in `commits` and `isolation` allocates through the harness's `counting_allocator`, so their rows end in `loaded bytes/key`, the bytes held once the keys are in, and `timed bytes/key`, the bytes held as the first thread's timed loop ends, with its open transaction and any versions not yet reclaimed.

### Devices

The row kits are the CPU's devices: each case that runs per kit, like `test_kit_rows(row_kit_t::<kit>_k)` in `_row_search` and `test_kit(row_kit_t::<kit>_k)` in `_flat_set`, `_hash_index` and `_elias_fano`, runs every kit this build compiled and this machine supports, and reports `skipped: needs row_kit_t::<kit>_k` for the rest.
`test/atomic_cuda.cu` drives the pinned lock-free table from kernels and from the host over one managed allocation, and is built only where `nvcc` is visible, for the code `SMASHTABLE_CUDA_ARCH` names.
On a machine without a device its cases report `skipped: no GPU device`, on a device older than compute capability 7.0 `skipped: needs compute capability 7.0`, and the mixed case without concurrent managed access `skipped: needs cudaDevAttrConcurrentManagedAccess`.

### Private Surface Waivers

None: every test reaches the library through `include/smashtable/` and the Python module.

## Code Styling Guidelines

A batch operation applies wholly or not at all.
That holds against a refused allocation and against an element refusing its own copy alike, so a range modifier builds every element outside the destination and absorbs them in one step that cannot fail part-way.
A method that cannot honour this says so in its own docblock, the way a bounded ring says it takes what fits.
Never duplicate a batch element with `value_t(*first)`: that expression cannot report a refusal and will not compile over a move-only element, so route it through `stage_each`, which copies through `copy_safely` and moves an rvalue.
The `batches_atomically` concept checks the shape; the rollback itself is pinned by `test_batch_atomicity.hpp`, which refuses the allocator at every point a batch asks for memory.

A call that returns an error releases every lock it took, in every store and in every split protocol to come, and only a success may hand a lock on to the next step, as `validate_for_commit` hands its store's mutex to `publish_under`.

Internal `private` data and functions should be suffixed with an underscore (`_`).

Every all-caps name starts with the full project name, `SMASHTABLE_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                       | Form                        | Example                     |
| ---------------------------- | --------------------------- | --------------------------- |
| ISA tier, backend, GPU layer | `SMASHTABLE_TARGET_<TIER>`  | `SMASHTABLE_TARGET_HASWELL` |
| Architecture fact            | `SMASHTABLE_ARCH_<ARCH>_`   | `SMASHTABLE_ARCH_X8664_`    |
| Operating-system fact        | `SMASHTABLE_OS_<OS>_`       | `SMASHTABLE_OS_LINUX_`      |
| Platform fact                | `SMASHTABLE_HAS_<FEATURE>_` | `SMASHTABLE_HAS_AUXV_`      |

Architectures are spelled `X8664`, `X8632`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`.
Every name in these families is always defined, as 0 or 1, and tested with `#if`.
`defined(...)` appears only in a `!defined` default and in a guard that an `#error` follows.
Only the Platform Identity region of `shared.hpp` reads operating-system predefines like `__linux__`.
Compiler predefines like `__clang__`, `__GNUC__` and `_MSC_VER` are read raw where they matter.

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

- `mapping` replaces `std::pair` for key-value pairs in associative containers (maps).
  It has a `key` member for the lack of a better name, but the second one isn't a `value` - it's in the `mapped` variable.
- `expected` replaces `std::optional` augmenting the semantics with error codes.
  It has an `outcome` always initialized member instead of a generic `value` or `object`, often nested inside some union for uninitialized states.

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
  It always carries a direction — `@param[in]`, `@param[out]` or `@param[in,out]` — since the pre-commit hook rejects a bare one.
- `@return` names what the answer means rather than its type.
  `@retval` is unused here: a function that can fail lists the statuses its contract adds in a closing `@par Statuses:` block, while allocation and I/O failures, cancellation, deferral and defects are covered once by the categories in the `status_t` docblock.
- `@warning` for the edge a caller can fall off, such as a lock held across a callback or a moved-from store whose open transactions land nowhere.
  `@note` for an aside a reader needs but can act on without alarm — the cost of a walk, what a callback answering `walk_control_t` may do, which isolation levels a sentence holds for.
- `@tparam` for template parameters, `@p` to mention a parameter, `@c` for a type or method name, `@b` for bold and `@a` for italics.
  Backticks and Markdown emphasis print literally through Doxygen, and the hook rejects both inside a block.
- `@see` for an external reference, `@sa` for one inside the repository, `@code` and `@endcode` around a multi-token snippet, and `@section` with an anchor before its title.
- A tag's continuation lines indent 6 spaces; a body paragraph's sit unindented at `* `.
  A tag keeps whitespace on both sides — `[ @p lower, @p upper )` rather than `[@p lower, @p upper)` — and never glues to the character before it.
- Document every member of a type or none of them, with a hovering `/** … */` and never a trailing `//!<` or a bare `//`.
  The shared explanation belongs in the type's own docblock, where one sentence covers what a column of markers would repeat.

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message carries no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles are exempt from the title rules.
A `Release: vX.Y.Z` title is exempt too, but its commit must stamp every version site, and no other commit may change `VERSION`.
`.githooks/pre-commit` formats and judges the staged lines, and `.githooks/selftest` runs both hooks against their fixtures.
On a pull request, `.github/scripts/check-commits.sh` replays `commit-msg` over every commit and runs `pre-commit` over the whole range.

## Releases

`.github/scripts/mint-release.sh X.Y.Z` cuts a release from a clean checkout of the branch it ships.
It stamps every version site and commits `Release: vX.Y.Z`.
`.github/scripts/check-release.sh` then checks that commit, even in a clone where `core.hooksPath` is unset, before the script tags it and pushes both.
Then it publishes the GitHub release, marking `X.Y.Z-rc.N` candidates as prereleases.
Publishing starts every `release-<binding>.yml` at once, and each runs `check-release.sh` before it builds anything, so each registry trusts only its own binding's file.
