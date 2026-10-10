# Verification

Model checking for the protocols the stores promise: the two-phase group commit, the partitioned commit under one stamp, the commit order's ring and reader census, the slot lock of the atomic hash table, the shared mutex every locked store takes, and the staged batch every range modifier runs.
[Spin](https://spinroot.com) checks each as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks two of them as C++ under SC, RC11 and IMM, the model of compiled code on x86, Arm and POWER.

- `weak_memory.pml` is the C++ memory model as views, for Promela, under three models chosen by `-Dmemory=`: `sc`, one copy of every word; `rc11`, the default, RC11's release-acquire-relaxed fragment; and `far`, which also posts a relaxed no-return add the way RAO-INT does.
- `monitor_wait.pml` is a wait that can miss its wake-up: a waiter arms, re-checks the word, and blocks until a writer's `wake`, so a lost notify reads as `stuck`.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `spin_shared_mutex.pml` and `commit_order_steps.pml` are protocols several components include.

## Writing a model

A component is one mechanism of the code under test: a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second, holding a `protocol.pml` with its words, values, knobs and inlines, one file per scenario, and a `client.cpp` when GenMC checks the same words.
A scenario defines its shape, `thread_count`, `location_count` and `history_depth`, includes its protocol, and starts its roles from one `init { atomic { run <role>(<thread>); … } }`; the depth is the exact minimum every `rc11` and `far` line of the scenario accepts.
There is no conditional compilation: a knob is a choice the code under test makes, defaulted under `#ifndef` and read in a plain `if :: knob -> … :: else fi`, and memory-model differences are expressions on `memory`.
A wait on a word another thread moves goes through `monitor_wait.pml`'s `wait_on`, and the writer that ends it calls `wake`, so a dropped wake-up or a missing re-check comes out `stuck`.
A client spells the same knobs as fields of a `knobs_t`, one `constexpr knobs_t <variant>_k` per variant behind a `template <knobs_t const &knobs_>`, and one `extern "C"` entry per variant, `<scenario>` or `<scenario>_<knob>_<value>`.

Every expected verdict is a `@verify` line in the docblock of the file it checks:

```
@verify <pass|fail|stuck> <memory>[,<memory>...] [<knob>=<value> ...][: <finding>]   in a scenario
@verify <pass|fail|stuck> <memory>[,<memory>...] <entry>[: <finding>]                in a client
```

`pass` is a search that completed with no error, `fail` a violated assertion, and `stuck` an invalid end state in Spin or a liveness violation in GenMC; a search cut short, a knob defined twice, or a write past the scenario's shape is `broken`.
Every scenario has a `pass` line, every knob appears on some line, and adding a scenario, a knob or a finding never edits `check.sh`.

## What the models found

Each finding is the `@verify` line that replays it, where the full counterexample is written out.

- `validate_for_commit` holds a locked store's mutex exclusively until the publication, the rollback or the reset that follows; dropped between the phases, a writer commits over a watched key in the gap: [`transaction_group/`](transaction_group/).
- Two groups holding across their phases take the stores ascending, or they deadlock: [`transaction_group/`](transaction_group/).
- A refused validation gives its hold straight back, and so do the stores asked before it, or the caller's next read waits on its own thread: [`transaction_group/`](transaction_group/).
- The atomic hash table counts a slot populated under the slot's lock, or an eraser slipping in between takes the count through zero, and its slot lock acquires, or two emplacers of one key populate two slots: [`atomic_hash_table/`](atomic_hash_table/).
- A locked store's writers acquire its mutex and its readers acquire the shared side, or a commit is lost or read torn, and `unlock_shared` wakes parked writers only when the last reader leaves, or a writer parked behind the readers sleeps for good: [`locked_store.pml`](locked_store.pml).
- The commit order stands on three orderings in place of a mutex: a reader joins a bucket and reads the watermark through a read-modify-write, a mark reads the watermark before scanning the buckets, and a landing commit reads the watermark through a read-modify-write before walking the ring: [`commit_order/`](commit_order/), [`commit_order_publication.pml`](commit_order_publication.pml).
- `end_commit` wakes parked committers once its walk moves the watermark, or a committer waiting for ring room sleeps through the walks that made room: [`commit_order_publication.pml`](commit_order_publication.pml).
- A store-level window write over partitions sharing a clock publishes every partition under one stamp, or a reader drawing between two publications sees the window half written: [`partitioned_erase.pml`](partitioned_erase.pml), [`partitioned_erase.cpp`](partitioned_erase.cpp).
- `ordered_cursor_t` reads fronts under the shared lock and the write count is stepped with a release, either of which keeps a front fresh, and a store publishes every part it wrote: [`partitioned_store.pml`](partitioned_store.pml).
- The claim a pinned reader holds, and the one an adopting transaction links beside it, keep a prune off the version each reads, and the mark seals against the newest published stamp, or a later reader's claim lets a prune free its version: [`snapshot_reader/`](snapshot_reader/).
- A staged batch meets the allocator's refusal and an element's refusal while the range is still outside the destination, and checks for a key before the absorb writes it: [`test/batch_atomicity.hpp`](../test/batch_atomicity.hpp).

## What is not covered

The in-turn commit of stores that do not split theirs is documented as a tear when a later participant refuses, and `whole_across_stores=true` is the assertion that shows it rather than a fix.
The stage's unwind discards each rollback's status; an inner rollback fails only with a store-level fault the participant reports again on its next call, and the stage already returns the refusal that matters.
The trees, the vectors and the single-writer stores are documented one-thread cores, and their staged batch is checked by the C++ tests in `test/batch_atomicity.hpp`, not by a model.
The GenMC clients spell their mutexes as one exchanged word rather than `spin_shared_mutex_t`, whose shape `spin_shared_mutex.pml` already checks, and they run only where GenMC is installed.
That `apply_each_` cannot refuse once `reserve_more` has returned rests on the growth threshold leaving a quarter of the slots free and on a bounded probe walking all of them, which is `hash_layout.hpp`'s property and has no model of its own.

Only a futex-style park can lose a wake-up, so the waits run as parks through `monitor_wait.pml`, and the spinning, pausing and address-wait policies, which re-read the word itself, add nothing a park does not already cover.
`await_published` is not modelled on its own; it parks on the same word, woken by the same walk, as the ring-room wait in `commit_order_steps.pml`.

## Running

```sh
./check.sh                                  # everything; GenMC is skipped when absent
./check.sh transaction_group/two_pass.pml   # one scenario
GENMC=~/genmc/build/bin/genmc ./check.sh    # with the clients
```

Verdicts run `VERIFY_JOBS` at a time, four by default, since each pan holds a hash table of its own.

A green suite is narrower than it looks, and this is worth saying plainly.
Six defects in `basic_commit_order` were found by the C++ tests while every model passed: two of them no model here can express, since a livelock leaves no invalid end state under `-DSAFETY` and two unordered relaxed stores are a shape rather than a value, and the other four were arithmetic and object lifetime, which these models abstract away.
