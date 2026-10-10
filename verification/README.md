# Verification

Model checking for the protocols the stores promise: the two-phase group commit, the partitioned commit under one stamp, the commit order's ring and reader census, the slot lock of the atomic hash table, the shared mutex every locked store takes, and the staged batch every range modifier runs.
[Spin](https://spinroot.com) checks each as a Promela model, and [GenMC](https://github.com/MPI-SWS/genmc) checks two of them as C++ under SC, RC11 and IMM, the model of compiled code on x86, Arm and POWER.

- `weak_memory.pml` is the C++ memory model as views, for Promela, under three models chosen by `-Dmemory=`: `sc`, one copy of every word; `rc11`, the default, RC11's release-acquire-relaxed fragment; and `far`, which also posts a relaxed no-return add the way RAO-INT does.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `waiting_policy.pml`, `spin_shared_mutex.pml` and `commit_order_steps.pml` are protocols several components include.

## Writing a model

A component is one mechanism of the code under test: a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second, holding a `protocol.pml` with its words, values, knobs and inlines, one file per scenario, and a `client.cpp` when GenMC checks the same words.
A scenario defines its shape, `thread_count`, `location_count` and `history_depth`, includes its protocol, and starts its roles from one `init { atomic { run <role>(<thread>); … } }`; the depth is the exact minimum every `rc11` and `far` line of the scenario accepts.
There is no conditional compilation: a knob is a choice the code under test makes, defaulted under `#ifndef` and read in a plain `if :: knob -> … :: else fi`, and memory-model differences are expressions on `memory`.
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
- The atomic hash table counts a slot populated under the slot's lock, or an eraser slipping in between takes the count through zero: [`atomic_hash_table/`](atomic_hash_table/).
- The commit order stands on four orderings in place of a mutex: a reader counts into a bucket before reading the watermark, a mark reads the watermark before scanning the buckets, a landing commit reads the watermark through a read-modify-write before walking the ring, and a bucket shuts to arrivals before its floor is replaced: [`commit_order/`](commit_order/), [`commit_order_publication.pml`](commit_order_publication.pml).
- A store-level window write over partitions sharing a clock publishes every partition under one stamp, or a reader drawing between two publications sees the window half written: [`partitioned_erase.pml`](partitioned_erase.pml), [`partitioned_erase.cpp`](partitioned_erase.cpp).
- The claim a pinned reader holds, and the one an adopting transaction links beside it, keep a prune off the version each reads: [`snapshot_reader/`](snapshot_reader/).
- A staged batch meets the allocator's refusal and an element's refusal while the range is still outside the destination, and checks for a key before the absorb writes it: [`staged_batch/`](staged_batch/).

## What is not covered

The in-turn commit of stores that do not split theirs is documented as a tear when a later participant refuses, and `whole_across_stores=true` is the assertion that shows it rather than a fix.
The stage's unwind discards each rollback's status; an inner rollback fails only with a store-level fault the participant reports again on its next call, and the stage already returns the refusal that matters.
The trees, the vectors and the single-writer stores are documented one-thread cores, and only their staged batch has a model.
The GenMC clients spell their mutexes as one exchanged word rather than `spin_shared_mutex_t`, whose shape `spin_shared_mutex.pml` already checks, and they run only where GenMC is installed.

`staged_batch/` runs three keys and a range of three, which is enough for a key already there, a key that is free and a key the range repeats, and not enough for the probe of an open table.
That `apply_each_` cannot refuse once `reserve_more` has returned rests on the growth threshold leaving a quarter of the slots free and on a bounded probe walking all of them, which is `hash_layout.hpp`'s property and has no model of its own.
A table whose element duplicates without refusing stages nothing and writes the caller's range straight in; `staged_batch/table.pml` covers that path as the same one allocation, since the two differ in what they duplicate rather than in what a refusal leaves behind.
Only `basic_avl_tree` spells `update` over a range, so the model's fourth verb stands for the tree alone.

The waiting policies run over `locked_store.pml` and `atomic_hash_table/`, the smallest model over each of the two locks; the models that layer a store protocol on the same mutex run under the default, since a policy admitting more interleavings for the lock admits them for everything above it.
A policy that sleeps on a notification of its own, rather than on the word the lock lives in, is covered only for when it retries: the wake lives in the word's own history here, so a wake-up lost between a waiter deciding to sleep and a releaser looking for waiters is not a shape these models can express.

## Running

```sh
./check.sh                                  # everything; GenMC is skipped when absent
./check.sh staged_batch/flat.pml            # one scenario
GENMC=~/genmc/build/bin/genmc ./check.sh    # with the clients
```

Verdicts run `VERIFY_JOBS` at a time, four by default, since each pan holds a hash table of its own.

A green suite is narrower than it looks, and this is worth saying plainly.
Six defects in `basic_commit_order` were found by the C++ tests while every model passed: two of them no model here can express, since a livelock leaves no invalid end state under `-DSAFETY` and two unordered relaxed stores are a shape rather than a value, and the other four were arithmetic and object lifetime, which these models abstract away.
