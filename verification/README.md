# Verification

Model checking for the protocols the stores promise: the two-phase group commit, the partitioned commit under one stamp, the snapshot clock, the slot lock of the atomic hash table, the shared mutex every locked store takes, and the staged batch every range modifier runs.
[Spin](https://spinroot.com) checks each as a Promela model under the memory models of [ForkUnion's `verification/`](https://github.com/ashvardanian/ForkUnion), which sits beside this repository in whichever superproject vendors both, and is checked out beside it in CI; `./check.sh` runs everything here and compares each verdict with the expected one.

- `weak_memory.pml` — the forwarder to ForkUnion's memory module.
- `waiting_policy.pml` — what a loser of either lock does between two attempts, shared by the two models that spell a lock.
  `-Dwaiting=spinning`, the default, re-reads the word until it admits the attempt; `-Dwaiting=pausing` adds a step that touches no location; `-Dwaiting=on_the_address` wakes on any move of the word; `-Dwaiting=parking` blocks on nothing and retries whenever it is scheduled.
- `spin_shared_mutex.pml` — `spin_shared_mutex_t` as one word, the lock an acquire exchange and the unlock a release, shared by every model here.
  `-Dwaiting=` for the policy its two retry loops plug in.
- `locked_store.pml` — `locked_store`: one shared mutex per call, exclusion, and a reader inside the lock seeing a commit whole.
- `transaction_group.pml` — `transaction_group`: staging in address order and the unwind of a refused prefix, the two-pass commit that asks every participant before any writes, the in-turn commit's tear, and a participant holding its lock across the phases.
  `-Dscenario=one_stamp` for stores on one clock: the group's stamp drawn before any store is written and the watermark moved after the last, against a reader that must see the group whole.
- `partitioned_store.pml` — `partitioned_store`: the reached partitions held ascending, the stamp drawn before any write and the watermark moved after the last, and the cursor's epoch.
  `-Dscenario=cursor` for the epoch.
- `commit_order.pml` — `basic_commit_order`: the stamp ring, the watermark walk, the reader buckets and the marks they bound, over the steps in `commit_order_steps.pml`.
- `atomic_hash_table.pml` — `atomic_hash_table`: the slot lock driven up with acquire and released with an xor, the counters under it, the finder, and a full table.
  `-Dscenario=exhausted` for one slot, and `-Dwaiting=` for the policy the slot lock's retry loop plugs in.
- `snapshot_reader.pml` — `snapshot_store::reader_t`: its claim joining the census under the clock's mutex, reads at its stamp while commits prune the key's version run, and a transaction adopting the stamp.
  `-Dscenario=adoption` for the adoption.
- `partitioned_erase.pml` — `partitioned_store`'s store-level window writes: every partition held, one stamp drawn once all of them staged, stamped into each, and the watermark moved after the last.
- `staged_batch.pml` — the range modifiers of `basic_avl_tree`, `basic_flat_set` and `basic_hash_table`: the range staged in a container of the destination's own kind, the two causes that can refuse met there, and the absorb that asks the allocator at most once.
  `-Dscenario=tree`, the default, for the merge that relinks and asks for nothing; `-Dscenario=flat` for the merged array; `-Dscenario=table` for `reserve_more`.
- `snapshot_reader.cpp` — the pinned reader and its adoption as a GenMC client over `std::atomic`.
- `partitioned_erase.cpp` — the one-stamp window write as a GenMC client over `std::atomic`.

## What the models found, and what changed for it

A `locked_store` dropped its mutex between the two phases of a split commit.
`validate_for_commit` took the shared lock and released it, `publish_under` took the unique lock after, and the inner publication is documented "only ever called after `validate_for_commit` answered success, with nothing since": a writer committing over a watched key in the gap was the lost update the watch was taken against.
The partition locks of a `partitioned_store` cover the gap, but a group built straight from locked stores, the README's own example, had nothing over it.
`transaction_group.pml` found it with one writer against two groups; the validation now takes the mutex exclusively and keeps it until the publication, the rollback or the reset that follows, and `-Dwithout_held_validation` keeps the counterexample.
Two groups holding across their phases cannot deadlock because both take the stores ascending, which the group already argued and `-Dwithout_address_order` now shows by reversing one of them.

The atomic hash table counted a slot populated after it had unlocked it.
`emplace` released the slot and then added to `populated_count`, while `erase` subtracted under the lock, so an eraser slipping in between took the unsigned count through zero.
The count is a statistic and nothing reads it for a decision, but `atomic_hash_table.pml` showed the dip, the add moved under the lock, and `-Dwithout_count_under_lock` keeps the old order.
The slot lock also re-issued its `fetch_or` on every miss, a store to a header thirty-two slots share; it reads until the two bits are not both set before it claims now, the shape the model's own lock had.

The commit order carries no mutex at all, and four orderings stand in its place.
A reader counts its snapshot into a bucket before reading the watermark, a mark reads the watermark before it scans the buckets, a landing commit reads the watermark through a read-modify-write before walking the ring, and a bucket is shut to arrivals before its floor is replaced.
`commit_order.pml` drops each in turn under `-Dwithout_*`, and `low_water_mark_` is read without any lock by the pruner, where a stale read is a lower mark, which frees less and never a version a live claim names.

A store-level window write over partitions sharing a clock drew one stamp per partition.
`erase_range`, `erase_from`, `erase_up_to` and `update_range` held every partition exclusively and called each part's own, whose publication drew, stamped and published a stamp of its own in turn, while a snapshot is drawn from the clock without any partition lock.
A reader drawing between two of those publications named the first partition's stamp and not the second's, and saw the window half written once the locks came free.
`partitioned_erase.pml` finds it with one eraser and one reader; the partitions now stage into publications and one stamp publishes all of them, and `-Dwithout_one_stamp` keeps the counterexample.

`snapshot_reader.pml` found nothing to change.
It confirms that the claim a pinned reader holds, and the claim an adopting transaction links beside it, are what keep a prune off the version each reads, and its two variants without a claim show the prune that would land otherwise.

`staged_batch.pml` found nothing to change either.
It confirms that both causes a batch has to survive, the allocator refusing a request and an element refusing its own duplication, are met while the range is still outside the destination, and that the absorb past them asks for room once or not at all.
Spin reports the tree's refusal branch as unreachable under `-Dscenario=tree`, which is the merge relinking rather than allocating, stated as a verdict rather than as a docblock.
Its three variants are the batch before it staged, a merge that copied rather than relinked, and the key check moved past the absorb.
The last is the sharpest: with the absorb ahead of it, `update`'s check for a key that is not here finds the key the absorb has just written, so the batch answers success over a destination it should never have touched.

## What is not covered

The in-turn commit of stores that do not split theirs is documented as a tear when a later participant refuses, and `-Dwhole_across_stores` is the assertion that shows it rather than a fix.
The stage's unwind discards each rollback's status; an inner rollback fails only with a store-level fault the participant reports again on its next call, and the stage already returns the refusal that matters.
The trees, the vectors and the single-writer stores are documented one-thread cores, and only their staged batch has a model.
The GenMC clients spell their mutexes as one exchanged word rather than `spin_shared_mutex_t`, whose shape `spin_shared_mutex.pml` already checks, and they run only where GenMC is installed.

`staged_batch.pml` runs three keys and a range of three, which is enough for a key already there, a key that is free and a key the range repeats, and not enough for the probe of an open table.
That `apply_each_` cannot refuse once `reserve_more` has returned rests on the growth threshold leaving a quarter of the slots free and on a bounded probe walking all of them, which is `hash_layout.hpp`'s property and has no model of its own.
A table whose element duplicates without refusing stages nothing and writes the caller's range straight in; `-Dscenario=table` covers that path as the same one allocation, since the two differ in what they duplicate rather than in what a refusal leaves behind.
Only `basic_avl_tree` spells `update` over a range, so the model's fourth verb stands for the tree alone.

The waiting policies run over `locked_store.pml` and `atomic_hash_table.pml`, the smallest model over each of the two locks; the models that layer a store protocol on the same mutex run under the default, since a policy admitting more interleavings for the lock admits them for everything above it.
A policy that sleeps on a notification of its own, rather than on the word the lock lives in, is covered only for when it retries: the wake lives in the word's own history here, so a wake-up lost between a waiter deciding to sleep and a releaser looking for waiters is not a shape these models can express.

## Running

```sh
./check.sh
```

Inside a superproject the runner finds ForkUnion two directories up, as the forwarder does; standalone, check ForkUnion out beside this repository, which is what CI does.
Every `verify` line names a model, the expected verdict and the defines, so a new variant is one line.
Counting the verdicts here would drift the moment a line is added, so the file is the count.

A green suite is narrower than it looks, and this is worth saying plainly.
Six defects in `basic_commit_order` were found by the C++ tests while every model passed: two of them no model here can express — a livelock leaves no invalid end state under `-DSAFETY`, and two unordered relaxed stores are a shape rather than a value — and the other four were arithmetic and object lifetime, which these models abstract away.
