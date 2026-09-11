# Verification

Model checking for the protocols the stores promise: the two-phase group commit, the partitioned commit under one stamp, the snapshot clock, the slot lock of the atomic hash table, and the shared mutex every locked store takes.
[Spin](https://spinroot.com) checks each as a Promela model under the memory models of [ForkUnion's `verification/`](https://github.com/ashvardanian/ForkUnion), which sits beside this repository inside USearch and is checked out beside it in CI; `./check.sh` runs everything here and compares each verdict with the expected one.

| File                    | What it checks                                                                                                                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weak_memory.pml`       | The forwarder to ForkUnion's memory module.                                                                                                                                                                                           |
| `spin_shared_mutex.pml` | `spin_shared_mutex_t` as one word, the lock an acquire exchange and the unlock a release, shared by every model here.                                                                                                                 |
| `locked_store.pml`      | `locked_store`: one shared mutex per call, exclusion, and a reader inside the lock seeing a commit whole.                                                                                                                             |
| `transaction_group.pml` | `transaction_group`: staging in address order and the unwind of a refused prefix, the two-pass commit that asks every participant before any writes, the in-turn commit's tear, and a participant holding its lock across the phases. |
| `partitioned_store.pml` | `partitioned_store`: the reached partitions held ascending, the stamp drawn before any write and the watermark moved after the last, and the cursor's epoch; `-Dscenario=cursor` for the epoch.                                       |
| `snapshot_clock.pml`    | `snapshot_clock_t`: leases, commits in flight, the low-water mark, and the watermark stored and read relaxed under its mutex.                                                                                                         |
| `atomic_hash_table.pml` | `atomic_hash_table`: the slot lock driven up with acquire and released with an xor, the counters under it, the finder, and a full table; `-Dscenario=exhausted` for one slot.                                                         |

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

The snapshot clock's relaxed words are ordered by its mutex, and the model says which reads may stand outside it.
`published_stamp_` is stored and read under `mutex_` by every transactional path, and `snapshot_clock.pml` shows the bare accessor, the one path outside it, naming a stamp whose versions a reader cannot yet see under the view model, while under sequential consistency it is fine: that is the docblock's warning made visible, and no code changed for it.
`low_water_mark_` is read without the mutex by the pruner, and a stale read is a lower mark, which frees less and never a version a live lease names.

## What is not covered

The in-turn commit of stores that do not split theirs is documented as a tear when a later participant refuses, and `-Dwhole_across_stores` is the assertion that shows it rather than a fix.
The stage's unwind discards each rollback's status; an inner rollback fails only with a store-level fault the participant reports again on its next call, and the stage already returns the refusal that matters.
The trees, the vectors and the single-writer stores are documented one-thread cores and have no model.
No client runs under GenMC yet; the runner's `verify_client` is there when one does.

## Running

```sh
./check.sh
```

Inside USearch the runner finds ForkUnion two directories up, as the forwarder does; standalone, check ForkUnion out beside this repository, which is what CI does.
Every `verify` line names a model, the expected verdict and the defines, so a new variant is one line.
The suite's 27 verdicts take about ten seconds four at a time, which is the default.
