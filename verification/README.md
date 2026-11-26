# Verification

Model checking for the protocols the stores promise, with [Spin](https://spinroot.com) models and [GenMC](https://github.com/MPI-SWS/genmc) clients.

- `atomic_hash_table/`: the slot lock and unlock of `atomic_hash_table`, and the counters under it.
- `atomic_intrusive_stack.cpp`: two pushers of `atomic_intrusive_stack`, one of them pushing a chain, and a taker draining as they go.
- `commit_order/`: the census of `basic_commit_order`, the buckets readers join, the floors they carry, and the mark.
- `commit_order_publication.pml`: the word of landings and the watermark of `basic_commit_order`, with one version per committer.
- `commit_order_steps.pml`: `basic_commit_order` as inlines, shared by every model that publishes commits or counts readers.
- `locked_store.pml`: one shared mutex per call, and a reader inside it seeing a commit whole.
- `partitioned_erase.pml`, `partitioned_erase.cpp`: a window write spanning every partition of `partitioned_store` under one stamp.
- `partitioned_store.pml`: a cursor stepping past a store-wide write, and a commit over one partition beside them.
- `snapshot_reader/`: `snapshot_store::reader_t` and a transaction adopting its stamp, against a commit that prunes behind them.
- `shared_mutex_order.pml`: three writers and a reader on one `spin_shared_mutex_t`, the fewest where one writer can bypass another.
- `spin_shared_mutex.pml`: `spin_shared_mutex_t` as one word, the serving ticket at the bottom, readers counted above it and queued writers on top, with the writers' order asserted in every model over it.
- `transaction_group/`: `transaction_group` over two locked stores, in turn, in two passes, and on one stamp, then over a locked and a partitioned store held through the prune; `one_stamp.pml` runs only with `VERIFY_BACKGROUND=1`.

## Running

`./check.sh` runs every `@verify` line here, or only the files it is given, and the GenMC clients once `GENMC` names its executable.
`check.sh`, `weak_memory.pml`, `monitor_wait.pml` and `genmc.hpp` are byte-identical copies of ForkUnion's: change them there and copy them here.
[ForkUnion's verification README](https://github.com/ashvardanian/ForkUnion/blob/main/verification/README.md) documents the runner, the `@verify` grammar, the two include roots and the memory budget.

## What the models found

- `validate_for_commit` holds a locked store's mutex exclusively until the publication, the rollback or the reset that follows: [`transaction_group/`](transaction_group/).
- Two groups holding across their phases take the stores ascending, or they deadlock: [`transaction_group/`](transaction_group/).
- A refused validation gives its hold straight back, and so do the stores asked before it: [`transaction_group/`](transaction_group/).
- A store reclaims what it published before giving its hold up, since taking it back while later stores are held deadlocks two groups: [`transaction_group/`](transaction_group/).
- A store keeping its hold from the publication to the prune needs every store of the group to keep it: a locked store given up at its publication and taken back at its prune deadlocks two groups, and a group against a reader of a locked store over a partitioned one: [`transaction_group/held_prune.pml`](transaction_group/held_prune.pml).
- A group awaiting its own publication under its holds waits on no commit that needs them, a full ring included, and without the wait it reopens below its own stamp: [`transaction_group/held_prune.pml`](transaction_group/held_prune.pml).
- The atomic hash table counts a slot populated under the slot's lock, and its slot lock acquires: [`atomic_hash_table/`](atomic_hash_table/).
- A push onto the intrusive stack lands with a releasing compare-exchange and a take with an acquiring exchange, or the taker meets a record whose payload and link it is not ordered after: [`atomic_intrusive_stack.cpp`](atomic_intrusive_stack.cpp).
- A locked store's writers acquire its mutex and its readers acquire the shared side, and `unlock_shared` wakes parked writers only when the last reader leaves: [`locked_store.pml`](locked_store.pml).
- A reader entering the mutex with one blind add wakes the parked threads when it takes the last unit back out, retries with the bounded add, and needs room in the tally above the admission limit, since a refused unit added to a full tally reads as a waiting writer over no readers: [`locked_store.pml`](locked_store.pml).
- Writers enter the shared mutex in the order they took their tickets, and no reader is admitted while one is queued; with fewer tickets than writers queued at once, a writer takes the ticket of the one inside and enters beside it: [`shared_mutex_order.pml`](shared_mutex_order.pml).
- The commit order stands on three orderings in place of a mutex: a reader joins a bucket and reads the watermark through a read-modify-write, a mark reads the watermark before scanning the buckets, and a landing commit flips its bit and reads the word back before it loads the watermark: [`commit_order/`](commit_order/), [`commit_order_publication.pml`](commit_order_publication.pml).
- `end_commit` wakes parked committers once its raise moves the watermark: [`commit_order_publication.pml`](commit_order_publication.pml).
- A landing that flips one bit of a parity word returns once the watermark passed its own stamp, or reads the watermark before the flip, and the wait for room in the window is what keeps a stamp a lap above from sharing its bit: [`commit_order_publication.pml`](commit_order_publication.pml).
- A census join by one unconditional add leaves through the retire where the bucket was recycled ahead of it, and the member it held meanwhile only lowers a mark: [`commit_order/`](commit_order/).
- A store-level window write over partitions sharing a clock publishes every partition under one stamp: [`partitioned_erase.pml`](partitioned_erase.pml), [`partitioned_erase.cpp`](partitioned_erase.cpp).
- `ordered_cursor_t` reads fronts under the shared lock and the write count is stepped with a release, either of which keeps a front fresh, and a store publishes every part it wrote: [`partitioned_store.pml`](partitioned_store.pml).
- The claim a pinned reader holds, and the one an adopting transaction links beside it, keep a prune off the version each reads, and the mark seals against the newest published stamp: [`snapshot_reader/`](snapshot_reader/).
- A staged batch meets the allocator's refusal and an element's refusal while the range is still outside the destination: [`test/batch_atomicity.hpp`](../test/batch_atomicity.hpp).

## What is not covered

- A caller's own store offering only `commit` tears a group when a later participant refuses, and `whole_across_stores=true` is the assertion that shows it; no store here reaches that path.
- The trees, the vectors and the single-writer stores are documented one-thread cores, and their staged batch is checked by the C++ tests in `test/batch_atomicity.hpp`, not by a model.
- The GenMC clients spell their mutexes as one exchanged word rather than `spin_shared_mutex_t`, whose shape `spin_shared_mutex.pml` already checks.
- That `apply_each_` cannot refuse once `reserve_more` has returned rests on the growth threshold leaving a quarter of the slots free and on a bounded probe walking all of them, which is `hash_layout.hpp`'s property and has no model of its own.
- Only a futex-style park can lose a wake-up, so the waits run as parks through `monitor_wait.pml`.
- The parity word runs at four bits with four committers of one stamp each, so a fifth stamp sharing a bit with the first is reached only at two bits.
- A joiner refused under the unconditional add re-reads the head until it moved, since reading the same stale head again would only repeat the refusal.
- `await_published` is not modelled on its own; it parks on the same word, woken by the same walk, as the ring-room wait in `commit_order_steps.pml`.
- Six defects in `basic_commit_order` were found by the C++ tests while every model passed: two of them no model here can express, and the other four were arithmetic and object lifetime, which these models abstract away.
