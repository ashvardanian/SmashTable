#!/usr/bin/env bash
# Runs every Promela model under the memory models it is meant for, and compares Spin's verdict
# with the expected one: `pass` means no assertion fails, `fail` means the model admits the
# outcome the assertion forbids - the variants that drop an order or hold a lock less long. The
# memory models and the runner's functions live in ForkUnion's `verification/`, beside this
# repository in whichever superproject vendors both, and checked out beside it in CI.
# Needs `spin` and a C compiler; a GenMC on the path, or named by `GENMC`, also runs the clients.

set -u
cd "$(dirname "$0")"
source ../../forkunion/verification/check.sh

section "locked_store.pml: one shared mutex per call, and what a reader inside it sees"
verify locked_store.pml pass -Dmemory=sequential
verify locked_store.pml pass
verify locked_store.pml fail -Dwithout_unlock_release
verify locked_store.pml fail -Dwithout_lock_acquire
verify locked_store.pml pass -Dwaiting=pausing
verify locked_store.pml pass -Dwaiting=on_the_address
verify locked_store.pml pass -Dwaiting=parking
verify locked_store.pml pass -Dqueued
verify locked_store.pml fail -Dqueued -Dwithout_unlock_release

section "transaction_group.pml: staging in address order, the two-pass commit, the in-turn tear, the held validation, and its release at a refusal"
verify transaction_group.pml pass -Dmemory=sequential
verify transaction_group.pml pass
verify transaction_group.pml pass -Dmemory=sequential -Dscenario=in_turn
verify transaction_group.pml fail -Dmemory=sequential -Dscenario=in_turn -Dwhole_across_stores
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_address_order
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_prefix_rollback
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_held_validation
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_refusal_release
verify transaction_group.pml pass -Dmemory=sequential -Dscenario=one_stamp
verify transaction_group.pml pass -Dscenario=one_stamp
verify transaction_group.pml fail -Dmemory=sequential -Dscenario=one_stamp -Dwithout_one_stamp

section "partitioned_store.pml: ascending partition locks, the stamp, the watermark, and the cursor's epoch"
verify partitioned_store.pml pass -Dmemory=sequential
verify partitioned_store.pml pass
verify partitioned_store.pml fail -Dmemory=sequential -Dwithout_ascending_order
verify partitioned_store.pml fail -Dwithout_watermark_mutex
verify partitioned_store.pml pass -Dscenario=cursor
verify partitioned_store.pml fail -Dscenario=cursor -Dwithout_epoch_release
verify partitioned_store.pml fail -Dmemory=sequential -Dwithout_held_partitions
verify partitioned_store.pml fail -Dwithout_held_partitions

section "commit_order.pml: the stamp ring, the watermark walk, the reader buckets and the marks they bound"
verify commit_order.pml pass -Dmemory=sequential
verify commit_order.pml pass
verify commit_order.pml fail -Dwithout_done_release
verify commit_order.pml pass -Dmemory=sequential -Dwithout_done_release
verify commit_order.pml fail -Dwithout_advance_rmw
verify commit_order.pml pass -Dmemory=sequential -Dwithout_advance_rmw
verify commit_order.pml pass -Dmemory=sequential -Dcommitters=3
verify commit_order.pml fail -Dmemory=sequential -Dcommitters=3 -Dwithout_ring_check
verify commit_order.pml pass -Dmemory=sequential -Dscenario=census
verify commit_order.pml pass -Dscenario=census
verify commit_order.pml fail -Dmemory=sequential -Dscenario=census -Dwithout_watermark_first
verify commit_order.pml fail -Dscenario=census -Dwithout_watermark_first
verify commit_order.pml pass -Dmemory=sequential -Dscenario=rotation
verify commit_order.pml pass -Dscenario=rotation
verify commit_order.pml pass -Dmemory=sequential -Dscenario=sharing
verify commit_order.pml pass -Dscenario=sharing
verify commit_order.pml pass -Dmemory=far -Dscenario=sharing
verify commit_order.pml fail -Dmemory=far -Dscenario=sharing -Dwithout_share_release
verify commit_order.pml pass -Dmemory=sequential -Dscenario=retention
verify commit_order.pml fail -Dmemory=sequential -Dscenario=retention -Dwithout_head_watermark
verify commit_order.pml fail -Dmemory=sequential -Dscenario=retention -Dbuckets=1

section "atomic_hash_table.pml: the slot lock and unlock, the counters, the finder, and a full table"
verify atomic_hash_table.pml pass -Dmemory=sequential
verify atomic_hash_table.pml pass
verify atomic_hash_table.pml fail -Dwithout_unlock_release
verify atomic_hash_table.pml fail -Dwithout_lock_acquire
verify atomic_hash_table.pml fail -Dmemory=sequential -Dwithout_count_under_lock
verify atomic_hash_table.pml pass -Dmemory=sequential -Dscenario=exhausted
verify atomic_hash_table.pml fail -Dmemory=far -Dwithout_count_release
verify atomic_hash_table.pml pass -Dwaiting=pausing
verify atomic_hash_table.pml pass -Dwaiting=on_the_address
verify atomic_hash_table.pml pass -Dwaiting=parking

section "snapshot_reader.pml: a pinned reader's claim against commits that prune, and a transaction adopting its stamp"
verify snapshot_reader.pml pass -Dmemory=sequential
verify snapshot_reader.pml pass
verify snapshot_reader.pml fail -Dmemory=sequential -Dwithout_held_claim
verify snapshot_reader.pml fail -Dwithout_held_claim
verify snapshot_reader.pml fail -Dmemory=sequential -Dwithout_join_first
verify snapshot_reader.pml fail -Dwithout_join_first
verify snapshot_reader.pml pass -Dmemory=sequential -Dscenario=adoption
verify snapshot_reader.pml pass -Dscenario=adoption
verify snapshot_reader.pml fail -Dmemory=sequential -Dscenario=adoption -Dwithout_shared_claim
verify snapshot_reader.pml fail -Dscenario=adoption -Dwithout_shared_claim
verify snapshot_reader.pml pass -Dmemory=sequential -Dcommitters=2
verify snapshot_reader.pml fail -Dmemory=sequential -Dcommitters=2 -Dwithout_watermark_first

section "partitioned_erase.pml: a store-level window write spanning partitions under one stamp"
verify partitioned_erase.pml pass -Dmemory=sequential
verify partitioned_erase.pml pass
verify partitioned_erase.pml pass -Dmemory=far
verify partitioned_erase.pml fail -Dmemory=sequential -Dwithout_one_stamp
verify partitioned_erase.pml fail -Dwithout_one_stamp
verify partitioned_erase.pml fail -Dmemory=far -Dwithout_one_stamp

section "staged_batch.pml: the range built beside the destination, and the refusal that leaves it as it was"
verify staged_batch.pml pass -Dscenario=tree
verify staged_batch.pml fail -Dscenario=tree -Dwithout_staging
verify staged_batch.pml fail -Dscenario=tree -Dwithout_secured_room
verify staged_batch.pml fail -Dscenario=tree -Dwithout_prior_check
verify staged_batch.pml pass -Dscenario=flat
verify staged_batch.pml fail -Dscenario=flat -Dwithout_staging
verify staged_batch.pml fail -Dscenario=flat -Dwithout_secured_room
verify staged_batch.pml fail -Dscenario=flat -Dwithout_prior_check
verify staged_batch.pml pass -Dscenario=table
verify staged_batch.pml fail -Dscenario=table -Dwithout_staging
verify staged_batch.pml fail -Dscenario=table -Dwithout_secured_room
verify staged_batch.pml fail -Dscenario=table -Dwithout_prior_check

section "snapshot_reader.cpp and partitioned_erase.cpp: the same two protocols under GenMC"
if genmc_ready; then
    verify_client snapshot_reader.cpp pass
    verify_client snapshot_reader.cpp fail -Dwithout_held_claim
    verify_client snapshot_reader.cpp fail -Dwithout_join_first
    verify_client snapshot_reader.cpp pass -Dscenario=adoption
    verify_client snapshot_reader.cpp fail -Dscenario=adoption -Dwithout_shared_claim
    verify_client snapshot_reader.cpp fail -Dscenario=adoption -Dwithout_join_first
    verify_client partitioned_erase.cpp pass
    verify_client partitioned_erase.cpp fail -Dwithout_one_stamp
fi

finish
