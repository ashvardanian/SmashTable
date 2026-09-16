#!/usr/bin/env bash
# Runs every Promela model under the memory models it is meant for, and compares Spin's verdict
# with the expected one: `pass` means no assertion fails, `fail` means the model admits the
# outcome the assertion forbids - the variants that drop an order or hold a lock less long. The
# memory models and the runner's functions live in ForkUnion's `verification/`, beside this
# repository inside USearch and checked out beside it in CI.
# Needs `spin` and a C compiler; a GenMC on the path, or named by `GENMC`, also runs the clients.

set -u
cd "$(dirname "$0")"
source ../../forkunion/verification/check.sh

section "locked_store.pml: one shared mutex per call, and what a reader inside it sees"
verify locked_store.pml pass -Dmemory=sequential
verify locked_store.pml pass
verify locked_store.pml fail -Dwithout_unlock_release
verify locked_store.pml fail -Dwithout_lock_acquire

section "transaction_group.pml: staging in address order, the two-pass commit, the in-turn tear, and the held validation"
verify transaction_group.pml pass -Dmemory=sequential
verify transaction_group.pml pass
verify transaction_group.pml pass -Dmemory=sequential -Dscenario=in_turn
verify transaction_group.pml fail -Dmemory=sequential -Dscenario=in_turn -Dwhole_across_stores
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_address_order
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_prefix_rollback
verify transaction_group.pml fail -Dmemory=sequential -Dwithout_held_validation

section "partitioned_store.pml: ascending partition locks, the stamp, the watermark, and the cursor's epoch"
verify partitioned_store.pml pass -Dmemory=sequential
verify partitioned_store.pml pass
verify partitioned_store.pml fail -Dmemory=sequential -Dwithout_ascending_order
verify partitioned_store.pml fail -Dwithout_watermark_mutex
verify partitioned_store.pml pass -Dscenario=cursor
verify partitioned_store.pml fail -Dscenario=cursor -Dwithout_epoch_release

section "snapshot_clock.pml: leases, in-flight commits, the low-water mark, and the watermark under its mutex"
verify snapshot_clock.pml pass -Dmemory=sequential
verify snapshot_clock.pml pass
verify snapshot_clock.pml pass -Dmemory=sequential -Dwithout_watermark_mutex
verify snapshot_clock.pml fail -Dwithout_watermark_mutex

section "atomic_hash_table.pml: the slot lock and unlock, the counters, the finder, and a full table"
verify atomic_hash_table.pml pass -Dmemory=sequential
verify atomic_hash_table.pml pass
verify atomic_hash_table.pml fail -Dwithout_unlock_release
verify atomic_hash_table.pml fail -Dwithout_lock_acquire
verify atomic_hash_table.pml fail -Dmemory=sequential -Dwithout_count_under_lock
verify atomic_hash_table.pml pass -Dmemory=sequential -Dscenario=exhausted

section "snapshot_reader.pml: a pinned reader's claim against commits that prune, and a transaction adopting its stamp"
verify snapshot_reader.pml pass -Dmemory=sequential
verify snapshot_reader.pml pass
verify snapshot_reader.pml fail -Dmemory=sequential -Dwithout_lease
verify snapshot_reader.pml pass -Dmemory=sequential -Dscenario=adoption
verify snapshot_reader.pml pass -Dscenario=adoption
verify snapshot_reader.pml fail -Dmemory=sequential -Dscenario=adoption -Dwithout_shared_lease

section "partitioned_erase.pml: a store-level window write spanning partitions under one stamp"
verify partitioned_erase.pml pass -Dmemory=sequential
verify partitioned_erase.pml pass
verify partitioned_erase.pml fail -Dmemory=sequential -Dwithout_one_stamp

section "snapshot_reader.cpp and partitioned_erase.cpp: the same two protocols under GenMC"
if genmc_ready; then
    verify_client snapshot_reader.cpp pass
    verify_client snapshot_reader.cpp fail -Dwithout_lease
    verify_client snapshot_reader.cpp fail -Dwithout_shared_lease
    verify_client partitioned_erase.cpp pass
    verify_client partitioned_erase.cpp fail -Dwithout_one_stamp
fi

finish
