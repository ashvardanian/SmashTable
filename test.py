"""
PyTest suite for Map Python bindings.

Tests thread-safe partitioned AVL tree with support for:
- Python 3.14t free-threading (GIL-free)
- Sub-interpreter data exchange
- Atomic batch operations via dict
- Transactions with watch/CAS semantics
- Monotonic Atomic View consistency
"""

import pytest
import threading
from concurrent.futures import ThreadPoolExecutor

import smashtable as st


class TestMapBasics:
    """Basic CRUD operations on Map"""

    def test_create_empty(self):
        """Create an empty Map"""
        tree = st.Map()
        assert tree.size() == 0
        assert tree.empty()

    def test_upsert_find_single(self):
        """Insert and find a single key-value pair"""
        tree = st.Map()

        assert tree.upsert(1, 100)
        assert tree.size() == 1
        assert not tree.empty()

        value = tree.find(1)
        assert value == 100

    def test_upsert_batch_dict(self):
        """Batch insert using dict"""
        tree = st.Map()

        # Batch upsert with dict
        count = tree.upsert({1: 100, 2: 200, 3: 300})
        assert count == 3
        assert tree.size() == 3

        assert tree.find(1) == 100
        assert tree.find(2) == 200
        assert tree.find(3) == 300

    def test_find_batch_list(self):
        """Batch find using list"""
        tree = st.Map()

        tree.upsert({1: 100, 2: 200, 3: 300, 4: 400})

        # Batch find with list
        results = tree.find([1, 2, 4])
        assert results == {1: 100, 2: 200, 4: 400}

        # Missing keys not in result
        results = tree.find([1, 999])
        assert results == {1: 100}

    def test_find_batch_tuple(self):
        """Batch find using tuple"""
        tree = st.Map()

        tree.upsert({10: 1000, 20: 2000})

        # Batch find with tuple
        results = tree.find((10, 20))
        assert results == {10: 1000, 20: 2000}

    def test_upsert_update(self):
        """Update existing key"""
        tree = st.Map()

        tree.upsert(1, 100)
        tree.upsert(1, 200)  # Update

        assert tree.size() == 1
        assert tree.find(1) == 200

    def test_find_missing(self):
        """Find on missing key returns None"""
        tree = st.Map()
        assert tree.find(999) is None

    def test_erase(self):
        """Erase a key"""
        tree = st.Map()

        tree.upsert(1, 100)
        assert tree.find(1) == 100

        assert tree.erase(1)
        assert tree.find(1) is None
        assert tree.size() == 0

    def test_erase_missing(self):
        """Erase missing key returns False"""
        tree = st.Map()
        assert not tree.erase(999)

    def test_clear(self):
        """Clear all entries"""
        tree = st.Map()

        tree.upsert({i: i * 10 for i in range(100)})
        assert tree.size() == 100

        tree.clear()
        assert tree.size() == 0
        assert tree.empty()


class TestMapTransactions:
    """Transaction support with watch/stage/commit"""

    def test_transaction_basic(self):
        """Basic transaction with stage/commit"""
        tree = st.Map()

        txn = tree.transaction()
        txn.upsert(1, 100)
        txn.upsert(2, 200)

        # Not visible until commit
        assert tree.find(1) is None

        txn.stage()
        assert tree.find(1) is None  # Still invisible (staged but not committed)

        txn.commit()
        assert tree.find(1) == 100  # Now visible
        assert tree.find(2) == 200

    def test_transaction_rollback(self):
        """Transaction rollback undoes staged changes"""
        tree = st.Map()
        tree.upsert(1, 100)

        txn = tree.transaction()
        txn.upsert(1, 999)
        txn.stage()
        txn.rollback()

        assert tree.find(1) == 100  # Original value preserved

    def test_transaction_watch_success(self):
        """Watch succeeds when key unchanged"""
        tree = st.Map()
        tree.upsert(1, 100)

        txn = tree.transaction()
        txn.watch(1)
        txn.upsert(1, 200)

        txn.stage()
        txn.commit()

        assert tree.find(1) == 200

    def test_transaction_watch_conflict(self):
        """Watch fails when key changes (CAS semantics)"""
        tree = st.Map()
        tree.upsert(1, 100)

        txn1 = tree.transaction()
        txn1.watch(1)

        # Another transaction modifies key 1
        tree.upsert(1, 999)

        txn1.upsert(1, 200)

        with pytest.raises(RuntimeError, match="consistency"):
            txn1.stage()

    def test_transaction_concurrent_conflict(self):
        """Two transactions on same key - first wins"""
        tree = st.Map()
        tree.upsert(1, 100)

        txn1 = tree.transaction()
        txn2 = tree.transaction()

        txn1.watch(1)
        txn2.watch(1)

        txn1.upsert(1, 200)
        txn2.upsert(1, 300)

        txn1.stage()
        txn1.commit()

        with pytest.raises(RuntimeError, match="consistency"):
            txn2.stage()

    def test_transaction_disjoint_keys(self):
        """Transactions on different keys both succeed"""
        tree = st.Map()

        txn1 = tree.transaction()
        txn2 = tree.transaction()

        txn1.upsert(1, 100)
        txn2.upsert(2, 200)

        txn1.stage()
        txn2.stage()

        txn1.commit()
        txn2.commit()

        assert tree.find(1) == 100
        assert tree.find(2) == 200


class TestMapConcurrency:
    """Thread-safety and GIL-free operation"""

    def test_concurrent_upserts(self):
        """Multiple threads can safely upsert"""
        tree = st.Map()

        def worker(start, count):
            for i in range(start, start + count):
                tree.upsert(i, i * 100)

        threads = []
        for t in range(4):
            thread = threading.Thread(target=worker, args=(t * 250, 250))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert tree.size() == 1000

        # Verify subset of values
        results = tree.find([0, 100, 500, 999])
        assert results[0] == 0
        assert results[100] == 10000
        assert results[500] == 50000
        assert results[999] == 99900

    def test_concurrent_batch_upserts(self):
        """Concurrent batch upserts"""
        tree = st.Map()

        def worker(offset):
            batch = {i: i * 10 for i in range(offset, offset + 100)}
            tree.upsert(batch)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i * 100) for i in range(4)]
            for f in futures:
                f.result()

        assert tree.size() == 400

    def test_concurrent_executor(self):
        """ThreadPoolExecutor concurrent access (GIL-free)"""
        tree = st.Map()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(tree.upsert, i, i * 10) for i in range(500)]
            results = [f.result() for f in futures]

        assert all(results)
        assert tree.size() == 500

    def test_concurrent_find_while_upsert(self):
        """Concurrent reads while writing"""
        tree = st.Map()

        # Pre-populate
        tree.upsert({i: i for i in range(100)})

        found_count = [0]

        def reader():
            for _ in range(100):
                results = tree.find(list(range(100)))
                found_count[0] += len(results)

        def writer():
            tree.upsert({i: i for i in range(100, 200)})

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert found_count[0] > 0
        assert tree.size() >= 100


class TestMapMixedTypes:
    """Heterogeneous key and value types (str, int, float, bytes)"""

    def test_string_keys_values(self):
        """String keys with various value types"""
        tree = st.Map()

        tree.upsert("name", "Alice")
        tree.upsert("age", 30)
        tree.upsert("pi", 3.14159)
        tree.upsert("data", b"binary")

        assert tree.find("name") == "Alice"
        assert tree.find("age") == 30
        assert tree.find("pi") == 3.14159
        assert tree.find("data") == b"binary"

    def test_int_keys_mixed_values(self):
        """Integer keys with mixed value types"""
        tree = st.Map()

        tree.upsert(1, "string_value")
        tree.upsert(2, 42)
        tree.upsert(3, 2.718)
        tree.upsert(4, b"bytes_value")

        assert tree.find(1) == "string_value"
        assert tree.find(2) == 42
        assert tree.find(3) == 2.718
        assert tree.find(4) == b"bytes_value"

    def test_float_keys(self):
        """Float keys with various values"""
        tree = st.Map()

        tree.upsert(1.5, "one-half")
        tree.upsert(3.14, 314)
        tree.upsert(2.718, 2.718)

        assert tree.find(1.5) == "one-half"
        assert tree.find(3.14) == 314
        assert tree.find(2.718) == 2.718

    def test_bytes_keys(self):
        """Bytes keys are stored as strings internally"""
        tree = st.Map()

        tree.upsert(b"key1", "value1")
        tree.upsert(b"key2", 200)

        # Bytes are converted to strings
        assert tree.find(b"key1") == "value1"
        assert tree.find(b"key2") == 200

    def test_negative_and_large_integers(self):
        """Edge case integers: negative, zero, large"""
        tree = st.Map()

        tree.upsert(-100, "negative")
        tree.upsert(0, "zero")
        tree.upsert(2**60, "large_positive")

        assert tree.find(-100) == "negative"
        assert tree.find(0) == "zero"
        assert tree.find(2**60) == "large_positive"

    def test_batch_mixed_types(self):
        """Batch upsert with mixed key and value types"""
        tree = st.Map()

        count = tree.upsert({"name": "Bob", 42: "answer", 3.14: "pi", -1: b"bytes", b"bytes_key": 999})

        assert count == 5
        assert tree.size() == 5

        # Verify all types
        assert tree.find("name") == "Bob"
        assert tree.find(42) == "answer"
        assert tree.find(3.14) == "pi"
        assert tree.find(-1) == b"bytes"
        assert tree.find(b"bytes_key") == 999

    def test_batch_find_mixed_types(self):
        """Batch find with heterogeneous key types"""
        tree = st.Map()

        tree.upsert({"str_key": 100, 42: 200, 3.14: 300, b"bytes_key": 400})

        # Find with mixed types
        results = tree.find(["str_key", 42, 3.14, b"bytes_key"])

        assert results == {"str_key": 100, 42: 200, 3.14: 300, b"bytes_key": 400}

    def test_type_ordering(self):
        """Verify type ordering: string < int64 < uint64 < double"""
        tree = st.Map()

        # Insert in random order
        tree.upsert(100, "int")  # int
        tree.upsert("aaa", "string")  # string
        tree.upsert(3.14, "float")  # float
        tree.upsert(2**63, "large_uint")  # uint64

        # All should be findable
        assert tree.find("aaa") == "string"
        assert tree.find(100) == "int"
        assert tree.find(2**63) == "large_uint"
        assert tree.find(3.14) == "float"

    def test_transaction_mixed_types(self):
        """Transactions with heterogeneous types"""
        tree = st.Map()

        tree.upsert("key1", 100)

        txn = tree.transaction()
        txn.watch("key1")
        txn.upsert("key1", 200)
        txn.upsert(42, "new_key")

        txn.stage()
        txn.commit()

        assert tree.find("key1") == 200
        assert tree.find(42) == "new_key"

    def test_update_value_type_change(self):
        """Update same key with different value types"""
        tree = st.Map()

        tree.upsert("key", 100)  # int value
        assert tree.find("key") == 100

        tree.upsert("key", "string")  # change to string
        assert tree.find("key") == "string"

        tree.upsert("key", 3.14)  # change to float
        assert tree.find("key") == 3.14

        tree.upsert("key", b"bytes")  # change to bytes
        assert tree.find("key") == b"bytes"

    def test_empty_batch_operations(self):
        """Edge case: empty collections"""
        tree = st.Map()

        # Empty dict
        count = tree.upsert({})
        assert count == 0

        # Empty list
        results = tree.find([])
        assert results == {}

    def test_unicode_strings(self):
        """Unicode string keys and values"""
        tree = st.Map()

        tree.upsert("你好", "世界")
        tree.upsert("emoji", "🚀🎉")
        tree.upsert("Здравствуй", "мир")

        assert tree.find("你好") == "世界"
        assert tree.find("emoji") == "🚀🎉"
        assert tree.find("Здравствуй") == "мир"


class TestSet:
    """Set collection with heterogeneous keys"""

    def test_create_empty(self):
        """Create an empty Set"""
        s = st.Set()
        assert s.size() == 0
        assert s.empty()
        assert len(s) == 0

    def test_add_single(self):
        """Add single keys"""
        s = st.Set()
        assert s.add(1)
        assert s.add("hello")
        assert s.add(3.14)
        assert s.size() == 3
        assert len(s) == 3

    def test_add_batch(self):
        """Add batch of keys"""
        s = st.Set()
        count = s.add([1, 2, 3, 4, 5])
        assert count == 5
        assert s.size() == 5

    def test_contains_method(self):
        """Test contains() method"""
        s = st.Set()
        s.add("key1")
        s.add(42)

        assert s.contains("key1")
        assert s.contains(42)
        assert not s.contains("missing")

    def test_contains_operator(self):
        """Test 'in' operator"""
        s = st.Set()
        s.add("key1")
        s.add(42)
        s.add(3.14)

        assert "key1" in s
        assert 42 in s
        assert 3.14 in s
        assert "missing" not in s

    def test_remove(self):
        """Remove keys"""
        s = st.Set()
        s.add("key1")
        s.add(42)

        assert s.remove("key1")
        assert not s.contains("key1")
        assert not s.remove("missing")  # Returns False for missing

    def test_discard(self):
        """Discard doesn't error on missing keys"""
        s = st.Set()
        s.add("key1")

        s.discard("key1")  # Should succeed
        s.discard("missing")  # Should not raise
        assert s.size() == 0

    def test_clear(self):
        """Clear all keys"""
        s = st.Set()
        s.add([1, 2, 3, 4, 5])
        assert s.size() == 5

        s.clear()
        assert s.size() == 0
        assert s.empty()

    def test_mixed_types(self):
        """Set with mixed key types"""
        s = st.Set()
        s.add([
            "string",
            42,
            3.14,
            -1,
            b"bytes",
        ])

        assert s.size() == 5
        assert "string" in s
        assert 42 in s
        assert 3.14 in s
        assert -1 in s
        assert b"bytes" in s

    def test_concurrent_add(self):
        """Concurrent adds to set"""
        s = st.Set()

        def worker(start, count):
            for i in range(start, start + count):
                s.add(i)

        threads = []
        for t in range(4):
            thread = threading.Thread(target=worker, args=(t * 100, 100))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert s.size() == 400


class TestSetTransaction:
    """Transaction support for Set"""

    def test_transaction_basic(self):
        """Basic transaction with stage/commit"""
        s = st.Set()

        txn = s.transaction()
        txn.add(1)
        txn.add(2)

        # Not visible until commit
        assert 1 not in s

        txn.stage()
        assert 1 not in s  # Still invisible (staged but not committed)

        txn.commit()
        assert 1 in s  # Now visible
        assert 2 in s

    def test_transaction_rollback(self):
        """Transaction rollback undoes staged changes"""
        s = st.Set()
        s.add(1)

        txn = s.transaction()
        txn.remove(1)
        txn.stage()
        txn.rollback()

        assert 1 in s  # Original value preserved

    def test_transaction_watch_success(self):
        """Watch succeeds when key unchanged"""
        s = st.Set()
        s.add(1)

        txn = s.transaction()
        txn.watch(1)
        txn.remove(1)

        txn.stage()
        txn.commit()

        assert 1 not in s

    def test_transaction_watch_conflict(self):
        """Watch fails when key changes (CAS semantics)"""
        s = st.Set()
        s.add(1)

        txn1 = s.transaction()
        txn1.watch(1)

        # Another transaction modifies key 1
        s.remove(1)

        txn1.add(2)

        with pytest.raises(RuntimeError, match="consistency"):
            txn1.stage()

    def test_transaction_concurrent_conflict(self):
        """Two transactions on same key - first wins"""
        s = st.Set()
        s.add(1)

        txn1 = s.transaction()
        txn2 = s.transaction()

        txn1.watch(1)
        txn2.watch(1)

        txn1.remove(1)
        txn2.remove(1)

        txn1.stage()
        txn1.commit()

        with pytest.raises(RuntimeError, match="consistency"):
            txn2.stage()

    def test_transaction_disjoint_keys(self):
        """Transactions on different keys both succeed"""
        s = st.Set()

        txn1 = s.transaction()
        txn2 = s.transaction()

        txn1.add(1)
        txn2.add(2)

        txn1.stage()
        txn2.stage()

        txn1.commit()
        txn2.commit()

        assert 1 in s
        assert 2 in s

    def test_transaction_watch_missing_key(self):
        """Watch on missing key, ensure it stays missing"""
        s = st.Set()

        txn = s.transaction()
        txn.watch(999)  # Watch missing key
        txn.add(1)

        txn.stage()
        txn.commit()

        assert 1 in s
        assert 999 not in s

    def test_transaction_watch_missing_conflict(self):
        """Watch missing key, fail if someone adds it"""
        s = st.Set()

        txn = s.transaction()
        txn.watch(999)  # Watch missing key

        # Another operation adds it
        s.add(999)

        txn.add(1)

        with pytest.raises(RuntimeError, match="consistency"):
            txn.stage()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
