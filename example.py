"""A runnable tour of the Python containers, in the order you would meet them.

Every section asserts what it claims and prints what it saw, so this file fails rather than lies if
the behaviour drifts. Read it top to bottom; run it to watch it happen.

Every diagram has one shape: what happens, then a rule, then what everyone outside can see.

    ↦   maps to
    ·   staged, and visible to nobody
    ✓   committed, and visible to everyone
    ✗   discarded, as if it never happened
    ⚡  refused

Run:
    pip install -e . && python example.py
"""

import smashtable as st


def section(title: str) -> None:
    """Prints a heading, so the output reads as a walkthrough rather than a log."""
    print(f"\n{title}\n{'─' * len(title)}")


def shows(label: str, value) -> None:
    """Prints one observation, aligned so a column of them can be compared at a glance."""
    print(f"    {label:<32} {value}")


# region One Update, Several Containers

section("Two indexes that must move together")

# A lookup by name yields an identifier that a lookup by identifier must then resolve. Let one move
# without the other and the pair lies, so both writes have to land together or neither may.
#
#     by_id      42        ↦  'carol'
#     by_name    'carol'   ↦  42
#
#     ids[42] = 'carol'                 ·
#     names['carol'] = 42               ·
#     ──────────────────────────────────────────────────
#     the block ends                    ✓  both, together
by_id = st.SortedMap(key=int)
by_name = st.SortedMap(key=str)

with st.atomic(by_id, by_name) as (ids, names):
    ids[42] = "carol"
    names["carol"] = 42

shows("by_id[42]", by_id[42])
shows("by_name['carol']", by_name["carol"])
assert by_id[42] == "carol" and by_name["carol"] == 42

section("A block that raises applies nothing")

#     ids[43] = 'dave'                  ·
#     names['dave'] = 43                ·
#     raise ValueError                  ✗
#     ──────────────────────────────────────────────────
#     the block ends                    ✗  neither, and no torn state in between
try:
    with st.atomic(by_id, by_name) as (ids, names):
        ids[43] = "dave"
        names["dave"] = 43
        raise ValueError("failed validation")
except ValueError:
    pass

shows("43 in by_id", 43 in by_id)
shows("'dave' in by_name", "dave" in by_name)
assert 43 not in by_id and "dave" not in by_name

section("A group may mix maps and sets, and key types")

#     ids[7] = 'eve'                    ·  a map takes assignment
#     names['eve'] = 7                  ·  and its own key type
#     labels.add('staff')               ·  a set takes add
#     ──────────────────────────────────────────────────
#     the block ends                    ✓  all three, together
tags = st.SortedSet(key=str)

with st.atomic(by_id, by_name, tags) as (ids, names, labels):
    ids[7] = "eve"
    names["eve"] = 7
    labels.add("staff")

shows("by_id[7]", by_id[7])
shows("'staff' in tags", "staff" in tags)
assert by_id[7] == "eve" and "staff" in tags

# endregion One Update, Several Containers

# region The Ordinary Surface

section("Nothing above costs you the vocabulary you know")

scores = st.SortedMap(key=str)
scores.update({"carol": 91, "alice": 74, "bob": 88})

shows("len(scores)", len(scores))
shows("'alice' in scores", "alice" in scores)
shows("list(scores)", list(scores))
shows("dict(scores)", dict(scores))
shows("scores == a plain dict", scores == {"alice": 74, "bob": 88, "carol": 91})
assert list(scores) == ["alice", "bob", "carol"]
assert scores == {"alice": 74, "bob": 88, "carol": 91}

names_view = scores.keys()
scores["dan"] = 65
shows("a view taken before a write", list(names_view))
assert "dan" in list(names_view)

section("The order is real, so a window over it is cheap")

# The window is half-open, so it takes everything from the start up to but not including the stop,
# and neither bound has to be a key that exists.
#
#     'alice'   'bob'   'carol'   'dan'
#               └────────────┘
#               scan('b', 'd')
shows("scores.scan('b', 'd')", scores.scan("b", "d"))
assert [key for key, _ in scores.scan("b", "d")] == ["bob", "carol"]

# `popitem` takes the smallest key. An ordered store reaches it in one lookup, where the largest
# would cost a full walk, so this is where it reasonably differs from `dict`.
shows("scores.popitem()", scores.popitem())

# endregion The Ordinary Surface

# region Keys Are Typed

section("Keys are typed, which is what makes lookups cheap")

# A container names its key layout once, and every comparison after that skips type dispatch.
#
#     key=int       ↦  signed 64-bit
#     key='uint'    ↦  unsigned 64-bit, the only way to ask, since Python has no unsigned type
#     key=str       ↦  UTF-8, ordered bytewise
#     key=bytes     ↦  opaque, never decoded
counters = st.SortedMap(key=int)
shows("counters.key_type", counters.key_type)

# Values stay rich even so, and come back as the type they went in as.
counters[1] = 3.5
counters[2] = True
counters[3] = b"payload"
shows("a float value stays float", isinstance(counters[1], float))
shows("a bool value stays bool", counters[2] is True)
assert isinstance(counters[1], float) and counters[2] is True

# Keys are stricter than `dict`, deliberately, and the reason differs for each.
#
#     counters[1.0]     ⚡  ordering would depend on a total order over NaN
#     counters[True]    ⚡  bool subclasses int, so it would alias two key spaces
#     counters['one']   ⚡  a container holds one layout, never a mixture
for rejected in (1.0, True, "one"):
    try:
        counters[rejected] = 0
        raise AssertionError(f"{rejected!r} should not be a valid key")
    except TypeError as error:
        shows(f"counters[{rejected!r}] = 0", f"⚡ {str(error).split(';')[0]}")

section("Values may be arbitrary objects, if you ask")

# The mode is what makes the scalar guarantee true, so it is named rather than inferred.
#
#     value='scalar'    ↦  int, float, bool, str, bytes; the GIL is released around the store
#     value='object'    ↦  anything at all; the GIL is held, because a refcount is at stake
documents = st.SortedMap(key=str, value="object")
payload = {"nested": [1, 2, {"deep": True}]}
documents["doc"] = payload

shows("documents.value_mode", documents.value_mode)
shows("documents['doc']", documents["doc"])
shows("the same object, not a copy", documents["doc"] is payload)
assert documents["doc"] is payload

try:
    counters[4] = {"nested": True}
    raise AssertionError("a scalar container should refuse an object")
except TypeError:
    shows("a scalar container refuses it", "⚡ pass value='object' to store one")

# endregion Keys Are Typed

# region Sets

section("Sets, with the algebra you expect")

languages = st.SortedSet(key=str)
languages.update(["python", "rust", "c++"])
languages.add("python")  # Idempotent

shows("list(languages)", list(languages))
shows("intersection({'rust', 'go'})", sorted(languages.intersection({"rust", "go"})))
shows("difference({'rust'})", sorted(languages.difference({"rust"})))
shows("isdisjoint({'go', 'java'})", languages.isdisjoint({"go", "java"}))
shows("subset of a bigger set", languages <= {"c++", "python", "rust", "go"})
assert set(languages.intersection({"rust", "go"})) == {"rust"}
assert languages <= {"c++", "python", "rust", "go"}

# endregion Sets

# region Losing an Update, and Not

section("Two writers, one key: the update that would be lost")

# Without a watch, a group is atomic but not serializable, and the loss is silent.
#
#     A reads 100                       ·
#     B writes 500                      ✓
#     A writes 90                       ✓  computed from 100, so B's write is gone
#     ──────────────────────────────────────────────────
#     the value is 90                   ✗  and nothing reported a problem
#
# A watch turns that into a refusal, which a retry loop can act on.
#
#     A reads 100, watches 'alice'      ·
#     B writes 500                      ✓
#     A stages                          ⚡  ConflictError, nothing applied
#     A retries, reads 500              ·
#     A writes 490                      ✓  computed from what is actually there
#     ──────────────────────────────────────────────────
#     the value is 490                  ✓  no update was lost
accounts = st.SortedMap(key=str)
accounts["alice"] = 100

attempts = 0
interfered = False
while True:
    attempts += 1
    try:
        group = st.atomic(accounts)
        (view,) = group.begin()
        view.watch("alice")
        view["alice"] = accounts["alice"] - 10

        if not interfered:  # Someone else writes the watched key, exactly once
            accounts["alice"] = 500
            interfered = True

        group.stage()
        group.commit()
        break
    except st.ConflictError:
        continue  # Nothing was applied, so retrying is safe

shows("attempts", attempts)
shows("accounts['alice']", accounts["alice"])
assert attempts == 2 and accounts["alice"] == 490

section("The two phases, when the decision depends on staging")

# `with` is exactly begin, the body, stage, commit. Splitting them apart is for when what you do
# next depends on what staging reported.
#
#     begin()                           ·
#     view['widget'] = 0                ·
#     stage()                           ·  validated and reserved, still invisible
#     ──────────────────────────────────────────────────
#     rollback()                        ✗  nobody ever saw it, so nothing to undo publicly
#     commit() instead                  ✓  would have published it
inventory = st.SortedMap(key=str)
inventory["widget"] = 5

group = st.atomic(inventory)
(view,) = group.begin()
view["widget"] = 0
group.stage()
shows("after stage, outside sees", inventory["widget"])
assert inventory["widget"] == 5

group.rollback()
shows("after rollback", inventory["widget"])
assert inventory["widget"] == 5

with inventory.transaction() as (solo,):  # One container opens a group over itself
    solo["gadget"] = 3
shows("inventory['gadget']", inventory["gadget"])
assert inventory["gadget"] == 3

# endregion Losing an Update, and Not

print("\nevery assertion held")
