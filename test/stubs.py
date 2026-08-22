"""The stub is the only description a type checker sees, so it is checked against the runtime.

A stub cannot be verified by running it, and it drifts silently the moment a method is added or
renamed in the C layer. These tests compare the two surfaces directly, so a missing entry is a
red test rather than a wrong completion in somebody's editor.
"""

import ast
import pathlib

import pytest

import smashtable as st

from .base import container_class_names

# region Reading the stub

STUB_PATH = pathlib.Path(__file__).resolve().parent.parent / "python" / "smashtable.pyi"


@pytest.fixture(scope="module")
def stub() -> ast.Module:
    """The parsed stub, read from the source tree rather than from site-packages."""
    assert STUB_PATH.is_file(), f"the stub moved away from {STUB_PATH}"
    return ast.parse(STUB_PATH.read_text())


def declared_names(stub: ast.Module) -> set[str]:
    """Every public class and function the stub declares at module level."""
    return {
        node.name
        for node in stub.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_")
    }


def member_name(member: ast.stmt) -> str | None:
    """The name one statement in a class body declares, or None for a docstring or a comment."""
    if isinstance(member, ast.FunctionDef):
        return member.name
    if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
        return member.target.id
    return None


def declared_classes(stub: ast.Module) -> dict[str, set[str]]:
    """Every class the stub declares, mapped to the names it declares inside."""
    return {
        node.name: {name for name in map(member_name, node.body) if name is not None}
        for node in stub.body
        if isinstance(node, ast.ClassDef)
    }


# What every heap type carries whatever its methods are, plus the constructor the stub spells as
# `__init__`, so stating any of them would say nothing about this module.
UNIVERSAL_WAIVERS = frozenset({"__doc__", "__module__", "__new__"})

# One `tp_richcompare` slot fills all six comparison names, so a map answering NotImplemented for
# the ordering four still carries them; `__setitem__` on a set exists only to raise.
CLASS_WAIVERS = {
    "SortedMap": frozenset({"__lt__", "__le__", "__gt__", "__ge__"}),
    "SortedSet": frozenset({"__setitem__"}),
}


def runtime_members(subject: type, class_name: str) -> set[str]:
    """Every name the type itself defines, read from the type dictionaries rather than from `dir`.

    `dir` folds in everything inherited from `object`, and subtracting those by name would hide the
    dunders the C layer genuinely overrides - `__eq__`, `__repr__`, `__hash__` and the ordering
    surface all share a name with an inherited one.
    """
    defined = {name for klass in subject.__mro__ if klass is not object for name in vars(klass)}
    return defined - UNIVERSAL_WAIVERS - CLASS_WAIVERS.get(class_name, frozenset())


# endregion Reading the stub

# region Surface Parity

# The lazy views are returned by `keys`, `values` and `items` rather than exported, so the stub
# names them privately and they are reached through an instance instead of through the module.
VIEW_NAMES = ["_KeysView", "_ValuesView", "_ItemsView"]
TRANSACTION_NAMES = ["Transaction", "Participant"]


def view_types() -> dict[str, type]:
    container = st.SortedMap(key="int")
    container[1] = 1
    return {
        "_KeysView": type(container.keys()),
        "_ValuesView": type(container.values()),
        "_ItemsView": type(container.items()),
    }


def test_the_stub_declares_everything_the_module_exports(stub):
    """A name reachable as `st.<name>` that the stub omits is invisible to a checker."""
    exported = {name for name in dir(st) if not name.startswith("_")}
    missing = exported - declared_names(stub)
    assert not missing, f"exported at runtime but absent from the stub: {sorted(missing)}"


def test_the_stub_declares_nothing_the_module_lacks(stub):
    """A name only the stub knows about is a promise the module does not keep."""
    exported = {name for name in dir(st) if not name.startswith("_")}
    invented = declared_names(stub) - exported
    assert not invented, f"declared in the stub but missing at runtime: {sorted(invented)}"


@pytest.mark.parametrize("class_name", [*container_class_names, *TRANSACTION_NAMES, *VIEW_NAMES])
def test_every_class_states_its_whole_surface(stub, class_name):
    """Each method and property a class carries has to appear in the stub, and nothing else."""
    subject = view_types()[class_name] if class_name in VIEW_NAMES else getattr(st, class_name)
    declared = declared_classes(stub)[class_name]
    actual = runtime_members(subject, class_name)

    assert not actual - declared, f"{class_name} carries {sorted(actual - declared)}, unmentioned by the stub"
    unkept = declared - actual - {"__init__"}
    assert not unkept, f"{class_name} promises {sorted(unkept)}, which it does not carry"


@pytest.mark.parametrize(
    "alias, keyword, reader",
    [
        ("_IsolationName", "isolation", "isolation"),
        ("_SharingName", "sharing", "sharing"),
        ("_KeyTypeName", "key", "key_type"),
        ("_ValueModeName", "value", "value_mode"),
    ],
)
def test_every_literal_spells_what_the_parser_accepts(stub, alias, keyword, reader):
    """A name the parser takes but the `Literal` omits reads as a type error on correct code."""
    spelled = next(
        {literal.value for literal in node.value.slice.elts}
        for node in stub.body
        if isinstance(node, ast.Assign) and node.targets[0].id == alias
    )
    for name in spelled:
        arguments = {"key": "int"} | {keyword: name}
        container = st.SortedMap(**arguments)
        assert getattr(container, reader) == name, f"{name} was accepted but reads back as something else"


def test_a_name_outside_the_literal_is_refused():
    """The other half of the pair: a spelling the stub excludes must not quietly work."""
    with pytest.raises(ValueError):
        st.SortedMap(key="int", isolation="strict-serializable")
    with pytest.raises(ValueError):
        st.SortedMap(key="int", isolation="monotonic")


# endregion Surface Parity
