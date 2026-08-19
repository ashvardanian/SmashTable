"""The stub is the only description a type checker sees, so it is checked against the runtime.

A stub cannot be verified by running it, and it drifts silently the moment a method is added or
renamed in the C layer. These tests compare the two surfaces directly, so a missing entry is a
red test rather than a wrong completion in somebody's editor.
"""

import ast
import pathlib

import pytest

import smashtable as st

# region Reading the stub

STUB_PATH = pathlib.Path(__file__).resolve().parent.parent / "python" / "smashtable.pyi"


@pytest.fixture(scope="module")
def stub() -> ast.Module:
    """The parsed stub, read from the source tree rather than from site-packages."""
    assert STUB_PATH.is_file(), f"the stub moved away from {STUB_PATH}"
    return ast.parse(STUB_PATH.read_text())


def declared_classes(stub: ast.Module) -> dict[str, set[str]]:
    """Every class the stub declares, mapped to the names it declares inside."""
    return {
        node.name: {member.name for member in node.body if isinstance(member, ast.FunctionDef)}
        for node in stub.body
        if isinstance(node, ast.ClassDef)
    }


def runtime_members(subject: type) -> set[str]:
    """Every name a class carries that `object` does not, which is the surface worth stating."""
    inherited = set(dir(object)) | {"__module__"}
    return {name for name in dir(subject) if name not in inherited}


# endregion Reading the stub

# region Surface Parity

# The lazy views are returned by `keys`, `values` and `items` rather than exported, so they are
# reached through an instance instead of through the module.
VIEW_NAMES = ["KeysView", "ValuesView", "ItemsView"]
CONTAINER_NAMES = ["SortedMap", "SortedSet", "HashMap", "HashSet"]
TRANSACTION_NAMES = ["Transaction", "Participant"]


def view_types() -> dict[str, type]:
    container = st.SortedMap(key="int")
    container[1] = 1
    return {
        "KeysView": type(container.keys()),
        "ValuesView": type(container.values()),
        "ItemsView": type(container.items()),
    }


def test_the_stub_declares_everything_the_module_exports(stub):
    """A name reachable as `st.<name>` that the stub omits is invisible to a checker."""
    exported = {name for name in dir(st) if not name.startswith("_")}
    declared = {node.name for node in stub.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert not exported - declared, "exported at runtime but absent from the stub"


def test_the_stub_declares_nothing_the_module_lacks(stub):
    """A name only the stub knows about is a promise the module does not keep."""
    exported = {name for name in dir(st) if not name.startswith("_")}
    declared = {node.name for node in stub.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert not declared - exported - set(VIEW_NAMES), "declared in the stub but missing at runtime"


@pytest.mark.parametrize("class_name", CONTAINER_NAMES + TRANSACTION_NAMES + VIEW_NAMES)
def test_every_class_states_its_whole_surface(stub, class_name):
    """Each method and property a class carries has to appear in the stub, and nothing else."""
    subject = view_types()[class_name] if class_name in VIEW_NAMES else getattr(st, class_name)
    declared = declared_classes(stub)[class_name]
    actual = runtime_members(subject)

    # `__setitem__` on a set exists only to raise, so the stub leaves it out on purpose - a checker
    # rejecting the assignment outright beats a signature that always fails.
    if class_name == "SortedSet":
        actual -= {"__setitem__"}

    assert not actual - declared, f"{class_name} carries names the stub never mentions"
    assert not declared - actual - {"__init__"}, f"{class_name} promises names it does not carry"


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
