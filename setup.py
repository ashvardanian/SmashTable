import os
import pathlib
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class build_ext_with_stubs(build_ext):
    """Ships the stub and the typing marker beside the extension they describe.

    Hooked onto `build_ext` rather than `build_py`, because with no pure modules to build the
    latter never runs.
    """

    typing_files = ["smashtable.pyi", "py.typed"]

    def run(self) -> None:
        super().run()
        self.mkpath(self.build_lib)
        for name in self.typing_files:
            self.copy_file(str(pathlib.Path("python") / name), str(pathlib.Path(self.build_lib) / name))


def get_compile_args() -> tuple[list[str], list[str]]:
    """Get platform-specific compile and link arguments"""
    compile_args = []
    link_args = []

    if sys.platform == "linux" or sys.platform.startswith("freebsd"):
        compile_args = [
            "-std=c++20",
            "-O3",
            "-fdiagnostics-color=always",
            "-Wno-unknown-pragmas",
            "-fPIC",
            "-fvisibility=hidden",
            "-pthread",
        ]
        link_args = [
            "-fPIC",
            "-pthread",
        ]

    elif sys.platform == "darwin":
        min_macos = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "11.0")
        compile_args = [
            "-std=c++20",
            "-O3",
            "-fcolor-diagnostics",
            "-Wno-unknown-pragmas",
            "-fPIC",
            "-fvisibility=hidden",
            f"-mmacosx-version-min={min_macos}",
        ]
        link_args = ["-fPIC"]

    elif sys.platform == "win32":
        compile_args = [
            "/std:c++20",
            "/O2",
            "/W3",
            "/wd4365",  # signed/unsigned mismatch
            "/wd4820",  # padding added
        ]
        link_args = []

    return compile_args, link_args


def main():
    compile_args, link_args = get_compile_args()

    # One translation unit per domain, all compiled into the single `smashtable` extension. The
    # directory is never on sys.path, so it is a source layout rather than a Python package.
    sources = sorted(str(path) for path in pathlib.Path("python").glob("*.cpp"))

    # Headers are listed as dependencies because setuptools rebuilds a translation unit only when
    # its own `.cpp` is newer than the object file. Without this, editing a header leaves a stale
    # extension in place and every test afterwards reports on code that is no longer there.
    headers = sorted(str(path) for path in pathlib.Path("python").glob("*.hpp"))
    headers += sorted(str(path) for path in pathlib.Path("include/smashtable").glob("*.hpp"))

    ext_modules = [
        Extension(
            "smashtable",
            sources=sources,
            include_dirs=["include", "python"],
            depends=headers,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language="c++",
        ),
    ]

    setup(
        name="smashtable",
        version="0.3.1",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_with_stubs},
        description="Safer associative containers with DBMS-like transactions in Python",
    )


if __name__ == "__main__":
    main()
