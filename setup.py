import os
import pathlib
import sys

from setuptools import Extension, setup


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
    sources = sorted(str(path) for path in pathlib.Path("python/smashtable").glob("*.cpp"))

    # Headers are listed as dependencies because setuptools rebuilds a translation unit only when
    # its own `.cpp` is newer than the object file. Without this, editing a header leaves a stale
    # extension in place and every test afterwards reports on code that is no longer there.
    headers = sorted(str(path) for path in pathlib.Path("python/smashtable").glob("*.hpp"))
    headers += sorted(str(path) for path in pathlib.Path("include/smashtable").glob("*.hpp"))

    ext_modules = [
        Extension(
            "smashtable",
            sources=sources,
            include_dirs=["include", "python/smashtable"],
            depends=headers,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language="c++",
        ),
    ]

    setup(
        name="smashtable",
        version="0.1.0",
        ext_modules=ext_modules,
        description="Safer associative containers with DBMS-like transactions in Python",
    )


if __name__ == "__main__":
    main()
