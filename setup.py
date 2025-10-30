import os
import sys
import platform
from setuptools import setup, Extension
from typing import List, Tuple


def get_compile_args() -> Tuple[List[str], List[str]]:
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
    # Enforce Python 3.14+ requirement for sub-interpreter support
    if sys.version_info < (3, 14):
        print("ERROR: SmashTable requires Python 3.14 or later for sub-interpreter support", file=sys.stderr)
        print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}", file=sys.stderr)
        sys.exit(1)

    compile_args, link_args = get_compile_args()

    # Check if building for Python 3.14t (free-threading)
    is_free_threaded = hasattr(sys, "abiflags") and "t" in sys.abiflags
    if is_free_threaded:
        print("✓ Building for Python 3.14t with FREE-THREADING support (GIL-free)")
    else:
        print("✓ Building for Python 3.14+ with sub-interpreter support")

    ext_modules = [
        Extension(
            "smashtable",
            sources=["python/smashtable.cpp"],
            include_dirs=["include"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language="c++",
        ),
    ]

    setup(
        name="smashtable",
        version="0.1.0",
        ext_modules=ext_modules,
        description="Concurrent Thread-Safe Atomic & Consistent Collections for Parallel Python",
    )


if __name__ == "__main__":
    main()
