#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/smashtable_docs_<version>.tar.gz
set -euo pipefail
version=$(cat VERSION)
name=smashtable_docs_$version out=build/smashtable_docs_$version
# Rustdoc's JSON is unstable, so it runs on a pinned beta with the bootstrap switch.
toolchain=${SMASHTABLE_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" build/docs-scratch && mkdir -p "$out"/{doxygen,headers,python,rust} build/docs-scratch/stub/smashtable

rm -rf build/xml && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
git ls-files ':(glob)include/smashtable/*.hpp' | xargs cp --parents -t "$out/headers"
# griffe cannot load a lone stub, so it reads one dressed as a package.
cp python/smashtable.pyi build/docs-scratch/stub/smashtable/__init__.pyi
griffe dump smashtable --search build/docs-scratch/stub --no-inspection --docstyle google --full \
    --output "$out/python/smashtable.json"
cp build/docs-scratch/stub/smashtable/__init__.pyi "$out/python/"
RUSTC_BOOTSTRAP=1 cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/smashtable.json" "$out/rust/"
# griffe, rustdoc and Doxygen's body locations hold absolute paths, which would leak the build machine's directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/python/smashtable.json" "$out/rust/smashtable.json" "$out"/doxygen/*.xml
printf '{"project":"smashtable","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
