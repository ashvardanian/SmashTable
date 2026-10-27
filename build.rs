//! Compiles nothing, since SmashTable is header-only: it reports the tree as `DEP_SMASHTABLE_SOURCE`, so a
//! dependent's CMake adds the checkout Cargo resolved as a subdirectory.
//!
//! File: build.rs
//! Author: Ash Vardanian
//! Date: October 14, 2026

fn main() {
    let manifest =
        std::env::var("CARGO_MANIFEST_DIR").expect("Cargo always sets CARGO_MANIFEST_DIR");
    println!("cargo:source={manifest}");
}
