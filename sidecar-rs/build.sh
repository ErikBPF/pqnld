#!/usr/bin/env bash
# Build the Rust decision sidecar. Toolchain and target dir live under /work so
# they persist on the host mount across container restarts.
set -euo pipefail

export RUSTUP_HOME=/work/rustup
export CARGO_HOME=/work/cargo
export CARGO_TARGET_DIR=/work/cargo-target
export PATH="$CARGO_HOME/bin:$PATH"

if ! command -v cargo >/dev/null 2>&1; then
  echo "installing rustup (minimal stable)..."
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
fi

cd /work/pqnld-rs
echo "rustc: $(rustc --version)"
cargo build --release
ls -l "$CARGO_TARGET_DIR/release/pqnld-rs"
