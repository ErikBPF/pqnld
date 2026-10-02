#!/usr/bin/env bash
# Transfer only declarative task source; every Rust process runs inside Apollo.
set -euo pipefail
cd "$(dirname "$0")/../.."
pm=/mnt/data/ai/validation/decision-index-2026-10-01/pm.sh
workspace=/work/pqnld-reliability-20261002
set -- Makefile sidecar-rs/Cargo.toml sidecar-rs/Cargo.lock sidecar-rs/src sidecar-rs/models
if [[ -d sidecar-rs/tests ]]; then set -- "$@" sidecar-rs/tests; fi
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cf - "$@" |
  ssh apollo "$pm exec -i dibuild python3 -c 'import hashlib,io,os,sys,tarfile; p=\"$workspace\"; assert open(p+\"/.task-id\").read()==\"pqnld-reliability-20261002-7342\\n\"; data=sys.stdin.buffer.read(); t=tarfile.open(fileobj=io.BytesIO(data)); assert all(m.name==\"Makefile\" or m.name.startswith(\"sidecar-rs/\") for m in t.getmembers()); assert all(m.isfile() or m.isdir() for m in t.getmembers()); t.extractall(p,filter=\"data\"); [os.utime(p+\"/\"+m.name,None) for m in t.getmembers() if m.isfile()]; print(\"snapshot sha256:\",hashlib.sha256(data).hexdigest())'"
ssh apollo "$pm exec -w $workspace dibuild env CARGO_HOME=/work/cargo RUSTUP_HOME=/work/rustup PATH=/work/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin CARGO_TARGET_DIR=$workspace/target CARGO_NET_OFFLINE=true CARGO_TERM_COLOR=never RUSTC_WRAPPER= make test"
