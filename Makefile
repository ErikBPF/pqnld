.PHONY: help build test release run mcp

help:
	@echo "make build   - debug build of the Rust sidecar"
	@echo "make test    - unit tests (no GPU, no network)"
	@echo "make release - optimized static binary (x86_64-unknown-linux-musl)"
	@echo "make run     - run the HTTP sidecar (needs an OpenAI-compatible engine)"
	@echo "make mcp     - run the MCP stdio server"

MANIFEST = sidecar-rs/Cargo.toml

build:
	cargo build --manifest-path $(MANIFEST)

test:
	cargo test --manifest-path $(MANIFEST)

release:
	cargo build --release --target x86_64-unknown-linux-musl --manifest-path $(MANIFEST)

run:
	cargo run --release --manifest-path $(MANIFEST) -- --vllm-url http://127.0.0.1:11542 --models-dir sidecar-rs/models

mcp:
	cargo run --release --manifest-path $(MANIFEST) -- --mcp --vllm-url http://127.0.0.1:11542 --models-dir sidecar-rs/models
