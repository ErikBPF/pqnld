.PHONY: help install test selftest test-all build image run

help:
	@echo "make install   - editable install with dev extras and the tester"
	@echo "make test      - unit tests (no GPU)"
	@echo "make selftest  - benchmark self-tests (no GPU)"
	@echo "make test-all  - test + selftest"
	@echo "make build     - build sdist + wheel into dist/"
	@echo "make image     - build the container image"
	@echo "make run       - run the sidecar (pqnld)"

install:
	python -m pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -v

selftest:
	python benchmarks/bench_parallel.py --selftest
	python benchmarks/bench_mixed.py --selftest

test-all: test selftest

build:
	python -m build

image:
	docker build -t pqnld:dev .

run:
	python -m pqnld
