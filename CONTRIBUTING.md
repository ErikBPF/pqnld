# Contributing to pqnld

Thanks for helping. This project is small on purpose: one stdlib-only core, one
optional tester, two benchmark scripts. Keep it that way.

## Development setup

```sh
git clone https://github.com/ErikBPF/pqnld
cd pqnld
python -m venv .venv && . .venv/bin/activate
make install        # editable install with dev + tester extras
make test-all       # unit tests + benchmark self-tests (no GPU needed)
```

Nothing here needs a GPU or a real model: the tests run the HTTP surface against
a stub vLLM.

## What to work on

- **Bugs and readout correctness** — the answer-slot mapping, cache keying,
  descriptor resolution, error paths.
- **New model descriptors** — a `src/pqnld/models/<name>.json` for a model you
  serve. Include the chat template exactly, and say which values you verified.
- **New readouts** — only if a model genuinely cannot use the lettered or echo
  path; explain the failure with a reproducible request.
- **Docs** — clearer explanations, worked examples, corrections.

## Ground rules

- The core (`src/pqnld/decision.py`) must stay **stdlib-only**. Third-party
  dependencies belong in the `tester` extra.
- Every behavior change comes with a test in `tests/`. Tests must run without a
  GPU and without network access.
- Keep changes vertical and small; one concern per pull request.

## Tests

```sh
make test       # python -m unittest discover -s tests -v
make selftest   # benchmark --selftest paths
```

If you change the readout, add or update a scenario in
[docs/decision_readout.feature](docs/decision_readout.feature) and bind it to a
test. A `.feature` scenario counts as automated only when a test actually
executes it.

## Adding a model descriptor

1. Add `src/pqnld/models/<name>.json` (copy `qwen38-27b-nvfp4.json` and edit).
2. Point the server at it: `pqnld --model <served-name> --descriptor <name>`.
3. Verify live with the tester (`pqnld-tester`) or a `curl` to `/v1/decide`.
4. Note in the pull request the exact engine revision/flags you tested.

## Commit style

Conventional Commits, scoped when useful:

```
feat(readout): fall back to echo for >26 options
fix(cache): key the LRU by readout mode as well as model
docs(wire): describe the 422 capacity rejection
```

## Pull requests

- Describe the observable behavior change and how you verified it.
- Include the command output for the tests you ran.
- Small diffs review faster than large ones.

## Releases

Maintainers tag `vX.Y.Z`; the release workflow builds the container and pushes
it to GitHub Container Registry. `pyproject.toml` carries the version and must
match the tag.

## Code of conduct

Participation is covered by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
