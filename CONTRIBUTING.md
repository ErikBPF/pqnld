# Contributing to pqnld

`pqnld` turns an OpenAI-compatible endpoint into a typed decision engine by
reading the model's answer-slot logprobs. The core is small and deliberate: a
stdlib-only readout server, an optional tester, and two benchmark scripts. This
file is the human contract for working here.

Design premise, for context: a decision is just a `max_tokens=1` + `logprobs`
chat request, so one engine serves chat and decisions. Changes that move the
project away from that premise are design changes and need discussion first.

## What lands easily

- Bug reports with a minimal reproducer: the failing request plus expected vs.
  observed response.
- Security findings (see [`SECURITY.md`](SECURITY.md)).
- New model descriptors for models you actually serve.
- Documentation fixes: typos, stale links, incorrect claims.
- Corrections to benchmark numbers that do not reproduce on your hardware.

## What needs more discussion

- New core dependencies. `src/pqnld/decision.py` must stay stdlib-only;
  third-party packages belong in the `tester` extra. Adding one to the core
  requires a stated justification.
- New behavior beyond typed decisions (free text, reasoning, tool calls). The
  narrow scope is intentional; extends it only with a measured reason.
- A second readout path. The lettered path plus the echo fallback cover the
  models we know; a new one needs a reproducible failing request.
- Changes to the `/v1/decide` wire format or the `/v1/systemone` alias, which are
  pinned to the reference Decision Index kit.

## Environment

Nothing here needs a GPU or a real model; the tests run the HTTP surface against
a stub vLLM.

```sh
git clone https://github.com/ErikBPF/pqnld
cd pqnld
python -m venv .venv && . .venv/bin/activate
make install        # editable install with dev + tester extras
make test-all       # unit tests + benchmark self-tests, no GPU
```

Make targets: `test`, `selftest`, `test-all`, `build` (sdist + wheel), `image`
(container).

## The rules

Keep these true; they are what makes the project reviewable.

1. **The core is stdlib-only.** `src/pqnld/decision.py` imports the standard
   library and nothing else. Enforced by CI.
2. **Behavior ships with its contract.** A readout, caching, or wire change gets
   a scenario in [`docs/decision_readout.feature`](docs/decision_readout.feature)
   and a test that fails before the implementation. A `.feature` without bound,
   executing steps is a draft, not a test.
3. **Tests run without a GPU and without network access,** against a stub
   endpoint on an ephemeral port.
4. **One concern per pull request.** Small diffs review faster and revert
   cleaner.
5. **Refuse instead of guess.** A question pqnld cannot answer is refused
   (`422`), never truncated or approximated.
6. **No output outside the supplied options.** The model only ever scores the
   caller's options; any change that lets it emit beyond that set is a
   regression.
7. **Descriptors state what you verified.** Ship the chat template exactly and
   record the engine revision and flags you tested against.

## Process

1. Open an issue first for anything larger than a typo.
2. Branch naming: `<kind>/<short-slug>` (for example `fix/cache-key-readout`,
   `docs/contributing-guide`).
3. Commits follow Conventional Commits, scoped when useful:

   ```
   feat(readout): fall back to echo for >26 options
   fix(cache): key the LRU by readout mode as well as model
   docs(wire): describe the 422 capacity rejection
   ```

4. Behavior changes require a test that goes red if the change is reverted.
5. Run the full battery before requesting review: `make test-all`.
6. Human review: the reviewer reads the diff in `tuicr`; treat `issue` comments
   as blocking and answer every other comment.

## Adding a model descriptor

1. Add `src/pqnld/models/<name>.json` (copy `qwen38-27b-nvfp4.json` and edit).
2. Point the server at it: `pqnld --model <served-name> --descriptor <name>`.
3. Verify live with the tester (`pqnld-tester`) or a `curl` to `/v1/decide`.
4. Note in the pull request the exact engine revision and flags you tested.

Field reference, lookup order, and the `auto` probe contract:
[`docs/descriptor.md`](docs/descriptor.md).

## Behavior before implementation

1. Describe the behavior as a concrete scenario in the `.feature`.
2. Watch it fail for the right reason.
3. Implement the smallest change that makes it pass.
4. Re-read the scenario against the diff before running the tooling. The
   question the tests cannot answer is whether the implementation drifted from
   what the scenario declared.

## Releasing

There is no release cadence yet; `main` is the truth. Maintainers tag `vX.Y.Z`;
the release workflow builds the container and pushes it to GitHub Container
Registry. `pyproject.toml` carries the version and must match the tag.

## Code of conduct

Participation is covered by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
