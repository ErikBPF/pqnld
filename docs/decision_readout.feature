# Behavior contract for the decision readout sidecar (S0/S1/S2).
# Automated: bound to test_decision_sidecar.py (stdlib unittest), which was
# observed failing (ModuleNotFoundError) before decision_sidecar.py existed.
# Scenarios marked @unautomated are contract only; no step binding exists yet.
# Seed: homelab/docs/plans/apollo-decision-model/one-pager.md, SHA256
# recorded on the plan; seed quote: "add a small lora layer to this model or a
# sidecar to also serve it as a decision model".
# Plan: homelab/docs/plans/apollo-decision-model/one-pager.md.
# Wire format: apolinario/decision-index @ 52a698928a9ae5bdf16b75687c903871db29c6e5.
# Readout: options are rendered with single-letter labels and ONE
# /v1/chat/completions request per question asks for exactly one option letter;
# the sidecar reads the letter distribution at the answer slot, ignores
# non-letter tokens, maps letters back to keys, and softmaxes. More than 26
# options has no single-token label and falls back to the chunked echo readout
# (echo the context, then echo each context+key and sum the key tokens). The
# remaining ceiling is context capacity: a question whose options do not fit is
# refused (422), never truncated.
# Readout route is per-model: a models/<name>.json descriptor names the readout
# (lettered|echo|auto), the chat template, the thinking toggle, the letters and
# the temperature. An "auto" descriptor probes the model at startup and keeps
# lettered only when a single option letter is the top token for a fixture.
# Cache is a bounded LRU keyed by model + descriptor + question.
# DR-07 receipt: Apollo results/s2b-kit-results.jsonl, produced by run-s2-kit.sh.
# DR-15 receipt: Apollo results/parallel.json and results/parallel-numseq8-*.json,
# produced by bench_parallel.py. --max-num-seqs 2 / --each 4: parallelism 5.59 of 8,
# 2.58 req/s. --max-num-seqs 8 / --each 4: parallelism 6.37, 5.74 req/s. / --each 8
# (16 in flight): parallelism 10.49 of 16, 5.19 req/s. 0 failures in every run:
# decisions and chat both complete while the engine batches them. DR-15 stays
# @unautomated: the measurement is manual, no failing-first step binding exists.

Feature: Decision readout sidecar over a served Apollo model
  A Decision Index request is answered with a typed distribution over exactly the
  supplied options, read from the model's own answer-slot probabilities.
  A question the engine cannot answer is refused, never guessed.

  @DR-01
  Scenario: A choice question returns a valid and correct distribution
    Given a state that names a colour and a two-option colour question
    When the readout sidecar answers the question
    Then the response model is echoed from the request
    And the answer type is choice
    And the probability keys equal the question criteria keys
    And the probabilities sum to one
    And the most likely option is the named colour

  @DR-02
  Scenario: A noul question returns a true-probability
    Given a state that names a colour and a yes-or-no question
    When the readout sidecar answers the question
    Then the answer type is noul
    And the noul value is a probability between zero and one

  @DR-03
  Scenario: A multi-token option key is ranked by its letter
    Given a question whose option key tokenizes to more than one token
    When the readout sidecar answers the question
    Then the multi-token option can be the most likely answer
    And its probability is the letter distribution read at the answer slot
    And the probabilities still sum to one

  @DR-04
  Scenario: A capacity rejection passes through unchanged
    Given a state that exceeds the model's maximum context length
    When the readout sidecar answers the question
    Then the sidecar returns 422
    And the model's capacity message is preserved in the error

  @DR-05
  Scenario: A question type the engine cannot answer is refused
    Given a question of an unsupported type
    When the readout sidecar answers the question
    Then the sidecar returns 422
    And no answer is invented for that question

  @DR-06 @unautomated @S3
  Scenario: The readout is scored on the frozen suite
    Given the pinned decision-index-suite rows file
    When the reproduction kit runs its http engine against the sidecar
    Then every response passes the kit's validate
    And the Decision Index score and calibration error are recorded
    And no score is reported without the pinned suite hash

  @DR-07 @unautomated @S2
  Scenario: The authoritative kit accepts the sidecar's wire responses
    Given the reproduction kit's http engine pointed at the sidecar
    When the kit runs over a suite-shaped sample
    Then every response passes the kit's validate
    And each recorded status is ok

  @DR-08
  Scenario: A non-letter token at the answer slot is ignored
    Given the model ranks a non-letter token first
    When the readout sidecar answers the question
    Then the non-letter token is skipped
    And the probabilities still sum to one

  @DR-09
  Scenario: More options than letters falls back to the echo readout
    Given a question with more options than there are single-token letters
    When the readout sidecar answers the question
    Then every option is scored by the chunked echo readout
    And the probabilities still sum to one

  @DR-10 @S1
  Scenario: A model descriptor selects the readout route
    Given a model descriptor that selects the lettered readout
    When the readout sidecar answers a question
    Then the answer is read from the answer-slot letter distribution
    And the upstream is asked for one chat token with logprobs
    And a descriptor that selects the echo readout scores by echoing keys

  @DR-11 @S1
  Scenario: A model that does not rank an option letter first falls back to echo
    Given a model descriptor that requests the automatic readout
    And a model that ranks a non-letter token first at the answer slot of the probe
    When the readout sidecar resolves its readout route
    Then it selects the echo readout
    And a question it answers is scored by the echo readout

  @DR-12 @S1
  Scenario: A declared echo readout without a chat template is refused
    Given a model descriptor that selects the echo readout
    And that descriptor has no chat template
    When the readout sidecar is constructed
    Then it refuses with an unsupported error
    And no request is served

  @DR-13 @S0
  Scenario: The decision cache is a bounded LRU
    Given a warm cache entry for a question
    When the same question is asked again
    Then the answer is served without a second upstream request
    And the least recently used entry is evicted beyond the cache bound

  @DR-14 @S2
  Scenario: Concurrent decisions are not serialized by the sidecar
    Given an upstream that records the number of requests in flight
    When several decisions are requested at once
    Then every response is a valid distribution
    And the upstream observed more than one request in flight

  @DR-15 @unautomated @S2
  Scenario: One engine serves chat and decisions concurrently
    Given the full engine profile bound to the Apollo tailnet
    When a mix of chat completions and decisions is issued concurrently
    Then both kinds complete without error
    And decisions are not serialized behind chat
    And no port or firewall rule is broadened to reach it

  @DR-16 @S0
  Scenario: The decisions endpoint is /v1/decide and the kit alias still answers
    Given the readout sidecar serves POST /v1/decide
    When a decision is posted to /v1/decide
    Then the sidecar returns a valid typed answer
    And a decision posted to the frozen kit's /v1/systemone alias also returns a valid typed answer