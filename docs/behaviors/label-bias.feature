@label-bias @engine-validation
Feature: Label invariance and engine readout validation
  # Human basis: the LocalLLaMA thread on pqnld (2026-10-13 seed) and
  # docs/proposals in the homelab coordination repo:
  #   docs/proposals/2026-10-13-pqnld-label-bias-engine-validation.md
  #   docs/plans/2026-10-13-pqnld-label-bias-llamacpp-strata.md
  # Thresholds, the counterbalanced design and the unresolved band are fixed in
  # the PL artifact before measurement. This feature is a behavior contract, not
  # a standalone Gherkin runner; Rust bindings and observed RED/GREEN receipts
  # must be recorded before claiming any scenario has run.

  Scenario: Seeded label binding is deterministic and independent of option order
    Given a choice question with k options and k <= 26
    And the sidecar is started with a label permutation seed
    When the same question is submitted twice with the option order unchanged
    Then both submissions bind the same label to the same option
    And the label bound to an option does not follow that option's position

  Scenario: Default readout is unchanged by the permutation control
    Given the sidecar is started without a label permutation seed
    When any supported choice question is rendered
    Then labels are bound in option order exactly as before the control existed

  Scenario: An unbiased engine reports no material label effect
    Given a restart-separated noise floor is measured for the arm
    And the engine's scores do not depend on the label bound to an option
    When the probe runs the counterbalanced fixture set
    Then the reported per-option spread is at most the support band
    And the top-1 flip rate stays within the pre-registered bound
    And no known-answer fixture inverts under label permutation
    And the measured noise floor is reported next to the result

  Scenario: A noise-limited measurement yields no verdict
    Given the restart-separated noise floor exceeds the pre-registered band
    When the probe summarises the arm
    Then the result is reported as noise-limited
    And no supported or rejected verdict is issued

  Scenario: A planted label bias is detected
    Given an engine whose scores are biased toward one label by a known amount
    When the probe runs the counterbalanced fixture set
    Then the probe reports a material label effect
    And it names the biased label in the per-label offset report

  Scenario: A decision that inverts a known-correct answer is reported
    Given a fixture whose expected answer is known
    When label binding moves the decision away from that expected answer
    Then the inversion is counted in the result
    And the value is reported, not smoothed into an aggregate

  Scenario: Missing option labels are refused, never renormalized
    Given an engine response that omits at least one option label
    When a client requests that decision
    Then the request is refused
    And no probability distribution over a subset of options is returned

  Scenario: llama.cpp exact-ID requests are refused honestly
    Given the sidecar is configured with the llama.cpp engine kind
    And the requested descriptor opts into exact token ID scoring
    When a client requests a decision
    Then the request is refused as unsupported by that engine
    And the reported engine capabilities state that exact ids are unavailable

  Scenario: llama.cpp cannot silently approximate more than 26 options
    Given the sidecar is configured with the llama.cpp engine kind
    When a client requests a choice question with 27 or more options
    Then the request is refused
    And no label outside the configured single-token alphabet is scored

  Scenario: Sequential repeats of one question are identical
    Given the sidecar runs with its sequential worker default
    When the same question is submitted repeatedly against the same running engine
    Then the returned scores are identical across repeats
    And the repeated-run result is reported as an observation about that engine's
      run, not as a universal reproducibility claim
    And it is not reported as the noise floor for label-bias measurement

  Scenario: Label and position effects are separated
    Given a counterbalanced fixture in which each label occupies each position once
    When the probe runs the fixture with option order held fixed
    Then a per-label mean offset is reported
    And a per-position mean offset is reported for the same run
    And each is reported with the permutation count it was computed from

  Scenario: Large top-k requests neither truncate silently nor inflate a decision
    Given the llama.cpp engine kind and a requested top-k above the server's coverage
    When a client requests a decision
    Then either every option label is present in the response
    Or the request is refused
    And a decision is never returned from a truncated score set

  Scenario: The generic read widens itself until every label is visible
    Given an engine without exact token id scoring
    And a descriptor that declares no top-k floor
    When a question's option labels fall outside the first read width
    Then the readout retries at the next documented width
    And it names each escalation while it does
    And it returns a complete distribution once every label is visible

  Scenario: A refusal names what was tried and what would help
    Given every documented read width failed to see all option labels
    When a client requests that decision
    Then the refusal names the missing labels
    And it names the widths that were tried
    And it names the descriptor fields that could change the outcome

  Scenario: The llama.cpp lettered readout uses the chat surface by default
    Given a llama.cpp engine and a descriptor without a lettered surface
    When a client requests a lettered decision
    Then the request is sent to the chat completions surface with the letter
      system prompt
    And a descriptor that declares the native surface keeps the raw prompt path

  Scenario: Two engines agree on the decision for the same prompt
    Given identical fixtures, prompts and pinned model files on two engines
    When the probe runs both arms sequentially
    Then the choices agree on at least the pre-registered share of questions
    And any score divergence is reported with its magnitude and the model hashes
    And no bit-identical claim is made from a decision agreement
