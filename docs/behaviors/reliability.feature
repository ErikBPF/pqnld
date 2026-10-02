@reliability
Feature: Fail-closed typed decisions
  # Human basis: pqnld-one-pager.md, reliability priority, and "lets do improvements".
  # Named Rust bindings and observed RED/GREEN live in reliability-evidence/receipt.md.
  # This feature is a behavior contract, not a standalone Gherkin test runner.

  Scenario: Malformed choices do not reach the engine
    Given a choice question with missing or empty criteria
    When a client requests a decision
    Then the request is refused without an engine scoring call
    And the service does not panic

  Scenario: Choice cardinality follows the wire contract
    Given a choice question with fewer than 2 or more than 255 criteria
    When a client requests a decision
    Then the request is refused before scoring

  Scenario: Structured instructions and descriptions remain supported
    Given a choice question with JSON object instructions
    And its criteria descriptions contain structured JSON values
    When a client requests a decision
    Then the existing JSON-aware renderer preserves those supplied values
    And a complete supported decision is returned through HTTP or MCP

  Scenario: Instructions remain required without imposing a string-only schema
    Given a question without an instructions field
    When a client requests a decision
    Then the request is refused before scoring

  Scenario: Missing echo evidence is not a uniform distribution
    Given an engine response without usable continuation log probabilities
    When a client requests an echo-scored decision
    Then the request is refused
    And no successful probability distribution is returned

  Scenario: Incomplete echo batches are refused
    Given an engine returns a different number of scores than requested options
    When a client requests an echo-scored decision
    Then the request is refused instead of omitting or fabricating option scores

  Scenario: Invalid temperatures cannot configure scoring
    Given a non-finite or non-positive scoring temperature
    When the sidecar starts
    Then configuration is refused

  Scenario: Supported decisions retain their response contract
    Given a supported question and complete finite engine evidence
    When a client requests a decision
    Then choice answers contain exactly the supplied criteria keys
    And choice probabilities are finite and sum to 1 within the documented tolerance
    And noul answers are numeric probabilities between 0 and 1
