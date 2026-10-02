@reliability
Feature: Worker limits bound sidecar scoring globally
  # Accepted Q-PQNLD-3: Global bound. Direct engine traffic remains outside scope.
  # Rust bindings and actual RED/GREEN evidence must be recorded before claiming
  # these scenarios have run. Cache hits are not fresh-engine determinism evidence.

  Scenario: Independent clients share the worker budget
    Given the sidecar is configured with 1 worker
    And two clients submit uncached decisions at the same time
    When the sidecar scores their questions
    Then at most 1 sidecar scoring operation is active at a time
    And both clients receive complete responses

  Scenario: Sequential execution preserves question order
    Given the sidecar is configured with 1 worker
    And a request contains several questions in stored order
    When the sidecar scores that request
    Then its questions are scored in their stored order

  Scenario: Initial readout probing respects the same budget
    Given the sidecar is configured with 1 worker
    And readout mode has not yet been determined
    When two clients request decisions at the same time
    Then sidecar scoring probes and question scoring do not exceed the worker budget

  Scenario: A failed concurrent request does not leave detached scoring
    Given a request has several concurrently scheduled questions
    And an engine scoring operation fails
    When the sidecar refuses that decision
    Then its remaining scoring tasks are cancelled and drained
    And worker permits become available to subsequent requests

  Scenario: Direct engine chat is not governed by sidecar workers
    Given ordinary chat is sent directly to the model endpoint
    When the sidecar scores a decision
    Then no engine-wide deterministic guarantee is made by the sidecar worker limit
