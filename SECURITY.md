# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue.

- Open a private advisory: <https://github.com/ErikBPF/pqnld/security/advisories/new>
- Or email the maintainer. The contact address is on the
  <https://github.com/ErikBPF> profile.

Include a description, a reproduction, and the impact you believe it has. You
can expect an acknowledgement within a few days.

## Deployment notes

pqnld is a thin proxy in front of a model endpoint and holds no credentials of
its own. Two things are still worth stating explicitly, because they are easy to
get wrong:

- **Bind it deliberately.** The default host is `127.0.0.1`. If you expose it on
  a network, put authentication or a firewall in front of it — an unauthenticated
  decision endpoint is an unauthenticated proxy to your model.
- **The chat shim is a shim.** `/v1/chat/completions` answers any Decision Index
  JSON user message as a decision and otherwise returns a notice. It is not a
  general chat gateway; use your real model endpoint for chat.

## Scope

In scope: the code in this repository. Out of scope: the model you point it at,
your serving stack (vLLM, LiteLLM, a gateway), and your deployment's network
policy.
