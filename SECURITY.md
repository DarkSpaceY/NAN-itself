# Security

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/DarkSpaceY/NAN-itself/security/advisories/new)
rather than a public issue. Please include a reproduction and the
affected paths. You can expect an initial response within a week.

## Security model

The framework is designed to run locally, on the user's own machine:

- The only inbound listener is the HTTP gateway, bound to
  `127.0.0.1` by default (this is verified by the test suite).
- All repo-relative paths resolve through a single anchoring module;
  the process never depends on its working directory and writes no
  state outside the repository directories.

## Trust boundaries

- **MCP servers and skill scripts are trusted subprocesses.** They
  inherit the user's environment and are unrestricted by design —
  review a tool config or skill before adding it.
- **Workspace sources are code.** Files under `workspace/` hot-reload
  into the running process with the same privileges as NAN-itself
  itself; treat them like you treat your own scripts.
- **The LLM provider receives conversation content.** Whatever the
  agent observes (task text, module projections, tool results) is sent
  to the configured LLM endpoint.
