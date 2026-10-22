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

## nan/ migration — current trust & security model

> Record of the security model as implemented in the `nan/` package
> (migration branch). Fold into the sections above once the migration
> replaces the old runtime; the MCP/skill trust boundary text above is
> backend-era and does not apply to `nan/`.

Trusted machine: like the CLI-agent baseline (Claude Code, Codex),
`nan/` trusts the user's own machine. The security model defends
against remote parties, not against malicious local processes.

Network listeners (both loopback-only):

- **Gateway** (browser-facing). Entry guard at the HTTP and
  WebSocket handshakes:
  - `Host` must be a loopback form of the gateway's own port
    (`127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>`), which
    rejects DNS-rebinding requests carrying the attacker's domain.
  - `Origin`, when present, must be the gateway's own loopback
    origin. Native clients send no Origin and pass; browsers always
    send one and cannot forge it.
- **Injection registry** (service-facing). No token: registration,
  heartbeat and turn delivery are open to loopback. Local process
  trust is the boundary, consistent with the trusted-machine model.

Process boundaries:

- **Execution sandbox.** Shell verbs run under an OS-level sandbox
  (macOS `sandbox-exec` SBPL, Linux bubblewrap): writes confined to
  the workspace and temp dirs; persona, config and package files
  read-only. Modes: enforcing / degraded (sandbox tooling missing,
  loud warning) / required (missing tooling refuses to execute).
  Services and the autostart script run with user authority, outside
  the sandbox.
- **Agent authority limits.** `shell_timeout` and
  `shell_output_limit` come from user configuration; tools and
  skills are plain files reached through the sandboxed shell, and
  the persona file is core config the agent cannot rewrite.
- **LLM provider.** As above: the configured endpoint receives the
  assembled conversation, including tool/skill output the agent
  observed.

Known residual risks (recorded, unresolved):

- A model-generated service/tool file may reference paths outside
  the workspace; the sandbox confines execution, but file *writes*
  through non-shell channels are not separately fenced.
- Sub-27B models may emit malformed tool-call JSON, which currently
  fails the turn loudly (backoff + retry) rather than being repaired.
