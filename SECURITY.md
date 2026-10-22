# Security

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/DarkSpaceY/NAN-itself/security/advisories/new)
rather than a public issue. Please include a reproduction and the
affected paths. You can expect an initial response within a week.

## Trust model

NAN runs locally, on the user's own machine. Like the CLI-agent
baseline (Claude Code, Codex), it trusts the machine: the security
model defends against remote parties, not against malicious local
processes.

## Network listeners

Both listeners bind loopback only (verified by the test suite).

**Gateway** (browser-facing). An entry guard runs at the HTTP and
WebSocket handshakes:

- `Host` must be a loopback form of the gateway's own port
  (`127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>`). A
  DNS-rebinding request carries the attacker's domain in `Host`
  and is rejected.
- `Origin`, when present, must be the gateway's own loopback
  origin. Native clients (curl, ws tools) send no `Origin` and
  pass; browsers always send one and cannot forge it, which is
  what makes this an authentication boundary.

**Injection registry** (service-facing). No token: registration,
heartbeat and turn delivery are open to loopback. Local process
trust is the boundary, consistent with the trust model above.

## Process boundaries

- **Execution sandbox.** Shell verbs run under an OS-level sandbox
  (macOS `sandbox-exec` SBPL, Linux bubblewrap): writes confined
  to the workspace and temp dirs; the nan package (persona,
  config) and builtin resources read-only. Modes: enforcing /
  degraded (sandbox tooling missing, loud warning) / required
  (`sandbox.required: true` -- missing tooling refuses to
  execute). Services and the autostart script run with user
  authority, outside the sandbox.
- **Agent authority limits.** `shell_timeout` and
  `shell_output_limit` come from user configuration; tools and
  skills are plain files reached through the sandboxed shell; the
  persona file is core config the agent cannot rewrite.
- **LLM endpoint.** The configured endpoint receives the
  assembled conversation, including tool/skill output the agent
  observed. Point NAN at a local model
  (`llm.providers`) when that matters.

## Known residual risks

- A model-generated service/tool file may reference paths outside
  the workspace; the sandbox confines execution, but file *writes*
  through non-shell channels are not separately fenced.
- Smaller models may emit malformed tool-call JSON, which
  currently fails the turn loudly (backoff + retry) rather than
  being repaired.
