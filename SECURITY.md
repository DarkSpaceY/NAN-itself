# Security

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/DarkSpaceY/NAN-itself/security/advisories/new)
rather than a public issue. Please include a reproduction and the
affected paths. You can expect an initial response within a week.

## Trust model

NAN-itself runs locally, on the user's own machine. It trusts the
machine, like the AI-agent baseline (Claude Code, Codex): the
security model defends against remote parties, not against
malicious local processes.

## Network listeners

Both listeners bind loopback by default; `host` is user-settable
and inbound requests are additionally Host/Origin-guarded.

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

An iframe without `allow-same-origin` has an opaque origin: its
requests carry `Origin: null`, which the guard rejects, and its
WebSocket handshake is rejected the same way. A service panel
therefore cannot reach the gateway from the browser; it talks to
its own backend, and that backend declares the panel.

**Injection registry** (service-facing). The same guard applies to
`127.0.0.1:8901`. Registration, heartbeat and turn delivery are
open to loopback, with no token: local process trust is the
boundary, consistent with the trust model above.

**Outbound destinations.** Every URL the core calls or hands to
the desktop must resolve textually to loopback (`127.0.0.0/8`,
`::1`, `localhost`):

- a service `endpoint` outside loopback is rejected at
  registration, so no turn payload can leave the machine;
- a panel `url` outside loopback is rejected, so the desktop
  never loads an internet page;
- a panel `icon` must be a local path or a loopback URL, so the
  rail cannot leak a request to the internet.

The check is on the literal URL, with no DNS resolution, so a
loopback name cannot be rebound to a public address.

## Process boundaries

**Execution sandbox.** Shell verbs run under an OS-level sandbox
(macOS `sandbox-exec` SBPL, Linux bubblewrap): writes confined to
the workspace and temp dirs; the nan package (persona, config) and
builtin resources read-only. Sandboxing is always on when a tool
exists; without one, `sandbox.required: true` refuses to start and
the default (`false`) degrades to a loud one-time warning.

Services and `builtin/autostart.sh` run with user authority,
outside the sandbox. `workspace/autostart.sh` is the opposite:
agents may write it to set up their own boot services, and it runs
inside the same sandbox as their shell commands, so its writes
stay confined.

**Agent authority limits.** `shell_timeout` and
`shell_output_limit` come from user configuration; tools and
skills are plain files reached through the sandboxed shell, with
no registration path that bypasses it; the persona file is core
config the agent cannot rewrite.

**LLM endpoint.** The configured endpoint receives the assembled
conversation, including tool/skill output the agent observed.
Point NAN-itself at a local model (`llm.base_url`) when that
matters.

## Known residual risks

- A model-generated service/tool file may reference paths outside
  the workspace; the sandbox confines execution, but file *writes*
  through non-shell channels are not separately fenced.