# Security

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/DarkSpaceY/NAN-itself/security/advisories/new).
Please include a reproduction and the affected paths. You can expect
an initial response within a week.

## Trust model

NAN-itself runs locally, on the user's own machine. It trusts the
machine, like the AI-agent baseline (Claude Code, Codex): the
security model covers remote parties, and local processes run
with the user's authority.

## Network listeners

Both listeners bind loopback by default; `host` is user-settable
and inbound requests are additionally Host/Origin-guarded.

**Gateway** (browser-facing). An entry guard runs at the HTTP and
WebSocket handshakes:

- `Host` must name a loopback host, on the gateway's port when
  the header carries one. A DNS-rebinding request carries the
  attacker's domain in `Host` and is rejected.
- `Origin`, when present, must be loopback on that same port.
  Native clients (curl, ws tools) send no `Origin` and pass;
  browsers attach one, and page scripts have no way to set it,
  which is what makes this an authentication boundary.

An iframe with an opaque origin sends `Origin: null`, which the
guard rejects, and its WebSocket handshake is rejected the same
way. A service panel talks to its own backend, and that backend
declares the panel.

**Injection registry** (service-facing). The same guard applies to
`127.0.0.1:8901`. Registration, heartbeat and turn delivery
authenticate by loopback alone; local process trust is the
boundary, consistent with the trust model above.

**Outbound destinations.** Service endpoints and panel URLs must
be loopback (`127.0.0.0/8`, `::1`, `localhost`):

- a service `endpoint` outside loopback is rejected at
  registration, so turn payloads stay on the machine;
- a panel `url` outside loopback is rejected, which keeps the
  desktop on local pages;
- a panel `icon` must be a local path or a loopback URL, which
  keeps the rail from fetching a remote asset.

The check is on the literal URL, with no DNS resolution, so a
loopback name stays loopback even if its DNS record points
elsewhere.

## Process boundaries

**Execution sandbox.** Shell verbs run under an OS-level sandbox
(macOS `sandbox-exec` SBPL, Linux bubblewrap): writes confined to
the workspace and temp locations; the nan package (persona),
`config/` and builtin resources read-only. Reads and network access
stay open. When the sandbox tool is present, every shell command runs
inside it; without one, `sandbox.required: true` refuses to start,
and the default (`false`) runs commands unwrapped with a one-time
warning.

Before wrapping, a configurable regex list (`sandbox.deny`) rejects
known destructive command shapes: root deletes, raw disk writes,
`mkfs`, download-to-shell pipes, and similar. The refusal and its
reason go back to the model, and the command does not run. The list
catches obvious mistakes; the OS sandbox remains the limit on what
a command can touch.

Services and `builtin/autostart.sh` run with user authority,
outside the sandbox. `workspace/autostart.sh` runs inside the same
sandbox as shell commands, and agents may write it to set up their
own boot services, so its writes stay confined.

**Agent authority limits.** `shell_timeout` and
`shell_output_limit` come from user configuration; tools and
skills are plain files reached through the sandboxed shell. The
persona file is core config the agent cannot rewrite.

**LLM endpoint.** The configured endpoint receives the assembled
conversation, including tool/skill output the agent observed.
Point NAN-itself at a local model (`llm.base_url`) when that
matters.

## Known residual risks

- Shell commands have network access: the sandbox confines what
  they write, not where they connect.
- A model-generated service/tool file may reference paths outside
  the workspace; the sandbox confines execution, but file *writes*
  through non-shell channels are not separately fenced.