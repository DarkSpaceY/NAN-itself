# Configuration

The single source is `config/nan.yaml`, loaded once into the
process-wide `nan.config.settings` object — every component reads
that one object; nothing is threaded through constructor arguments.
Nothing else is read: no environment variables, no dotenv files, no
CLI overrides. Paths are not configuration — they live in the same
object (`settings.paths`, see [Paths](#paths)).

## llm

| Key | Default | Meaning |
|-----|---------|---------|
| `provider` | `"openai"` | Protocol: `"openai"` or `"anthropic"` (anything else fails at load) |
| `base_url` | `""` | Endpoint URL; empty = provider default |
| `api_key` | `""` | Empty = SDK default behavior (env fallback inside the SDK) |
| `model` | `""` | Model name sent to the endpoint |
| `timeout` | `120.0` | Per-request timeout (seconds) |
| `max_retries` | `2` | SDK-level retry count |

Example -- a local 27B model behind llama.cpp:

    llm:
      provider: openai            # OpenAI-compatible protocol
      base_url: http://127.0.0.1:8080/v1
      model: qwen-27b
      api_key: none

## agent

| Key | Default | Meaning |
|-----|---------|---------|
| `autonomous_interval` | `0.0` | `>0`: the root agent self-rotates with this pacing (seconds); `<=0`: event-driven (turns happen when input arrives) |
| `history_char_limit` | `100000` | Rendered-history size cap; exceeding it clears the whole history (a reset, not a rewrite) |
| `shell_timeout` | `30.0` | `shell` verb execution timeout; the model cannot change it |
| `shell_output_limit` | `20000` | `shell` verb output truncation size (head + tail kept, middle elided) |
| `backoff` | `[1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0]` | Turn-failure retry ladder (seconds); a failed turn is retried whole after these delays |
| `max_depth` | `3` | Spawn depth limit; `-1` = unlimited |
| `max_children` | `5` | Per-agent children limit; `-1` = unlimited |

## sandbox

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Wrap `shell` commands in the OS-level sandbox |
| `required` | `false` | `true`: refuse to execute when no sandbox tool exists (instead of running degraded) |

## injection

The registry the external prompt-contributor services talk to.

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `127.0.0.1` | Registry listen address |
| `port` | `8901` | Registry port |
| `timeout` | `2.0` | Per-service, per-turn collection timeout |
| `heartbeat_ttl` | `15.0` | A service silent longer than this is evicted from the live set |

## gateway

| Key | Default | Meaning |
|-----|---------|---------|
| `host` | `127.0.0.1` | Listen address |
| `port` | `8765` | HTTP + WebSocket port |

## autostart

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Run both boot scripts once: `builtin/autostart.sh` (user-owned, outside the sandbox) and `workspace/autostart.sh` (agent-writable, inside the sandbox) |

Both scripts are launched once at boot, fire-and-forget: output
goes to the log file, and a failure in either script does not
block the core.

## Paths

Paths are not configuration. They are code defaults on
`nan.config.Paths`, carried by the same `settings` object every
component reads, so the whole process resolves one location:

| Constant | Value | Meaning |
|-----|---------|---------|
| `paths.workspace` | `<repo>/workspace` | The agents' writable territory |
| `paths.builtin` | `<repo>/builtin` | Read-only resource root (tools/skills as plain files) |
| `paths.package` | `nan/` | The package root (read-only to agents) |
| `paths.persona` | `nan/persona.md` | The global persona, re-read every turn |
| `paths.frontend` | `<repo>/frontend/dist` | The built desktop served by the gateway |
| `paths.autostart` | `builtin/autostart.sh` | Boot script, owned by the user (unsandboxed) |
| `paths.agent_autostart` | `workspace/autostart.sh` | Boot script, writable by agents (sandboxed) |
| `paths.config` | `<repo>/config/nan.yaml` | This file |
