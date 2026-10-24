# NAN-itself documentation

NAN-itself is a local-first agent core: one asyncio task per agent, handles
for multi-agent coordination, external processes for prompt
injection, and the shell as the universal tool surface.

## Reading order

> Diagrams use [Mermaid](https://mermaid.js.org/); GitHub renders
> them inline. VS Code needs the Markdown Preview Mermaid Support
> extension.

| Path | Audience |
|------|----------|
| [for-users/get-started.md](for-users/get-started.md) | Run the core for the first time |
| [for-users/configuration.md](for-users/configuration.md) | Every key in `config/nan.yaml` |
| [for-users/security.md](for-users/security.md) | Trust model, entry guard, sandbox |
| [for-service-authors/injection-service.md](for-service-authors/injection-service.md) | Write a prompt-contributor service |
| [for-contributors/architecture.md](for-contributors/architecture.md) | How the core is built |
| [for-contributors/principles.md](for-contributors/principles.md) | The design rules behind it |
| [for-contributors/frontend-protocol.md](for-contributors/frontend-protocol.md) | The event stream contract for the frontend |
| [for-contributors/develop.md](for-contributors/develop.md) | Test, lint, type-check |

## Repository layout

    nan/          the core package (the only runtime code)
    builtin/      read-only resources the agent reaches through shell
    config/       nan.yaml -- the single configuration source
    frontend/     the gateway's web UI (built to frontend/dist)
    workspace/    the agents' writable territory (services live in
                  workspace/services/, one uv project each)
    tests/        pytest suite
    docs/         this documentation
