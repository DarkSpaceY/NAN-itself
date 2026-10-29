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
| [Security](../SECURITY.md) | Trust model, entry guard, sandbox |
| [for-service-authors/injection-service.md](for-service-authors/injection-service.md) | Write a prompt-contributor service |
| [for-contributors/architecture.md](for-contributors/architecture.md) | How the core is built |
| [for-contributors/principles.md](for-contributors/principles.md) | The design rules behind it |
| [for-contributors/frontend-protocol.md](for-contributors/frontend-protocol.md) | The event stream contract for the frontend |
| [for-contributors/frontend-architecture.md](for-contributors/frontend-architecture.md) | How the frontend renderer is laid out |
| [for-contributors/develop.md](for-contributors/develop.md) | Test, lint, type-check |

## Repository layout

    nan/          the core runtime package
    lib/          nan-service-kit, the shared library services run on
    builtin/      user-owned resources: skills, services/, autostart.sh
    workspace/    agent-writable territory: services/, autostart.sh
    config/       nan.yaml, the configuration file
    frontend/     the desktop UI, built to frontend/dist
    scripts/      install.sh
    tests/        the core pytest suite
    docs/         this documentation
    design/       design notes and the backlog
    infraend/     notes for a separate inference backend
    paper_search/ research experiments (idea generation, agent memory)
    legacy/       archived prototypes
    dist/         build output (`uv build`)
