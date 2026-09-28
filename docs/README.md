# NAN-itself Documentation

NAN-itself is a local-first autonomous agent framework. These documents are
organized by who you are: find the path below that matches what you are here
to do, and follow the reading order inside it.

## Who are you?

### You want to run the framework

You cloned the repository and want a working agent on your machine. Start
with [for-users/](for-users/):

1. [Get started](for-users/get-started.md) — from a fresh clone to a first
   conversation.
2. [Configuration](for-users/configuration.md) — `settings.yaml` and
   per-module `config/modules/*.yaml`.

### You want to write plugins

You want to extend the agent with your own tools, modules or skills. Start
with [for-plugin-authors/](for-plugin-authors/):

1. [Add a tool](for-plugin-authors/add-a-tool.md) — an MCP or local Python
   tool provider.
2. [Add a module](for-plugin-authors/add-a-module.md) — an ambient or
   reactive module.
3. [Add a skill](for-plugin-authors/add-a-skill.md) — a skill package.
4. Reference: [tools](for-plugin-authors/tools.md),
   [modules](for-plugin-authors/modules.md),
   [skills](for-plugin-authors/skills.md).

### You want to modify the framework

You want to work on NAN-itself itself. Start with
[for-contributors/](for-contributors/):

1. [Develop](for-contributors/develop.md) — setup, tests, conventions and
   pull requests.
2. [Core principles](for-contributors/principles.md) — the load-bearing
   principles behind every design decision.
3. [Architecture](for-contributors/architecture.md) — how the system is
   shaped, and why.

## Document index

| Document | Contents |
|---|---|
| [for-users/get-started.md](for-users/get-started.md) | Quick start: install, configure, start, converse. |
| [for-users/configuration.md](for-users/configuration.md) | `settings.yaml` and per-module `config/modules/*.yaml`. |
| [for-plugin-authors/add-a-tool.md](for-plugin-authors/add-a-tool.md) | Add an MCP or local Python tool provider. |
| [for-plugin-authors/add-a-module.md](for-plugin-authors/add-a-module.md) | Add an ambient or reactive module. |
| [for-plugin-authors/add-a-skill.md](for-plugin-authors/add-a-skill.md) | Add a skill package. |
| [for-plugin-authors/tools.md](for-plugin-authors/tools.md) | The tool subsystem: provider kinds, discovery, reload, verbs. |
| [for-plugin-authors/modules.md](for-plugin-authors/modules.md) | The module subsystem: surface, lifecycle, DataSpace, channels. |
| [for-plugin-authors/skills.md](for-plugin-authors/skills.md) | The skill subsystem: format, discovery, resources. |
| [for-contributors/develop.md](for-contributors/develop.md) | Set up, test, and follow the developer conventions; open a PR. |
| [for-contributors/principles.md](for-contributors/principles.md) | The load-bearing design principles. |
| [for-contributors/architecture.md](for-contributors/architecture.md) | How the system is shaped, and why. |

For the project overview, install requirements and license, see the
[root README](../README.md). For the security model, see
[SECURITY.md](../SECURITY.md).
