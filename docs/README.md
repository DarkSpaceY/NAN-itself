# NAN-itself Documentation

This is the documentation for NAN-itself, a local-first autonomous agent
framework; it is organized with the Diataxis framework, so every document
serves exactly one of four purposes and you can go straight to the one
that matches what you need.

## The four quadrants

Every document answers one kind of question. To place a new topic, ask
two questions: are you *studying* or *working*, and do you need
*practical steps* or *theoretical knowledge*?

```
                     PRACTICAL
                        │
         Tutorials      │      How-to guides
        (learning)      │      (task-oriented)
                        │
   ACQUISITION ─────────┼───────── APPLICATION
                        │
        Explanation     │      Reference
      (understanding)   │      (information)
                        │
                    THEORETICAL
```

| Quadrant | Answers | Documents |
|---|---|---|
| [Tutorials](tutorials/) | "Can you teach me to...?" | [first-run.md](tutorials/first-run.md) |
| [How-to guides](how-to/) | "How do I...?" | [develop.md](how-to/develop.md), [add-a-tool.md](how-to/add-a-tool.md), [add-a-module.md](how-to/add-a-module.md), [add-a-skill.md](how-to/add-a-skill.md) |
| [Reference](reference/) | "What is...?" | [modules.md](reference/modules.md), [tools.md](reference/tools.md), [skills.md](reference/skills.md), [configuration.md](reference/configuration.md) |
| [Explanation](explanation/) | "Why...?" | [principles.md](explanation/principles.md), [architecture.md](explanation/architecture.md) |

## Start here

Follow this path the first time:

1. **Learn by doing** — [Your first run](tutorials/first-run.md) takes a
   fresh clone to a working agent.
2. **Do a task** — the [how-to guides](how-to/) cover the day-to-day
   developer workflow.
3. **Look it up** — the [reference](reference/) describes the machinery
   precisely.

If you want the why behind the system, read
[principles.md](explanation/principles.md) and then
[architecture.md](explanation/architecture.md).

## Document index

| Document | Purpose |
|---|---|
| [tutorials/first-run.md](tutorials/first-run.md) | End-to-end first run: install, configure, start, converse. |
| [how-to/develop.md](how-to/develop.md) | Set up, test, and follow the developer conventions; open a PR. |
| [how-to/add-a-tool.md](how-to/add-a-tool.md) | Add an MCP or local Python tool provider. |
| [how-to/add-a-module.md](how-to/add-a-module.md) | Add an ambient or reactive module. |
| [how-to/add-a-skill.md](how-to/add-a-skill.md) | Add a skill package. |
| [reference/modules.md](reference/modules.md) | The module subsystem: surface, lifecycle, DataSpace, channels. |
| [reference/tools.md](reference/tools.md) | The tool subsystem: provider kinds, discovery, reload, verbs. |
| [reference/skills.md](reference/skills.md) | The skill subsystem: format, discovery, resources. |
| [reference/configuration.md](reference/configuration.md) | `settings.yaml` and per-module `config/modules/*.yaml`. |
| [explanation/principles.md](explanation/principles.md) | The load-bearing design principles. |
| [explanation/architecture.md](explanation/architecture.md) | How the system is shaped, and why. |
| [WRITING.md](WRITING.md) | The documentation standard and terminology glossary. |

For the project overview, install requirements and license, see the
[root README](../README.md). For the security model, see
[SECURITY.md](../SECURITY.md).

## Writing documentation

Before adding or changing a document, read [WRITING.md](WRITING.md) — it
defines which quadrant a document belongs to, the writing rules, and the
project terminology glossary.
