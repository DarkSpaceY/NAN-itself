# Documentation standard

This is the writing standard for NAN-itself's documentation. It adapts
the Diataxis framework and a set of developer-docs best practices to this
project. It is meant to be followed, not admired: keep it practical.

All documentation is written in English, like the code, comments and
identifiers.

## 1. Pick the right quadrant

Every document serves exactly one of four purposes. To place a new
document, ask two questions in order:

1. **Studying or working?** Studying → the left column. Working → the
   right column.
2. **Practical steps or theoretical knowledge?** Practical → the top row.
   Theoretical → the bottom row.

|  | Practical steps | Theoretical knowledge |
|---|---|---|
| **Studying** | **Tutorial** — teach by doing | **Explanation** — clarify the why |
| **Working** | **How-to guide** — solve a task | **Reference** — describe the machinery |

Where each quadrant lives in this repository:

| Quadrant | Folder | Voice |
|---|---|---|
| Tutorial | `docs/tutorials/` | "We", encouraging, a lesson |
| How-to guide | `docs/how-to/` | Direct, imperative, second person |
| Reference | `docs/reference/` | Neutral, descriptive, austere |
| Explanation | `docs/explanation/` | Discursive, contextual |

## 2. Rules

### Content architecture

- **One purpose per document.** Never mix quadrants in one file.
- **A tutorial is not a how-to guide.** Tutorials teach by doing for a
  beginner; how-to guides solve a task for someone who already knows the
  basics. Do not label one as the other.
- **Reference describes only.** Record the machinery neutrally — no
  instructions, no rationale, no opinions.
- **Explanation has no steps.** Provide context and reasoning; never
  include a step-by-step procedure.
- **Document outcomes, not features.** Describe what the reader can
  achieve, not merely that a component exists.
- **Show, don't tell.** Support abstract statements with a concrete
  example, a command, or a diagram.

### Writing style

- **Active voice, second person.** Address the reader as "you" in how-to
  and tutorials; use the present tense to describe.
- **Code examples must work.** Every snippet is copy-pasteable and
  correct; validate it against the real source before publishing.
- **One term per concept.** Use the glossary below everywhere; never
  alternate between synonyms for the same thing.
- **Global readability.** No idioms, cultural references, or jokes that
  do not translate; spell out an acronym on first use.
- **Minimize admonitions.** At most a few callouts per page — if
  everything is a warning, nothing is.
- **Tone matches the type.** Encouraging in tutorials, direct in how-to
  guides, neutral in reference, conversational in explanation.

### Information architecture

- **Organize by type, not by component.** Structure docs by quadrant
  (tutorials, how-to, reference, explanation), not by the internal module
  a topic belongs to.
- **Two levels of navigation, maximum.** Do not nest deeper than two
  folder levels under `docs/`.
- **Cross-link, no dead ends.** Every document links to its prerequisites
  and to the next step; if a fact lives elsewhere, link to it instead of
  restating it.
- **Each fact has one home.** Reference and explanation must not
  duplicate; keep one authoritative copy and link to it from everywhere
  else.

### Governance

- **Docs are part of done.** A change is not complete until its
  documentation is written or updated. Contract changes require a doc
  update.
- **Freshness and ownership.** Update a document in the same change that
  makes it stale. A moved file must not leave a duplicate behind.

## 3. Terminology glossary

One term per concept. The identifier column is the name used in the code;
use it verbatim in prose and code. The "do not call it" column lists
rejected synonyms.

| Term | Identifier | Definition | Do not call it |
|---|---|---|---|
| Agent core | `Agent` (`agent/core.py`) | The turn-driven cognitive loop; it holds the last `Turn` and runs the engine. | bot, assistant, runner |
| Engine | `StepEngine` (`agent/engine.py`) | The single-round executor that builds the snapshot, calls the LLM and renders the turn. | runtime, orchestrator, scheduler |
| Turn | `Turn` (`agent/model.py`) | The sole record of one LLM round: identity, observation inputs, the round's flow, and its outputs. | round, iteration, message |
| Verb | `list_*` / `show_*` / `invoke_*` (`agent/verbs.py`) | A cognitive action the model can take; every interface face uses the same list/show/invoke triple. | command, action, function |
| Tool provider | `Provider` / `ProviderRuntime` (`tools/`) | One source file exposing a group of tools; the runtime reconciles and reloads all providers. | plugin, connector, tool manager |
| MCP provider | kind `"mcp"` (`tools/spec.py`) | A tool provider backed by a stdio MCP server described by a YAML file. | remote tool, external tool |
| Local provider | kind `"local"` (`tools/spec.py`) | A tool provider backed by an in-process Python class. | in-process tool, Python tool |
| ToolSet | `ToolSet` (`tools/local.py`) | The base class for a local provider; holds its `@tool` methods. | LocalToolProvider, Tool, Provider |
| `@tool` | `@tool` (`tools/local.py`) | The decorator that marks a `ToolSet` method as an exposed tool; its schema derives from the type annotations. | @tool_method, @action |
| Tool | `mcp.types.Tool` | One callable a provider exposes, addressed as `provider/tool`. | function, capability |
| Module | `Module` (`modules/model.py`) | A long-lived background service that senses, computes and acts at its own rhythm and contributes ambient context. | plugin, service, sensor, daemon |
| Facade | `Facade` (`modules/runtime.py`) | The module runtime supervisor: discovery, hot reload, dependency graph, lifecycle, channels. | ModuleManager, registry |
| Channel | `channels` / `ChannelSpec` (`modules/model.py`) | A model-writable downlink endpoint a module opts into; the model feeds it, the module consumes at its own tick. | output, queue, command |
| DataSpace | `DataSpace` / `DataSpaceReader` (`modules/model.py`) | A module's published state; the owner is the only writer, readers get detached deep-copied snapshots. | state store, blackboard, bus |
| Skill | `SKILL.md` directory (`skills/`) | A capability package: a directory with `SKILL.md` (YAML frontmatter) plus optional resources. | plugin, prompt, tool |
| Gateway | `Gateway` (`gateway.py`) | The single inbound HTTP/WebSocket listener, bound to loopback by default, serving the API and the web UI. | server, backend, API |
| builtin | `builtin/` | The sources shipped with the repository. | core, default, system |
| workspace | `workspace/` | The user-editable root, scanned with the same hot-reload semantics as `builtin/`. | overrides, user dir, local dir |
| Persona | `workspace/persona.md` | The user-editable prompt identity prepended to the agent's context. | system prompt, character |
| Ambient context | `Module.ask()` | The per-turn projection a module contributes at turn start; a cheap read of already-computed state. | perception, telemetry, sensing |
| Hot reload | reload transaction (`tools/runtime.py`, `modules/reload.py`) | Replacing a live source: the replacement starts unregistered and is committed only once fully connected. | refresh, restart, rescan |
