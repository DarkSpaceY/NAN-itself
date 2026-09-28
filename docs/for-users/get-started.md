# Get started

This quick start takes you from a fresh clone to a working local agent:
install the framework, point it at an LLM, start it, and have a first
conversation.

You need a Unix-like shell with `git`, `uv` and Node.js (`npm`).

## 1. Clone the repository

```bash
git clone https://github.com/DarkSpaceY/NAN-itself.git
cd NAN-itself
```

## 2. Install the framework

NAN-itself is managed with [uv](https://docs.astral.sh/uv/). `uv sync`
installs the framework and the dev group; `--all-extras` adds what the
builtin plugins need — the perception stack behind the audio, voice and
vision modules:

```bash
uv sync --all-extras
```

## 3. Point it at an LLM

NAN-itself talks only to an LLM endpoint you configure — everything else
runs locally. Open `config/settings.yaml` and set the `llm` block:

```yaml
llm:
  provider: openai          # client protocol; openai-compatible endpoints work
  api_key: xxx              # your key (local servers accept any placeholder)
  model: qwen3.5:4b-mlx     # the model identifier
  base_url: http://127.0.0.1:11434/v1
  timeout: 600
  max_retries: 2
```

Two common setups:

- **A local server** (Ollama, LM Studio, vLLM): keep `provider: openai`,
  set `base_url` to the server's `/v1` endpoint, set `model` to the model
  you have pulled, and any placeholder `api_key` works.
- **A hosted OpenAI-compatible provider**: set `base_url` to the
  provider's endpoint, `api_key` to your real key, and `model` to the
  model name.

If a field does not match a model you actually have, the first turn will
fail — check `model` and `base_url` here before moving on.

## 4. Build the web UI

The gateway serves the web UI as static files from `frontend/app/dist`,
so build it once:

```bash
cd frontend/app
npm install
npm run build
cd ../..
```

If Node.js is not available to you, `frontend/app/README.md` describes
running the UI as a Vite dev server instead.

## 5. Start the agent

```bash
uv run nan-itself
```

The process starts the tool providers, the modules and skills, then the
agent loop and the HTTP gateway. Log lines about providers loading and
modules starting should appear, and NAN-itself opens your browser at
`http://127.0.0.1:8765/`. If it does not, open that URL yourself.

On the first run the perception modules (audio, vision, voice) try to
provision their model weights. A module whose weights are missing comes
up loudly failed and keeps retrying with backoff; this is expected and
does not stop the agent. The modules needed for this quick start —
`system` and `inbox` — need no weights.

## 6. Have your first conversation

In the web UI composer, type a message and send it:

> Hello! Who are you?

The message enters through the gateway and lands in the `inbox` module;
the agent reads it on its next turn and streams a reply back to the UI.
The reply appears in the conversation.

## 7. Try ambient context

Now ask something the agent could only answer from ambient perception:

> What time is it for me right now, and which app is in the foreground?

The `system` module samples your machine every few seconds and publishes
facts — local time, idle time, the foreground app, CPU and memory. At the
start of every turn the agent asks each running module for its ambient
context (`ask()`), so the `system` module's facts are part of what the
model sees. A correct answer here means a module is contributing ambient
context to the turn, not just echoing your message.

## Next steps

- [Develop NAN-itself](../for-contributors/develop.md) — set up, test and
  follow the developer conventions.
- [Add a tool](../for-plugin-authors/add-a-tool.md)
- [Add a module](../for-plugin-authors/add-a-module.md)
- [Add a skill](../for-plugin-authors/add-a-skill.md)

For the design behind what you just ran, see
[../for-contributors/architecture.md](../for-contributors/architecture.md)
and
[../for-contributors/principles.md](../for-contributors/principles.md).
