# Your first run

In this lesson we take NAN-itself from a fresh clone to a working local
agent: we install it, point it at an LLM, start it, open the web UI,
have a first conversation, and watch a module hand the agent ambient
context it could not have known otherwise. Every step is here — follow
along and you will end with a running agent on your machine.

We assume a Unix-like shell and that `git`, `uv` and Node.js (`npm`) are
available.

## 1. Clone the repository

```bash
git clone https://github.com/DarkSpaceY/NAN-itself.git
cd NAN-itself
```

## 2. Install the framework

NAN-itself is managed with [uv](https://docs.astral.sh/uv/). One command
installs the runtime and the dev group:

```bash
uv sync
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
agent loop and the HTTP gateway. You should see log lines about providers
loading and modules starting, and NAN-itself opens your browser at
`http://127.0.0.1:8765/`. If it does not, open that URL yourself.

On the first run the perception modules (audio, vision, voice) try to
provision their model weights. A module whose weights are missing comes
up loudly failed and keeps retrying with backoff; this is expected and
does not stop the agent. The modules we care about here — `system` and
`inbox` — need no weights.

## 6. Have your first conversation

In the web UI composer, type a message and send it:

> Hello! Who are you?

The message enters through the gateway and lands in the `inbox` module;
the agent reads it on its next turn and streams a reply back to the UI.
You should see the reply appear in the conversation.

## 7. Watch a module contribute context

Now ask something the agent could only answer from ambient perception:

> What time is it for me right now, and which app is in the foreground?

The `system` module samples your machine every few seconds and publishes
facts — local time, idle time, the foreground app, CPU and memory. At the
start of every turn the agent asks each running module for its ambient
context (`ask()`), so the `system` module's facts are part of what the
model sees. A correct answer here means a module is contributing ambient
context to the turn, not just echoing your message.

## What just happened

You now have NAN-itself running locally. Here is the shape of what you
saw:

- **One turn = one LLM round-trip.** Your message became an observation;
  the engine built the model's snapshot from the previous turn plus that
  observation, and recorded everything the round produced in a single
  `Turn` record.
- **Modules are background daemons.** `system` measured continuously and
  `ask()` returned a cheap projection of already-computed facts at turn
  start — the agent reads projections, it never runs the measurement
  loop.
- **Tools are hot-reloadable and bounded.** The tool providers were loaded
  from `builtin/tools/` and `workspace/tools/` at startup; editing a
  source file reloads it live, and every tool call is timeout-bounded.
- **Everything is local-first.** The only external call is the LLM
  endpoint you configured; the gateway itself is bound to loopback.

From here, head to the guides for the task you have in mind.

- [How to develop NAN-itself](../how-to/develop.md)
- [Add a tool](../how-to/add-a-tool.md)
- [Add a module](../how-to/add-a-module.md)
- [Add a skill](../how-to/add-a-skill.md)

For the design behind what you just ran, see
[../explanation/architecture.md](../explanation/architecture.md) and
[../explanation/principles.md](../explanation/principles.md).
