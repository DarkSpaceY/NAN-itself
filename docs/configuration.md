# Configuration

## settings.yaml

The main configuration file is `config/settings.yaml`. All fields have
sensible defaults; only the `llm` block usually needs editing.

```yaml
llm:
  provider: openai          # client protocol (openai-compatible endpoints work)
  api_key: xxx              # your key (local servers accept any placeholder)
  model: qwen3.5:4b-mlx     # model identifier
  base_url: http://127.0.0.1:11434/v1
  timeout: 600
  max_retries: 2

gateway:
  host: 127.0.0.1           # keep on loopback unless you know what you expose
  port: 8765

agent:
  max_subagent_depth: 3

runtime:
  inbox:
    max_size: 256
  turn:
    grace: 5.0
  retry:
    backoff: [1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0]

modules:
  retry_interval: 1.0
  scan_interval: 1.0

providers:
  scan_interval: 1.0        # hot-reload scan period (seconds)
  tool_timeout: 300.0       # per tool call, seconds

events:
  history_limit: 500
  subscriber_queue_size: 2000
  input_dedup_cache_size: 256
```

## Environment variables

| Variable | Purpose |
|---|---|
| `NAN_DATA_DIR` | Override the repo-anchored `data/` directory. |
| `NAN_MODELS_DIR` | Override the repo-anchored `models/` directory. |
| `NAN_VISION_VLM` | Set `0` to disable the vision VLM backend. |
| `NAN_VISION_VLM_DIR` | Override the VLM weights directory (`models/vision/vlm/...`). |
| `NAN_SEARXNG_SETTINGS` | Point the built-in `search` tool at a different searxng-cli settings file (replaces `config/searxng.yml`). |
| `HF_ENDPOINT` | HuggingFace endpoint mirror, used by model auto-download. |

`SEARXNG_CLI_SETTINGS` is injected automatically by the built-in search
tool on every invocation; you normally never set it by hand.

## Data and models

- `data/` (gitignored) holds runtime state, e.g. the vision person
  registry at `data/databases/vision/faces.json`.
- `models/` (gitignored) holds model weights. The vision VLM weights
  (`SmolVLM2-500M-Video-Instruct`) are downloaded automatically on first
  use; a failed download degrades that backend to `unavailable` without
  affecting other layers.

All of these locations resolve through
[`backend/nan_itself/utils/paths.py`](../backend/nan_itself/utils/paths.py) and are
therefore independent of the process working directory.

## Tool-level settings

Tool providers may carry their own config files under `config/`, e.g.
`config/searxng.yml` for the built-in search tool (SearXNG outgoing
proxy/timeouts, injected into every searxng-cli invocation).
