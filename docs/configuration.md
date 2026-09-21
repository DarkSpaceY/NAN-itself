# Configuration

Configuration lives in two layers, and there are no environment
variables to set:

- `config/settings.yaml` — core framework parameters only. Modules
  never appear here.
- `config/modules/<module_id>.yaml` — one file per module, loaded and
  validated by the module itself.

## settings.yaml

Core settings only (LLM, gateway, agent, module/provider machinery,
events). The file ships with every field written out; values may be
edited in place, and code defaults back anything you delete.

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
  history_char_limit: 100000

skills:
  resource_char_limit: 100000
  script_timeout: 300.0

runtime:
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

## Module configuration

Each module owns exactly one YAML file:
`config/modules/<module_id>.yaml`. The module declares a pydantic
config model whose defaults are the shipped constants; the YAML
(when present) overrides them and is validated at construction
time — a malformed file is a loud startup failure, never a silent
half default. Modules read their config in `__init__`, so a hot
reload (instance rebuild) picks up edits automatically.

Shipped files:

| File | Module | Fields |
|---|---|---|
| `config/modules/audio.yaml` | audio | mic `device`, speaker/tagger/emotion model paths, voices registry |
| `config/modules/voice.yaml` | voice | STT model/language, SLM weights + repo, CosyVoice checkout/weights/reference wav, `tts_speed`, `fast_path_enabled`, diarization (pyannote repo/token/threshold, speaker registry) |
| `config/modules/vision.yaml` | vision | camera `device`, faces/OCR/VLM toggles, registry + weights paths |

Path convention: relative paths resolve against the repository
root; absolute paths (and `~/...`) pass through unchanged.

## Data and models

- `data/` (gitignored) holds runtime state, e.g. the vision person
  registry at `data/databases/vision/faces.json`.
- `models/` (gitignored) holds model weights. The vision VLM weights
  (`SmolVLM2-500M-Video-Instruct`) are downloaded automatically on first
  use; a failed download crashes the module at startup and it retries
  with backoff until the weights are present.

All of these locations are fixed derivatives of the repository root
([`backend/nan_itself/utils/paths.py`](../backend/nan_itself/utils/paths.py))
and are therefore independent of the process working directory.
`HF_ENDPOINT` is honoured by the `huggingface_hub` library itself for
download mirrors; NAN sets no environment variables of its own.

## Tool-level settings

Tool providers may carry their own config files under `config/`, e.g.
`config/searxng.yml` for the built-in search tool (SearXNG outgoing
proxy/timeouts). The tool injects its settings file into every
searxng-cli invocation via `SEARXNG_CLI_SETTINGS`; you never set that
variable by hand.
