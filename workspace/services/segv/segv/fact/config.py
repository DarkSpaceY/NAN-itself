"""Fact layer knobs.

Defaults are the frozen SEGV v2 values; nothing here is an experiment
switch. A key that changes retrieval behaviour lives in this file so the
service config only carries where the models are.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .prompts import CUE_PROMPT, CUE_SYSTEM


class FactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SlmConfig(FactModel):
    """The cue model, reached through the OpenAI chat-completions API."""

    model: str = "qwen3.5:4b-mlx"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 220
    timeout: float = 300.0
    attempts: int = 4
    # qwen3.5 reasons by default; the cue contract wants one sentence, not a
    # thinking trace. Ollama maps "none" onto think=false.
    reasoning_effort: str | None = "none"


class EmbedConfig(FactModel):
    """EmbeddingGemma through the OpenAI embeddings API."""

    model: str = "embeddinggemma:300m"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    dim: int = 768
    batch_size: int = 32
    timeout: float = 600.0
    attempts: int = 4
    # v2 prefixes: the model was selected and measured with these exact bytes.
    doc_prefix: str = "title: none | text: "
    query_prefix: str = "task: search result | query: "


class BM25Config(FactModel):
    k1: float = 1.5
    b: float = 0.0
    tokenizer: str = "stop"  # "plain" or "stop"


class RerankConfig(FactModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    device: str = "auto"
    fp16: bool = True
    batch_size: int = 16
    max_length: int = 512
    candidates: int = 100
    weight: float = 0.75


class FusionConfig(FactModel):
    k: int = 2
    pool: int = 100
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "raw_bm25": 0.5,
            "raw_dense": 1.0,
            "event_dense": 0.75,
            "event_bm25": 0.25,
        }
    )
    order: list[str] = Field(
        default_factory=lambda: [
            "raw_bm25",
            "raw_dense",
            "event_dense",
            "event_bm25",
        ]
    )


class CueConfig(FactModel):
    system: str = CUE_SYSTEM
    prompt: str = CUE_PROMPT
    ctx_chars: int = 256
    workers: int = 2
    fallback_on_error: bool = True


class ChunkConfig(FactModel):
    """One chunk is one fact: two texts, two vectors, one turn pointer."""

    # 512 keeps a Chinese chunk inside the CE's 512-token window and avoids
    # diluting the embedding with unrelated messages.
    max_chars: int = 512
    ctx_chars: int = 256
    include_tool_calls: bool = True


class FactConfig(FactModel):
    enabled: bool = True
    # Relative to the L0 store directory: <data>/<fact.data_dir>.
    data_dir: str = "fact"
    hnsw_connectivity: int = 16
    slm: SlmConfig = Field(default_factory=SlmConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    cue: CueConfig = Field(default_factory=CueConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    raw_bm25: BM25Config = Field(default_factory=BM25Config)
    event_bm25: BM25Config = Field(default_factory=BM25Config)
