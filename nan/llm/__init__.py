"""
LLM boundary package: protocol vocabulary + provider adapter.

`nan.llm.Message` etc. keep working -- everything is re-exported.
Import discipline: any domain may import the TYPES; only
nan.agent.engine may call `LLMProvider` (the transport edge).
"""

from .provider import LLMProvider
from .types import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    Message,
    ProviderName,
    ToolCall,
    ToolDefinition,
    Usage,
)

__all__ = [
    "FinishReason",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamEvent",
    "Message",
    "ProviderName",
    "ToolCall",
    "ToolDefinition",
    "Usage",
]
