from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from nan_agent.model.types import MultiModalInput, MultiModalOutput


@dataclass
class InferenceRequest:
    input: MultiModalInput
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 16384
    stream: bool = True
    stop: list[str] = field(default_factory=list)


class ModelProvider(ABC):
    @abstractmethod
    async def infer(self, request: InferenceRequest) -> MultiModalOutput:
        ...

    @abstractmethod
    def infer_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
