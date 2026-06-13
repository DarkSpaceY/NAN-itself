from abc import ABC, abstractmethod
from typing import Any


class BaseStore(ABC):
    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...