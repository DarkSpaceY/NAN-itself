import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from nan_agent.exceptions import ModelError
from nan_agent.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AdaptorMeta:
    name: str
    path: str
    label: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activation_count: int = 0
    base_model: str = ""
    rank: int = 8
    alpha: int = 16
    size_mb: float = 0.0
    merged_from: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ModelError(
                "adaptor name must not be empty",
                error_code="E200",
            )
        if not self.path or not self.path.strip():
            raise ModelError(
                f"adaptor path must not be empty for '{self.name}'",
                error_code="E200",
            )
        if self.rank < 1:
            raise ModelError(
                f"adaptor rank must be >= 1 for '{self.name}', got {self.rank}",
                error_code="E200",
            )
        if self.alpha < 1:
            raise ModelError(
                f"adaptor alpha must be >= 1 for '{self.name}', got {self.alpha}",
                error_code="E200",
            )
        if self.size_mb < 0:
            raise ModelError(
                f"adaptor size_mb must be >= 0 for '{self.name}', got {self.size_mb}",
                error_code="E200",
            )


@dataclass
class _ActiveEntry:
    adaptor_name: str


class PEFTManager:
    def __init__(
        self,
        model: Optional[object] = None,
        max_active: int = 4,
    ):
        self._model = model
        self._max_active = max_active
        self._adaptors: dict[str, AdaptorMeta] = {}
        self._active: OrderedDict[str, _ActiveEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        logger.info(
            "peft_manager_initialized",
            max_active=max_active,
            has_model=model is not None,
        )

    async def register_adaptor(self, meta: AdaptorMeta) -> None:
        async with self._lock:
            if meta.name in self._adaptors:
                raise ModelError(
                    f"adaptor '{meta.name}' is already registered",
                    error_code="E200",
                    details={"name": meta.name},
                )
            self._adaptors[meta.name] = meta
            logger.info(
                "adaptor_registered",
                name=meta.name,
                label=meta.label,
                rank=meta.rank,
                size_mb=meta.size_mb,
            )

    async def activate_adaptor(self, name: str) -> None:
        async with self._lock:
            if name not in self._adaptors:
                raise ModelError(
                    f"cannot activate unknown adaptor '{name}'",
                    error_code="E200",
                    details={"name": name},
                )
            if name in self._active:
                self._active.move_to_end(name)
                self._adaptors[name] = AdaptorMeta(
                    name=self._adaptors[name].name,
                    path=self._adaptors[name].path,
                    label=self._adaptors[name].label,
                    version=self._adaptors[name].version,
                    created_at=self._adaptors[name].created_at,
                    activation_count=self._adaptors[name].activation_count + 1,
                    base_model=self._adaptors[name].base_model,
                    rank=self._adaptors[name].rank,
                    alpha=self._adaptors[name].alpha,
                    size_mb=self._adaptors[name].size_mb,
                    merged_from=list(self._adaptors[name].merged_from),
                )
                logger.debug("adaptor_already_active", name=name)
                return

            while len(self._active) >= self._max_active:
                evicted_name, _ = self._active.popitem(last=False)
                await self._do_deactivate(evicted_name)
                logger.info(
                    "lru_evicted",
                    evicted=evicted_name,
                    reason="max_active_limit",
                )

            await self._do_activate(name)
            self._active[name] = _ActiveEntry(
                adaptor_name=name,
            )
            self._adaptors[name] = AdaptorMeta(
                name=self._adaptors[name].name,
                path=self._adaptors[name].path,
                label=self._adaptors[name].label,
                version=self._adaptors[name].version,
                created_at=self._adaptors[name].created_at,
                activation_count=self._adaptors[name].activation_count + 1,
                base_model=self._adaptors[name].base_model,
                rank=self._adaptors[name].rank,
                alpha=self._adaptors[name].alpha,
                size_mb=self._adaptors[name].size_mb,
                merged_from=list(self._adaptors[name].merged_from),
            )

    async def deactivate_adaptor(self, name: str) -> None:
        async with self._lock:
            if name not in self._active:
                raise ModelError(
                    f"adaptor '{name}' is not active",
                    error_code="E200",
                    details={"name": name},
                )
            await self._do_deactivate(name)
            del self._active[name]

    async def remove_adaptor(self, name: str) -> None:
        async with self._lock:
            if name not in self._adaptors:
                raise ModelError(
                    f"cannot remove unknown adaptor '{name}'",
                    error_code="E200",
                    details={"name": name},
                )
            if name in self._active:
                await self._do_deactivate(name)
                del self._active[name]
            del self._adaptors[name]
            logger.info("adaptor_removed", name=name)

    def list_adaptors(self) -> list[AdaptorMeta]:
        return list(self._adaptors.values())

    def get_active_adaptors(self) -> List[str]:
        return list(self._active.keys())

    async def merge_adaptors(self, names: List[str]) -> object:
        if len(names) < 2:
            raise ModelError(
                "merge_adaptors requires at least 2 adaptor names",
                error_code="E200",
                details={"names": names},
            )

        unknown = [n for n in names if n not in self._adaptors]
        if unknown:
            raise ModelError(
                f"unknown adaptors: {', '.join(unknown)}",
                error_code="E200",
                details={"unknown": unknown},
            )

        if self._model is None:
            raise ModelError(
                "no base model set for merge_adaptors",
                error_code="E200",
            )

        try:
            from peft import PeftModel
        except ImportError:
            raise ModelError(
                "peft library is required for merge_adaptors but not installed",
                error_code="E200",
            )

        merged_model = self._model

        async with self._lock:
            for name in names:
                adaptor = self._adaptors[name]
                merged_model = PeftModel.from_pretrained(
                    merged_model,
                    adaptor.path,
                    adapter_name=name,
                )

            for i in range(1, len(names)):
                merged_model = merged_model.merge_and_unload(
                    adapter_names=names[: i + 1],
                )

        logger.info(
            "adaptors_merged",
            names=names,
            count=len(names),
        )
        return merged_model

    async def merge_and_save(
        self,
        names: List[str],
        output_name: str,
        output_path: str,
    ) -> AdaptorMeta:
        merged_model = await self.merge_adaptors(names)

        try:
            from peft import PeftModel, LoraConfig, get_peft_model
        except ImportError:
            raise ModelError(
                "peft library is required for merge_and_save but not installed",
                error_code="E200",
            )

        sample_adaptor = self._adaptors[names[0]]
        lora_config = LoraConfig(
            r=sample_adaptor.rank,
            lora_alpha=sample_adaptor.alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        peft_model = get_peft_model(merged_model, lora_config)
        peft_model.save_pretrained(output_path)

        meta = AdaptorMeta(
            name=output_name,
            path=output_path,
            label=sample_adaptor.label,
            version="1.0.0",
            base_model=sample_adaptor.base_model,
            rank=sample_adaptor.rank,
            alpha=sample_adaptor.alpha,
            size_mb=sample_adaptor.size_mb,
            merged_from=list(names),
        )

        async with self._lock:
            if output_name in self._adaptors:
                raise ModelError(
                    f"adaptor '{output_name}' already exists",
                    error_code="E200",
                    details={"name": output_name},
                )
            self._adaptors[output_name] = meta

        logger.info(
            "adaptor_merged_and_saved",
            output_name=output_name,
            output_path=output_path,
            source_names=names,
        )
        return meta

    async def _do_activate(self, name: str) -> None:
        adaptor = self._adaptors[name]
        if self._model is not None:
            try:
                from peft import PeftModel
            except ImportError:
                logger.warning(
                    "peft_not_installed_skip_load",
                    name=name,
                )
                return

            try:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adaptor.path,
                    adapter_name=name,
                )
                self._model.set_adapter(name)
                logger.info(
                    "adaptor_activated",
                    name=name,
                    path=adaptor.path,
                )
            except Exception as e:
                raise ModelError(
                    f"failed to activate adaptor '{name}': {e}",
                    error_code="E200",
                    details={"name": name, "error": str(e)},
                )

    async def _do_deactivate(self, name: str) -> None:
        adaptor = self._adaptors[name]
        if self._model is not None:
            try:
                from peft import PeftModel
            except ImportError:
                logger.warning(
                    "peft_not_installed_skip_unload",
                    name=name,
                )
                return

            try:
                if hasattr(self._model, "delete_adapter"):
                    self._model.delete_adapter(name)
                logger.info(
                    "adaptor_deactivated",
                    name=name,
                )
            except Exception as e:
                logger.warning(
                    "adaptor_deactivate_failed",
                    name=name,
                    error=str(e),
                )

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def max_active(self) -> int:
        return self._max_active