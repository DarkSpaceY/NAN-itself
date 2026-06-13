from pathlib import Path
from typing import List, Optional

from nan_agent.logging.logger import get_logger
from nan_agent.model.peft_manager import AdaptorMeta, PEFTManager

logger = get_logger(__name__)


class HotSwapManager:
    def __init__(self, peft_manager: PEFTManager):
        self._peft_manager = peft_manager
        logger.info("hotswap_manager_initialized")

    async def load_adaptor(
        self,
        name: str,
        path: str,
        label: str = "",
    ) -> Optional[AdaptorMeta]:
        is_ollama_model = "/" not in path and "\\" not in path
        resolved_path = path

        if is_ollama_model:
            resolved_path = path
        else:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(
                    "adaptor_path_not_found",
                    name=name,
                    path=path,
                )
                return None
            resolved_path = str(path_obj.resolve())

        meta = AdaptorMeta(
            name=name,
            label=label,
            path=resolved_path,
        )
        await self._peft_manager.register_adaptor(meta)
        logger.info(
            "adaptor_loaded",
            name=name,
            label=label,
            path=resolved_path,
        )
        return meta

    async def activate(self, name: str) -> bool:
        try:
            await self._peft_manager.activate_adaptor(name)
            return True
        except Exception:
            logger.warning("activate_failed", name=name, exc_info=True)
            return False

    async def deactivate(self, name: str) -> None:
        await self._peft_manager.deactivate_adaptor(name)

    async def unload(self, name: str) -> None:
        active_names = self._peft_manager.get_active_adaptors()
        if name in active_names:
            await self._peft_manager.deactivate_adaptor(name)
        await self._peft_manager.remove_adaptor(name)
        logger.info("adaptor_unloaded", name=name)

    def list_active(self) -> List[AdaptorMeta]:
        active_names = self._peft_manager.get_active_adaptors()
        all_adaptors = self._peft_manager.list_adaptors()
        return [m for m in all_adaptors if m.name in active_names]

    def list_all(self) -> List[AdaptorMeta]:
        return self._peft_manager.list_adaptors()

    async def a_b_test(self, new_name: str, old_name: Optional[str] = None) -> bool:
        all_adaptors = self._peft_manager.list_adaptors()
        new_adaptor = next(
            (m for m in all_adaptors if m.name == new_name), None
        )
        if new_adaptor is None:
            logger.warning(
                "a_b_test_adaptor_not_found",
                new_name=new_name,
            )
            return False

        await self._peft_manager.activate_adaptor(new_name)
        logger.info(
            "a_b_test_activated",
            new_name=new_name,
            old_name=old_name,
        )
        return True

    async def rollback(self, current_name: str, fallback_name: str) -> None:
        await self._peft_manager.deactivate_adaptor(current_name)
        await self._peft_manager.activate_adaptor(fallback_name)
        logger.info(
            "rollback_completed",
            deactivated=current_name,
            activated=fallback_name,
        )