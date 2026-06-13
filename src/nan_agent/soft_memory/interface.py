import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from nan_agent.logging.logger import get_logger
from nan_agent.model.peft_manager import AdaptorMeta, PEFTManager
from nan_agent.soft_memory.data_gen import TrainingDataGenerator
from nan_agent.soft_memory.hotswap import HotSwapManager
from nan_agent.soft_memory.screener import ContentScreener, ScreenedContent
from nan_agent.soft_memory.trainer import ORPOTrainer
from nan_agent.soft_memory.triggers import TriggerManager

logger = get_logger(__name__)


class SoftMemory:
    def __init__(
        self,
        cognition,
        hard_memory=None,
        peft_manager: Optional[PEFTManager] = None,
        config: Optional[dict] = None,
    ):
        self.cognition = cognition
        self._hard_memory = hard_memory

        cfg = config or {}

        self._enabled = cfg.get("enabled", True)
        self._peft = peft_manager or PEFTManager()
        self._hotswap = HotSwapManager(self._peft)
        self._triggers = TriggerManager(**cfg.get("triggers", {}))
        self._screener = ContentScreener(cognition)
        self._data_gen = TrainingDataGenerator(cognition)
        self._trainer = ORPOTrainer(cfg.get("lora", {}))

        self._ollama_model = cfg.get("ollama_model", "tinyllama:latest")

    @property
    def trainer(self) -> ORPOTrainer:
        return self._trainer

    async def load_adaptor(self, name: str, path: str, label: str = ""):
        return await self._hotswap.load_adaptor(name, path, label)

    async def activate_adaptor(self, name: str) -> bool:
        return await self._hotswap.activate(name)

    async def deactivate_adaptor(self, name: str):
        await self._hotswap.deactivate(name)

    def get_active_adaptors(self) -> list:
        return self._peft.get_active_adaptors()

    def list_adaptors(self) -> list:
        return self._hotswap.list_all()

    def process_feedback(self, feedback: str):
        self._triggers.trigger_feedback()
        logger.info("feedback_processed", feedback=feedback[:200])

    def track_learning_opportunity(self, context: dict):
        """Record a learning opportunity from the GoT engine for potential future training.

        当 GoT 引擎产生新节点时调用，记录学习机会供后续 soft memory 训练周期使用。

        Args:
            context: 包含学习机会上下文的字典，如 {"new_nodes": 3, "step": 5}
        """
        logger.debug("learning_opportunity_tracked", **context)

    def record_failure(self, task_type: str):
        self._triggers.record_failure(task_type)

    async def screen_content(self, candidates: list[dict]) -> list[ScreenedContent]:
        return await self._screener.screen(candidates)

    async def run_learning_cycle(self, memcell_count: int = 0) -> Optional[str]:
        if not self._enabled:
            logger.debug("run_learning_cycle_disabled")
            return None
        triggers = self._triggers.get_active_triggers(memcell_count)
        if not triggers:
            logger.debug("run_learning_cycle_no_triggers")
            return None

        logger.info(
            "run_learning_cycle_start",
            trigger_count=len(triggers),
            trigger_types=[t.type.value for t in triggers],
        )

        candidates = []
        if self._hard_memory is not None:
            try:
                raw = await self._hard_memory.recollect("", k=200)
                if raw:
                    for item in raw:
                        candidates.append({
                            "id": str(item.get("id", item.get("memcell_id", ""))),
                            "content": str(item.get("content", item.get("summary", ""))),
                            "source": str(item.get("source", "hard_memory")),
                        })
            except Exception as e:
                logger.warning("recollect_failed", error=str(e))

        if not candidates:
            logger.info("run_learning_cycle_no_candidates")
            return None

        screened = await self._screener.screen(candidates)
        if not screened:
            logger.info("run_learning_cycle_no_screened")
            return None

        questions = await self._data_gen.generate_questions(screened)
        if not questions:
            logger.info("run_learning_cycle_no_questions")
            return None

        pairs = await self._data_gen.generate_pairs(questions, screened)
        pairs = self._data_gen.filter_quality(pairs)
        pairs = self._data_gen.augment(pairs)

        valid_count = self._trainer.prepare_training(pairs)
        logger.info("run_learning_cycle_prepared", valid_pairs=valid_count)

        if valid_count < 1:
            return None

        adaptor_path = await self._trainer.train(pairs)

        adaptor_name = f"learned_{int(time.time())}"
        ollama_model_name = f"nan-learned-{int(time.time())}"

        ollama_created = await self._ollama_create(adaptor_path, ollama_model_name)

        if ollama_created:
            await self._hotswap.load_adaptor(
                name=adaptor_name, path=ollama_model_name, label="knowledge",
            )
            logger.info("learning_cycle_registered_ollama", model=ollama_model_name)
        else:
            await self._hotswap.load_adaptor(
                name=adaptor_name, path=adaptor_path, label="knowledge",
            )
            logger.info("learning_cycle_registered_stub", path=adaptor_path)

        return adaptor_path

    async def _ollama_create(self, adapter_dir: str, model_name: str) -> bool:
        adapter_path = Path(adapter_dir)
        model_file = adapter_path / "model.safetensors"
        config_file = adapter_path / "adapter_config.json"

        if not model_file.exists() or not config_file.exists():
            logger.info(
                "ollama_create_skipped_stub",
                reason="no real adapter files, stub training",
                adapter_dir=adapter_dir,
            )
            return False

        modelfile_path = adapter_path / "Modelfile"
        modelfile_content = (
            f"FROM {self._ollama_model}\n"
            f"ADAPTER {adapter_dir}\n"
        )
        modelfile_path.write_text(modelfile_content)

        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "create", model_name,
                "-f", str(modelfile_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                logger.info(
                    "ollama_create_success",
                    model_name=model_name,
                    adapter_dir=adapter_dir,
                )
                return True
            else:
                stderr_text = stderr.decode() if stderr else "unknown error"
                logger.error(
                    "ollama_create_failed",
                    model_name=model_name,
                    returncode=proc.returncode,
                    stderr=stderr_text[:500],
                )
                return False
        except FileNotFoundError:
            logger.error("ollama_not_found")
            return False
        except Exception as e:
            logger.error("ollama_create_exception", error=str(e))
            return False

    def _extract_A_matrices(self, adaptor_path: str) -> dict[str, np.ndarray] | None:
        import safetensors.torch
        model_file = Path(adaptor_path) / "model.safetensors"
        if not model_file.exists():
            return None
        state = safetensors.torch.load_file(str(model_file))
        A_matrices = {}
        for key, tensor in state.items():
            if "lora_A" in key:
                A_matrices[key] = tensor.numpy()
        return A_matrices if A_matrices else None

    async def incremental_train(self, pairs: list[dict], label: str = "") -> Optional[str]:
        if not self._enabled:
            logger.debug("incremental_train_disabled")
            return None
        if not pairs:
            return None

        active = self._peft.get_active_adaptors()
        if active:
            active_name = active[0]
            active_meta = self._peft._adaptors.get(active_name)
            if active_meta:
                A_ref = self._extract_A_matrices(active_meta.path)
                self._trainer.set_orth_reference(A_ref)

        new_path = await self._trainer.train(pairs)

        adaptor_name = f"learned_{int(time.time())}"
        ollama_name = f"nan-learned-{int(time.time())}"
        ollama_ok = await self._ollama_create(new_path, ollama_name)
        path = ollama_name if ollama_ok else new_path
        await self._hotswap.load_adaptor(adaptor_name, path, label)
        await self._hotswap.activate(adaptor_name)

        self._trainer.set_orth_reference(None)
        return new_path