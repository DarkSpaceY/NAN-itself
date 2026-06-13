"""硬记忆（Hard Memory）顶层接口模块。

HardMemory 是硬记忆系统的统一入口。
双存储架构：
- Episodic Store：Episode + Fact（发生了什么）→ 检索用
- Skill Store：Exp（该怎么做）→ 行为指导用
"""

import json
from typing import Optional
import asyncio

from nan_agent.hard_memory.dream_engine import DreamEngine
from nan_agent.hard_memory.memory import Memory
from nan_agent.hard_memory.retrieve import RetrieveEngine
from nan_agent.hard_memory.skill_store import SKILL_COLLECTION, SkillStore
from nan_agent.hard_memory.store import COLLECTION_NAME, Store
from nan_agent.logging.logger import get_logger
from nan_agent.model.cognition import Cognition
from nan_agent.storage.blob_store import BlobStore
from nan_agent.storage.state_store import StateStore
from nan_agent.storage.vector_store import VectorStore

logger = get_logger(__name__)


class HardMemory:
    """硬记忆系统顶层接口。

    双存储架构：
    - Episodic Store (Store): Episode + Fact → 事实性检索
    - Skill Store (SkillStore): Exp → 行为指导

    Attributes:
        _store: Episodic Store — 记忆存储层。
        _skill_store: SkillStore — 技能存储层。
        _retrieval: RetrieveEngine — 混合检索引擎。
        _dream: DreamEngine — 巩固引擎。
        _cognition: 认知模型接口。
        _blob_store: BlobStore — 多模态附件二进制存储。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        ollama_provider,
        cognition: Cognition,
        state_store: Optional[StateStore] = None,
        blob_store: Optional[BlobStore] = None,
    ):
        self._store = Store(ollama_provider, vector_store, cognition)
        self._skill_store = SkillStore(ollama_provider, vector_store)
        self._retrieval = RetrieveEngine(self._store)
        self._dream = DreamEngine(self._store, self._skill_store, cognition)
        self._cognition = cognition
        self._state_store = state_store
        self._blob_store = blob_store
        self._write_lock = asyncio.Lock()

    # ── 生命周期 ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """初始化向量集合并从 ChromaDB 恢复内存索引。"""
        await self._store._vs.ensure_collection(COLLECTION_NAME)
        await self._skill_store._vs.ensure_collection(SKILL_COLLECTION)
        # 从 ChromaDB 恢复 BM25 / 时间图 / 内存索引
        await self._store.load()
        await self._skill_store.load()

    async def close(self) -> None:
        if self._state_store:
            await self._state_store.close()

    # ── 记忆写入 ──────────────────────────────────────────────

    async def add_memcell(
        self,
        episode: str,
        source: str = "task_agent",
        emotional_state: Optional[dict] = None,
        multimodal_attachments: Optional[list] = None,
        timestamp: Optional[str] = None,
    ) -> list[str]:
        """添加一条记忆单元。

        Args:
            episode: 记忆文本内容。
            source: 来源标识。
            emotional_state: 情感状态 {"valence": float, "arousal": float}。
            multimodal_attachments: 多模态附件列表，
                [{"type": "image"|"audio"|"video", "blob_id": "...", "description": "..."}]。
                如果附件中有 data 字段但无 blob_id，会自动存入 BlobStore 并生成 blob_id。
            timestamp: 时间戳。
        """
        async with self._write_lock:
            # 处理多模态附件：自动存入 BlobStore
            processed_attachments = None
            if multimodal_attachments and self._blob_store:
                processed_attachments = []
                for att in multimodal_attachments:
                    if "blob_id" not in att and "data" in att:
                        # 自动存入 BlobStore
                        blob_id = f"mem_{att.get('type', 'blob')}_{id(att)}"
                        data = att.pop("data")
                        if isinstance(data, str):
                            data = data.encode("utf-8")
                        await self._blob_store.put(
                            blob_id, data,
                            metadata={"type": att.get("type", ""), "description": att.get("description", "")}
                        )
                        att["blob_id"] = blob_id
                    processed_attachments.append(att)
            elif multimodal_attachments:
                processed_attachments = multimodal_attachments

            mem = await self._store.add(
                episode,
                source=source,
                session_date=timestamp or "",
                emotional_state=emotional_state,
                attachments=processed_attachments,
            )
        logger.info("memory_stored", id=mem.id, source=source, has_emotion=bool(emotional_state), has_attachments=bool(multimodal_attachments))
        return [mem.id]

    # ── 记忆检索 ──────────────────────────────────────────────

    async def recollect(
        self,
        query: str,
        k: int = 10,
        query_timestamp: str = "",
        temporal_window_days: int = 3,
        secondary: bool = False,
        current_emotion: Optional[dict] = None,
    ) -> list[Memory]:
        """三路混合检索：向量 + BM25 + 时间图 → RRF 融合 + 情绪引导 + 可选二次检索。"""
        result = await self._retrieval.search(
            query,
            top_k=k,
            query_timestamp=query_timestamp,
            temporal_window_days=temporal_window_days,
            secondary=secondary,
            current_emotion=current_emotion,
        )
        if result:
            logger.info("memory_recall", query_preview=query[:80], hits=len(result))
        return result

    # ── 技能检索 ──────────────────────────────────────────────

    async def match_skills(self, situation: str, top_k: int = 5) -> list[Memory]:
        """按情境匹配检索 Exp。

        Args:
            situation: 当前情境描述。
            top_k: 返回的最大 Exp 数。
        """
        return await self._skill_store.search(situation, top_k=top_k)

    # ── 附件检索 ──────────────────────────────────────────────

    async def get_attachment(self, blob_id: str) -> bytes | None:
        """通过 blob_id 获取附件的二进制数据。"""
        if self._blob_store:
            return await self._blob_store.get(blob_id)
        return None

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def state_store(self) -> Optional[StateStore]:
        return self._state_store

    @property
    def total_count(self) -> int:
        return self._store.count

    @property
    def skill_count(self) -> int:
        return self._skill_store.count

    # ── 记忆巩固 ──────────────────────────────────────────────

    async def dream(self, max_items: int = 50) -> dict:
        """执行一次异步记忆巩固（三阶段流水线）。

        Phase 1: Episode → Fact
        Phase 2: Fact → Exp
        Phase 3: Exp lifecycle (prune)
        """
        return await self._dream.dream(max_items=max_items)
