"""记忆存储模块（Store）。

负责：
1. 向量嵌入：qwen3-embedding → ChromaDB
2. BM25 关键词索引
3. 时间图索引：按日期确定性建图，支持时间窗过滤
4. 启动时从 ChromaDB 恢复内存索引（BM25 + 时间图 + _memories）
"""

import json
import re
from typing import Optional

from nan_agent.hard_memory.common import BM25Index, new_id, tokenize
from nan_agent.hard_memory.memory import Memory
from nan_agent.hard_memory.temporal_graph import TemporalGraph
from nan_agent.logging.logger import get_logger
from nan_agent.model.cognition import Cognition
from nan_agent.storage.vector_store import VectorStore

logger = get_logger(__name__)

COLLECTION_NAME = "memories"


def _parse_session_date(session_date: str) -> str:
    """将 LoCoMo 格式的 session_date 转为 ISO 8601 日期。

    "1:56 pm on 8 May, 2023" → "2023-05-08"
    """
    m = re.search(
        r"(\d{1,2})\s+(January|February|March|April|May|June|"
        r"July|August|September|October|November|December),?\s+(\d{4})",
        session_date,
    )
    if m:
        day = int(m.group(1))
        month_str = m.group(2)
        year = int(m.group(3))
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        month = month_map[month_str]
        return f"{year:04d}-{month:02d}-{day:02d}"
    return session_date


class Store:
    """记忆存储层。

    封装 embedding + ChromaDB + BM25 + 时间图。每条记忆独立 embed，不在存储层合并。
    支持启动时从 ChromaDB 恢复所有内存索引。

    Attributes:
        _ollama: Ollama 模型提供者，提供 embed() 方法。
        _vs: ChromaDB VectorStore 实例。
        _bm25: BM25 关键词索引。
        _temporal: 时间图索引。
        _memories: 内存中的 Memory 索引（id → Memory）。
    """

    def __init__(self, ollama_provider, vector_store: VectorStore, cognition: Optional[Cognition] = None):
        self._ollama = ollama_provider
        self._vs = vector_store
        self._bm25 = BM25Index()
        self._temporal = TemporalGraph()
        self._memories: dict[str, Memory] = {}

    async def load(self) -> None:
        """从 ChromaDB 恢复所有内存索引（BM25 + 时间图 + _memories）。

        在 HardMemory.initialize() 中调用，确保重启后检索功能可用。
        """
        try:
            client = self._vs._get_client()
            col = client.get_or_create_collection(name=COLLECTION_NAME)
            count = col.count()
            if count == 0:
                return

            # ChromaDB get() 最多一次取全量
            results = col.get(include=["metadatas", "documents"])
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            for i, mid in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else None
                if content is None:
                    content = meta.get("content", "")
                if not content:
                    continue  # 跳过空内容记录

                # 从 ChromaDB metadata 恢复 Memory 对象
                timestamp = meta.get("timestamp") or ""
                mem = Memory(
                    id=mid,
                    type=meta.get("type", "episode"),
                    content=content,
                    episode_text="",
                    source=meta.get("source", ""),
                    timestamp=timestamp,
                    parent_id=meta.get("parent_id", ""),
                )

                # 恢复 metadata 中的扩展字段
                extra_meta = {}
                if meta.get("emotional_state"):
                    try:
                        extra_meta["emotional_state"] = json.loads(meta["emotional_state"])
                    except (json.JSONDecodeError, TypeError):
                        extra_meta["emotional_state"] = meta["emotional_state"]
                if meta.get("attachments"):
                    try:
                        extra_meta["attachments"] = json.loads(meta["attachments"])
                    except (json.JSONDecodeError, TypeError):
                        extra_meta["attachments"] = meta["attachments"]
                if meta.get("consolidated"):
                    extra_meta["consolidated"] = meta["consolidated"] in ("true", "True", True)
                if extra_meta:
                    mem.metadata = extra_meta

                self._memories[mid] = mem
                self._bm25.add(mid, content)
                if timestamp:
                    self._temporal.add(mid, timestamp)

            logger.info("store_loaded", count=len(self._memories))
        except Exception as e:
            logger.warning("store_load_failed", error=str(e))

    async def add(
        self,
        episode_text: str,
        source: str = "task_agent",
        session_date: str = "",
        emotional_state: Optional[dict] = None,
        attachments: Optional[list] = None,
    ) -> Memory:
        """添加一条记忆，同时写入 ChromaDB / BM25 / 时间图 / 内存索引。

        Args:
            episode_text: 记忆文本。
            source: 来源标识。
            session_date: 会话日期。
            emotional_state: 情感状态，写入 ChromaDB metadata 以持久化。
            attachments: 多模态附件列表，写入 ChromaDB metadata 以持久化。
        """
        content = episode_text[:4000]
        timestamp = _parse_session_date(session_date) if session_date else ""
        embedding = await self._ollama.embed(content)

        mem = Memory(
            id=new_id(),
            type="episode",
            content=content,
            episode_text=episode_text,
            source=source,
            timestamp=timestamp,
        )

        # 构建 ChromaDB metadata（含持久化的扩展字段）
        chroma_meta: dict = {"content": content, "source": source, "type": "episode"}
        if timestamp:
            chroma_meta["timestamp"] = timestamp
        if emotional_state:
            mem.metadata["emotional_state"] = emotional_state
            chroma_meta["emotional_state"] = json.dumps(emotional_state, ensure_ascii=False)
        if attachments:
            mem.metadata["attachments"] = attachments
            chroma_meta["attachments"] = json.dumps(attachments, ensure_ascii=False)

        await self._vs.add(COLLECTION_NAME, mem.id, embedding, metadata=chroma_meta)
        self._bm25.add(mem.id, content)
        self._temporal.add(mem.id, timestamp)
        self._memories[mem.id] = mem

        logger.debug("memory_stored", id=mem.id, source=source)
        return mem

    async def write_fact(self, fact_mem: Memory) -> None:
        """将 DreamEngine 提取的 fact 写入存储（封装内部操作）。

        替代 DreamEngine 直接操作 _vs/_bm25/_temporal/_memories，
        保持封装一致性。

        Args:
            fact_mem: 已构建好的 fact Memory 对象。
        """
        embedding = await self._ollama.embed(fact_mem.content)

        chroma_meta: dict = {
            "content": fact_mem.content,
            "source": "dream",
            "type": "fact",
        }
        if fact_mem.timestamp:
            chroma_meta["timestamp"] = fact_mem.timestamp
        if fact_mem.parent_id:
            chroma_meta["parent_id"] = fact_mem.parent_id

        await self._vs.add(COLLECTION_NAME, fact_mem.id, embedding, metadata=chroma_meta)
        self._bm25.add(fact_mem.id, fact_mem.content)
        self._temporal.add(fact_mem.id, fact_mem.timestamp)
        self._memories[fact_mem.id] = fact_mem

    async def delete(self, mem_id: str) -> None:
        """删除一条记忆，同时清理 ChromaDB / BM25 / 时间图 / 内存索引。"""
        if mem_id not in self._memories:
            return
        self._memories.pop(mem_id, None)
        self._bm25.remove(mem_id)
        # 时间图暂不提供单条删除，影响可忽略
        try:
            await self._vs.delete(COLLECTION_NAME, mem_id)
        except Exception as e:
            logger.warning("store_delete_vs_failed", id=mem_id, error=str(e))

    async def search_vector(self, query: str, top_k: int = 10) -> list[Memory]:
        """纯向量语义搜索。"""
        if not self._memories:
            return []
        query_emb = await self._ollama.embed(query)
        results = await self._vs.search(COLLECTION_NAME, query_emb, top_k=top_k)
        return [self._memories[r["id"]] for r in results if r["id"] in self._memories]

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[Memory, float]]:
        """纯 BM25 关键词搜索，返回 (Memory, score)。"""
        results = self._bm25.search(query, top_k=top_k)
        return [(self._memories[doc_id], score)
                for doc_id, score in results
                if doc_id in self._memories]

    def search_temporal(self, query_timestamp: str, window_days: int = 3, top_k: int = 20) -> list[tuple[str, float]]:
        """时间图搜索：返回查询时间 ±window_days 内的记忆，按时间接近度排序。

        Returns:
            [(mem_id, score)] 按分数降序。
        """
        ids = self._temporal.search_by_date(query_timestamp, window_days=window_days)
        if not ids:
            return []
        scored = self._temporal.rank_by_temporal_proximity(query_timestamp, ids)
        return scored[:top_k]

    def expand_temporal(self, seed_ids: list[str], window_days: int = 1) -> list[str]:
        """时间图扩展：从种子记忆的时间戳出发，扩展同时段记忆。"""
        return self._temporal.expand(seed_ids, window_days=window_days)

    def get(self, mem_id: str) -> Memory | None:
        return self._memories.get(mem_id)

    def get_all(self) -> list[Memory]:
        return list(self._memories.values())

    @property
    def count(self) -> int:
        return len(self._memories)
